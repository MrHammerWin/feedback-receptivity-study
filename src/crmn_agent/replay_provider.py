"""Exact-prefix replay: logical inference is charged, subscription launches are not.

The cached prefix is immutable experimental evidence, not a response cache whose
misses can silently trigger a model call. Only explicit prefix exhaustion permits
new calls through the already authorized subscription provider.
"""

import hashlib
import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path

from pydantic import Field

from .codex_provider import CodexProvider
from .schemas import Schema


class ReplayPrefixError(RuntimeError):
    """Stop the run on incompatible or ambiguous historical evidence."""


class ReplayRow(Schema):
    output: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    inference_seconds: float = Field(ge=0, allow_inf_nan=False)
    backend_usage: dict | None = None
    subscription_launch: int = Field(gt=0)


class ReplayPrefix(Schema):
    schema_version: str = "codex-replay-prefix-v1"
    model: str
    reasoning_effort: str
    source_log_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: list[ReplayRow]
    source_log: str | None = None
    source_run_hash: str | None = None
    source_episode_hash: str | None = None
    episode_key: str | None = None
    variant: str | None = None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ReplayCodexProvider(CodexProvider):
    def __init__(self, config: dict, call_log: Path | None = None):
        super().__init__(config, call_log)
        self._replay_rows: list[ReplayRow] = []
        self._replay_position = 0
        self.replay_prefix_path: Path | None = None
        self.last_replayed = False
        self.last_replay_seconds = 0.0
        self.replayed_seconds = 0.0

    @property
    def replay_remaining(self) -> int:
        return len(self._replay_rows) - self._replay_position

    def assert_replay_exhausted(self) -> None:
        if self.replay_remaining:
            raise ReplayPrefixError(
                f"Historical prefix has {self.replay_remaining} unconsumed rows; do not discard an outcome"
            )

    def set_replay_prefix(self, path: Path | str | None) -> None:
        """Load a task's exact prefix, rejecting unread rows from the previous task."""
        self.assert_replay_exhausted()
        if path is None:
            self._replay_rows = []
            self._replay_position = 0
            self.replay_prefix_path = None
            return
        path = Path(path).resolve()
        prefix = ReplayPrefix.model_validate_json(path.read_text())
        if prefix.schema_version != "codex-replay-prefix-v1":
            raise ReplayPrefixError("Unsupported replay prefix version")
        if prefix.model != self.model:
            raise ReplayPrefixError("Replay model differs from the configured model")
        if prefix.reasoning_effort != self.config.get("reasoning_effort", "low"):
            raise ReplayPrefixError("Replay reasoning setting differs from the configured setting")
        if prefix.source_log is not None:
            source = Path(prefix.source_log)
            source = source if source.is_absolute() else path.parent / source
            if not source.is_file() or _sha256(source.read_bytes()) != prefix.source_log_hash:
                raise ReplayPrefixError("Original source log is missing or its hash changed")
        self._replay_rows = prefix.rows
        self._replay_position = 0
        self.replay_prefix_path = path

    def _generate(self, system, user, schema, cap):
        self.last_replayed = False
        self.last_replay_seconds = 0.0
        self.last_launch = None
        self.last_backend_usage = None
        if self.replay_remaining:
            row = self._replay_rows[self._replay_position]
            prompt_hash = _sha256((system + user).encode())
            if prompt_hash != row.prompt_sha256:
                raise ReplayPrefixError(
                    f"Exact replay prompt mismatch at cached row {self._replay_position + 1}; no model call made"
                )
            self._replay_position += 1
            self.last_replayed = True
            self.last_replay_seconds = row.inference_seconds
            self.replayed_seconds += row.inference_seconds
            self.last_launch = row.subscription_launch
            self.last_backend_usage = deepcopy(row.backend_usage)
            return row.output, row.input_tokens, row.output_tokens
        return super()._generate(system, user, schema, cap)


def _integer(value, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReplayPrefixError(f"Expected a nonnegative integer for {field}")
    return value


def prepare_replay_prefixes(
    oldrun_dir: Path | str,
    ordered_tasks: list[dict | str],
    output_dir: Path | str,
) -> dict[str, str]:
    """Partition a frozen, single-process log using ordered completed episodes.

    ``ordered_tasks`` must give the complete historical task order, not a selected
    successful subset. Variants follow the original run configuration. Provider
    error rows, cumulative-counter resets, missing episodes, incomplete episodes,
    and usage mismatches stop preparation before any prefix files are emitted.
    A valid model response rejected later by an agent's action policy is reusable;
    this is distinct from replaying a failed provider request or hiding its retry.
    """
    oldrun_dir = Path(oldrun_dir).resolve()
    output_dir = Path(output_dir).resolve()
    run_path = oldrun_dir / "run.json"
    log_path = oldrun_dir / "model_calls.jsonl"
    run_raw, log_raw = run_path.read_bytes(), log_path.read_bytes()
    run = json.loads(run_raw)
    config = run["config"]
    model = config["provider"]["model"]
    reasoning = config["provider"].get("reasoning_effort", "low")
    variants = config["variants"]
    if not variants or len(set(variants)) != len(variants):
        raise ReplayPrefixError("Original run must list distinct ordered variants")
    keys = [task["key"] if isinstance(task, dict) else task for task in ordered_tasks]
    if not keys or len(set(keys)) != len(keys):
        raise ReplayPrefixError("Supply distinct task keys in their complete original order")
    if any(not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", key) for key in keys):
        raise ReplayPrefixError("Task keys must be safe episode basenames")
    keys = [key for key in keys if key in config.get("task_keys", keys)]
    rows = []
    previous_inputs = previous_outputs = 0
    previous_seconds = 0.0
    for number, line in enumerate(log_raw.decode().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise ReplayPrefixError(f"Incomplete or malformed historical call-log line {number}") from error
        if record.get("error_type") is not None:
            raise ReplayPrefixError(f"Historical provider error at line {number}; no implicit retries allowed")
        if record.get("replayed") or record.get("last_replayed"):
            raise ReplayPrefixError("Prepare from original subscription calls, not a nested replay log")
        if record.get("model") != model:
            raise ReplayPrefixError("Historical log mixes model identities")
        usage = record["usage_cumulative"]
        if _integer(usage["calls"], "cumulative calls") != len(rows) + 1:
            raise ReplayPrefixError("Historical cumulative calls reset or skip; slicing would be ambiguous")
        cumulative_inputs = _integer(usage["input_tokens"], "cumulative input tokens")
        cumulative_outputs = _integer(usage["output_tokens"], "cumulative output tokens")
        inputs, outputs = cumulative_inputs - previous_inputs, cumulative_outputs - previous_outputs
        if min(inputs, outputs) < 0:
            raise ReplayPrefixError("Historical token counters decreased")
        previous_inputs, previous_outputs = cumulative_inputs, cumulative_outputs
        cumulative_seconds = float(usage["seconds"])
        elapsed = cumulative_seconds - previous_seconds
        if not math.isfinite(cumulative_seconds) or elapsed < 0:
            raise ReplayPrefixError("Historical inference-time counters decreased or are invalid")
        previous_seconds = cumulative_seconds
        raw_output = record.get("output", "")
        try:
            parsed = json.loads(raw_output)
        except (TypeError, ValueError) as error:
            raise ReplayPrefixError("Historical successful call has no valid JSON response") from error
        if not isinstance(parsed, dict):
            raise ReplayPrefixError("Historical response is not a structured JSON object")
        backend_usage = record.get("backend_usage")
        if backend_usage is not None and any(
            backend_usage.get(field) != value
            for field, value in (("input_tokens", inputs), ("output_tokens", outputs))
        ):
            raise ReplayPrefixError("Historical backend usage disagrees with cumulative token deltas")
        rows.append(ReplayRow(
            output=raw_output, prompt_sha256=record["prompt_sha256"],
            input_tokens=inputs, output_tokens=outputs, backend_usage=backend_usage,
            inference_seconds=elapsed,
            subscription_launch=record["subscription_launch"],
        ))
    prepared = {}
    cursor = 0
    missing_episode = False
    for key in keys:
        for variant in variants:
            episode_path = oldrun_dir / "episodes" / f"{key}--{variant}.json"
            if not episode_path.is_file():
                missing_episode = True
                continue
            if missing_episode:
                raise ReplayPrefixError("A later episode exists after a missing historical predecessor")
            raw = episode_path.read_bytes()
            episode = json.loads(raw)
            if not episode.get("finished"):
                raise ReplayPrefixError("An unfinished historical episode cannot be sliced safely")
            if (episode.get("key") != key or episode.get("variant") != variant
                    or episode.get("model") != model):
                raise ReplayPrefixError("Episode identity does not match its historical position")
            usage = episode["usage"]
            calls = _integer(usage["calls"], "episode calls")
            selected = rows[cursor:cursor + calls]
            if len(selected) != calls:
                raise ReplayPrefixError("Episode usage extends beyond the historical call log")
            for field in ("input_tokens", "output_tokens"):
                if sum(getattr(row, field) for row in selected) != _integer(usage[field], "episode " + field):
                    raise ReplayPrefixError("Episode token usage does not match its adjacent call-log slice")
            historical_seconds = sum(row.inference_seconds for row in selected)
            if not math.isclose(historical_seconds, float(usage["seconds"]), rel_tol=1e-9, abs_tol=1e-9):
                raise ReplayPrefixError("Episode inference time does not match its adjacent call-log slice")
            name = f"{key}--{variant}"
            prepared[name] = ReplayPrefix(
                model=model, reasoning_effort=reasoning, source_log_hash=_sha256(log_raw),
                source_log=os.path.relpath(log_path, output_dir),
                source_run_hash=_sha256(run_raw), source_episode_hash=_sha256(raw),
                episode_key=key, variant=variant, rows=selected,
            )
            cursor += calls
    if cursor != len(rows):
        raise ReplayPrefixError("Historical calls remain outside completed ordered episode slices")
    if not prepared:
        raise ReplayPrefixError("No completed episode prefixes were found")
    if log_path.read_bytes() != log_raw or run_path.read_bytes() != run_raw:
        raise ReplayPrefixError("Historical source changed while preparing replay evidence")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReplayPrefixError("Refusing to overwrite existing replay evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for name, prefix in prepared.items():
        path = output_dir / f"{name}.json"
        path.write_text(prefix.model_dump_json(indent=2) + "\n")
        mapping[name] = str(path)
    (output_dir / "index.json").write_text(json.dumps({
        "schema_version": "codex-replay-index-v1", "model": model,
        "reasoning_effort": reasoning, "source_log_hash": _sha256(log_raw),
        "source_run_hash": _sha256(run_raw), "logical_calls": len(rows),
        "prefixes": {name: Path(path).name for name, path in mapping.items()},
    }, indent=2) + "\n")
    return mapping
