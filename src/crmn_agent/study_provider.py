"""Private, scope-bounded replay for checkpoint resume and paired QA controls.

The sequence cache is an exact request prefix. The shared cache couples identical
requests inside one explicitly frozen QA case/condition. Neither mode creates a
ledger, changes authorization, repairs invalid JSON, or retries failed requests.
"""

import fcntl
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Literal

from pydantic import Field

from .codex_provider import CAMPAIGN, CodexProvider
from .schemas import Schema


class StudyCacheError(RuntimeError):
    """An incompatible or incomplete cache must stop the study, not trigger a miss."""


class CacheIdentity(Schema):
    model: str
    reasoning_effort: str
    task_kind: str = Field(min_length=1)
    source_identity: str = Field(min_length=1)


class ScopeMetadata(Schema):
    schema_version: Literal["scoped-codex-cache-v1"] = "scoped-codex-cache-v1"
    scope: str = Field(min_length=1)
    mode: Literal["sequence", "shared"]
    identity: CacheIdentity


class CachedCompletion(Schema):
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    inference_seconds: float = Field(ge=0, allow_inf_nan=False)
    backend_usage: dict | None
    original_campaign: str
    subscription_launch: int = Field(gt=0)


class ScopeRecords(Schema):
    schema_version: Literal["scoped-codex-records-v1"] = "scoped-codex-records-v1"
    entries: list[CachedCompletion]


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".pending-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ScopedCodexProvider(CodexProvider):
    """One explicitly configured private cache root; one active scope at a time."""

    def __init__(self, config: dict, call_log: Path | None = None):
        for key in ("study_source_identity", "task_kind"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"A nonempty {key} is required for study cache isolation")
        super().__init__(config, call_log)
        self.cache_root = Path(config.get("study_cache_dir", "local/cache")).resolve()
        self._metadata: ScopeMetadata | None = None
        self._scope_dir: Path | None = None
        self._position = 0
        self.last_replayed = False
        self.last_replay_seconds = 0.0
        self.replayed_seconds = 0.0
        self.last_original_campaign: str | None = None

    def _identity(self) -> CacheIdentity:
        return CacheIdentity(
            model=self.model,
            reasoning_effort=self.config.get("reasoning_effort", "low"),
            task_kind=self.config["task_kind"],
            source_identity=self.config["study_source_identity"],
        )

    @contextmanager
    def _locked(self, directory: Path):
        with (directory / ".lock").open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def _read_locked(self, expected: ScopeMetadata) -> ScopeRecords:
        metadata_path = self._scope_dir / "metadata.json"
        records_path = self._scope_dir / "records.json"
        if not metadata_path.is_file() or not records_path.is_file():
            raise StudyCacheError("Incomplete scope cache; no implicit regeneration is allowed")
        try:
            actual = ScopeMetadata.model_validate_json(metadata_path.read_text())
            records = ScopeRecords.model_validate_json(records_path.read_text())
        except (OSError, ValueError) as error:
            raise StudyCacheError("Malformed scope cache; no model call was made") from error
        if actual != expected:
            raise StudyCacheError("Scope, mode or frozen configuration identity differs from the cache")
        if actual.identity != self._identity():
            raise StudyCacheError("Provider identity changed after the scope began")
        if actual.mode == "shared":
            keys = [entry.request_sha256 for entry in records.entries]
            if len(keys) != len(set(keys)):
                raise StudyCacheError("Shared cache contains duplicate request identities")
        return records

    def begin_scope(self, scope: str, mode: Literal["sequence", "shared"] = "sequence") -> None:
        """Sequence: frozen episode; shared: one frozen QA case AND condition.

        The caller must include both QA case and condition in a shared scope ID.
        These identifiers are never added to model input. A different string
        creates a disjoint namespace; semantic correctness of caller IDs remains
        a runner responsibility.
        """
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("A nonempty explicit scope is required")
        if mode not in {"sequence", "shared"}:
            raise ValueError("Unknown study cache mode")
        if mode == "shared" and self.config["task_kind"] != "qa":
            raise ValueError("Shared coupling is permitted only for task_kind='qa'")
        self.assert_scope_exhausted()
        expected = ScopeMetadata(scope=scope, mode=mode, identity=self._identity())
        directory = self.cache_root / _digest(scope.encode())
        directory.mkdir(parents=True, exist_ok=True)
        previous_directory, previous_metadata = self._scope_dir, self._metadata
        self._scope_dir = directory
        try:
            with self._locked(directory):
                present = [(directory / name).exists() for name in ("metadata.json", "records.json")]
                if not any(present):
                    _atomic_json(directory / "records.json", ScopeRecords(entries=[]).model_dump())
                    _atomic_json(directory / "metadata.json", expected.model_dump())
                self._read_locked(expected)
        except Exception:
            self._scope_dir, self._metadata = previous_directory, previous_metadata
            raise
        self._metadata = expected
        self._position = 0

    @property
    def replay_remaining(self) -> int:
        if self._metadata is None or self._metadata.mode == "shared":
            return 0
        with self._locked(self._scope_dir):
            return len(self._read_locked(self._metadata).entries) - self._position

    def assert_scope_exhausted(self) -> None:
        if self.replay_remaining:
            raise StudyCacheError("Unconsumed sequence responses cannot be discarded by ending or switching scopes")

    def assert_replay_exhausted(self) -> None:
        self.assert_scope_exhausted()

    def _request(self, system, user, schema) -> tuple[str, str, str]:
        schema_hash = _digest(_canonical(schema.model_json_schema()))
        prompt_hash = _digest((system + user).encode())
        key = _digest(_canonical({
            "system": system, "user": user, "schema_sha256": schema_hash,
            "identity": self._identity().model_dump(),
        }))
        return key, prompt_hash, schema_hash

    def _generate(self, system, user, schema, cap):
        self.last_replayed = False
        self.last_replay_seconds = 0.0
        self.last_original_campaign = None
        self.last_launch = None
        self.last_backend_usage = None
        if self._metadata is None:
            raise StudyCacheError("begin_scope must be called before a study request")
        key, prompt_hash, schema_hash = self._request(system, user, schema)
        with self._locked(self._scope_dir):
            records = self._read_locked(self._metadata)
            cached = None
            if self._metadata.mode == "sequence":
                if self._position < len(records.entries):
                    cached = records.entries[self._position]
                    if cached.request_sha256 != key:
                        raise StudyCacheError("Exact sequence-prefix request mismatch; no model call was made")
                elif self._position != len(records.entries):
                    raise StudyCacheError("Sequence cache unexpectedly lost an existing response")
            else:
                cached = next((entry for entry in records.entries if entry.request_sha256 == key), None)
            if cached is not None:
                self.last_replayed = True
                self.last_replay_seconds = cached.inference_seconds
                self.replayed_seconds += cached.inference_seconds
                self.last_launch = cached.subscription_launch
                self.last_backend_usage = deepcopy(cached.backend_usage)
                self.last_original_campaign = cached.original_campaign
                self._position += 1
                return cached.output, cached.input_tokens, cached.output_tokens
            started = time.monotonic()
            # Exactly one delegated attempt. Errors propagate, without a cached
            # response, retry, model-output validation or answer replacement.
            output, inputs, outputs = super()._generate(system, user, schema, cap)
            elapsed = time.monotonic() - started
            campaign = self.config.get("subscription_campaign", CAMPAIGN)
            entry = CachedCompletion(
                request_sha256=key, prompt_sha256=prompt_hash, schema_sha256=schema_hash,
                output=output, input_tokens=inputs, output_tokens=outputs,
                inference_seconds=elapsed, backend_usage=deepcopy(self.last_backend_usage),
                original_campaign=campaign, subscription_launch=self.last_launch,
            )
            records.entries.append(entry)
            _atomic_json(self._scope_dir / "records.json", records.model_dump())
            self.last_original_campaign = campaign
            self._position += 1
            return output, inputs, outputs
