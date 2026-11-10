#!/usr/bin/env python3
"""Render attributed, auditable coursework results without any model calls.

Input: experiment.py result directories and optional diagnostic-battery results.
Example: python scripts/build_report.py --candidate diagnostic --e2e RUN_A RUN_B \
    --battery BATTERY_DIRECTORY --out reports/latest

Only completed evaluable episodes enter SR. Campaign-budget censoring and
unfinished checkpoints remain visible in coverage. Episode compute caps count
as failures. Pilot/held-out, models and protocols are never silently pooled.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import platform
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

METHOD_NAMES = {
    "diagnostic": "Диагностический контроллер",
    "feedback": "Контроллер восприимчивости",
    "stateact": "StateAct · official prompts",
    "reflact_paper": "ReflAct · по статье",
    "raw_actor": "Исходный актор · абляция",
}
BATTERY_METRICS = {
    "diagnosis_accuracy": "Совпадение с заданным условием",
    "source_fidelity_accuracy": "Точность фактов",
    "bad_to_good": "Исправлено неверных",
    "good_to_bad": "Испорчено верных",
    "beneficial_correction": "Полезная коррекция (проба)",
    "harmful_corruption": "Вредная порча (проба)",
}
BATTERY_MODES = {
    "controlled": "Наш контроллер",
    "diagnostic": "Диагностический контроллер",
    "always_critique": "Всегда критик",
    "facts_only": "Только факты",
    "literal_only_ablation": "Только literal (абляция)",
}
BATTERY_CONDITIONS = {
    "representation_loss": "Потеря представления",
    "decoder_failure": "Сбой декодера",
    "unresolved": "Неопределённость",
    "clean": "Без ошибки",
    "all": "Все случаи",
    "initially_wrong": "Изначально неверно",
    "initially_correct": "Изначально верно",
    "local_quote_proxy": "Проверка цитаты",
    "all_synthetic": "Все синтетические случаи",
}
BASELINES = ("stateact", "reflact_paper")
OPTIONAL_ABLATIONS = ("raw_actor",)
CENSORED = {"campaign_budget_censored", "running", "interrupted", "unfinished"}
LABELS = {
    "pilot": "Пилот",
    "heldout": "Отложенная выборка",
    "unknown": "Разбиение не указано",
    "heldout-derived": "На основе отложенных задач · микрозадачи",
    "pilot-derived": "На основе пилота · микрозадачи",
    "train-derived": "На основе train · микрозадачи",
}
SECRET_KEYS = {"api_key", "authorization", "access_token", "refresh_token", "client_secret", "password"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def redact(value):
    """Snapshot experiment metadata without copying credential values."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in SECRET_KEYS else redact(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    """Two-sided Wilson score interval. Empty groups have no estimated rate."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Expected 0 <= successes <= total")
    if total == 0:
        return None, None
    p = successes / total
    divisor = 1 + z * z / total
    center = (p + z * z / (2 * total)) / divisor
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / divisor
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar(wins: int, losses: int) -> float:
    """Exact two-sided paired binomial test, conditioned on discordant pairs."""
    if wins < 0 or losses < 0:
        raise ValueError("Negative discordant count")
    n = wins + losses
    if not n:
        return 1.0
    probability = sum(
        math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2))
        for k in range(min(wins, losses) + 1)
    )
    return min(1.0, 2 * probability)


def holm_adjust(values):
    """Holm familywise adjustment, preserving the caller's order."""
    if any(not 0 <= x <= 1 for x in values):
        raise ValueError("p-values must be in [0, 1]")
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [None] * len(values)
    previous = 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def phase(path: Path, run: dict) -> str:
    config = run.get("config", {})
    for value in (config.get("phase"), config.get("split"), config.get("manifest"), str(path)):
        text = str(value or "").lower()
        if "heldout" in text or "held_out" in text:
            return "heldout"
        if "pilot" in text:
            return "pilot"
    return "unknown"


def evaluation_reason(episode: dict) -> str:
    if episode.get("finished") is not True:
        return "unfinished"
    if episode.get("status") in CENSORED:
        return episode["status"]
    if not isinstance(episode.get("success"), bool):
        return "invalid_success_flag"
    return "included"


def protocol_id(episode: dict, run: dict) -> str:
    """Separate different environments/horizons/backbone settings in summaries."""
    config = run.get("config", {})
    benchmark = episode.get("benchmark")
    provider = config.get("provider", {})
    controlled = {
        "environment": episode.get("protocol"),
        "horizon": config.get("horizons", {}).get(benchmark),
        "episode_call_cap": config.get("episode_call_cap"),
        "reasoning_effort": provider.get("reasoning_effort", provider.get("reasoning")),
        "backend": provider.get("backend"),
    }
    return hashlib.sha256(json.dumps(controlled, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[
        :10
    ]


def task_identity(row: dict):
    return (
        row["split"],
        row["benchmark"],
        row["model"],
        row["protocol_id"],
        row["key"],
        row["task_id"],
        row.get("seed"),
    )


def group_identity(row: dict):
    return row["split"], row["benchmark"], row["model"], row["protocol_id"]


def run_for_episode(path: Path):
    for parent in [path.parent, *path.parents]:
        candidate = parent / "run.json"
        if candidate.is_file():
            return candidate
        if parent == path.anchor:
            break
    return None


def episode_paths(inputs):
    paths = set()
    for item in inputs:
        p = Path(item).expanduser().resolve()
        if p.is_file():
            paths.add(p)
        elif p.is_dir():
            if p.name == "episodes":
                paths.update(p.glob("*.json"))
            paths.update(p.glob("**/episodes/*.json"))
    return sorted(paths)


def load_episodes(inputs, candidate):
    rows, warnings, provenance, run_records = [], [], [], {}
    seen, conflicts = {}, set()
    methods = {candidate, *BASELINES, *OPTIONAL_ABLATIONS}
    for path in episode_paths(inputs):
        try:
            raw = read_json(path)
            if not isinstance(raw, dict) or "variant" not in raw or "benchmark" not in raw:
                continue
            method = "reflact_paper" if raw["variant"] == "reflact" else raw["variant"]
            if method not in methods:
                continue
            run_path = run_for_episode(path)
            run = {}
            if run_path:
                if str(run_path) not in run_records:
                    run_records[str(run_path)] = read_json(run_path)
                run = run_records[str(run_path)]
            usage = raw.get("usage", {})
            calls = usage.get("calls")
            if calls is not None and (not isinstance(calls, (int, float)) or calls < 0):
                raise ValueError("Invalid usage.calls")
            row = {
                "split": phase(path, run),
                "benchmark": str(raw["benchmark"]),
                "model": str(raw.get("model", "unknown")),
                "variant": method,
                "key": str(raw.get("key", raw.get("task_id", path.stem))),
                "task_id": str(raw.get("task_id", raw.get("key", path.stem))),
                "seed": raw.get("seed"),
                "protocol_id": protocol_id(raw, run),
                "success": raw.get("success") is True and raw.get("status") != "episode_compute_cap",
                "status": str(raw.get("status", "unknown")),
                "included": evaluation_reason(raw) == "included",
                "exclusion": evaluation_reason(raw),
                "calls": calls,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "wall_seconds": raw.get("wall_seconds"),
                "steps": raw.get("steps", len(raw.get("trajectory", []))),
                "source_sha256": raw.get("source_sha256"),
                "path": str(path),
                "trajectory": raw.get("trajectory", []),
                "aliases": [],
            }
            digest = sha256(path)
            provenance.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                    "kind": "episode",
                    "split": row["split"],
                    "variant": method,
                }
            )
            identity = (*task_identity(row), method)
            if identity in seen:
                if seen[identity][0] == digest:
                    seen[identity][1]["aliases"].append(str(path))
                else:
                    conflicts.add(identity)
                continue
            seen[identity] = digest, row
            rows.append(row)
        except (OSError, ValueError, TypeError, KeyError) as error:
            warnings.append(f"Не прочитан {path}: {error}")
    if conflicts:
        rows = [row for row in rows if (*task_identity(row), row["variant"]) not in conflicts]
        warnings.append(
            f"Конфликтующие повторы одного эпизода исключены: {len(conflicts)}. Нужна ручная проверка."
        )
    # Read plans even if a run has not yet written its first episode.
    for item in inputs:
        root = Path(item).expanduser().resolve()
        metadata_paths = sorted(root.glob("**/run.json")) if root.is_dir() else []
        for metadata in metadata_paths:
            if str(metadata) in run_records:
                continue
            try:
                run = read_json(metadata)
                if set(run.get("config", {}).get("variants", [])) & methods:
                    run_records[str(metadata)] = run
            except (OSError, ValueError, TypeError) as error:
                warnings.append(f"Метаданные {metadata} не прочитаны: {error}")
    for path_str, run in run_records.items():
        p = Path(path_str)
        provenance.append(
            {"path": path_str, "sha256": sha256(p), "kind": "run_metadata", "metadata": redact(run)}
        )
        snapshot = p.parent / "source_snapshot.zip"
        if snapshot.exists():
            provenance.append(
                {
                    "path": str(snapshot),
                    "sha256": sha256(snapshot),
                    "bytes": snapshot.stat().st_size,
                    "kind": "experiment_source_snapshot",
                }
            )
    return rows, warnings, provenance, run_records


def resolve_frozen_manifest(run_path, run):
    """Resolve relocated evidence without changing the authentic run metadata.

    A basename-only fallback is never trusted: an included ancestor manifest
    must match the run's recorded manifest_sha256 byte for byte.
    """
    value = run.get("config", {}).get("manifest")
    if not value:
        raise FileNotFoundError("run.json не содержит config.manifest")
    configured = Path(str(value)).expanduser()
    direct = configured if configured.is_absolute() else Path(run_path).parent / configured
    expected = run.get("manifest_sha256")
    expected = expected.lower() if isinstance(expected, str) else None
    if direct.is_file() and (not expected or sha256(direct) == expected):
        return direct, {
            "configured_manifest": str(configured),
            "manifest_resolution": "declared_path",
            "expected_manifest_sha256": expected,
        }
    if not expected or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise FileNotFoundError(
            "Исходный manifest недоступен; переносимый fallback требует записанного manifest_sha256"
        )
    for parent in Path(run_path).resolve().parents:
        candidate = parent / configured.name
        if candidate.is_file() and sha256(candidate) == expected:
            return candidate, {
                "configured_manifest": str(configured),
                "manifest_resolution": "ancestor_basename_and_sha256",
                "expected_manifest_sha256": expected,
            }
    raise FileNotFoundError(
        "Manifest не найден либо SHA-256 не совпадает; непроверенная подстановка запрещена"
    )


def expected_coverage(run_records, rows):
    """Manifest-based planned coverage; missing tasks are not failures or wins."""
    result, warnings, manifest_sources = [], [], []
    seen = set()
    for run_path, run in run_records.items():
        config = run.get("config", {})
        try:
            manifest, resolution = resolve_frozen_manifest(run_path, run)
        except (OSError, ValueError) as error:
            warnings.append(f"Манифест плана недоступен для {run_path}; полное покрытие неизвестно: {error}.")
            continue
        try:
            frozen = read_json(manifest)
            manifest_sources.append(
                {
                    "path": str(manifest.resolve()),
                    "sha256": sha256(manifest),
                    "kind": "task_manifest",
                    "metadata": redact(frozen),
                    **resolution,
                }
            )
            selected = config.get("task_keys")
            for task in frozen.get("tasks", []):
                if selected is not None and task.get("key") not in selected:
                    continue
                for method in config.get("variants", []):
                    match = [
                        r
                        for r in rows
                        if r["key"] == task.get("key")
                        and r["variant"] == method
                        and str(Path(r["path"]).parent.parent) == str(Path(run_path).parent)
                    ]
                    key = (phase(Path(run_path), run), task.get("benchmark"), method, task.get("key"))
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append(
                        {
                            "split": key[0],
                            "benchmark": key[1],
                            "variant": method,
                            "key": key[3],
                            "record_present": bool(match),
                            "evaluable": bool(match and match[0]["included"]),
                        }
                    )
        except (OSError, ValueError, TypeError) as error:
            warnings.append(f"План {manifest} не прочитан: {error}")
    return result, warnings, manifest_sources


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(*group_identity(row), row["variant"])].append(row)
    summaries = []
    for (split, benchmark, model, protocol, method), values in sorted(grouped.items()):
        completed = [r for r in values if r["included"]]
        n, successes = len(completed), sum(r["success"] for r in completed)
        lo, hi = wilson(successes, n)
        known = [r["calls"] for r in completed if r["calls"] is not None]
        all_known = [r["calls"] for r in values if r["calls"] is not None]
        summaries.append(
            {
                "split": split,
                "benchmark": benchmark,
                "model": model,
                "protocol_id": protocol,
                "variant": method,
                "n": n,
                "successes": successes,
                "success_rate": successes / n if n else None,
                "ci_low": lo,
                "ci_high": hi,
                "records": len(values),
                "excluded": len(values) - n,
                "compute_cap_failures": sum(r["status"] == "episode_compute_cap" for r in completed),
                "model_error_failures": sum(r["status"] == "model_error" for r in completed),
                "horizon_failures": sum(r["status"] == "action_horizon" for r in completed),
                "calls_total_all_records": sum(all_known),
                "calls_total_evaluable": sum(known),
                "calls_coverage": len(known),
                "calls_per_evaluable_episode": sum(known) / len(known) if known else None,
                "calls_per_solved_episode": sum(known) / successes if successes and len(known) == n else None,
                "total_steps": sum(r["steps"] or 0 for r in completed),
                "source_hashes": ";".join(sorted({str(r["source_sha256"]) for r in values})),
            }
        )
    return summaries


def paired_comparisons(rows, candidate):
    groups = defaultdict(dict)
    for row in rows:
        if row["included"]:
            groups[group_identity(row)].setdefault(row["variant"], {})[task_identity(row)] = row
    comparisons = []
    for group, by_method in sorted(groups.items()):
        ours = by_method.get(candidate, {})
        optional = [method for method in OPTIONAL_ABLATIONS if ours and by_method.get(method)]
        for baseline in (*BASELINES, *optional):
            theirs = by_method.get(baseline, {})
            shared = sorted(set(ours) & set(theirs), key=str)
            exploratory = baseline in OPTIONAL_ABLATIONS
            if exploratory and not shared:
                continue
            wins = sum(ours[k]["success"] and not theirs[k]["success"] for k in shared)
            losses = sum(not ours[k]["success"] and theirs[k]["success"] for k in shared)
            both = sum(ours[k]["success"] and theirs[k]["success"] for k in shared)
            comparisons.append(
                {
                    "split": group[0],
                    "benchmark": group[1],
                    "model": group[2],
                    "protocol_id": group[3],
                    "candidate": candidate,
                    "baseline": baseline,
                    "baseline_label": METHOD_NAMES.get(baseline, baseline),
                    "comparison_kind": "exploratory_ablation" if exploratory else "published_baseline",
                    "paired_n": len(shared),
                    "wins": wins,
                    "losses": losses,
                    "both_success": both,
                    "both_failure": len(shared) - wins - losses - both,
                    "candidate_unpaired": len(ours) - len(shared),
                    "baseline_unpaired": len(theirs) - len(shared),
                    "delta_pp": 100 * (wins - losses) / len(shared) if shared else None,
                    "p_exact": exact_mcnemar(wins, losses) if shared and not exploratory else None,
                    "p_holm": None,
                }
            )
    # Each split is a separate family; pilot is exploratory and never strengthens held-out evidence.
    for split in {r["split"] for r in comparisons}:
        family = [r for r in comparisons if r["split"] == split and r["p_exact"] is not None]
        for row, adjusted in zip(family, holm_adjust([r["p_exact"] for r in family]), strict=True):
            row["p_holm"] = adjusted
    return comparisons


def route_counts(rows, candidate):
    counts = Counter()
    for episode in rows:
        if episode["variant"] != candidate:
            continue
        for turn in episode["trajectory"]:
            diagnostic = turn.get("diagnostics") or {}
            counts[
                (
                    episode["split"],
                    episode["benchmark"],
                    diagnostic.get("intervention", "unreported"),
                    diagnostic.get("diagnosis", diagnostic.get("gate_reason", "unreported")),
                    "evaluable" if episode["included"] else "excluded_episode",
                )
            ] += 1
    return [
        {
            "split": split,
            "benchmark": benchmark,
            "intervention": route,
            "reason": reason,
            "episode_scope": scope,
            "turns": n,
        }
        for (split, benchmark, route, reason, scope), n in sorted(counts.items())
    ]


def canonical_battery_source(case):
    """Condition IDs are not independent source states.

    The original case/provenance remains untouched. Use only source identity,
    never condition, probe IDs, mutated views or the arbitrary state_id suffix.
    """
    provenance = case.get("provenance") or {}
    task = provenance.get("task_id", case.get("task_id"))
    step = provenance.get("step", case.get("step"))
    source = case.get("source")
    observation_hash = (
        hashlib.sha256(source.encode()).hexdigest()
        if isinstance(source, str)
        else provenance.get("observation_sha256")
    )
    if task is None or step is None or not observation_hash:
        return None, None
    benchmark = provenance.get("benchmark", case.get("benchmark", "unknown"))
    physical = [benchmark, task, step, observation_hash]
    underlying_task = [benchmark, task]
    return (
        "source:" + hashlib.sha256(json.dumps(physical, ensure_ascii=False).encode()).hexdigest(),
        "task:" + hashlib.sha256(json.dumps(underlying_task, ensure_ascii=False).encode()).hexdigest(),
    )


def battery_split_metadata(*sources):
    """A transformed source split is never the original stock test result.

    Unlike episode selection, battery splits are not guessed from filenames.
    Only an explicit split/source_split metadata field supports this label.
    """
    for source in sources:
        if not isinstance(source, dict):
            continue
        value = source.get("split", source.get("source_split"))
        if not isinstance(value, str) or value.lower() in {"", "unknown"}:
            continue
        normalized = value.lower()
        if normalized in {"test", "heldout", "held_out", "valid_unseen", "heldout-derived"}:
            return "heldout-derived"
        if normalized in {"pilot", "pilot-derived"}:
            return "pilot-derived"
        if normalized in {"train", "train-derived"}:
            return "train-derived"
        return value
    return "unknown"


def load_battery(paths):
    """Support diagnostic records.jsonl and feedback-battery-results-v1 JSON."""
    rows, provenance, warnings, seen, seen_rows = [], [], [], set(), set()
    for item in paths:
        p = Path(item).expanduser().resolve()
        files = [p] if p.is_file() else sorted(p.glob("**/records.jsonl")) if p.is_dir() else []
        if p.is_dir():
            files += sorted(p.glob("**/*battery*.json"))
            files += sorted(p.glob("**/results.json"))
        for file in files:
            try:
                digest = sha256(file)
                if digest in seen:
                    continue
                seen.add(digest)
                if file.suffix == ".jsonl":
                    values = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
                    metadata = {}
                else:
                    metadata = read_json(file)
                    values = metadata.get("rows", [])
                if not isinstance(values, list):
                    raise ValueError("Expected battery rows list")
                run_metadata_path = file.parent / "run_metadata.json"
                run_metadata = read_json(run_metadata_path) if run_metadata_path.is_file() else {}
                case_manifest = file.parent / "manifest.json"
                manifest_metadata = read_json(case_manifest) if case_manifest.is_file() else {}
                cases = manifest_metadata.get("cases", [])
                cases_by_id = {case["case_id"]: case for case in cases if "case_id" in case}
                for value in values:
                    if isinstance(value, dict):
                        # A checkpoint JSON and its JSONL export are the same cases.
                        row_hash = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
                        if row_hash in seen_rows:
                            continue
                        seen_rows.add(row_hash)
                        case = cases_by_id.get(value.get("case_id"), {})
                        source_id, task_id = canonical_battery_source(case)
                        if "diagnosis_correct" in value and source_id is None:
                            warnings.append(
                                f"Случай {value.get('case_id')}: исходное состояние не подтверждено manifest provenance; state_n не восстанавливается из condition ID."
                            )
                        rows.append(
                            {
                                **value,
                                "original_state_id": value.get("state_id"),
                                "canonical_source_state_id": source_id,
                                "canonical_underlying_task_id": task_id,
                                "source_state_provenance": case.get("provenance"),
                                "battery_model": metadata.get("model")
                                or run_metadata.get("model")
                                or "unknown",
                                "battery_split": battery_split_metadata(
                                    value, metadata, run_metadata, manifest_metadata, case.get("provenance")
                                ),
                                "battery_protocol": metadata.get("protocol", "custom_diagnostic_battery"),
                                "source_path": str(file),
                            }
                        )
                provenance.append({"path": str(file), "sha256": digest, "kind": "battery_results"})
                for extra in (file.parent / "manifest.json", file.parent / "summary.json", run_metadata_path):
                    if extra.is_file():
                        provenance.append(
                            {
                                "path": str(extra),
                                "sha256": sha256(extra),
                                "kind": "battery_metadata",
                                "metadata": redact(read_json(extra)),
                            }
                        )
            except (OSError, ValueError, TypeError) as error:
                warnings.append(f"Батарея {file} не прочитана: {error}")
    return rows, provenance, warnings


def battery_summary(rows):
    """Case rates are descriptive: multiple cases share a source state.

    Do not display independent-Bernoulli confidence intervals for these clusters.
    A larger future study can pre-register state-cluster resampling instead.
    """
    grouped = defaultdict(list)
    state_sets = defaultdict(set)
    task_sets = defaultdict(set)
    for row in rows:
        prefix = (
            row.get("battery_split", "unknown"),
            row.get("battery_model", "unknown"),
            row.get("benchmark", "source_states"),
        )
        state_id = row.get("canonical_source_state_id")
        if state_id is None and "diagnosis_correct" not in row:
            state_id = row.get("state_id")
        task_id = row.get("canonical_underlying_task_id") or row.get("task_id")

        def add(metric, condition, method, value, prefix=prefix, state_id=state_id, task_id=task_id):
            key = (*prefix, metric, condition, method)
            grouped[key].append(bool(value))
            if state_id is not None:
                state_sets[key].add(str(state_id))
            if task_id is not None:
                task_sets[key].add(str(task_id))

        if "diagnosis_correct" in row:
            if isinstance(row.get("diagnosis_correct"), bool):
                add(
                    "diagnosis_accuracy",
                    row.get("condition", "unknown"),
                    "diagnostic",
                    row["diagnosis_correct"],
                )
                add("diagnosis_accuracy", "all_synthetic", "diagnostic", row["diagnosis_correct"])
            if isinstance(row.get("ablation_correct"), bool):
                add(
                    "diagnosis_accuracy",
                    row.get("condition", "unknown"),
                    "literal_only_ablation",
                    row["ablation_correct"],
                )
                add("diagnosis_accuracy", "all_synthetic", "literal_only_ablation", row["ablation_correct"])
        elif isinstance(row.get("after_correct"), bool):
            mode = row.get("mode", "unknown")
            add("source_fidelity_accuracy", "all", mode, row["after_correct"])
            if row.get("before_correct") is False:
                add("bad_to_good", "initially_wrong", mode, row["after_correct"])
            elif row.get("before_correct") is True:
                add("good_to_bad", "initially_correct", mode, not row["after_correct"])
            profile = row.get("profile")
            if isinstance(profile, dict) and profile.get("observed_pairs") == 1:
                for metric in ("beneficial_correction", "harmful_corruption"):
                    if profile.get(metric) in (0, 1):
                        add(metric, "local_quote_proxy", mode, profile[metric])
    result = []
    for key, values in sorted(grouped.items()):
        split, model, benchmark, metric, condition, method = key
        n, hits = len(values), sum(values)
        result.append(
            {
                "split": split,
                "model": model,
                "benchmark": benchmark,
                "metric": metric,
                "condition": condition,
                "method": method,
                "n": n,
                "state_n": len(state_sets[key]) or None,
                "underlying_task_n": len(task_sets[key]) or None,
                "count": hits,
                "rate": hits / n,
                "ci_low": None,
                "ci_high": None,
                "interval_policy": "omitted_clustered_cases_small_state_count",
            }
        )
    return result


def battery_inventory(rows):
    """Audit denominators across conditions without asserting independence."""
    groups = {"all": rows}
    diagnostic = [row for row in rows if "diagnosis_correct" in row]
    if diagnostic:
        groups["synthetic_scored"] = [
            row for row in diagnostic if isinstance(row.get("diagnosis_correct"), bool)
        ]
        groups["ecological_unlabeled"] = [row for row in diagnostic if row.get("diagnosis_correct") is None]
    result = []
    for label, values in groups.items():
        states = {
            r.get("canonical_source_state_id")
            or (r.get("state_id") if "diagnosis_correct" not in r else None)
            for r in values
        }
        tasks = {r.get("canonical_underlying_task_id") or r.get("task_id") for r in values}
        result.append(
            {
                "scope": label,
                "cases": len(values),
                "source_states": len(states - {None}) or None,
                "underlying_tasks": len(tasks - {None}) or None,
            }
        )
    return result


def write_csv(path: Path, rows: list[dict], fields=None):
    keys = fields or list(dict.fromkeys(k for row in rows for k in row)) or ["no_data"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=1):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(rows, columns):
    if not rows:
        return '<p class="empty">Нет наблюдений.</p>'
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(fmt(row.get(key), 4 if key.startswith('p_') else 1))}</td>"
                for key, _ in columns
            )
            + "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def make_plots(out, summaries, pairs, routes, battery, candidate):
    # Import lazily: statistics and validation require only the standard library.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    colors = {"stateact": "#708393", "reflact_paper": "#c3a269", "raw_actor": "#9b85b5", candidate: "#087f8c"}
    plots = []

    def save(fig, slug, caption):
        fig.savefig(out / (slug + ".png"), dpi=170, bbox_inches="tight", facecolor="white")
        fig.savefig(out / (slug + ".svg"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        plots.append((slug, caption))

    groups = sorted({(s["split"], s["benchmark"], s["model"], s["protocol_id"]) for s in summaries})
    if not groups:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.axis("off")
        ax.text(0.5, 0.55, "Результаты экспериментов ещё не загружены", ha="center", fontsize=15)
        ax.text(0.5, 0.32, "Пустой набор не означает 0% успеха", ha="center", color="#667085")
        save(fig, "success", "Нет данных: результат не оценивается.")
    else:
        fig, axes = plt.subplots(len(groups), 1, figsize=(10, max(3.4, len(groups) * 3.2)), squeeze=False)
        for ax, group in zip(axes.flat, groups, strict=True):
            entries = [
                s
                for s in summaries
                if tuple(s[k] for k in ("split", "benchmark", "model", "protocol_id")) == group
            ]
            for i, row in enumerate(entries):
                if row["n"]:
                    rate = row["success_rate"]
                    ax.bar(i, rate, color=colors.get(row["variant"], "#7b8794"), width=0.62)
                    ax.errorbar(
                        i,
                        rate,
                        yerr=[[max(0, rate - row["ci_low"])], [max(0, row["ci_high"] - rate)]],
                        fmt="none",
                        color="#24364b",
                        capsize=5,
                    )
                    ax.text(
                        i, min(1.16, row["ci_high"] + 0.055), f"{row['successes']}/{row['n']}", ha="center"
                    )
                else:
                    ax.text(i, 0.05, "нет завершённых", rotation=90, ha="center")
            ax.set_xticks(
                range(len(entries)), [METHOD_NAMES.get(s["variant"], s["variant"]) for s in entries]
            )
            ax.set_ylim(0, 1.25)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylabel("Успех задач")
            ax.set_title(
                f"{LABELS.get(group[0], group[0])} · {group[1]} · {group[2]} · протокол {group[3]}",
                loc="left",
            )
            ax.grid(axis="y", alpha=0.16)
        fig.tight_layout(h_pad=2.2)
        save(
            fig,
            "success",
            "Успех среды: число успехов / оцениваемые эпизоды; интервалы Wilson 95%. Пилот и отложенная выборка разделены.",
        )

    valid_pairs = [r for r in pairs if r["paired_n"]]
    if valid_pairs:
        fig, ax = plt.subplots(figsize=(11, max(3, 0.8 * len(valid_pairs))))
        names = []
        for i, row in enumerate(valid_pairs):
            ax.barh(i, row["delta_pp"], color="#087f8c" if row["delta_pp"] >= 0 else "#bd5946")
            names.append(
                f"{row['split']} · {row['benchmark']} · {row['baseline_label']}\n{row['model']} · {row['protocol_id']}"
            )
            inferential_label = (
                f"p Holm={row['p_holm']:.3g}"
                if row.get("p_holm") is not None
                else "исследовательская абляция; без p"
            )
            ax.annotate(
                f"  Δ={row['delta_pp']:+.1f} п.п.; n={row['paired_n']}; +{row['wins']}/−{row['losses']}; "
                + inferential_label,
                (row["delta_pp"], i),
                ha="left" if row["delta_pp"] >= 0 else "right",
                va="center",
                fontsize=8,
            )
        ax.axvline(0, color="#667085", linewidth=1)
        ax.set_yticks(range(len(valid_pairs)), names)
        ax.set_xlim(-115, 115)
        ax.set_xlabel("Разность доли успеха кандидата и baseline, процентные пункты")
        fig.tight_layout()
        save(
            fig,
            "paired_delta",
            "Только общие оцениваемые задачи пары; +/− — дискордантные исходы. Для двух опубликованных baseline: exact McNemar и Holm отдельно внутри разбиения. Исходный актор — отдельная исследовательская абляция без p-value.",
        )

    known_cost = [s for s in summaries if s["calls_per_evaluable_episode"] is not None]
    if known_cost:
        fig, ax = plt.subplots(figsize=(11, max(3, 0.55 * len(known_cost))))
        ax.barh(
            range(len(known_cost)),
            [s["calls_per_evaluable_episode"] for s in known_cost],
            color=[colors.get(s["variant"], "#7b8794") for s in known_cost],
        )
        ax.set_yticks(
            range(len(known_cost)),
            [
                f"{s['split']} · {s['benchmark']} · {METHOD_NAMES.get(s['variant'], s['variant'])}\n{s['model']} · {s['protocol_id']}"
                for s in known_cost
            ],
        )
        ax.set_xlabel("Среднее число CLI-вызовов / оцениваемый эпизод, включая неуспешные")
        fig.tight_layout()
        save(
            fig,
            "calls",
            "Стоимость показана отдельно от качества; большая вычислительная нагрузка может объяснять различия качества.",
        )

    evaluated_routes = [r for r in routes if r["episode_scope"] == "evaluable"]
    if evaluated_routes:
        counts = Counter()
        for row in evaluated_routes:
            counts[(row["split"], row["benchmark"], row["intervention"])] += row["turns"]
        labels, values = zip(*sorted(counts.items()), strict=True)
        fig, ax = plt.subplots(figsize=(11, max(3, 0.45 * len(labels))))
        ax.barh(range(len(values)), values, color="#087f8c")
        ax.set_yticks(range(len(values)), [" · ".join(k) for k in labels])
        ax.set_xlabel("Число реально выполненных шагов с данным маршрутом")
        fig.tight_layout()
        save(
            fig,
            "routes",
            "Частоты вмешательств описывают поведение механизма; выбор маршрута сам по себе не доказывает его полезность.",
        )

    if battery:
        fig, axes = plt.subplots(1, 1, figsize=(11, max(4, 0.55 * len(battery))), squeeze=False)
        ax = axes[0, 0]
        for i, row in enumerate(battery):
            ax.barh(
                i,
                row["rate"],
                color="#bd5946" if row["metric"] in {"good_to_bad", "harmful_corruption"} else "#087f8c",
            )
            ax.text(
                min(1.12, row["rate"] + 0.02),
                i,
                f"{row['count']}/{row['n']} · состояний={fmt(row['state_n'])}",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(
            range(len(battery)),
            [
                f"{LABELS.get(r['split'], r['split'])} · {BATTERY_MODES.get(r['method'], r['method'])}\n"
                f"{BATTERY_METRICS.get(r['metric'], r['metric'])} · {BATTERY_CONDITIONS.get(r['condition'], r['condition'])}"
                for r in battery
            ],
        )
        ax.set_xlim(0, 1.24)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title("Отдельная диагностическая батарея · не успех stock benchmark", loc="left")
        fig.tight_layout()
        save(
            fig,
            "battery",
            "Доля верных ответов/изменений в преобразованных микрозадачах. Красные метрики означают вред; меньше лучше. Интервалы по случаям не показаны: случаи одного исходного состояния зависимы; состояний — число исходных состояний. Модель и точные идентификаторы метрик указаны в таблице.",
        )
    return plots


def build_report(e2e, candidate, out, battery_paths=()):
    out = Path(out).expanduser().resolve()
    html_path = out if out.suffix.lower() == ".html" else out / "report.html"
    directory = html_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    rows, warnings, provenance, runs = load_episodes(e2e, candidate)
    plan, coverage_warnings, plan_sources = expected_coverage(runs, rows)
    warnings.extend(coverage_warnings)
    provenance.extend(plan_sources)
    battery_rows, battery_sources, battery_warnings = load_battery(battery_paths)
    provenance.extend(battery_sources)
    warnings.extend(battery_warnings)
    stats = summarize(rows)
    pairs = paired_comparisons(rows, candidate)
    routes = route_counts(rows, candidate)
    battery_stats = battery_summary(battery_rows)
    battery_sources = [
        {
            "case_id": row.get("case_id"),
            "original_state_id": row.get("original_state_id", row.get("state_id")),
            "canonical_source_state_id": row.get("canonical_source_state_id"),
            "canonical_underlying_task_id": row.get("canonical_underlying_task_id"),
            "source_state_provenance": row.get("source_state_provenance"),
            "condition": row.get("condition"),
            "mode": row.get("mode"),
        }
        for row in battery_rows
    ]
    source_inventory = battery_inventory(battery_rows)
    selected = {candidate, *BASELINES, *OPTIONAL_ABLATIONS}
    plan = [r for r in plan if r["variant"] in selected]
    row_fields = (
        [k for k in rows[0] if k not in {"trajectory", "aliases"}]
        if rows
        else ["split", "benchmark", "variant", "included"]
    )
    for name, values, fields in [
        ("episodes.csv", rows, row_fields),
        ("success.csv", stats, None),
        ("paired.csv", pairs, None),
        ("coverage.csv", plan, None),
        ("routes.csv", routes, None),
        ("battery.csv", battery_stats, None),
        ("battery_sources.csv", battery_sources, None),
    ]:
        write_csv(directory / name, values, fields)
    plots = make_plots(directory, stats, pairs, routes, battery_stats, candidate)
    source_dir = directory / "provenance"
    source_dir.mkdir(exist_ok=True)
    script = Path(__file__).resolve()
    source_copy = source_dir / "build_report.py"
    if script != source_copy:
        shutil.copyfile(script, source_copy)
    snapshot = {
        "created_at": datetime.now(UTC).isoformat(),
        "candidate": candidate,
        "command_inputs": {"e2e": list(map(str, e2e)), "battery": list(map(str, battery_paths))},
        "reporter_sha256": sha256(script),
        "python_version": platform.python_version(),
        "inputs": provenance,
        "warnings": warnings,
        "rules": {
            "unfinished": "exclude from denominator, report coverage",
            "campaign_budget_censored": "exclude from denominator, report coverage",
            "episode_compute_cap": "failure in compute-capped protocol",
            "global_sota_comparison": False,
            "pilot_and_heldout_pooled": False,
        },
    }
    (source_dir / "input_manifest.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results = {
        "summary": stats,
        "paired": pairs,
        "coverage": plan,
        "routes": routes,
        "battery_summary": battery_stats,
        "battery_source_inventory": source_inventory,
        "battery_case_sources": battery_sources,
        "warnings": warnings,
        "evaluable_episodes": sum(r["included"] for r in rows),
        "excluded_episodes": sum(not r["included"] for r in rows),
    }
    (directory / "report_data.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    title = METHOD_NAMES[candidate]
    blocks = [
        f"<h1>{html.escape(title)}</h1>",
        '<p class="lead">Сравнение с опубликованными сильными scaffold-методами на общей модели. '
        "Это не проверка превосходства над текущим мировым SOTA.</p>",
    ]
    included = sum(r["included"] for r in rows)
    excluded = len(rows) - included
    blocks.append(
        f'<div class="metrics"><span><b>{included}</b> оцениваемых эпизодов</span>'
        f"<span><b>{excluded}</b> исключённых / незавершённых</span>"
        f"<span><b>{len(battery_rows)}</b> случаев отдельной батареи</span></div>"
    )
    limitations = [
        "Небольшая выборка даёт широкую неопределённость. Интервалы и точные парные тесты не заменяют достаточный объём данных.",
        "Одинаковый лимит вызовов не означает одинаковое число действий, токенов или FLOPs. Различия акторов и промптов также влияют на результат; вклад одного контроллера из общего сравнения не изолирован.",
        "Общие baseline-траектории переиспользуются в отчётах двух кандидатов; это не независимые повторные эксперименты.",
        "Пилот и отложенная выборка показаны раздельно. Смена модели, протокола или горизонта создаёт отдельную группу.",
        "Незавершённые эпизоды и остановки общего бюджета исключены из SR, но отражены в покрытии. Лимит вычислений отдельного эпизода считается неуспехом.",
        "Микрозадачи батареи преобразованы из публичных состояний. Они проверяют локальную диагностику/точность цитаты, а не успех исходной задачи среды.",
    ]
    if any(row["variant"] == "raw_actor" for row in rows):
        limitations.append(
            "«Исходный актор · абляция» — дополнительный актор без критика и проб, а не опубликованный SOTA. "
            "Его сравнение исследовательское и не входит в семейство проверок двух опубликованных baseline."
        )
        limitations.append(
            "Этот актор точно совпадает с первым запросом FeedbackAgent и использует отдельные генерации. "
            "Для DiagnosticAgent это дополнительный простой планировщик, а не строгая абляция всей архитектуры."
        )
    if any(r["split"] == "unknown" for r in rows):
        limitations.append(
            "Часть входов не содержит различимого разбиения: они не считаются отложенной выборкой автоматически."
        )
    if not rows:
        limitations.append(
            "Входных эпизодов нет. Пустая таблица не является результатом 0%; эксперименты ещё не оценены."
        )
    blocks.append(
        "<aside><strong>Границы выводов</strong><ul>"
        + "".join(f"<li>{html.escape(x)}</li>" for x in limitations)
        + "</ul></aside>"
    )
    if warnings:
        blocks.append(
            '<aside class="warning"><strong>Проверка входов</strong><ul>'
            + "".join(f"<li>{html.escape(x)}</li>" for x in warnings)
            + "</ul></aside>"
        )
    for slug, caption in plots:
        data = base64.b64encode((directory / (slug + ".png")).read_bytes()).decode()
        blocks.append(
            f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        )
    blocks.append("<h2>Полные эпизоды: успех и стоимость</h2>")
    formatted_stats = [
        {
            **s,
            "rate_display": f"{s['successes']}/{s['n']}",
            "interval_display": f"{100 * s['ci_low']:.1f}–{100 * s['ci_high']:.1f}%" if s["n"] else "—",
        }
        for s in stats
    ]
    blocks.append(
        table(
            formatted_stats,
            [
                ("split", "Разбиение"),
                ("benchmark", "Среда"),
                ("model", "Модель"),
                ("protocol_id", "Протокол"),
                ("variant", "Метод"),
                ("rate_display", "Успех n/N"),
                ("interval_display", "Wilson 95%"),
                ("excluded", "Исключено"),
                ("compute_cap_failures", "Лимит эпизода"),
                ("model_error_failures", "Ошибки модели"),
                ("horizon_failures", "Лимит шагов"),
                ("calls_per_evaluable_episode", "Вызовов / эпизод"),
                ("calls_per_solved_episode", "Все вызовы / успехи"),
            ],
        )
    )
    blocks.append(
        "<p>«Все вызовы / успехи» включает стоимость неуспешных оцениваемых эпизодов; при отсутствии успехов значение не определено. "
        "Вызовы прерванных общим бюджетом запусков сохранены отдельно в CSV calls_total_all_records.</p>"
    )
    blocks.append("<h2>Парные сравнения</h2>")
    blocks.append(
        table(
            pairs,
            [
                ("split", "Разбиение"),
                ("benchmark", "Среда"),
                ("model", "Модель"),
                ("baseline_label", "Baseline / абляция"),
                ("comparison_kind", "Тип сравнения"),
                ("paired_n", "Общих задач"),
                ("wins", "Выигрыш"),
                ("losses", "Проигрыш"),
                ("delta_pp", "Δ, п.п."),
                ("p_exact", "p exact"),
                ("p_holm", "p Holm"),
                ("candidate_unpaired", "Кандидат без пары"),
                ("baseline_unpaired", "Baseline без пары"),
            ],
        )
    )
    blocks.append("<h2>Покрытие запланированных задач</h2>")
    coverage = defaultdict(Counter)
    for row in plan:
        g = coverage[(row["split"], row["benchmark"], row["variant"])]
        g["planned"] += 1
        g["present"] += row["record_present"]
        g["evaluable"] += row["evaluable"]
    coverage_rows = [
        {"split": s, "benchmark": b, "variant": m, **counts, "missing": counts["planned"] - counts["present"]}
        for (s, b, m), counts in sorted(coverage.items())
    ]
    blocks.append(
        table(
            coverage_rows,
            [
                ("split", "Разбиение"),
                ("benchmark", "Среда"),
                ("variant", "Метод"),
                ("planned", "По плану"),
                ("present", "Файлов"),
                ("evaluable", "Оцениваемых"),
                ("missing", "Не начато / отсутствует"),
            ],
        )
    )
    if not plan:
        blocks.append(
            "<p>Полный знаменатель плана неизвестен: доступных манифестов нет. Число найденных файлов не считается полным покрытием.</p>"
        )
    if battery_rows:
        blocks.append(
            "<h2>Отдельная батарея механизма</h2><p>diagnosis_accuracy относится к контролируемым сбоям "
            "протокола декодера/представления; такие сбои не следует называть естественными дефектами модели. "
            "source_fidelity_accuracy, bad_to_good и good_to_bad относятся к точности преобразованной фактической микрозадачи. "
            "Локальные quote-proxy показатели не доказывают правильность реального плана. Несколько случаев одного состояния зависимы; "
            "интервалы по случаям не показаны; статистическое обобщение на задачи не выполнялось.</p>"
        )
        blocks.append(
            table(
                battery_stats,
                [
                    ("split", "Разбиение"),
                    ("model", "Модель"),
                    ("metric", "Метрика"),
                    ("condition", "Условие"),
                    ("method", "Режим"),
                    ("count", "Числитель"),
                    ("n", "Знаменатель"),
                    ("rate", "Доля"),
                    ("state_n", "Исходных состояний"),
                    ("underlying_task_n", "Исходных задач"),
                ],
            )
        )
        blocks.append(
            "<h2>Проверка знаменателей батареи</h2><p>Разные condition ID одного наблюдения "
            "считаются одним физическим исходным состоянием по task_id, шагу и SHA-256 исходного текста "
            "в manifest provenance. Исходные state_id и provenance сохранены в battery_sources.csv и "
            "report_data.json; RAW-файлы не меняются. Состояния одного задания также не являются "
            "независимыми заданиями. Проценты по синтетическим случаям — описательные, без проверки значимости.</p>"
        )
        blocks.append(
            table(
                source_inventory,
                [
                    ("scope", "Набор"),
                    ("cases", "Случаев"),
                    ("source_states", "Физических источников"),
                    ("underlying_tasks", "Исходных задач"),
                ],
            )
        )
    blocks.append(
        "<h2>Происхождение и воспроизводимость</h2><p>StateAct: "
        '<a href="https://aclanthology.org/2025.realm-1.27/">REALM@ACL 2025</a>, '
        '<a href="https://github.com/ai-nikolai/StateAct/tree/8fb47be6f5f3978a38726608ae347987341f1d82">официальные prompts, закреплённый коммит</a>. '
        'ReflAct: <a href="https://aclanthology.org/2025.emnlp-main.1697/">EMNLP 2025</a>, '
        "воспроизведение по Appendix K; TextCraft — перенос метода. Опубликованные числа на других моделях не включены в сопоставимые столбцы. "
        "Подробности адаптаций: docs/BASELINES.md проекта. Этот HTML автономен: все графики встроены; внешние запросы не нужны.</p>"
    )
    blocks.append(
        "<p>CSV: episodes.csv, success.csv, paired.csv, coverage.csv, routes.csv, battery.csv, battery_sources.csv. "
        "Графики сохранены в PNG и SVG. Проверяемые данные: report_data.json; "
        "SHA-256 входов, копия генератора, metadata и путь к source_snapshot.zip — в provenance/.</p>"
    )
    css = (
        "body{font-family:system-ui,Arial,sans-serif;color:#24364b;background:#f5f7fa;margin:0;line-height:1.55}"
        "main{max-width:1220px;margin:0 auto;padding:36px 28px;background:white}h1{font-size:32px;margin:0 0 8px}"
        "h2{margin-top:34px;font-size:22px}.lead{font-size:18px;color:#526273}.metrics{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0}"
        ".metrics span{background:#eef5f6;padding:14px 18px;border-radius:8px}.metrics b{display:block;font-size:28px}"
        "aside{background:#fff8e8;border-left:4px solid #c3a269;padding:18px 22px;margin:22px 0}.warning{background:#fff0ec;border-color:#bd5946}"
        "li{margin:7px 0}figure{margin:28px 0}img{max-width:100%;height:auto}figcaption{font-size:14px;color:#667085}"
        ".table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;text-align:left;border-bottom:1px solid #e4e8ed}"
        "th{background:#edf2f5;white-space:nowrap}tr:nth-child(even){background:#fafbfd}.empty{color:#667085}a{color:#087f8c}"
        "@media(max-width:600px){main{padding:20px 14px}h1{font-size:25px}.metrics{gap:8px}aside{padding:12px}}"
        "@media print{body,main{background:white}main{max-width:none;padding:0}figure,table{break-inside:avoid}}"
    )
    document = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += (
        f"<title>{html.escape(title)} — результаты</title><style>{css}</style><body><main>"
        + "".join(blocks)
        + "</main></body></html>"
    )
    html_path.write_text(document, encoding="utf-8")
    return {
        "html": str(html_path),
        "evaluable": included,
        "excluded": excluded,
        "battery_cases": len(battery_rows),
        "plots": len(plots),
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e", nargs="*", default=[], help="Result directories or episode JSON files")
    parser.add_argument("--candidate", choices=["diagnostic", "feedback"], required=True)
    parser.add_argument(
        "--battery", nargs="*", default=[], help="Optional battery JSON/JSONL files or directory"
    )
    parser.add_argument("--out", type=Path, required=True, help="Report directory, or a .html output path")
    args = parser.parse_args()
    result = build_report(args.e2e, args.candidate, args.out, args.battery)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
