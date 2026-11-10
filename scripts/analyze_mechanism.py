#!/usr/bin/env python3
"""Describe executed controller traces and logical costs; never call a model.

Use --episodes (alias --episodesdirs) for experiment.py result directories and
--out for a directory or a .json/.md path. Both JSON and Markdown are produced.
This appendix describes selected-run observational associations, not causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

CENSORED = {"campaign_budget_censored", "running", "interrupted", "unfinished"}
BASELINES = {"stateact", "reflact_paper"}
OPTIONAL_ABLATIONS = {"raw_actor"}
ABLATION_LABEL = "Исходный актор · абляция"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase(path):
    text = str(path).lower()
    return "pilot" if "pilot" in text else "heldout" if "heldout" in text else "unknown"


def stage(path, config):
    if config.get("replay_prefixes") or "v2" in str(path).lower():
        return "v2_corrected_protocol"
    return "pilot" if phase(path) == "pilot" else "v1_original_protocol"


def fraction(count, denominator):
    return {"count": count, "n": denominator, "rate": count / denominator if denominator else None}


def eligible(value):
    return (
        value.get("finished") is True
        and value.get("status") not in CENSORED
        and isinstance(value.get("success"), bool)
    )


def episode_paths(inputs):
    result = set()
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            result.add(path)
        elif path.is_dir():
            if path.name == "episodes":
                result.update(path.glob("*.json"))
            result.update(path.glob("**/episodes/*.json"))
    return sorted(result)


def run_path(path):
    return next((p / "run.json" for p in path.parents if (p / "run.json").is_file()), None)


def load(inputs, candidate):
    episodes, excluded, warnings, sources, runs = [], [], [], [], {}
    for item in inputs:
        if not Path(item).expanduser().exists():
            warnings.append(f"Входной путь отсутствует: {item}")
    seen = {}
    conflicts = set()
    for path in episode_paths(inputs):
        try:
            value = read(path)
            if not isinstance(value, dict) or value.get("variant") not in {
                candidate,
                *BASELINES,
                *OPTIONAL_ABLATIONS,
            }:
                continue
            metadata = run_path(path)
            run = {}
            if metadata:
                if metadata not in runs:
                    runs[metadata] = read(metadata)
                run = runs[metadata]
            record = {
                **value,
                "path": str(path),
                "split": phase(path),
                "stage": stage(path, run.get("config", {})),
                "run_path": str(metadata) if metadata else None,
                "config": run.get("config", {}),
            }
            sources.append({"path": str(path), "sha256": digest(path), "kind": "episode"})
            identity = (
                record["split"],
                record["stage"],
                record.get("benchmark"),
                record.get("model"),
                record.get("task_id"),
                record.get("seed"),
                record["variant"],
            )
            if identity in seen:
                if seen[identity] != digest(path):
                    conflicts.add(identity)
                continue
            seen[identity] = digest(path)
            record["identity"] = identity
            if eligible(value):
                if record.get("status") == "episode_compute_cap":
                    record["success"] = False
                episodes.append(record)
            else:
                excluded.append(
                    {
                        "path": str(path),
                        "status": value.get("status"),
                        "finished": value.get("finished"),
                        "variant": value.get("variant"),
                    }
                )
        except (OSError, TypeError, ValueError, KeyError) as error:
            warnings.append(f"Не прочитан {path}: {error}")
    if conflicts:
        episodes = [e for e in episodes if e["identity"] not in conflicts]
        warnings.append(f"Исключены конфликтующие повторы эпизодов: {len(conflicts)}.")
    amendments = []
    inspected = set()
    for metadata, run in runs.items():
        sources.append(
            {
                "path": str(metadata),
                "sha256": digest(metadata),
                "kind": "run_metadata",
                "source_sha256": run.get("source_sha256"),
                "config_sha256": run.get("config_sha256"),
                "manifest_sha256": run.get("manifest_sha256"),
            }
        )
        for parent in metadata.parents:
            amendment = parent / "protocol_amendment_v2.json"
            if amendment.is_file() and amendment not in inspected:
                inspected.add(amendment)
                amendments.append(
                    {"path": str(amendment), "sha256": digest(amendment), "content": read(amendment)}
                )
                break
    return episodes, excluded, warnings, sources, runs, amendments


def resolve_frozen_manifest(metadata, run):
    """Use relocated manifests only with the authentic run's exact file hash."""
    value = run.get("config", {}).get("manifest")
    if not value:
        raise FileNotFoundError("run.json не содержит config.manifest")
    configured = Path(str(value)).expanduser()
    direct = configured if configured.is_absolute() else Path(metadata).parent / configured
    expected = run.get("manifest_sha256")
    expected = expected.lower() if isinstance(expected, str) else None
    if direct.is_file() and (not expected or digest(direct) == expected):
        return direct
    if not expected or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise FileNotFoundError("Переносимый fallback требует записанного manifest_sha256")
    for parent in Path(metadata).resolve().parents:
        candidate = parent / configured.name
        if candidate.is_file() and digest(candidate) == expected:
            return candidate
    raise FileNotFoundError(
        "Manifest не найден либо SHA-256 не совпадает; непроверенная подстановка запрещена"
    )


def run_call_accounting(metadata, run):
    """Map physical calls only when the frozen ordered episode slices reconcile.

    Count logical inference even for exact replay. Do not read the subscription
    auth state or equate this run's launch IDs with the whole campaign ledger.
    """
    config = run.get("config", {})
    directory = metadata.parent
    logfile = directory / "model_calls.jsonl"
    result = {"run": str(directory), "mapped_episodes": {}, "warnings": []}
    if not logfile.is_file():
        result["warnings"].append("Нет model_calls.jsonl: физические вызовы не восстановлены.")
        return result
    try:
        log = [json.loads(line) for line in logfile.read_text().splitlines() if line.strip()]
        manifest = resolve_frozen_manifest(metadata, run)
        result["resolved_manifest"] = str(manifest)
        result["manifest_sha256"] = digest(manifest)
        tasks = read(manifest)["tasks"]
        selected = config.get("task_keys")
        tasks = [t for t in tasks if selected is None or t["key"] in selected]
        cursor = 0
        missing_predecessor = False
        for task in tasks:
            for variant in config.get("variants", []):
                path = directory / "episodes" / f"{task['key']}--{variant}.json"
                if not path.is_file():
                    missing_predecessor = True
                    continue
                episode = read(path)
                # An unfinished final checkpoint is deliberately not assigned a slice.
                if episode.get("finished") is not True:
                    missing_predecessor = True
                    continue
                if missing_predecessor:
                    raise ValueError(
                        "Завершённый эпизод найден после пропущенного/незавершённого предшественника"
                    )
                count = episode.get("usage", {}).get("calls")
                if not isinstance(count, int) or count < 0 or isinstance(count, bool):
                    raise ValueError("Usage.calls отсутствует или некорректен")
                subset = log[cursor : cursor + count]
                if len(subset) != count:
                    raise ValueError("Сумма usage эпизодов выходит за границы журнала")
                if any(r.get("model") != episode.get("model") for r in subset):
                    raise ValueError("Модель в срезе журнала не совпадает с эпизодом")
                missions = {
                    r.get("input", {}).get("mission") for r in subset if r.get("input", {}).get("mission")
                }
                expected = task.get("mission")
                if expected and missions and missions != {expected}:
                    raise ValueError("Миссии не совпали с замороженным порядком эпизодов")
                ids = {
                    r["subscription_launch"] for r in subset if isinstance(r.get("subscription_launch"), int)
                }
                new_ids = {
                    r["subscription_launch"]
                    for r in subset
                    if not r.get("replayed", False) and isinstance(r.get("subscription_launch"), int)
                }
                replay_ids = {
                    r["subscription_launch"]
                    for r in subset
                    if r.get("replayed", False) and isinstance(r.get("subscription_launch"), int)
                }
                result["mapped_episodes"][str(path)] = {
                    "logical_calls": count,
                    "log_rows": len(subset),
                    "replayed_logical_calls": sum(bool(r.get("replayed")) for r in subset),
                    "new_physical_launches": len(new_ids),
                    "historical_launch_ids": sorted(replay_ids),
                    "all_original_launch_ids": sorted(ids),
                    "fresh_launch_ids": sorted(new_ids),
                    "rows_without_launch_id": sum(
                        not isinstance(r.get("subscription_launch"), int) for r in subset
                    ),
                    "source_log_sha256": digest(logfile),
                }
                cursor += count
        result["unmapped_log_rows"] = len(log) - cursor
        result["source_log_sha256"] = digest(logfile)
        if cursor < len(log):
            result["warnings"].append(
                "Есть строки текущего/незавершённого эпизода; они не приписаны завершённым задачам."
            )
    except (OSError, ValueError, TypeError, KeyError) as error:
        # Partial mappings could otherwise turn a sequencing error into a false cost claim.
        result["mapped_episodes"] = {}
        result["warnings"].append(f"Срезы журнала не подтверждены: {error}")
    return result


def summary_group(episodes, candidate):
    selected = [e for e in episodes if e["variant"] == candidate]
    steps = [(e, turn.get("diagnostics") or {}) for e in selected for turn in e.get("trajectory", [])]
    count = len(steps)
    result = {
        "episodes": len(selected),
        "successes": sum(e["success"] for e in selected),
        "executed_steps": count,
        "logical_calls": sum(e.get("usage", {}).get("calls", 0) for e in selected),
        "status_counts": dict(Counter(e.get("status", "unknown") for e in selected)),
        "interventions": dict(Counter(d.get("intervention", "unreported") for _, d in steps)),
        "step_scope": "Only executed actions with retained trajectory; a failed final act may spend calls without a logged turn.",
    }
    if candidate == "diagnostic":
        measured = sum(d.get("probe_count", 0) > 0 for _, d in steps)
        result.update(
            {
                "measured_steps": fraction(measured, count),
                "no_public_slots_steps": fraction(
                    sum(d.get("reason") == "no_public_measurable_slots" for _, d in steps), count
                ),
                "full_audits": fraction(sum(d.get("full_audit") is True for _, d in steps), count),
                "diagnoses": dict(Counter(d.get("diagnosis", "unreported") for _, d in steps)),
                "reasons": dict(Counter(d.get("reason", "unreported") for _, d in steps)),
                "qualified_roles": {
                    role: fraction(
                        sum((d.get("qualified_decoders") or {}).get(role) is True for _, d in steps),
                        sum(role in (d.get("qualified_decoders") or {}) for _, d in steps),
                    )
                    for role in ("literal", "relational")
                },
                "measurement_limit": "Exact source quote/subject/slot checks, not a semantic truth oracle or proof that all plan-relevant facts were retained.",
            }
        )
    else:
        profiles = [(e, d["profile"]) for e, d in steps if isinstance(d.get("profile"), dict)]
        complete = [(e, p) for e, p in profiles if p.get("observed_pairs") == 1]
        healthy = [
            (e, p)
            for e, p in complete
            if p.get("beneficial_correction") == 1 and p.get("harmful_corruption") == 0
        ]
        healthy_episode_ids = {e["path"] for e, _ in healthy}
        result.update(
            {
                "critic_requested_revision": fraction(
                    sum(d.get("critic_requested_revision") is True for _, d in steps), count
                ),
                "critic_evidence_literal_present": fraction(
                    sum(d.get("critic_evidence_present") is True for _, d in steps), count
                ),
                "profile_steps": fraction(len(profiles), count),
                "complete_profile_pairs": len(complete),
                "beneficial_correction": fraction(
                    sum(p.get("beneficial_correction") == 1 for _, p in complete), len(complete)
                ),
                "harmful_corruption": fraction(
                    sum(p.get("harmful_corruption") == 1 for _, p in complete), len(complete)
                ),
                "healthy_quote_profile": fraction(len(healthy), len(complete)),
                "episodes_with_healthy_profile": len(healthy_episode_ids),
                "profile_passed_but_task_failed": fraction(
                    sum(e["path"] in healthy_episode_ids and not e["success"] for e in selected),
                    len(healthy_episode_ids),
                ),
                "measurement_limit": "Quote repair/preservation is a narrow source-fidelity proxy; critic evidence presence does not certify the critic conclusion or task plan.",
            }
        )
    return result


def controlled_key(episode):
    config = episode.get("config", {})
    provider = config.get("provider", {})
    return (
        episode["split"],
        episode.get("benchmark"),
        episode.get("model"),
        episode.get("task_id"),
        episode.get("seed"),
        config.get("horizons", {}).get(episode.get("benchmark")),
        config.get("episode_call_cap"),
        provider.get("reasoning_effort"),
        provider.get("backend"),
        episode.get("protocol"),
    )


def cost_pairs(episodes, candidate):
    methods = defaultdict(list)
    for e in episodes:
        methods[(controlled_key(e), e["variant"])].append(e)
    pairs = []
    for ours in [e for e in episodes if e["variant"] == candidate]:
        for baseline in sorted(BASELINES | OPTIONAL_ABLATIONS):
            matches = methods.get((controlled_key(ours), baseline), [])
            if len(matches) != 1:
                continue
            theirs = matches[0]
            oc = ours.get("usage", {}).get("calls")
            bc = theirs.get("usage", {}).get("calls")
            pairs.append(
                {
                    "split": ours["split"],
                    "candidate_stage": ours["stage"],
                    "benchmark": ours.get("benchmark"),
                    "model": ours.get("model"),
                    "task_id": ours.get("task_id"),
                    "key": ours.get("key"),
                    "baseline": baseline,
                    "baseline_label": ABLATION_LABEL if baseline == "raw_actor" else baseline,
                    "comparison_kind": "exploratory_ablation"
                    if baseline in OPTIONAL_ABLATIONS
                    else "published_baseline",
                    "candidate_success": ours["success"],
                    "baseline_success": theirs["success"],
                    "candidate_logical_calls": oc,
                    "baseline_logical_calls": bc,
                    "logical_call_ratio": oc / bc if isinstance(oc, (int, float)) and bc else None,
                    "candidate_status": ours.get("status"),
                    "baseline_status": theirs.get("status"),
                    "candidate_path": ours["path"],
                    "baseline_path": theirs["path"],
                    "claim": "Same-task descriptive cost comparison, not global SOTA or causal attribution.",
                }
            )
    return pairs


def failure_examples(episodes, candidate, call_maps):
    examples = []
    for episode in sorted(
        episodes, key=lambda e: (e["split"], e["stage"], e.get("benchmark", ""), e.get("key", ""))
    ):
        if episode["variant"] != candidate or episode["success"]:
            continue
        turns = []
        for turn in episode.get("trajectory", [])[-3:]:
            d = turn.get("diagnostics") or {}
            turns.append(
                {
                    "step": turn.get("step"),
                    "observation": turn.get("observation", "")[:600],
                    "action": turn.get("action"),
                    "next_observation": turn.get("next_observation", "")[:600],
                    "intervention": d.get("intervention"),
                    "diagnosis": d.get("diagnosis"),
                    "reason": d.get("reason", d.get("gate_reason")),
                    "full_audit": d.get("full_audit"),
                    "profile": d.get("profile"),
                }
            )
        examples.append(
            {
                "path": episode["path"],
                "split": episode["split"],
                "stage": episode["stage"],
                "key": episode.get("key"),
                "benchmark": episode.get("benchmark"),
                "status": episode.get("status"),
                "logical_calls": episode.get("usage", {}).get("calls"),
                "call_accounting": call_maps.get(episode["path"]),
                "last_public_turns": turns,
            }
        )
        if len(examples) == 3:
            break
    return examples


def markdown(data):
    def render(value):
        return json.dumps(value, ensure_ascii=False, indent=2)

    lines = [
        "# Наблюдения о работе механизма",
        "",
        "Анализ выбранных запусков; это не автоматическая окончательная сводка всей кампании.",
        "",
        "Анализ описательный: наблюдаемое совпадение вмешательства и исхода не доказывает причинность. "
        "Пилот, первоначальный v1 и исправленный v2 не являются независимыми повторениями одной процедуры.",
        "",
        "## Покрытие и стоимость",
        "",
        f"Оцениваемых завершённых эпизодов: {data['evaluable_episodes']}; исключённых: {len(data['excluded'])}.",
        "",
        "usage.calls — логические модельные вызовы, включая точное воспроизведение исторических ответов. "
        "new_physical_launches — новые уникальные ID запусков в подтверждённом срезе журнала; исторические ID считаются отдельно. "
        "Это не полный расход общего ledger подписки. Точное повторное использование ответа не является новой независимой выборкой модели.",
        "",
    ]
    if data["amendments"]:
        lines += [
            "## Раскрытие изменения протокола",
            "",
            "После первоначальных исходов исправлены неодинаковая обработка недоступных команд "
            "и консервативный разбор отрицаний. v1 сохранён как инженерный этап. В v2 используются одинаковая обратная связь среды "
            "и проверяемый replay только при совпадении SHA-256 запроса. Это раскрытое изменение после наблюдения результатов, а не подтверждающее исследование.",
            "",
        ]
        for item in data["amendments"]:
            lines += [
                f"Источник: `{item['path']}`, SHA-256 `{item['sha256']}`.",
                "",
                "```json",
                render(item["content"]),
                "```",
                "",
            ]
    lines += [
        "## Stock-задачи и узкие пробы — разные выводы",
        "",
        "Полезная коррекция/сохранение цитаты проверяют воспроизведение источника. Высокая точность, в том числе потолочный результат "
        "на source-copy микрозадаче, не доказывает перенос на выбор игровых действий, восстановление плана или task success. "
        "Тест батареи не входит в знаменатель успеха исходного benchmark. Здесь анализируются только журналы реальных эпизодов; "
        "результаты отдельной батареи не подмешиваются.",
        "",
        "Для диагностического механизма exact subject/quote/slot проверка не является проверкой полной семантической истины. "
        "Отсутствие доступных публичных слотов означает отсутствие измерения. Для feedback присутствие цитаты у критика не доказывает правильность совета.",
        "",
    ]
    for group in data["groups"]:
        lines += [
            f"## {group['split']} · {group['stage']} · {group['benchmark']} · {group['model']}",
            "",
            "Шаговые доли относятся только к выполненным действиям с сохранённой trajectory; последний неисполненный act "
            "может израсходовать вызовы без отдельного шага.",
            "",
            "```json",
            render(group["statistics"]),
            "```",
            "",
        ]
    lines += [
        "## Стоимость на одних и тех же задачах",
        "",
        "Сопоставлены только одинаковые модель, effort, задача, seed, публичный протокол, горизонт и лимит вызовов эпизода. "
        "Различия числа вызовов могут объяснять различия качества; отношения не доказывают эффективность механизма. "
        "v2 кандидата может сопоставляться с неизменённым baseline v1 при раскрытой поправке; исходные партии кандидата не объединяются.",
        "",
        "| Этап | Среда | Задача | Baseline | Успех кандидат / baseline | Вызовов кандидат / baseline | Отношение |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in data["same_task_cost_pairs"]:
        ratio = row["logical_call_ratio"]
        lines.append(
            f"| {row['split']} {row['candidate_stage']} | {row['benchmark']} | {row['key']} | {row['baseline_label']} | "
            f"{int(row['candidate_success'])} / {int(row['baseline_success'])} | "
            f"{row['candidate_logical_calls']} / {row['baseline_logical_calls']} | {ratio:.2f} |"
            if ratio is not None
            else f"| {row['split']} | {row['benchmark']} | {row['key']} | {row['baseline']} | — | — | — |"
        )
    if not data["same_task_cost_pairs"]:
        lines += ["", "Подтверждённых сопоставимых пар пока нет."]
    if any(row["baseline"] == "raw_actor" for row in data["same_task_cost_pairs"]):
        lines += [
            "",
            "«Исходный актор · абляция» — дополнительный актор без критика и проб. "
            "Это исследовательское сравнение собственного механизма, не опубликованный SOTA и не проверка значимости.",
        ]
    lines += [
        "",
        "## До трёх публичных примеров неуспешных эпизодов",
        "",
        "Примеры иллюстрируют зарегистрированную последовательность, не устанавливая причину неуспеха. "
        "Показаны последние три действия, а не полный скрытый мир или эталонный план.",
        "",
    ]
    for example in data["failure_examples"]:
        lines += ["```json", render(example), "```", ""]
    if not data["failure_examples"]:
        lines += ["Среди загруженных оцениваемых эпизодов неуспешных примеров нет; они не выдумываются.", ""]
    lines += ["## Предупреждения", ""] + [f"- {item}" for item in data["warnings"]]
    if not data["warnings"]:
        lines.append("Ошибок чтения и согласования журналов не обнаружено.")
    lines += [
        "",
        "Выводы ограничены переданными файлами. Отсутствие результата не равно неуспеху. "
        "Этот отчёт не подтверждает превосходство над абсолютным SOTA и не заменяет причинную абляцию.",
        "",
    ]
    return "\n".join(lines)


def analyze(inputs, candidate, output):
    episodes, excluded, warnings, sources, runs, amendments = load(inputs, candidate)
    accounting = [run_call_accounting(path, run) for path, run in runs.items()]
    call_maps = {path: value for run in accounting for path, value in run["mapped_episodes"].items()}
    warnings += [f"{run['run']}: {warning}" for run in accounting for warning in run["warnings"]]
    groups = defaultdict(list)
    for e in episodes:
        if e["variant"] == candidate:
            groups[
                (e["split"], e["stage"], e.get("benchmark"), e.get("model"), e.get("source_sha256"))
            ].append(e)
    grouped = []
    for (split, version, benchmark, model, source), values in sorted(groups.items(), key=str):
        stats = summary_group(values, candidate)
        mapped = [call_maps[e["path"]] for e in values if e["path"] in call_maps]
        stats["call_accounting"] = {
            "mapped_episodes": len(mapped),
            "episodes": len(values),
            "logical_calls_mapped": sum(v["logical_calls"] for v in mapped),
            "replayed_logical_calls": sum(v["replayed_logical_calls"] for v in mapped),
            "new_physical_launches": len({i for v in mapped for i in v["fresh_launch_ids"]}),
            "historical_unique_launch_ids": len({i for v in mapped for i in v["historical_launch_ids"]}),
            "not_campaign_ledger": True,
        }
        grouped.append(
            {
                "split": split,
                "stage": version,
                "benchmark": benchmark,
                "model": model,
                "source_sha256": source,
                "statistics": stats,
            }
        )
    data = {
        "schema_version": "mechanism-observational-appendix-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "selected_runs_analysis",
        "candidate": candidate,
        "evaluable_episodes": len(episodes),
        "excluded": excluded,
        "groups": grouped,
        "candidate_episodes": sum(e["variant"] == candidate for e in episodes),
        "same_task_cost_pairs": cost_pairs(episodes, candidate),
        "failure_examples": failure_examples(episodes, candidate, call_maps),
        "run_call_accounting": accounting,
        "amendments": amendments,
        "sources": sources,
        "analyzer_sha256": digest(__file__),
        "warnings": warnings,
        "causal_claim": False,
        "battery_in_stock_task_success": False,
        "absolute_sota_claim": False,
    }
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() in {".json", ".md"}:
        json_path, md_path = output.with_suffix(".json"), output.with_suffix(".md")
    else:
        json_path, md_path = output / "mechanism_analysis.json", output / "mechanism_analysis.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown(data), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "evaluable": len(episodes),
        "candidate_episodes": sum(e["variant"] == candidate for e in episodes),
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", "--episodesdirs", nargs="*", default=[])
    parser.add_argument("--candidate", choices=["diagnostic", "feedback"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.episodes, args.candidate, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
