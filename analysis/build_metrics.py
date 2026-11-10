"""Numeric-only metrics for selected study runs; never call models or emit QA text.

Example: python build_metrics.py --qa ../pilot-v2/qa-p0 --native ../pilot-v2/native
    --out ../analysis/pilot-v2
QA source versions/cohorts/models/repeats are kept separate. Infrastructure
outages are not silently scored as failures of the cognitive mechanism.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from paired_statistics import clopper_pearson, paired_binary

QA_MODES = (
    "controlled",
    "grounding_only",
    "always_critique",
    "facts_only",
    "no_feedback",
)
CONDITIONS = ("correct_advice", "false_advice")
CUTOFFS = (16, 32, 64, 128)
NATIVE_BASELINES = ("stateact", "reflact_paper")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashed(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path, warnings):
    result = []
    if not Path(path).exists():
        return result
    for index, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip():
            try:
                result.append(json.loads(line))
            except ValueError:
                warnings.append(f"Incomplete JSONL line excluded: {path.name}:{index}")
    return result


def selected_paths(inputs, pattern):
    result = set()
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            result.add(path)
        elif path.is_dir():
            result.update(path.glob(pattern))
    return sorted(result)


def cohort(path, split=None):
    values = [
        part
        for part in Path(path).parts
        if part.startswith(("pilot", "main", "heldout"))
    ]
    label = (
        values[-1]
        if values
        else "main"
        if split == "test"
        else "pilot"
        if split == "dev"
        else "unknown"
    )
    phase = (
        "pilot"
        if label.startswith("pilot")
        else "main"
        if label.startswith(("main", "heldout"))
        else "unknown"
    )
    return phase, label


def rate(hits, n):
    return hits / n if n else None


def interval(hits, n):
    return clopper_pearson(hits, n) if n else [None, None]


def pair_stats(a, b):
    if not a:
        return {
            "n": 0,
            "a_only": 0,
            "b_only": 0,
            "difference": None,
            "interval95": [None, None],
            "interval_method": "conservative exact discordant-cell Bonferroni",
            "mcnemar_exact_two_sided_p": None,
        }
    return paired_binary(a, b)


def transport_log(record, allow_legacy=False):
    """Use typed errors; an explicit v1 outage audit permits its old heuristic."""
    if record.get("error_type") == "BackendTransportError":
        return True
    if not allow_legacy:
        return False
    usage = record.get("usage_delta", {})
    return (
        record.get("error_type") == "ModelOutputError"
        and not record.get("output", "").strip()
        and record.get("backend_usage") is None
        and not usage.get("input_tokens", 0)
        and not usage.get("output_tokens", 0)
    )


def log_index(path, warnings, allow_legacy=False):
    records = lines(path, warnings)
    return {
        "records": records,
        "transport_ids": {
            r.get("subscription_launch")
            for r in records
            if transport_log(r, allow_legacy)
            and isinstance(r.get("subscription_launch"), int)
        },
        "by_prompt": {
            key: [r for r in records if r.get("prompt_sha256") == key]
            for key in {r.get("prompt_sha256") for r in records}
        },
    }


def nonnegative_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def numeric_sum(values):
    """Unknown measurements stay unknown rather than becoming a measured zero."""
    known = [v for v in values if nonnegative_number(v)]
    return sum(known) if known else None


def receipt_identity(record):
    """Proof that two log rows refer to the very same physical completion."""
    launch = record.get("subscription_launch")
    campaign, prompt = record.get("campaign"), record.get("prompt_sha256")
    output, backend = record.get("output"), record.get("backend_usage")
    if (
        not isinstance(launch, int)
        or isinstance(launch, bool)
        or not isinstance(campaign, str)
        or not campaign
        or not isinstance(prompt, str)
        or not prompt
        or not isinstance(output, str)
        or not isinstance(backend, dict)
    ):
        return None
    names = ("input_tokens", "output_tokens", "cached_input_tokens")
    if not all(nonnegative_number(backend.get(k)) for k in names):
        return None
    if backend["cached_input_tokens"] > backend["input_tokens"]:
        return None
    delta = record.get("usage_delta", {})
    if not all(
        nonnegative_number(delta.get(k))
        for k in ("calls", "input_tokens", "output_tokens")
    ):
        return None
    return (
        campaign,
        launch,
        prompt,
        hashlib.sha256(output.encode()).hexdigest(),
        record.get("error_type"),
        *(backend[k] for k in names),
        *(delta[k] for k in ("calls", "input_tokens", "output_tokens")),
    )


def usage_segment(usage, records):
    """Identify one log segment by its numeric usage fingerprint, without task text.

    Resumes may append old prefixes. Ambiguous matches stay unassigned, unless
    they are proven duplicate receipts for identical physical completions and
    differ by at most one microsecond of aggregate elapsed bookkeeping.
    """
    names = ("calls", "input_tokens", "output_tokens", "seconds")
    if not all(nonnegative_number(usage.get(k)) for k in names):
        return None
    if usage["calls"] == 0:
        return [] if usage["input_tokens"] == usage["output_tokens"] == 0 else None
    matches = []
    called = [r for r in records if r.get("usage_delta", {}).get("calls", 0) > 0]
    for start in range(len(called)):
        totals = dict.fromkeys(names, 0)
        for stop in range(start, len(called)):
            delta = called[stop].get("usage_delta", {})
            if not all(nonnegative_number(delta.get(k)) for k in names):
                break
            for name in names:
                totals[name] += delta[name]
            if totals["calls"] > usage["calls"]:
                break
            if totals["calls"] == usage["calls"]:
                if all(
                    math.isclose(totals[k], usage[k], rel_tol=1e-10, abs_tol=1e-6)
                    for k in names
                ):
                    matches.append(called[start : stop + 1])
                break
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        identities = [
            tuple(receipt_identity(r) for r in segment) for segment in matches
        ]
        elapsed = [
            sum(r["usage_delta"]["seconds"] for r in segment) for segment in matches
        ]
        if (
            all(
                all(identity is not None for identity in sequence)
                for sequence in identities
            )
            and all(sequence == identities[0] for sequence in identities[1:])
            and max(elapsed) - min(elapsed) <= 1e-6
        ):
            return matches[0]
    return None


def cost_unit(usage, records=None, transport=False):
    """Recorded token totals are lower bounds until every backend receipt is verified."""
    usage = usage if isinstance(usage, dict) else {}
    calls = usage.get("calls")
    result = {
        "logical_calls": calls if nonnegative_number(calls) else None,
        "input_tokens_lower_bound": usage.get("input_tokens")
        if nonnegative_number(usage.get("input_tokens"))
        else None,
        "output_tokens_lower_bound": usage.get("output_tokens")
        if nonnegative_number(usage.get("output_tokens"))
        else None,
        "inference_seconds": usage.get("seconds")
        if nonnegative_number(usage.get("seconds"))
        else None,
        "cached_input_tokens_lower_bound": None,
        "complete_token_telemetry": None,
        "complete_cached_token_telemetry": None,
        "transport_unknown": bool(transport),
    }
    if records is None:
        if transport:
            result["complete_token_telemetry"] = False
        return result
    called = [r for r in records if r.get("usage_delta", {}).get("calls", 0) > 0]
    if sum(r.get("usage_delta", {}).get("calls", 0) for r in called) != calls:
        return result
    receipts = [r.get("backend_usage") for r in called]
    complete = all(
        isinstance(r, dict)
        and all(nonnegative_number(r.get(k)) for k in ("input_tokens", "output_tokens"))
        for r in receipts
    )
    if complete:
        complete = all(
            all(
                nonnegative_number(record.get("usage_delta", {}).get(k))
                and record["backend_usage"][k] == record["usage_delta"][k]
                for k in ("input_tokens", "output_tokens")
            )
            for record in called
        )
    result["complete_token_telemetry"] = complete and not transport
    cached = [r.get("cached_input_tokens") for r in receipts if isinstance(r, dict)]
    result["cached_input_tokens_lower_bound"] = numeric_sum(cached) if called else 0
    result["complete_cached_token_telemetry"] = (
        complete
        and len(cached) == len(called)
        and all(nonnegative_number(v) for v in cached)
        and not transport
    )
    return result


def aggregate_cost(units):
    complete_n = sum(u["complete_token_telemetry"] is True for u in units)
    result = {
        "n_cost_known": complete_n,
        "n_complete_token_cost": complete_n,
        "n_token_cost_unknown": len(units) - complete_n,
        "n_transport_unknown": sum(u["transport_unknown"] for u in units),
        "n_recorded_token_prefix": sum(
            u["input_tokens_lower_bound"] is not None
            and u["output_tokens_lower_bound"] is not None
            for u in units
        ),
        "n_complete_cached_token_cost": sum(
            u["complete_cached_token_telemetry"] is True for u in units
        ),
        "n_logical_calls_known": sum(u["logical_calls"] is not None for u in units),
        "n_inference_time_known": sum(
            u["inference_seconds"] is not None for u in units
        ),
        "token_cost_complete": complete_n == len(units) and bool(units),
        "cost_telemetry_note": "Token sums are recorded lower bounds; n_cost_known means complete verified backend token receipts, not merely an aggregate usage dictionary",
    }
    for field in (
        "logical_calls",
        "input_tokens_lower_bound",
        "output_tokens_lower_bound",
        "inference_seconds",
        "cached_input_tokens_lower_bound",
    ):
        result[field + "_total"] = numeric_sum(u[field] for u in units)
    result["input_tokens_total"] = result["input_tokens_lower_bound_total"]
    result["output_tokens_total"] = result["output_tokens_lower_bound_total"]
    for kind in ("input", "output"):
        result[f"complete_{kind}_tokens_total"] = numeric_sum(
            u[f"{kind}_tokens_lower_bound"]
            for u in units
            if u["complete_token_telemetry"] is True
        )
    result["calls_total"] = result["logical_calls_total"]
    result["seconds_total"] = result["inference_seconds_total"]
    return result


def quote_format_flags(profile):
    """Predeclared secondary typography sensitivity; never changes runtime flags."""
    source = profile.get("source_quote")
    source_available = isinstance(source, str) and bool(source)

    def normalize(value, mode):
        if mode != "strict_whitespace":
            value = value.casefold()
        if mode == "casefold_punctuation_removed":
            value = "".join(
                c for c in value if not unicodedata.category(c).startswith("P")
            )
        return " ".join(value.split())

    strict = {}
    for arm, key in (
        ("correction", "corrected_quote"),
        ("corruption", "preserved_quote"),
    ):
        value = profile.get(key)
        match = (
            normalize(value, "strict_whitespace")
            == normalize(source, "strict_whitespace")
            if source_available and isinstance(value, str)
            else None
        )
        strict[arm] = (
            (match if arm == "correction" else not match) if match is not None else None
        )
    result = []
    for mode in (
        "strict_whitespace",
        "casefold_whitespace",
        "casefold_punctuation_removed",
    ):
        row = {"normalization": mode, "source_available": source_available}
        for arm, key, runtime_key in (
            ("correction", "corrected_quote", "beneficial_correction"),
            ("corruption", "preserved_quote", "harmful_corruption"),
        ):
            value = profile.get(key)
            match = (
                normalize(value, mode) == normalize(source, mode)
                if source_available and isinstance(value, str)
                else None
            )
            row[arm] = (
                (match if arm == "correction" else not match)
                if match is not None
                else None
            )
            raw = profile.get(runtime_key)
            row["runtime_" + arm] = bool(raw) if raw in (0, 1) else None
            row["strict_" + arm] = strict[arm]
        result.append(row)
    return result


def summarize_format_units(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in (*fields, "normalization"))].append(row)
    result = []
    for key, values in sorted(groups.items(), key=str):
        summary = dict(zip((*fields, "normalization"), key, strict=True))
        summary.update(
            {
                "n_profile_attempts": len(values),
                "source_available_n": sum(v["source_available"] for v in values),
                "analysis_kind": "secondary_descriptive_format_sensitivity",
                "unit_warning": "Logged profile attempts; native steps within a task are dependent. No semantic-equivalence or runtime-route claim.",
            }
        )
        for arm in ("correction", "corruption"):
            known = [v for v in values if isinstance(v[arm], bool)]
            events = sum(v[arm] for v in known)
            runtime = [v for v in known if isinstance(v["runtime_" + arm], bool)]
            summary.update(
                {
                    arm + "_observed_n": len(known),
                    arm + "_missing_n": len(values) - len(known),
                    arm + "_events": events,
                    arm + "_rate": rate(events, len(known)),
                    arm + "_runtime_comparable_n": len(runtime),
                    arm + "_runtime_disagreement_n": sum(
                        v[arm] != v["runtime_" + arm] for v in runtime
                    ),
                    arm + "_strict_disagreement_n": sum(
                        v[arm] != v["strict_" + arm] for v in known
                    ),
                }
            )
        result.append(summary)
    return result


def qa_key(row):
    return (
        row["phase"],
        row["cohort"],
        row.get("model"),
        row.get("source_identity"),
        row.get("repeat_id", 0),
        row.get("reasoning_effort", "unknown"),
    )


def qa_group_fields(key):
    return dict(
        zip(
            (
                "phase",
                "cohort",
                "model",
                "source_identity",
                "repeat_id",
                "reasoning_effort",
            ),
            key,
            strict=True,
        )
    )


def final_request(row, logs):
    hashes = row.get("request_sha256", [])
    allowed = set(row.get("new_launch_ids", []) + row.get("reused_launch_ids", []))
    for key in reversed(hashes):
        candidates = [
            r
            for r in logs["by_prompt"].get(key, [])
            if r.get("subscription_launch") in allowed
        ]
        for record in candidates:
            purpose = record.get("purpose", "")
            channel = (
                "critique"
                if purpose.startswith("Revise or retain the draft action")
                else "facts_only"
                if purpose.startswith("Reassess the draft using")
                else "rebuild"
                if purpose.startswith("Build a fresh understanding")
                else None
            )
            if channel:
                return key, channel
    return None, (row.get("diagnostics") or {}).get("intervention")


def load_qa(inputs, warnings, provenance):
    versions = defaultdict(list)
    for path in selected_paths(inputs, "**/outcomes.jsonl"):
        meta_path = path.parent / "run_metadata.json"
        metadata = read(meta_path) if meta_path.exists() else {}
        frozen = metadata.get("frozen", {})
        _, path_cohort = cohort(path, frozen.get("split"))
        logs = log_index(
            path.parent / "model_calls.jsonl",
            warnings,
            allow_legacy=path_cohort == "pilot",
        )
        provenance.append(
            {"file": str(path), "sha256": digest(path), "kind": "qa_outcomes"}
        )
        for value in lines(path, warnings):
            phase, label = cohort(path, value.get("study_split"))
            row = {
                **value,
                "phase": phase,
                "cohort": label,
                "reasoning_effort": frozen.get("reasoning_effort", "unknown"),
                "_file": str(path),
            }
            ids = set(
                value.get("new_launch_ids", [])
                + value.get("reused_launch_ids", [])
                + value.get("shared_draft_new_launch_ids", [])
                + value.get("shared_draft_reused_launch_ids", [])
            )
            row["_transport"] = value.get("status") in {
                "infrastructure_censored",
                "draft_infrastructure_censored",
            } or bool(ids & logs["transport_ids"])
            row["_final_prompt"], row["_route"] = final_request(row, logs)
            # A policy and its shared initial draft are not contiguous in this
            # log; their usage fingerprints are matched separately, then joined.
            policy_records = usage_segment(value.get("usage", {}), logs["records"])
            draft_records = usage_segment(
                value.get("shared_draft_usage", {}), logs["records"]
            )
            total_records = (
                policy_records + draft_records
                if policy_records is not None and draft_records is not None
                else None
            )
            row["_cost"] = cost_unit(
                value.get("total_policy_usage", value.get("usage", {})),
                total_records,
                row["_transport"],
            )
            row["_eligible"] = (
                value.get("finished") is True
                and not row["_transport"]
                and value.get("status") not in {"campaign_budget_censored", "running"}
                and isinstance(value.get("after_correct"), bool)
            )
            key = (*qa_key(row), row.get("record_id"))
            versions[key].append(row)
    selected = []
    for key, values in versions.items():
        unique = {
            hashed({k: v for k, v in row.items() if not k.startswith("_")}): row
            for row in values
        }
        finished = [r for r in unique.values() if r.get("finished") is True]
        if len(finished) > 1:
            warnings.append(
                f"Multiple terminal QA attempts require adjudication: {key[-1]}"
            )
            # Terminal model errors must not be retried into a successful answer.
            for r in finished:
                r["_eligible"] = False
        pool = finished or list(unique.values())
        selected.append(max(pool, key=lambda row: row.get("attempt", 0)))
    return selected


def qa_metrics(rows):
    summaries, pairs, routes, profiles, coupling, drafts, numeric, transfer = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    groups = defaultdict(list)
    format_units = []
    for row in rows:
        groups[qa_key(row)].append(row)
        d = row.get("diagnostics") or {}
        p = d.get("profile") or {}
        if p:
            format_units.extend(
                {
                    **qa_group_fields(qa_key(row)),
                    "question_id": row.get("question_id"),
                    "condition": row.get("condition"),
                    "mode": row.get("mode"),
                    **flags,
                }
                for flags in quote_format_flags(p)
            )
        numeric.append(
            {
                **qa_group_fields(qa_key(row)),
                "question_id": row.get("question_id"),
                "source_dataset": row.get("source_dataset"),
                "condition": row.get("condition"),
                "mode": row.get("mode"),
                "attempt": row.get("attempt"),
                "status": row.get("status"),
                "finished": row.get("finished"),
                "mechanism_eligible": row["_eligible"],
                "transport_contaminated": row["_transport"],
                "before_correct": row.get("before_correct"),
                "after_correct": row.get("after_correct"),
                "route": row.get("_route"),
                "beneficial_correction": p.get("beneficial_correction"),
                "harmful_corruption": p.get("harmful_corruption"),
                "complete_profile": p.get("observed_pairs") == 1,
                "logical_calls": row.get("usage", {}).get("calls"),
                "policy_logical_calls_including_draft": row.get(
                    "total_policy_usage", {}
                ).get("calls"),
                **{("cost_" + k): v for k, v in row["_cost"].items()},
            }
        )
    for key, values in sorted(groups.items(), key=str):
        fields = qa_group_fields(key)
        by_question = defaultdict(list)
        for row in values:
            by_question[row.get("question_id")].append(row)
        initial = []
        for variants in by_question.values():
            flags = {
                r.get("before_correct")
                for r in variants
                if isinstance(r.get("before_correct"), bool)
            }
            if len(flags) == 1 and not any(
                r["_transport"] and r.get("status") == "draft_model_error"
                for r in variants
            ):
                initial.append(next(iter(flags)))
        drafts.append(
            {
                **fields,
                "question_n_started": len(by_question),
                "valid_initial_n": len(initial),
                "initial_correct_n": sum(initial),
                "initial_wrong_n": len(initial) - sum(initial),
                "initial_accuracy": rate(sum(initial), len(initial)),
                "interval95": interval(sum(initial), len(initial)),
                "unit": "one shared draft per question; advice conditions not double-counted",
            }
        )
        for condition in CONDITIONS:
            condition_rows = [r for r in values if r.get("condition") == condition]
            for mode in QA_MODES:
                records = [r for r in condition_rows if r.get("mode") == mode]
                if not records:
                    continue
                valid = [r for r in records if r["_eligible"]]
                initially_correct = [
                    r for r in valid if r.get("before_correct") is True
                ]
                initially_wrong = [r for r in valid if r.get("before_correct") is False]
                successes = sum(r["after_correct"] for r in valid)
                retained = sum(r["after_correct"] for r in initially_correct)
                repaired = sum(r["after_correct"] for r in initially_wrong)
                operational = [r for r in records if r.get("finished") is True]
                costs = [r["_cost"] for r in records]
                summaries.append(
                    {
                        **fields,
                        "condition": condition,
                        "mode": mode,
                        "n": len(valid),
                        "n_questions": len({r["question_id"] for r in valid}),
                        "n_records": len(records),
                        "n_finished_total": len(operational),
                        "n_transport_contaminated": sum(
                            r["_transport"] for r in records
                        ),
                        "n_missing_or_censored": sum(
                            not r.get("finished") for r in records
                        ),
                        "successes": successes,
                        "accuracy": rate(successes, len(valid)),
                        "interval95": interval(successes, len(valid)),
                        "initially_correct_n": len(initially_correct),
                        "retained_n": retained,
                        "retention": rate(retained, len(initially_correct)),
                        "retention_interval95": interval(
                            retained, len(initially_correct)
                        ),
                        "initially_wrong_n": len(initially_wrong),
                        "repaired_n": repaired,
                        "repair": rate(repaired, len(initially_wrong)),
                        "repair_interval95": interval(repaired, len(initially_wrong)),
                        "model_error_n": sum(
                            r.get("status") in {"model_error", "draft_model_error"}
                            for r in valid
                        ),
                        "operational_n": len(operational),
                        "operational_accuracy": rate(
                            sum(r.get("after_correct") is True for r in operational),
                            len(operational),
                        ),
                        "operational_definition": "correct answer delivered per attempted terminal case; infrastructure nonresponse is not a mechanism error",
                        "cost_n_started": len(records),
                        "cost_scope": "all selected attempted policy records including shared draft, not only mechanism-evaluable answers",
                        **aggregate_cost(costs),
                    }
                )
                route_counts = Counter(r.get("_route") or "not_observed" for r in valid)
                routes.extend(
                    {
                        **fields,
                        "condition": condition,
                        "mode": mode,
                        "route": route,
                        "count": count,
                        "n": len(valid),
                    }
                    for route, count in sorted(route_counts.items())
                )
            by_mode = {
                mode: {
                    r["question_id"]: r
                    for r in condition_rows
                    if r.get("mode") == mode and r["_eligible"]
                }
                for mode in QA_MODES
            }
            ours = by_mode["controlled"]
            for other in QA_MODES[1:]:
                common = sorted(set(ours) & set(by_mode[other]))
                consistent = [
                    q
                    for q in common
                    if hashed(ours[q].get("draft"))
                    == hashed(by_mode[other][q].get("draft"))
                ]
                for endpoint in ("accuracy", "retention", "repair"):
                    if endpoint == "retention" and condition != "false_advice":
                        continue
                    if endpoint == "repair" and condition != "correct_advice":
                        continue
                    selected = [
                        q
                        for q in consistent
                        if endpoint == "accuracy"
                        or ours[q].get("before_correct") is (endpoint == "retention")
                    ]
                    if not selected:
                        continue
                    stat = pair_stats(
                        [ours[q]["after_correct"] for q in selected],
                        [by_mode[other][q]["after_correct"] for q in selected],
                    )
                    pairs.append(
                        {
                            **fields,
                            "condition": condition,
                            "endpoint": endpoint,
                            "a": "controlled",
                            "b": other,
                            "comparison_kind": "primary"
                            if endpoint == "retention"
                            and other == "grounding_only"
                            and fields["phase"] == "main"
                            and fields["repeat_id"] == 0
                            else "exploratory",
                            **stat,
                        }
                    )
                prompt_pairs = [
                    q
                    for q in consistent
                    if ours[q].get("_final_prompt")
                    and by_mode[other][q].get("_final_prompt")
                ]
                coupling.append(
                    {
                        **fields,
                        "condition": condition,
                        "a": "controlled",
                        "b": other,
                        "n": len(consistent),
                        "final_prompt_pair_n": len(prompt_pairs),
                        "same_route_n": sum(
                            ours[q].get("_route") == by_mode[other][q].get("_route")
                            for q in consistent
                        ),
                        "identical_final_prompt_n": sum(
                            ours[q]["_final_prompt"]
                            == by_mode[other][q]["_final_prompt"]
                            for q in prompt_pairs
                        ),
                        "identical_final_output_n": sum(
                            hashed(ours[q].get("final"))
                            == hashed(by_mode[other][q].get("final"))
                            for q in consistent
                        ),
                        "same_final_action_n": sum(
                            (ours[q].get("final") or {}).get("action")
                            == (by_mode[other][q].get("final") or {}).get("action")
                            for q in consistent
                        ),
                        "shared_draft_mismatch_n": len(common) - len(consistent),
                    }
                )
            controlled = list(ours.values())
            measured = [
                r
                for r in controlled
                if isinstance((r.get("diagnostics") or {}).get("profile"), dict)
            ]
            complete = [
                r
                for r in measured
                if r["diagnostics"]["profile"].get("observed_pairs") == 1
            ]
            healthy = [
                r
                for r in complete
                if r["diagnostics"]["profile"].get("beneficial_correction") == 1
                and r["diagnostics"]["profile"].get("harmful_corruption") == 0
            ]
            healthy_correct = [r for r in healthy if r.get("before_correct") is True]
            profiles.append(
                {
                    **fields,
                    "condition": condition,
                    "n": len(controlled),
                    "measured_n": len(measured),
                    "complete_n": len(complete),
                    "healthy_n": len(healthy),
                    "missing_n": len(controlled) - len(measured),
                    "healthy_rate": rate(len(healthy), len(complete)),
                    "healthy_initially_correct_n": len(healthy_correct),
                    "healthy_but_corrupted_n": sum(
                        not r["after_correct"] for r in healthy_correct
                    ),
                    "interpretation": "Quote-fidelity proxy, not semantic truth or calibrated probability",
                }
            )
            # Route-independent potential outcome: a suppression decision must
            # not be used as evidence that the rejected critique was harmless.
            always = by_mode["always_critique"]
            common = sorted(set(ours) & set(always))
            matched = [
                q
                for q in common
                if hashed(ours[q].get("draft")) == hashed(always[q].get("draft"))
                and hashed(ours[q].get("critique")) == hashed(always[q].get("critique"))
            ]
            for endpoint, initial in (
                ("critique_corruption", True),
                ("critique_repair", False),
            ):
                if condition != ("false_advice" if initial else "correct_advice"):
                    continue
                subset = [
                    q for q in matched if ours[q].get("before_correct") is initial
                ]
                if not subset:
                    continue
                for predictor, attribute in (
                    ("harm_flag", "harmful_corruption"),
                    ("beneficial_flag", "beneficial_correction"),
                ):
                    for flag in (False, True):
                        selected = []
                        for q in subset:
                            profile = (ours[q].get("diagnostics") or {}).get(
                                "profile"
                            ) or {}
                            value = profile.get(attribute)
                            if (
                                nonnegative_number(value)
                                and profile.get("observed_pairs", 0) > 0
                                and (value > 0) is flag
                            ):
                                selected.append(q)
                        events = sum(
                            always[q]["after_correct"] is (not initial)
                            for q in selected
                        )
                        transfer.append(
                            {
                                **fields,
                                "condition": condition,
                                "endpoint": endpoint,
                                "predictor": predictor,
                                "flag": flag,
                                "n": len(selected),
                                "events": events,
                                "rate": rate(events, len(selected)),
                                "interval95": interval(events, len(selected)),
                                "matched_case_n": len(subset),
                                "shared_draft_or_advice_mismatch_n": len(common)
                                - len(matched),
                                "profile_policy": "controlled",
                                "downstream_policy": "always_critique",
                                "comparison_kind": "secondary_diagnostic",
                                "interpretation": "Association with unconditional critique outcome on the same frozen draft/advice; not probability calibration or a causal profile effect",
                            }
                        )
    return {
        "qa_summary": summaries,
        "qa_pairs": pairs,
        "qa_routes": routes,
        "qa_draft_summary": drafts,
        "qa_profiles": profiles,
        "qa_coupling": coupling,
        "qa_proxy_transfer": transfer,
        "qa_format_units": format_units,
        "qa_format_sensitivity": summarize_format_units(
            format_units,
            (
                "phase",
                "cohort",
                "model",
                "source_identity",
                "repeat_id",
                "reasoning_effort",
                "condition",
                "mode",
            ),
        ),
    }, numeric


def resolve_manifest(path, run):
    configured = Path(run["config"]["manifest"])
    direct = configured if configured.is_absolute() else path.parent / configured
    expected = run.get("manifest_sha256")
    if direct.is_file() and (not expected or digest(direct) == expected):
        return direct
    for parent in path.parents:
        candidate = parent / configured.name
        if expected and candidate.is_file() and digest(candidate) == expected:
            return candidate
    return None


def native_success(episode, cutoff, allow_legacy_transport=False):
    """Per-cutoff observability; infrastructure's unreturned request is excluded."""
    if not isinstance(cutoff, int) or cutoff <= 0:
        raise ValueError("Positive integer call cutoff required")
    steps = episode.get("trajectory", [])
    if any(
        s.get("success") is True and s.get("end_logical_calls", math.inf) <= cutoff
        for s in steps
    ):
        return True
    status = episode.get("status")
    if status == "infrastructure_censored":
        complete = max(0, episode.get("usage", {}).get("calls", 0) - 1)
    elif (
        allow_legacy_transport
        and status == "model_error"
        and any(
            text in episode.get("error", "")
            for text in ("Codex failed", "Codex timed out", "missing usage evidence")
        )
    ):
        # Legacy engineering data may have swallowed multiple transport errors.
        complete = max([s.get("end_logical_calls", 0) for s in steps] or [0])
    else:
        complete = episode.get("usage", {}).get(
            "calls", max([s.get("end_logical_calls", 0) for s in steps] or [0])
        )
    if complete >= cutoff:
        return False
    if episode.get("finished") is True and status in {
        "success",
        "terminated",
        "action_horizon",
        "model_error",
    }:
        if status == "model_error" and complete != episode.get("usage", {}).get(
            "calls"
        ):
            return None
        return episode.get("success") is True
    return None


def native_protocol_id(environment, horizon, cap, effort):
    return hashed(
        {"environment": environment, "horizon": horizon, "cap": cap, "effort": effort}
    )[:12]


def native_metrics(inputs, warnings, provenance):
    rows, planned, log_cache = [], defaultdict(set), {}
    protocol_caps, protocol_efforts = {}, {}
    for path in selected_paths(inputs, "**/run_plan.json"):
        plan = read(path)
        if plan.get("schema_version") != "feedback-native-run-plan-v1":
            raise ValueError(f"Unknown native run-plan schema: {path.name}")
        phase, label = cohort(path)
        for task in plan["tasks"]:
            benchmark = task["benchmark"]
            protocol = native_protocol_id(
                plan["environment_protocols"][benchmark],
                plan["horizons"][benchmark],
                plan["episode_call_cap"],
                plan["reasoning_effort"],
            )
            protocol_caps[protocol] = plan["episode_call_cap"]
            protocol_efforts[protocol] = plan["reasoning_effort"]
            for variant in plan["variants"]:
                planned[
                    (
                        phase,
                        label,
                        benchmark,
                        plan["model"],
                        plan["source_sha256"],
                        protocol,
                        variant,
                    )
                ].add(task["key"])
        provenance.append(
            {"file": str(path), "sha256": digest(path), "kind": "native_run_plan"}
        )
    for path in selected_paths(inputs, "**/episodes/*.json"):
        episode = read(path)
        run_path = path.parent.parent / "run.json"
        run = read(run_path) if run_path.exists() else {"config": {}}
        config = run.get("config", {})
        phase, label = cohort(path)
        benchmark = episode.get("benchmark", "unknown")
        effort = config.get("provider", {}).get("reasoning_effort")
        protocol = native_protocol_id(
            episode.get("protocol"),
            config.get("horizons", {}).get(benchmark),
            config.get("episode_call_cap"),
            effort,
        )
        protocol_caps[protocol] = config.get("episode_call_cap")
        protocol_efforts[protocol] = effort
        episode = {
            **episode,
            "phase": phase,
            "cohort": label,
            "protocol_id": protocol,
            "reasoning_effort": config.get("provider", {}).get(
                "reasoning_effort", "unknown"
            ),
            "repeat_id": 0,
            "_path": str(path),
        }
        log_path = path.parent.parent / "model_calls.jsonl"
        if log_path not in log_cache:
            log_cache[log_path] = lines(log_path, warnings)
        receipts = usage_segment(episode.get("usage", {}), log_cache[log_path])
        transport_unknown = episode.get("status") == "infrastructure_censored" or (
            receipts is not None
            and any(transport_log(r, allow_legacy=label == "pilot") for r in receipts)
        )
        episode["_cost"] = cost_unit(episode.get("usage"), receipts, transport_unknown)
        rows.append(episode)
        provenance.append(
            {"file": str(path), "sha256": digest(path), "kind": "native_episode"}
        )
        if run_path.exists():
            manifest = resolve_manifest(run_path, run)
            if manifest:
                for task in read(manifest).get("tasks", []):
                    if task.get("key") in config.get("task_keys", [task.get("key")]):
                        for variant in config.get("variants", []):
                            planned[
                                (
                                    phase,
                                    label,
                                    task.get("benchmark"),
                                    episode.get("model"),
                                    episode.get("source_sha256"),
                                    protocol,
                                    variant,
                                )
                            ].add(task.get("key"))
    unique = {}
    for row in rows:
        key = (
            row["cohort"],
            row.get("model"),
            row.get("source_sha256"),
            row.get("protocol_id"),
            row.get("benchmark"),
            row.get("task_id"),
            row.get("seed"),
            row.get("variant"),
        )
        if key in unique and unique[key] is None:
            continue
        if key in unique and hashed(
            {k: v for k, v in unique[key].items() if not k.startswith("_")}
        ) != hashed({k: v for k, v in row.items() if not k.startswith("_")}):
            warnings.append(
                f"Conflicting native duplicate excluded: {row.get('key')}/{row.get('variant')}"
            )
            unique[key] = None
        elif key not in unique:
            unique[key] = row
    rows = [r for r in unique.values() if r is not None]
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["phase"],
                row["cohort"],
                row.get("benchmark"),
                row.get("model"),
                row.get("source_sha256"),
                row["protocol_id"],
                row.get("variant"),
            )
        ].append(row)
    for key in planned:
        groups.setdefault(key, [])
    summaries, costs, failures, pairs, numeric, episode_costs, format_units = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    indexed = defaultdict(dict)
    for key, values in sorted(groups.items(), key=str):
        phase, label, benchmark, model, source, protocol, variant = key
        fields = {
            "phase": phase,
            "cohort": label,
            "benchmark": benchmark,
            "model": model,
            "source_sha256": source,
            "protocol_id": protocol,
            "variant": variant,
            "repeat_id": 0,
            "reasoning_effort": values[0]["reasoning_effort"]
            if values
            else protocol_efforts.get(protocol, "unknown"),
            "configured_max_calls": protocol_caps.get(protocol),
        }
        for cutoff in CUTOFFS:
            cap = protocol_caps.get(protocol)
            within_requested = cutoff <= cap if isinstance(cap, int) else None
            comparative_allowed = within_requested is not False
            observed = [
                (r, native_success(r, cutoff, allow_legacy_transport=label == "pilot"))
                for r in values
            ]
            valid = [(r, success) for r, success in observed if success is not None]
            count = sum(success for _, success in valid)
            n_total = len(planned.get(key, set())) or len(values)
            summaries.append(
                {
                    **fields,
                    "cutoff": cutoff,
                    "within_requested_budget": within_requested,
                    "n_total": n_total,
                    "n_started": len(values),
                    "n_observed": len(valid) if comparative_allowed else 0,
                    "n_success": count if comparative_allowed else 0,
                    "n_missing": n_total - len(valid)
                    if comparative_allowed
                    else n_total,
                    "success_rate": rate(count, len(valid))
                    if comparative_allowed
                    else None,
                    "interval95": interval(count, len(valid))
                    if comparative_allowed
                    else [None, None],
                    "n_inferred_observed_beyond_requested_budget": len(valid)
                    if not comparative_allowed
                    else 0,
                    "budget_point_status": "not_requested"
                    if not comparative_allowed
                    else "requested"
                    if within_requested
                    else "requested_cap_unknown",
                }
            )
            for r, success in observed:
                numeric.append(
                    {
                        **fields,
                        "cutoff": cutoff,
                        "within_requested_budget": within_requested,
                        "key": r.get("key"),
                        "task_id": r.get("task_id"),
                        "observed": success is not None,
                        "comparative_observed": success is not None
                        and comparative_allowed,
                        "success": success,
                        "status": r.get("status"),
                        "logical_calls": r.get("usage", {}).get("calls"),
                        **{("cost_" + k): v for k, v in r["_cost"].items()},
                        "cost_repeated_across_cutoff_rows": True,
                    }
                )
            if comparative_allowed:
                indexed[(phase, label, benchmark, model, source, protocol, cutoff)][
                    variant
                ] = {(r.get("task_id"), r.get("seed")): success for r, success in valid}
        cost = {
            **fields,
            "n": len(values),
            "n_started": len(values),
            "physical_new_launches": None,
            "cost_scope": "full recorded trajectory up to max cap, not automatically cost at64",
            **aggregate_cost([r["_cost"] for r in values]),
        }
        cost["execution_wall_seconds_total"] = numeric_sum(
            r.get("execution_wall_seconds", r.get("wall_seconds")) for r in values
        )
        cost["status_counts"] = dict(
            Counter(r.get("status", "unknown") for r in values)
        )
        turns = [t for r in values for t in r.get("trajectory", [])]
        for episode in values:
            for turn in episode.get("trajectory", []):
                profile = (turn.get("diagnostics") or {}).get("profile")
                if isinstance(profile, dict):
                    format_units.extend(
                        {
                            **fields,
                            "key": episode.get("key"),
                            "task_id": episode.get("task_id"),
                            "step": turn.get("step"),
                            **flags,
                        }
                        for flags in quote_format_flags(profile)
                    )
        cost["executed_steps"] = len(turns)
        cost["outside_public_action_list_n"] = sum(
            bool(t.get("available_actions"))
            and t.get("action") not in t["available_actions"]
            for t in turns
        )
        cost["nothing_happens_feedback_n"] = sum(
            t.get("next_observation", "").strip().lower().startswith("nothing happens")
            for t in turns
        )
        cost["same_observation_text_n"] = sum(
            t.get("observation") == t.get("next_observation") for t in turns
        )
        repeat_episodes = 0
        for episode in values:
            run_length, prior_action, repeated = 0, None, False
            for turn in episode.get("trajectory", []):
                stagnant = turn.get("observation") == turn.get("next_observation")
                run_length = (
                    run_length + 1
                    if stagnant and turn.get("action") == prior_action
                    else int(stagnant)
                )
                repeated = repeated or run_length >= 3
                prior_action = turn.get("action")
            repeat_episodes += repeated
            episode_costs.append(
                {
                    **fields,
                    "key": episode.get("key"),
                    "task_id": episode.get("task_id"),
                    "seed": episode.get("seed"),
                    "status": episode.get("status"),
                    "executed_steps": len(episode.get("trajectory", [])),
                    "execution_wall_seconds": episode.get("execution_wall_seconds"),
                    "three_repeated_actions_same_feedback_text": repeated,
                    **episode["_cost"],
                }
            )
        cost["three_repeated_actions_same_feedback_text"] = repeat_episodes
        cost["repeated_action_interpretation"] = (
            "Episodes with three repeated actions and unchanged feedback text; "
            "this is not evidence of unchanged state or a semantic loop"
        )
        costs.append(cost)
        counts = Counter(r.get("status", "unknown") for r in values)
        failures.extend(
            {**fields, "failure_type": status, "count": count, "n": len(values)}
            for status, count in sorted(counts.items())
            if status != "success"
        )
    for key, methods in indexed.items():
        ours = methods.get("feedback", {})
        for baseline in ("grounding_only", "raw_actor", *NATIVE_BASELINES):
            theirs = methods.get(baseline, {})
            common = sorted(set(ours) & set(theirs), key=str)
            if not common:
                continue
            stat = pair_stats([ours[k] for k in common], [theirs[k] for k in common])
            phase, label, benchmark, model, source, protocol, cutoff = key
            pairs.append(
                {
                    "phase": phase,
                    "cohort": label,
                    "benchmark": benchmark,
                    "model": model,
                    "source_sha256": source,
                    "protocol_id": protocol,
                    "cutoff": cutoff,
                    "repeat_id": 0,
                    "a": "feedback",
                    "b": baseline,
                    "comparison_kind": "secondary_published"
                    if baseline in NATIVE_BASELINES
                    else "ablation",
                    **stat,
                }
            )
    families = defaultdict(list)
    for row in pairs:
        if row["comparison_kind"] == "secondary_published" and row["cutoff"] == 64:
            families[
                (row["phase"], row["cohort"], row["model"], row["source_sha256"])
            ].append(row)
    for family in families.values():
        previous = 0.0
        for rank, row in enumerate(
            sorted(family, key=lambda r: r["mcnemar_exact_two_sided_p"])
        ):
            previous = max(
                previous,
                min(1.0, (len(family) - rank) * row["mcnemar_exact_two_sided_p"]),
            )
            row["p_holm_secondary64"] = previous
    return {
        "native_summary": summaries,
        "native_pairs": pairs,
        "native_costs": costs,
        "native_failures": failures,
        "native_episode_costs": episode_costs,
        "native_format_units": format_units,
        "native_format_sensitivity": summarize_format_units(
            format_units,
            (
                "phase",
                "cohort",
                "benchmark",
                "model",
                "source_sha256",
                "protocol_id",
                "variant",
                "repeat_id",
                "reasoning_effort",
                "configured_max_calls",
            ),
        ),
    }, numeric


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row)) or ["no_data"]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def qa_run_inventory(inputs, warnings):
    metadata_paths = set()
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            metadata_paths.update(path.glob("**/run_metadata.json"))
        elif path.is_file() and (path.parent / "run_metadata.json").exists():
            metadata_paths.add(path.parent / "run_metadata.json")
    result = []
    for path in sorted(metadata_paths):
        metadata = read(path)
        frozen = metadata.get("frozen", {})
        phase, label = cohort(path, frozen.get("split"))
        invocation_path = path.parent / "invocation_selection.json"
        invocation = read(invocation_path) if invocation_path.exists() else None
        invocation_bound = isinstance(invocation, dict) and (
            invocation.get("split") == frozen.get("split")
            and invocation.get("repeat") == frozen.get("repeat_id", 0)
            and isinstance(invocation.get("question_ids"), list)
        )
        if invocation is not None and not invocation_bound:
            warnings.append(
                f"QA invocation split/repeat binding differs: {path.parent.name}"
            )
        halted = path.parent / "HALTED.json"
        logs = lines(path.parent / "model_calls.jsonl", warnings)
        fresh = {
            (r.get("campaign"), r.get("subscription_launch"))
            for r in logs
            if isinstance(r.get("subscription_launch"), int) and not r.get("replayed")
        }
        result.append(
            {
                "phase": phase,
                "cohort": label,
                "model": frozen.get("model"),
                "source_identity": frozen.get("source_identity"),
                "repeat_id": frozen.get("repeat_id", 0),
                "reasoning_effort": frozen.get("reasoning_effort", "unknown"),
                "split": frozen.get("split"),
                "manifest_sha256": frozen.get("manifest_sha256"),
                "requested_question_ids": sorted(set(invocation["question_ids"]))
                if invocation_bound
                else None,
                "invocation_sha256": digest(invocation_path)
                if invocation_path.exists()
                else None,
                "halted": halted.exists(),
                "halt_status": read(halted).get("status") if halted.exists() else None,
                "physical_launch_attempts_in_this_log": len(fresh),
                "transport_error_rows": sum(
                    transport_log(r, allow_legacy=label == "pilot") for r in logs
                ),
                "scope": "selected run log, not the full campaign ledger",
            }
        )
    return result


def build(qa_inputs, native_inputs, output, qa_manifest=None):
    warnings, provenance = [], []
    qa_rows = load_qa(qa_inputs, warnings, provenance)
    qa_stats, qa_numeric = qa_metrics(qa_rows)
    native_stats, native_numeric = native_metrics(native_inputs, warnings, provenance)
    inventory = qa_run_inventory(qa_inputs, warnings)
    qa_coverage = []
    if qa_manifest:
        manifest = read(qa_manifest)
        expected_hash = hashlib.sha256(
            json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        provenance.append(
            {
                "file": str(qa_manifest),
                "sha256": digest(qa_manifest),
                "kind": "public_qa_id_manifest",
            }
        )
        groups = defaultdict(list)
        for row in qa_rows:
            groups[qa_key(row)].append(row)
        requested = defaultdict(set)
        for run in inventory:
            if not run.get("model") or not run.get("source_identity"):
                continue
            if run.get("manifest_sha256") != expected_hash:
                warnings.append(
                    f"QA invocation manifest differs for {run['cohort']}; requested denominator unknown"
                )
                continue
            key = qa_key(run)
            groups.setdefault(key, [])
            if run.get("requested_question_ids") is not None:
                requested[key].update(run["requested_question_ids"])
        for key, values in groups.items():
            split = "test" if key[0] == "main" else "dev"
            if any(row.get("manifest_sha256") != expected_hash for row in values):
                warnings.append(
                    f"QA coverage manifest differs for {key[1]}; no planned denominator inferred"
                )
                continue
            expected = {v["id"] for v in manifest.get("splits", {}).get(split, [])}
            planned = requested.get(key)
            if planned is None and key[4] == 0:
                planned = expected
            if planned is not None and not planned <= expected:
                warnings.append(
                    f"QA invocation has IDs outside frozen split for {key[1]}; requested denominator unknown"
                )
                planned = None
            for condition in CONDITIONS:
                for mode in QA_MODES:
                    observed = [
                        r
                        for r in values
                        if r.get("condition") == condition and r.get("mode") == mode
                    ]
                    qa_coverage.append(
                        {
                            **qa_group_fields(key),
                            "condition": condition,
                            "mode": mode,
                            "planned_full_split": len(expected),
                            "planned_requested_questions": len(planned)
                            if planned is not None
                            else None,
                            "planned_denominator": len(planned)
                            if planned is not None
                            else None,
                            "records_present": len(observed),
                            "mechanism_evaluable": sum(
                                r["_eligible"] for r in observed
                            ),
                            "missing_full_split": len(expected) - len(observed),
                            "missing_requested": len(
                                planned - {r["question_id"] for r in observed}
                            )
                            if planned is not None
                            else None,
                            "records_outside_requested": sum(
                                r["question_id"] not in planned for r in observed
                            )
                            if planned is not None
                            else None,
                            "scope": "bound invocation ID union"
                            if key in requested
                            else "full frozen split for initial run"
                            if key[4] == 0
                            else "repeat invocation unavailable; requested coverage unknown",
                        }
                    )
    data = {
        "schema_version": "feedback-study-metrics-v1",
        "analysis_status": "selected_runs_metrics",
        "has_main_data": any(r["phase"] == "main" for r in qa_rows)
        or any(r["phase"] == "main" for r in native_numeric),
        "interval_method": "conservative exact discordant-cell Bonferroni; marginal exact Clopper-Pearson",
        "primary_qa_endpoint": "false_advice retention conditional on correct shared initial draft, controlled-grounding_only",
        "independence": "question per advice condition, not number of policies or repeated cases; repeat/cohort/source/model separate",
        "cost_warning": "Logical calls/tokens/historical inference cost per policy; cache physical savings are experimental coupling, not free deployment compute",
        **qa_stats,
        **native_stats,
        "qa_coverage": qa_coverage,
        "qa_run_inventory": inventory,
        "format_sensitivity_unicode_version": unicodedata.unidata_version,
        "warnings": warnings,
        "provenance": provenance,
        "raw_questions_or_evidence_exported": False,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "aggregated_stats.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )
    write_csv(output / "qa_scored_numeric.csv", qa_numeric)
    write_csv(output / "native_scored_numeric.csv", native_numeric)
    write_csv(output / "qa_coverage.csv", qa_coverage)
    for name, values in {**qa_stats, **native_stats}.items():
        write_csv(output / (name + ".csv"), values)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", nargs="*", default=[])
    parser.add_argument("--native", nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--qa-manifest", type=Path)
    args = parser.parse_args()
    data = build(args.qa, args.native, args.out, args.qa_manifest)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "has_main_data": data["has_main_data"],
                "qa_groups": len(data["qa_summary"]),
                "native_groups": len(data["native_summary"]),
                "warnings": data["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
