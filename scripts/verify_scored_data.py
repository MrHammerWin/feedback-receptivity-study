"""Recalculate published scores from numeric source data, without LLM access.

This verifies arithmetic, pairing and missing-value handling. Optional --qa-data
binds numeric targets to a separately downloaded pinned source file. It does not
establish semantic truth of gold labels, raw model provenance or unstarted tasks.
Run: python scripts/verify_scored_data.py --source-data source_data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

QA_GROUP = ("phase", "cohort", "model", "source_identity", "repeat_id", "reasoning_effort")
NATIVE_GROUP = (
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
)
NATIVE_PAIR_GROUP = ("phase", "cohort", "benchmark", "model", "source_sha256", "protocol_id", "repeat_id")
COST_FIELDS = {
    "logical_calls": "logical_calls_total",
    "input_tokens_lower_bound": "input_tokens_lower_bound_total",
    "output_tokens_lower_bound": "output_tokens_lower_bound_total",
    "inference_seconds": "inference_seconds_total",
    "cached_input_tokens_lower_bound": "cached_input_tokens_lower_bound_total",
}
QA_PARQUET_SHA256 = "8a09a9ddd3edfb4d171950954d90483a0d0a8a8a1b2581f065c9d1bf6ef700f7"


class InvalidSourceData(ValueError):
    pass


def optional_bool(value, field):
    if value is None or value == "":
        return None
    if value is True or value in ("True", "true", "1"):
        return True
    if value is False or value in ("False", "false", "0"):
        return False
    raise InvalidSourceData(f"Invalid Boolean in {field}; blank is missing, never false")


def number(value, field, integer=False, required=False):
    if value is None or value == "":
        if required:
            raise InvalidSourceData(f"Missing numeric field {field}")
        return None
    if isinstance(value, bool):
        raise InvalidSourceData(f"Boolean used as number in {field}")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise InvalidSourceData(f"Invalid number in {field}") from exc
    if not math.isfinite(result) or result < 0 or (integer and not result.is_integer()):
        raise InvalidSourceData(f"Invalid nonnegative {'integer' if integer else 'number'} in {field}")
    return int(result) if integer else result


def group(row, fields):
    result = []
    for field in fields:
        value = row.get(field)
        if field in {"repeat_id", "configured_max_calls"}:
            value = number(value, field, integer=True, required=field == "repeat_id")
        elif not isinstance(value, str) or not value:
            raise InvalidSourceData(f"Missing grouping field {field}; pooling models/sources is forbidden")
        result.append(value)
    return tuple(result)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise InvalidSourceData(f"Missing CSV header: {path.name}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise InvalidSourceData(f"Duplicate CSV columns: {path.name}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise InvalidSourceData(f"Malformed CSV row: {path.name}")
    return rows


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _letter_index(value):
    value = value.strip().upper() if isinstance(value, str) else ""
    return "ABCD".index(value) if value in {"A", "B", "C", "D"} else None


def load_qa_reference_data(path):
    """Explicit local-file validation; never download data automatically."""
    path = Path(path)
    if path.is_symlink():
        raise InvalidSourceData("Symlink QA source files are not accepted")
    if hashlib.sha256(path.read_bytes()).hexdigest() != QA_PARQUET_SHA256:
        raise InvalidSourceData("Local QA parquet differs from the fixed source SHA256")
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise InvalidSourceData("Install the optional qa extra (pyarrow) to use --qa-data") from exc
    references = {}
    for row in parquet.read_table(path).to_pylist():
        if row["id"] in references:
            raise InvalidSourceData("Duplicate upstream question ID")
        question = {
            "question": " ".join(row["question"].split()),
            "choices": {k: " ".join(row["choices"][k].split()) for k in "ABCD"},
        }
        references[row["id"]] = {
            "reference_choice_index": _letter_index(row["answer"]),
            "alternative_choice_index": _letter_index(row.get("alternative_supported_answer")),
            "n_choices": 4,
            "question_sha256": _canonical_hash(question),
            "row_sha256": _canonical_hash(row),
            "source_dataset": row["source_dataset"],
        }
    return references


def verify_choice_indices(rows, checks, references=None):
    """Recompute flags from choices; optional pinned data independently binds targets."""
    if not any("n_choices" in row for row in rows):
        if references is not None and rows:
            raise InvalidSourceData("Choice-index source columns are required for upstream rescoring")
        return 0
    checked = 0
    for row in rows:
        if row.get("n_choices") in (None, "") and optional_bool(row.get("finished"), "finished") is not True:
            continue
        count = number(row.get("n_choices"), "n_choices", integer=True, required=True)
        if count != 4:
            raise InvalidSourceData("This frozen QA protocol has exactly four choices")
        indices = {
            name: number(row.get(name), name, integer=True)
            for name in (
                "initial_choice_index",
                "final_choice_index",
                "reference_choice_index",
                "advice_choice_index",
            )
        }
        if any(value is not None and value >= count for value in indices.values()):
            raise InvalidSourceData("Choice index outside the 0-based action space")
        reference = indices["reference_choice_index"]
        if reference is None:
            raise InvalidSourceData("Missing reference choice index")
        condition = row.get("condition")
        advice = indices["advice_choice_index"]
        if advice is None and optional_bool(row.get("finished"), "finished") is True:
            raise InvalidSourceData("Finished QA advice target is missing")
        if advice is not None and (
            condition not in {"correct_advice", "false_advice"}
            or (advice == reference) != (condition == "correct_advice")
        ):
            raise InvalidSourceData("Advice choice index contradicts correct/false-advice condition")
        if references is not None:
            source = references.get(row.get("question_id"))
            if source is None:
                raise InvalidSourceData("Question ID absent from pinned upstream source")
            for field in ("reference_choice_index", "n_choices"):
                checks.compare(
                    number(row.get(field), field, integer=True), source[field], "upstream " + field
                )
            for field in ("question_sha256", "row_sha256"):
                if not row.get(field):
                    raise InvalidSourceData("Source hash is required for upstream target binding")
                checks.compare(row[field], source[field], "upstream " + field)
            if row.get("source_dataset"):
                checks.compare(row["source_dataset"], source["source_dataset"], "upstream topic")
            reference = source["reference_choice_index"]
            expected_advice = (
                reference if condition == "correct_advice" else source["alternative_choice_index"]
            )
            if advice is not None:
                checks.compare(advice, expected_advice, "upstream advice target")
        initial, final = indices["initial_choice_index"], indices["final_choice_index"]
        initial_present = optional_bool(row.get("initial_response_present"), "initial_response_present")
        final_present = optional_bool(row.get("final_response_present"), "final_response_present")
        if initial is not None and initial_present is False or final is not None and final_present is False:
            raise InvalidSourceData("Choice index provided for an absent response")
        status = row.get("status")
        censored = status in {
            "infrastructure_censored",
            "draft_infrastructure_censored",
            "campaign_budget_censored",
            "running",
        }
        before = initial == reference if initial is not None else None
        # The frozen infrastructure catch scores a present invalid draft false;
        # completed parsing failures keep initial correctness undefined.
        if (
            initial is None
            and initial_present is True
            and status in {"infrastructure_censored", "draft_infrastructure_censored"}
        ):
            before = False
        after = (
            None
            if censored or optional_bool(row.get("finished"), "finished") is not True
            else final == reference
        )
        checks.compare(
            optional_bool(row.get("before_correct"), "before_correct"),
            before,
            "choice-derived before_correct",
        )
        checks.compare(
            optional_bool(row.get("after_correct"), "after_correct"), after, "choice-derived after_correct"
        )
        checked += 1
    return checked


def rate(count, n):
    return count / n if n else None


def close(actual, expected):
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(close(a, b) for a, b in zip(actual, expected, strict=True))
        )
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (float, int))
            and not isinstance(actual, bool)
            and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10)
        )
    return actual == expected


class Checks:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.count = 0

    def compare(self, actual, expected, context):
        self.count += 1
        if not close(actual, expected):
            self.errors.append(f"{context}: reported={actual!r}, recomputed={expected!r}")


def unique_summary(rows, fields, extra, name):
    result = {}
    for row in rows:
        key = (*group(row, fields), *(row.get(k) for k in extra))
        if key in result:
            raise InvalidSourceData(f"Duplicate/conflicting {name} summary grouping")
        result[key] = row
    return result


def check_pair(claimed, a, b, checks, statistics, context):
    expected = statistics.paired_binary(a, b)
    for name in ("n", "a_only", "b_only", "difference", "interval95", "mcnemar_exact_two_sided_p"):
        checks.compare(claimed.get(name), expected[name], context + " " + name)
    if "interval_method" in claimed:
        checks.compare(claimed["interval_method"], expected["interval_method"], context + " interval method")
    return expected


def verify_cost(records, published, checks, prefix="cost_"):
    """Costs are optional; use published lower-bound columns, not inferred zeros."""
    if not records or not all(prefix + "logical_calls" in r for r in records):
        return
    for column, total_name in COST_FIELDS.items():
        values = [number(r.get(prefix + column), column) for r in records]
        known = [v for v in values if v is not None]
        total = sum(known) if known else None
        if total_name in published:
            checks.compare(published[total_name], total, f"cost {total_name}")
    counts = {
        "n_logical_calls_known": sum(r.get(prefix + "logical_calls") not in (None, "") for r in records),
        "n_inference_time_known": sum(r.get(prefix + "inference_seconds") not in (None, "") for r in records),
        "n_complete_token_cost": sum(
            optional_bool(r.get(prefix + "complete_token_telemetry"), "complete_token_telemetry") is True
            for r in records
        ),
        "n_transport_unknown": sum(
            optional_bool(r.get(prefix + "transport_unknown"), "transport_unknown") is True for r in records
        ),
    }
    for name, value in counts.items():
        if name in published:
            checks.compare(published[name], value, "cost " + name)


def verify_qa(rows, stats, checks, statistics):
    units, groups = {}, defaultdict(list)
    drafts = defaultdict(set)
    for index, row in enumerate(rows, 2):
        key = group(row, QA_GROUP)
        for field in ("question_id", "condition", "mode"):
            if not row.get(field):
                raise InvalidSourceData(f"Missing QA {field} at CSV row {index}")
        if row["condition"] not in {"false_advice", "correct_advice"}:
            raise InvalidSourceData("Unknown QA advice condition")
        unit = (*key, row["question_id"], row["condition"], row["mode"])
        if unit in units:
            kind = "duplicate" if row == units[unit] else "conflicting duplicate"
            raise InvalidSourceData(f"QA {kind} observational unit; attempts cannot inflate N")
        parsed = dict(row)
        for field in (
            "mechanism_eligible",
            "before_correct",
            "after_correct",
            "finished",
            "transport_contaminated",
        ):
            parsed[field] = optional_bool(row.get(field), field)
        if parsed["mechanism_eligible"] is None:
            raise InvalidSourceData("Missing QA mechanism_eligible flag")
        if parsed["mechanism_eligible"] and (
            parsed["after_correct"] is None
            or parsed["finished"] is not True
            or parsed["transport_contaminated"] is not False
            or row.get("status") in {"running", "campaign_budget_censored", "infrastructure_censored"}
        ):
            raise InvalidSourceData(
                "Eligible QA row has missing outcome or is censored; missing cannot be false"
            )
        units[unit] = row
        groups[(*key, row["condition"], row["mode"])].append(parsed)
        if parsed["before_correct"] is not None:
            drafts[(*key, row["question_id"])].add(parsed["before_correct"])
    if any(len(flags) > 1 for flags in drafts.values()):
        raise InvalidSourceData("Conflicting shared initial-draft correctness across policies/conditions")
    published = unique_summary(stats["qa_summary"], QA_GROUP, ("condition", "mode"), "QA")
    for key, records in groups.items():
        claimed = published.pop(key, None)
        if claimed is None:
            checks.errors.append("Missing QA summary for a model/source-specific numeric group")
            continue
        valid = [r for r in records if r["mechanism_eligible"]]
        correct = [r for r in valid if r["before_correct"] is True]
        wrong = [r for r in valid if r["before_correct"] is False]
        counts = {
            "n": len(valid),
            "n_questions": len({r["question_id"] for r in valid}),
            "n_records": len(records),
            "successes": sum(r["after_correct"] for r in valid),
            "initially_correct_n": len(correct),
            "retained_n": sum(r["after_correct"] for r in correct),
            "initially_wrong_n": len(wrong),
            "repaired_n": sum(r["after_correct"] for r in wrong),
        }
        counts.update(
            {
                "accuracy": rate(counts["successes"], counts["n"]),
                "retention": rate(counts["retained_n"], len(correct)),
                "repair": rate(counts["repaired_n"], len(wrong)),
            }
        )
        for name, value in counts.items():
            checks.compare(claimed.get(name), value, "QA " + name)
        terminal = [r for r in records if r["finished"] is True]
        for name, value in {
            "n_finished_total": len(terminal),
            "n_transport_contaminated": sum(r["transport_contaminated"] is True for r in records),
            "n_missing_or_censored": sum(r["finished"] is not True for r in records),
            "model_error_n": sum(r.get("status") in {"model_error", "draft_model_error"} for r in valid),
            "operational_n": len(terminal),
            "operational_accuracy": rate(sum(r["after_correct"] is True for r in terminal), len(terminal)),
        }.items():
            if name in claimed:
                checks.compare(claimed[name], value, "QA " + name)
        for name, successes, total in (
            ("interval95", counts["successes"], counts["n"]),
            ("retention_interval95", counts["retained_n"], len(correct)),
            ("repair_interval95", counts["repaired_n"], len(wrong)),
        ):
            checks.compare(claimed.get(name), statistics.clopper_pearson(successes, total), "QA " + name)
        verify_cost(records, claimed, checks)
    if published:
        checks.errors.append(
            "Extra QA summary groups without numeric source rows (possible pooled model/source)"
        )
    pairs = unique_summary(stats["qa_pairs"], QA_GROUP, ("condition", "endpoint", "a", "b"), "QA pair")
    eligible = defaultdict(lambda: defaultdict(dict))
    for key, records in groups.items():
        base, condition, mode = key[:-2], key[-2], key[-1]
        for row in records:
            if row["mechanism_eligible"]:
                eligible[(*base, condition)][mode][row["question_id"]] = row
    paired_groups, primary_groups = 0, 0
    for key, policies in eligible.items():
        base, condition = key[:-1], key[-1]
        controlled = policies["controlled"]
        for other in ("grounding_only", "always_critique", "facts_only", "no_feedback"):
            alternative = policies[other]
            common = sorted(set(controlled) & set(alternative))
            for question in common:
                for field in ("shared_draft_sha256", "shared_critique_sha256"):
                    a, b = controlled[question].get(field), alternative[question].get(field)
                    if a and b and a != b:
                        raise InvalidSourceData("Conflicting shared draft/advice digests in a QA pair")
            endpoints = ("accuracy", "retention") if condition == "false_advice" else ("accuracy", "repair")
            for endpoint in endpoints:
                selected = [
                    q
                    for q in common
                    if endpoint == "accuracy" or controlled[q]["before_correct"] is (endpoint == "retention")
                ]
                if not selected:
                    continue
                claimed = pairs.pop((*base, condition, endpoint, "controlled", other), None)
                if claimed is None:
                    checks.errors.append(
                        "Missing paired " + endpoint + " summary for a QA model/source group"
                    )
                    continue
                check_pair(
                    claimed,
                    [controlled[q]["after_correct"] for q in selected],
                    [alternative[q]["after_correct"] for q in selected],
                    checks,
                    statistics,
                    "paired " + endpoint,
                )
                primary = (
                    endpoint == "retention"
                    and other == "grounding_only"
                    and base[0] == "main"
                    and base[4] == 0
                )
                checks.compare(
                    claimed.get("comparison_kind"),
                    "primary" if primary else "exploratory",
                    "QA pair comparison kind",
                )
                primary_groups += primary
                paired_groups += 1
    if pairs:
        checks.errors.append("Extra QA pairs without corresponding eligible numeric pairs")
    if "qa_draft_summary" in stats:
        draft_summary = unique_summary(stats["qa_draft_summary"], QA_GROUP, (), "QA initial draft")
        by_group = defaultdict(list)
        for key, flags in drafts.items():
            base, question = key[:-1], key[-1]
            source_rows = [
                r
                for k, values in groups.items()
                if k[:-2] == base
                for r in values
                if r["question_id"] == question
            ]
            excluded = any(
                r["transport_contaminated"] and r.get("status") == "draft_model_error" for r in source_rows
            )
            if len(flags) == 1 and not excluded:
                by_group[base].append(next(iter(flags)))
        for base in {key[:-2] for key in groups}:
            values = by_group[base]
            claimed = draft_summary.pop(base, None)
            if claimed is None:
                checks.errors.append("Missing QA initial-draft summary")
                continue
            for name, value in {
                "valid_initial_n": len(values),
                "initial_correct_n": sum(values),
                "initial_wrong_n": len(values) - sum(values),
                "question_n_started": len(
                    {r["question_id"] for key, records in groups.items() if key[:-2] == base for r in records}
                ),
                "initial_accuracy": rate(sum(values), len(values)),
                "interval95": statistics.clopper_pearson(sum(values), len(values)),
            }.items():
                checks.compare(claimed.get(name), value, "QA draft " + name)
        if draft_summary:
            checks.errors.append("Extra QA initial-draft summary without numeric questions")
    if rows and not all(r.get("shared_draft_sha256") for r in rows):
        checks.warnings.append(
            "Draft digests absent in some rows: common-draft identity relies on the frozen protocol; raw-draft equality is not reverified"
        )
    return primary_groups, paired_groups


def verify_native_pairs(groups, stats, checks, statistics):
    published = {}
    for row in stats["native_pairs"]:
        key = (
            *group(row, NATIVE_PAIR_GROUP),
            number(row.get("cutoff"), "cutoff", integer=True, required=True),
            row.get("a"),
            row.get("b"),
        )
        if key in published:
            raise InvalidSourceData("Duplicate/conflicting native pair summary")
        published[key] = row
    settings, methods = {}, defaultdict(dict)
    for key, records in groups.items():
        metadata = dict(zip((*NATIVE_GROUP, "cutoff"), key, strict=True))
        core = (*group(metadata, NATIVE_PAIR_GROUP), metadata["cutoff"])
        setting = (metadata["reasoning_effort"], metadata["configured_max_calls"])
        if core in settings and settings[core] != setting:
            raise InvalidSourceData("Ambiguous native protocol ID maps to multiple effort/cap settings")
        settings[core] = setting
        indexed = {}
        for row in records:
            if not row["comparative_observed"]:
                continue
            seed = number(row.get("seed"), "native pair seed", integer=True, required=True)
            identity = (row["task_id"], seed)
            if identity in indexed:
                raise InvalidSourceData("Duplicate native task_id/seed pair identity across frozen keys")
            indexed[identity] = (row["success"], row["key"])
        methods[core][metadata["variant"]] = indexed
    reconstructed, compared = [], 0
    for core, policies in methods.items():
        candidate = policies.get("feedback", {})
        for other in ("grounding_only", "raw_actor", "stateact", "reflact_paper"):
            alternative = policies.get(other, {})
            common = sorted(set(candidate) & set(alternative))
            if not common:
                continue
            if any(candidate[q][1] != alternative[q][1] for q in common):
                raise InvalidSourceData("Different frozen native keys share a claimed task/seed pair")
            claimed = published.pop((*core, "feedback", other), None)
            if claimed is None:
                checks.errors.append("Missing native paired comparison for eligible common cases")
                continue
            for name, expected in zip(
                ("reasoning_effort", "configured_max_calls"), settings[core], strict=True
            ):
                if name in claimed:
                    value = (
                        number(claimed[name], name, integer=True)
                        if name == "configured_max_calls"
                        else claimed[name]
                    )
                    checks.compare(value, expected, "native pair " + name)
            result = check_pair(
                claimed,
                [candidate[q][0] for q in common],
                [alternative[q][0] for q in common],
                checks,
                statistics,
                "native pair",
            )
            kind = "secondary_published" if other in {"stateact", "reflact_paper"} else "ablation"
            checks.compare(claimed.get("comparison_kind"), kind, "native pair comparison kind")
            if kind == "secondary_published" and core[-1] == 64:
                reconstructed.append((core, other, claimed, result))
            elif "p_holm_secondary64" in claimed:
                checks.errors.append("Holm field attached to an ablation or nonprimary native cutoff")
            compared += 1
    if published:
        checks.errors.append("Extra native pairs without eligible common cases or at a not-requested cutoff")
    families = defaultdict(list)
    for core, other, claimed, result in reconstructed:
        phase, cohort, benchmark, model, source, protocol, repeat, _ = core
        effort, cap = settings[core]
        family = (phase, cohort, model, source, repeat, effort, cap)
        families[family].append((benchmark, other, protocol, claimed, result))
    summaries = []
    for family, values in families.items():
        identities = {(b, other) for b, other, _, _, _ in values}
        if len(identities) != len(values) or len(values) > 4:
            raise InvalidSourceData("Ambiguous native Holm family contains repeated contrasts/protocols")
        previous = 0.0
        for rank, (_, _, _, claimed, result) in enumerate(
            sorted(values, key=lambda v: v[4]["mcnemar_exact_two_sided_p"])
        ):
            adjusted = min(1.0, (len(values) - rank) * result["mcnemar_exact_two_sided_p"])
            previous = max(previous, adjusted)
            checks.compare(claimed.get("p_holm_secondary64"), previous, "native Holm64 adjusted p")
        if len(values) != 4:
            checks.errors.append("Incomplete native family; cannot verify planned4 Holm adjustment")
        summaries.append(
            {
                "phase": family[0],
                "cohort": family[1],
                "model": family[2],
                "source_sha256": family[3],
                "repeat_id": family[4],
                "reasoning_effort": family[5],
                "configured_max_calls": family[6],
                "planned_comparisons": 4,
                "observable_comparisons": len(values),
                "complete": len(values) == 4,
            }
        )
    return compared, summaries


def verify_native(rows, stats, checks, statistics):
    groups, seen, costs = defaultdict(list), {}, defaultdict(dict)
    for index, row in enumerate(rows, 2):
        key = group(row, NATIVE_GROUP)
        cutoff = number(row.get("cutoff"), "cutoff", integer=True, required=True)
        if cutoff <= 0 or not row.get("key") or not row.get("task_id"):
            raise InvalidSourceData(f"Missing native task identity/cutoff at row {index}")
        unit = (row["key"], row["task_id"], row.get("seed", ""))
        identity = (*key, cutoff, *unit)
        if identity in seen:
            raise InvalidSourceData("Duplicate/conflicting native task+cutoff row")
        seen[identity] = row
        parsed = dict(row)
        for field in ("observed", "comparative_observed", "success", "within_requested_budget"):
            parsed[field] = optional_bool(row.get(field), field)
        if parsed["observed"] is None or parsed["comparative_observed"] is None:
            raise InvalidSourceData("Native observation flags are missing")
        if parsed["observed"] != isinstance(parsed["success"], bool):
            raise InvalidSourceData(
                "Native missing success treated as false, or unobserved row has an outcome"
            )
        cap = key[-1]
        if cap is not None and parsed["within_requested_budget"] != (cutoff <= cap):
            raise InvalidSourceData("Native within_requested_budget disagrees with frozen maximum")
        if parsed["comparative_observed"] and (
            not parsed["observed"] or parsed["within_requested_budget"] is False
        ):
            raise InvalidSourceData("Unobserved/not-requested native result counted as comparative")
        if parsed["comparative_observed"] != (
            parsed["observed"] and parsed["within_requested_budget"] is not False
        ):
            raise InvalidSourceData("Native comparative observation flag dropped a known requested result")
        groups[(*key, cutoff)].append(parsed)
        old = costs[key].get(unit)
        fields = {k: v for k, v in row.items() if k.startswith("cost_")}
        if old is not None and {k: v for k, v in old.items() if k.startswith("cost_")} != fields:
            raise InvalidSourceData("Native cost differs across repeated cutoff rows")
        costs[key][unit] = parsed
    published = {}
    for row in stats["native_summary"]:
        key = (*group(row, NATIVE_GROUP), number(row.get("cutoff"), "cutoff", integer=True, required=True))
        if key in published:
            raise InvalidSourceData("Duplicate/conflicting native summary group")
        published[key] = row
    for key in set(groups) | set(published):
        records = groups.get(key, [])
        claimed = published.get(key)
        if claimed is None:
            checks.errors.append("Missing native summary for numeric model/source group")
            continue
        valid = [r for r in records if r["comparative_observed"]]
        success = sum(r["success"] for r in valid)
        for name, value in {
            "n_started": len(records),
            "n_observed": len(valid),
            "n_success": success,
            "success_rate": rate(success, len(valid)),
        }.items():
            checks.compare(claimed.get(name), value, "native " + name)
        checks.compare(
            claimed.get("interval95"), statistics.clopper_pearson(success, len(valid)), "native interval95"
        )
        within = key[-1] <= key[-2] if key[-2] is not None else None
        for name, value in {
            "within_requested_budget": within,
            "budget_point_status": "not_requested"
            if within is False
            else "requested"
            if within
            else "requested_cap_unknown",
            "n_inferred_observed_beyond_requested_budget": sum(r["observed"] for r in records)
            if within is False
            else 0,
        }.items():
            if name in claimed:
                checks.compare(claimed[name], value, "native " + name)
        if "n_total" in claimed and "n_missing" in claimed:
            total = number(claimed["n_total"], "n_total", integer=True, required=True)
            if total < len(records):
                checks.errors.append("Native planned N smaller than number of started units")
            checks.compare(claimed["n_missing"], total - len(valid), "native missing coverage")
    for claimed in stats.get("native_costs", []):
        records = list(costs.get(group(claimed, NATIVE_GROUP), {}).values())
        verify_cost(records, claimed, checks)
    pair_count, families = verify_native_pairs(groups, stats, checks, statistics)
    return len(groups), pair_count, families


def verify(source_data, stats_path=None, qa_data=None):
    source_data = Path(source_data)
    stats_path = Path(stats_path) if stats_path else source_data / "aggregated_stats.json"
    checks = Checks()
    report = {
        "schema_version": "scored-data-verification-v1",
        "passed": False,
        "scope": "Numeric score/choice reproducibility and optional pinned-source target binding; no raw prose exported, network or LLM calls",
        "checks": 0,
        "errors": checks.errors,
        "warnings": checks.warnings,
        "verification_boundary": "Checks scored-data arithmetic and target bindings, not private model provenance, intrinsic gold truth, hidden environment states or diagnostic-table statistics outside the named result sections",
    }
    try:
        qa_path, native_path = (
            source_data / "qa_scored_numeric.csv",
            source_data / "native_scored_numeric.csv",
        )
        stats = json.loads(stats_path.read_text())
        for name in ("qa_summary", "qa_pairs", "native_summary", "native_pairs"):
            if not isinstance(stats.get(name), list):
                raise InvalidSourceData(f"Missing statistics array {name}")
        qa, native = read_csv(qa_path), read_csv(native_path)
        references = load_qa_reference_data(qa_data) if qa_data is not None else None
        report["choice_index_rows_checked"] = verify_choice_indices(qa, checks, references)
        report["upstream_qa_reference_file_verified"] = qa_data is not None
        if qa_data is not None:
            report["upstream_qa_parquet_sha256"] = QA_PARQUET_SHA256
        helper = Path(__file__).resolve().parents[1] / "analysis" / "paired_statistics.py"
        spec = importlib.util.spec_from_file_location("published_paired_statistics", helper)
        statistics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(statistics)
        report["primary_pair_groups_checked"], report["qa_pair_groups_checked"] = verify_qa(
            qa, stats, checks, statistics
        )
        (
            report["native_groups_checked"],
            report["native_pair_groups_checked"],
            report["native_holm_families"],
        ) = verify_native(native, stats, checks, statistics)
        report.update(
            {
                "qa_rows": len(qa),
                "native_rows": len(native),
                "input_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (qa_path, native_path, stats_path, helper)
                },
            }
        )
        if not qa and not native:
            checks.warnings.append("No scored observations: only absence of numerical claims was checked")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        checks.errors.append(f"{type(exc).__name__}: {exc}")
    report["checks"] = checks.count
    report["passed"] = not checks.errors
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data", "--data", type=Path, default=Path(__file__).resolve().parents[1] / "source_data"
    )
    parser.add_argument("--stats", type=Path, help="Override aggregated_stats.json path")
    parser.add_argument(
        "--qa-data",
        type=Path,
        help="Optional separately downloaded pinned parquet; requires qa extra, never uses network",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable verification report")
    args = parser.parse_args(argv)
    report = verify(args.source_data, args.stats, args.qa_data)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['checks']} numerical checks")
        for message in report["errors"]:
            print("ERROR: " + message)
        for message in report["warnings"]:
            print("WARNING: " + message)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
