"""Descriptive topic-cluster sensitivity for the preplanned primary QA contrast.

Read numeric data only. No models, raw answers, labels or environment access.
The cluster interval supplements, and never replaces, the exact paired interval.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
VERIFIER = _module("topic_source_verifier", HERE.parent / "scripts/verify_scored_data.py")
PAIRED = _module("topic_paired_statistics", HERE / "paired_statistics.py")
GROUP = VERIFIER.QA_GROUP
InvalidSourceData = VERIFIER.InvalidSourceData
POLICIES = ("controlled", "grounding_only")
DIGEST_FIELDS = ("question_sha256", "shared_draft_sha256", "shared_critique_sha256")


def _optional_text(row, field):
    value = row.get(field)
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvalidSourceData(f"Invalid metadata field {field}")
    return value


def _validated_groups(rows):
    groups, seen, metadata = defaultdict(list), set(), {}
    for row in rows:
        key = VERIFIER.group(row, GROUP)
        for field in ("question_id", "condition", "mode"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise InvalidSourceData(f"Missing QA {field}")
        if row["condition"] not in {"false_advice", "correct_advice"}:
            raise InvalidSourceData("Unknown QA condition")
        unit = (*key, row["question_id"], row["condition"], row["mode"])
        if unit in seen:
            raise InvalidSourceData("Duplicate/conflicting QA observational unit; attempts cannot increase N")
        seen.add(unit)
        parsed = dict(row)
        for field in ("mechanism_eligible", "before_correct", "after_correct", "finished", "transport_contaminated"):
            parsed[field] = VERIFIER.optional_bool(row.get(field), field)
        if parsed["mechanism_eligible"] is None:
            raise InvalidSourceData("Missing mechanism_eligible flag")
        if parsed["mechanism_eligible"] and (
            parsed["after_correct"] is None or parsed["finished"] is not True
            or parsed["transport_contaminated"] is not False
            or row.get("status") in {"running", "campaign_budget_censored", "infrastructure_censored"}
        ):
            raise InvalidSourceData("Eligible row has a missing/censored outcome")
        parsed["source_dataset"] = _optional_text(row, "source_dataset")
        for field in DIGEST_FIELDS:
            parsed[field] = _optional_text(row, field)
        # Draft truth/topic/question identity are shared across conditions.
        # Advice identity is only shared inside one condition.
        for field in ("before_correct", "source_dataset", *DIGEST_FIELDS):
            value = parsed[field]
            if value is None:
                continue
            scope = row["condition"] if field == "shared_critique_sha256" else None
            meta_key = (*key, row["question_id"], scope, field)
            if meta_key in metadata and metadata[meta_key] != value:
                raise InvalidSourceData(f"Conflicting shared question metadata: {field}")
            metadata[meta_key] = value
        groups[key].append(parsed)
    return groups


def cluster_draws(topic_pairs, replicates=10000, seed=20261110):
    """Resample whole topics, preserving all eligible pairs and their multiplicity.

    An empty selected collection of eligible questions yields None, never a
    substituted zero or a hidden redraw. Topic lists may legitimately be empty.
    """
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    clusters = [(sum(values), len(values)) for _, values in sorted(topic_pairs.items())]
    if not clusters:
        return [None] * replicates
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        n = sum(size for _, size in sampled)
        draws.append(sum(total for total, _ in sampled) / n if n else None)
    return draws


def _percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def _summarize(key, rows, replicates, seed, expected_topics):
    target = [row for row in rows if row["condition"] == "false_advice" and row["mode"] in POLICIES]
    by_question = defaultdict(dict)
    topics = set()
    for row in target:
        by_question[row["question_id"]][row["mode"]] = row
        if row["source_dataset"] is not None:
            topics.add(row["source_dataset"])
    counters = {name: 0 for name in (
        "missing_policy_pair_n", "ineligible_pair_n", "initial_correctness_missing_pair_n",
        "initially_wrong_pair_n", "missing_topic_pair_n", "draft_digest_verified_pair_n",
        "advice_digest_verified_pair_n",
    )}
    outcomes, topic_pairs = [], {topic: [] for topic in sorted(topics)}
    for _, policies in sorted(by_question.items()):
        if any(mode not in policies for mode in POLICIES):
            counters["missing_policy_pair_n"] += 1
            continue
        a, b = (policies[mode] for mode in POLICIES)
        if not a["mechanism_eligible"] or not b["mechanism_eligible"]:
            counters["ineligible_pair_n"] += 1
            continue
        if a["before_correct"] is None or b["before_correct"] is None:
            counters["initial_correctness_missing_pair_n"] += 1
            continue
        if a["before_correct"] is not True or b["before_correct"] is not True:
            counters["initially_wrong_pair_n"] += 1
            continue
        outcomes.append((a["after_correct"], b["after_correct"]))
        for field, counter in (("shared_draft_sha256", "draft_digest_verified_pair_n"),
                               ("shared_critique_sha256", "advice_digest_verified_pair_n")):
            counters[counter] += a[field] is not None and b[field] is not None
        if a["source_dataset"] is None or b["source_dataset"] is None:
            counters["missing_topic_pair_n"] += 1
        else:
            topic_pairs[a["source_dataset"]].append(int(a["after_correct"]) - int(b["after_correct"]))
    n = len(outcomes)
    primary = PAIRED.paired_binary([a for a, _ in outcomes], [b for _, b in outcomes])
    if not n:
        primary["mcnemar_exact_two_sided_p"] = None
    status = ("undefined_no_eligible_pairs" if not n else
              "undefined_missing_topic" if counters["missing_topic_pair_n"] else
              "undefined_topic_coverage" if len(topics) != expected_topics else "defined")
    draws = cluster_draws(topic_pairs, replicates, seed) if status == "defined" else []
    valid_draws = sorted(value for value in draws if value is not None)
    if draws and len(valid_draws) != replicates:
        status = "undefined_zero_denominator_draw"
    interval = ([_percentile(valid_draws, .025), _percentile(valid_draws, .975)]
                if status == "defined" else [None, None])
    degenerate = bool(valid_draws) and valid_draws[0] == valid_draws[-1]
    leave_out = []
    if n and not counters["missing_topic_pair_n"] and len(topics) == expected_topics:
        total = sum(sum(values) for values in topic_pairs.values())
        for topic, values in topic_pairs.items():
            remaining = n - len(values)
            leave_out.append({"omitted_topic": topic, "remaining_pair_n": remaining,
                              "difference": (total - sum(values)) / remaining if remaining else None})
    leave_values = [row["difference"] for row in leave_out if row["difference"] is not None]
    warnings = []
    if counters["draft_digest_verified_pair_n"] != n or counters["advice_digest_verified_pair_n"] != n:
        warnings.append("Some pair digests are absent: shared inputs rely on the frozen protocol, not this numeric file")
    if degenerate:
        warnings.append("Degenerate bootstrap is not evidence of equivalence and does not replace the exact paired interval")
    if len(topics) != expected_topics:
        warnings.append("Observed topic universe differs from expected coverage; cluster interval is undefined")
    return {
        **dict(zip(GROUP, key, strict=True)), "condition": "false_advice", "endpoint": "retention",
        "a": POLICIES[0], "b": POLICIES[1], "candidate_question_n": len(by_question),
        **counters, "primary_exact": primary,
        "topic_n": len(topics), "expected_topic_n": expected_topics,
        "eligible_topic_n": sum(bool(values) for values in topic_pairs.values()),
        "topic_pair_n": {topic: len(values) for topic, values in topic_pairs.items()},
        "cluster_bootstrap": {
            "status": status, "seed": seed, "requested_replicates": replicates,
            "computed_replicates": len(draws), "defined_replicates": len(valid_draws),
            "zero_denominator_replicates": len(draws) - len(valid_draws),
            "difference": primary["difference"], "interval95": interval, "degenerate": degenerate,
            "method": "Whole-topic resampling with replacement; all eligible pairs retained; linear percentile95",
        },
        "leave_one_topic_out": {
            "status": "defined" if leave_out and len(leave_values) == len(leave_out) else "undefined",
            "defined_n": len(leave_values), "rows": leave_out,
            "difference_range": [min(leave_values), max(leave_values)]
            if leave_out and len(leave_values) == len(leave_out) else [None, None],
        }, "warnings": warnings,
    }


def analyze(rows, replicates=10000, seed=20261110, expected_topics=20):
    """Return separate summaries for every complete model/source/repeat identity."""
    if not isinstance(expected_topics, int) or isinstance(expected_topics, bool) or expected_topics < 1:
        raise ValueError("expected_topics must be a positive integer")
    cluster_draws({}, replicates, seed)  # Validate settings even when the CSV is empty.
    rows = list(rows)
    groups = _validated_groups(rows)
    return {
        "schema_version": "qa-topic-sensitivity-v1", "input_row_n": len(rows),
        "status": "defined_groups" if groups else "undefined_no_input_rows",
        "independence_unit": "Question pair; conditions, policies and repeats are not additional independent questions",
        "interpretation": "Preplanned descriptive topic sensitivity, not a replacement primary test or an equivalence test",
        "groups": [_summarize(key, values, replicates, seed, expected_topics)
                   for key, values in sorted(groups.items(), key=lambda item: str(item[0]))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--expected-topics", type=int, default=20)
    args = parser.parse_args()
    if args.out and args.out.resolve() == args.csv.resolve():
        raise ValueError("Analysis cannot overwrite its numeric source file")
    report = analyze(VERIFIER.read_csv(args.csv), args.replicates, args.seed, args.expected_topics)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
