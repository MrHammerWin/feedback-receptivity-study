"""Compare saved numeric episodes; no account, provider, model or raw replies needed."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "analysis"))
from paired_statistics import clopper_pearson, paired_binary

BASELINES = ("stateact", "reflact_paper")
PRIOR_ARMS = ("feedback_open_commands", "grounding_only", "raw_actor")
INTS = {"seed", "depth", "success", "success_16", "success_32", "calls", "commands", "rejected", "unrecognized",
        "complete_probe_pairs", "both_probes_passed", "recipe_parse_failures", "insufficient_items", "invalid_recipe",
        "unavailable_get_items", "other_explicit_rejections", "accepted_get", "accepted_craft", "accepted_inventory",
        "input_tokens", "output_tokens", "cached_input_tokens", "cache_telemetry_complete", "unexecuted_turn_calls"}
FLOATS = {"model_seconds", "execution_wall_seconds"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path, arms, tasks):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"goal", "task_id", "variant", "status", "success", "calls", "commands", "rejected", "unrecognized"}
        if not required <= set(reader.fieldnames) or set(reader.fieldnames) - (INTS | FLOATS | {"goal", "task_id", "variant", "status", "source_sha256"}):
            raise ValueError("Unexpected numeric CSV schema")
        rows = [r for r in reader if r["variant"] in arms]
    expected = {(key, arm) for key in tasks for arm in arms}
    if len(rows) != len(expected) or {(r["goal"], r["variant"]) for r in rows} != expected:
        raise ValueError("Missing, extra or duplicated goal/variant pair")
    for row in rows:
        if row["task_id"] != tasks[row["goal"]]["task_id"]:
            raise ValueError("Numeric episode belongs to a different goal")
        for key in row.keys() & INTS:
            value = row[key]
            row[key] = {"True": 1, "False": 0}.get(value, value)
            row[key] = int(row[key])
        for key in row.keys() & FLOATS:
            row[key] = float(row[key])
        if any(row[key] < 0 for key in row.keys() & INTS) or any(not math.isfinite(row[key]) or row[key] < 0 for key in row.keys() & FLOATS):
            raise ValueError("Negative or non-finite numeric metric")
        if (row["success"] not in {0, 1} or row["status"] not in {"success", "episode_compute_cap"}
                or bool(row["success"]) != (row["status"] == "success")
                or not 0 < row["calls"] <= 32 or not 0 <= row["unrecognized"] <= row["rejected"] <= row["commands"] <= row["calls"]):
            raise ValueError("Invalid observed endpoint or budget")
        if "success_16" in row and not 0 <= row["success_16"] <= row["success"]:
            raise ValueError("Invalid 16-call endpoint")
        if "success_32" in row and row["success_32"] != row["success"]:
            raise ValueError("Invalid 32-call endpoint")
        if "cached_input_tokens" in row and not 0 <= row["cached_input_tokens"] <= row["input_tokens"]:
            raise ValueError("Cached input must be a subset of total input")
    return rows


def aggregates(rows):
    n = len(rows)
    success = sum(r["success"] for r in rows)
    calls, commands = (sum(r[name] for r in rows) for name in ("calls", "commands"))
    rejected, unrecognized = (sum(r[name] for r in rows) for name in ("rejected", "unrecognized"))
    result = {"n": n, "n_goals": n, "successes": success, "success_rate": success / n,
              "success_at_32": success, "success_interval95": clopper_pearson(success, n),
              "logical_calls": calls, "calls": calls, "mean_calls_per_goal": calls / n,
              "executed_actions": commands, "commands": commands, "explicit_rejections": rejected, "rejected": rejected,
              "explicit_rejection_rate_per_executed_action": rejected / commands, "unrecognized_commands": unrecognized,
              "unrecognized": unrecognized, "unrecognized_rate_per_executed_action": unrecognized / commands,
              "status_counts": dict(Counter(r["status"] for r in rows))}
    if "success_16" in rows[0]:
        result.update(success_16=sum(r["success_16"] for r in rows), success_at_16=sum(r["success_16"] for r in rows))
    for name in ("input_tokens", "output_tokens", "cached_input_tokens", "model_seconds", "execution_wall_seconds", "unexecuted_turn_calls"):
        if name in rows[0]:
            result[name] = sum(r[name] for r in rows)
    if "cached_input_tokens" in result:
        result.update(cached_input_fraction=result["cached_input_tokens"] / result["input_tokens"],
                      cached_tokens_are_subset_of_input_tokens=True,
                      calls_with_complete_token_telemetry=sum(r["calls"] for r in rows if r["cache_telemetry_complete"]),
                      cache_telemetry_complete=all(r["cache_telemetry_complete"] == 1 for r in rows))
    categories = {"unrecognized_command": "unrecognized", "recipe_parse_failure": "recipe_parse_failures",
                  "insufficient_items": "insufficient_items", "invalid_recipe": "invalid_recipe",
                  "unavailable_get_item": "unavailable_get_items", "other_explicit_rejection": "other_explicit_rejections"}
    if all(name in rows[0] for name in categories.values()):
        result["rejection_categories"] = {key: sum(r[name] for r in rows) for key, name in categories.items()}
        result["command_outcomes"] = {**result["rejection_categories"],
                                      **{name: sum(r[name] for r in rows) for name in ("accepted_get", "accepted_craft", "accepted_inventory")}}
    return result


def verify(csv_path=None):
    checks = 0

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            raise ValueError(label)

    def equal(actual, expected, label):
        if isinstance(expected, dict):
            for key, value in expected.items():
                check(key in actual, label + "." + key)
                equal(actual[key], value, label + "." + key)
        elif isinstance(expected, list):
            check(len(actual) == len(expected), label)
            for a, b in zip(actual, expected, strict=True):
                equal(a, b, label)
        elif isinstance(expected, (int, float)):
            check(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), label)
        else:
            check(actual == expected, label)

    record = read(ROOT / "source_record.json")
    release = read(ROOT / "release_manifest.json")
    actual_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and p.name != "release_manifest.json" and "__pycache__" not in p.parts}
    check(actual_files == set(release["files"]), "Release file list differs")
    for relative, expected in release["files"].items():
        path = (ROOT / relative).resolve()
        check(path.is_relative_to(ROOT) and not (ROOT / relative).is_symlink(), "Unsafe release file")
        check(digest(path) == expected, "Release hash differs: " + relative)
    source = {}
    for name, expected in record["common_source_sha256"].items():
        source[name] = ROOT.parent / "textcraft/frozen_runner/src/crmn_agent" / name
        check(digest(source[name]) == expected, "Shared source differs: " + name)
    for name, expected in record["source_delta_sha256"].items():
        source[name] = ROOT / "source_delta" / name
        check(digest(source[name]) == expected, "Baseline source differs: " + name)
    package = hashlib.sha256()
    for name in sorted(source):
        package.update(name.encode())
        package.update(source[name].read_bytes())
    check(package.hexdigest() == record["executed_source_sha256"], "Assembled source identity differs")
    inputs = ROOT.parent / "textcraft/frozen_inputs"
    check(digest(inputs / "manifest.json") == record["recorded_manifest_sha256"], "Frozen goals differ")
    tasks = {t["key"]: t for t in read(inputs / "manifest.json")["tasks"]}
    check(len(tasks) == len({t["task_id"] for t in tasks.values()}) == 40, "Forty distinct goals required")
    baseline = load_rows(csv_path or ROOT / "source_data/episodes.csv", BASELINES, tasks)
    prior = load_rows(ROOT / "source_data/prior_episodes.csv", PRIOR_ARMS, tasks)
    summary = read(ROOT / "source_data/summary.json")
    combined = prior + baseline
    per_arm = {arm: aggregates([r for r in combined if r["variant"] == arm]) for arm in (*PRIOR_ARMS, *BASELINES)}
    pairs = {}
    for cutoff in (16, 32):
        if cutoff == 16 and "success_16" not in prior[0]:
            continue
        field = "success_16" if cutoff == 16 else "success"
        outcomes = {(r["goal"], r["variant"]): bool(r[field]) for r in combined}
        pairs[cutoff] = {arm: paired_binary([outcomes[(key, "feedback_open_commands")] for key in tasks],
                                          [outcomes[(key, arm)] for key in tasks]) for arm in BASELINES}
    if csv_path is None:
        for arm, value in per_arm.items():
            equal(summary["per_arm"][arm], value, "per_arm." + arm)
        for cutoff, values in pairs.items():
            for arm, value in values.items():
                equal(summary[f"paired_vs_feedback_at{cutoff}"][arm], value, f"paired{cutoff}." + arm)
        closed = read(ROOT / "source_data/closure.json")
        check(sum(r["calls"] for r in baseline) == closed["reserved_physical_launches"] == 777, "Recorded baseline call counters differ")
        check(closed["status"] == "closed" and closed["stop_after"] == 0 and closed["paid_api_calls"] == 0, "Closure differs")
        check(sum(r["success"] for r in baseline) == 77, "Recorded baseline successes differ")
        family = summary["published_family4_primary_caps"]["members"]
        check(len(family) == 4, "Incomplete four-test Holm family")
        with (ROOT.parents[1] / "source_data/native_pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
            old_pairs = list(csv.DictReader(stream))
        for member in family:
            if member["benchmark"] == "textcraft":
                equal(member["p"], pairs[32][member["b"]]["mcnemar_exact_two_sided_p"], "TextCraft family p")
            else:
                matches = [r for r in old_pairs if r["phase"] == "main" and r["benchmark"] == "alfworld" and r["cutoff"] == "64" and r["repeat_id"] == "0" and r["a"] == "feedback" and r["b"] == member["b"]]
                check(len(matches) == 1, "Missing historical ALFWorld pair")
                equal(member["p"], float(matches[0]["mcnemar_exact_two_sided_p"]), "ALFWorld family p")
        previous = 0.0
        for rank, index in enumerate(sorted(range(4), key=lambda i: family[i]["p"])):
            previous = max(previous, min(1.0, (4-rank) * family[index]["p"]))
            equal(family[index]["p_holm_family4"], previous, "Holm adjustment")
    return {"passed": True, "checks": checks, "mode": "recorded" if csv_path is None else "new_numeric_csv",
            "goals": 40, "baseline_episodes": len(baseline), "comparable_episodes": len(combined),
            "per_arm": per_arm, "paired_vs_feedback": pairs, "new_model_calls": 0,
            "recomputed_pair_cutoffs": list(pairs),
            "prior_detailed_token_metrics_recomputed": "input_tokens" in prior[0],
            "boundary": "Numeric consistency and published source identities; private dialogue provenance and server receipts are not reverified."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.csv), indent=2))
