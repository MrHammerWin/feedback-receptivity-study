"""Verify published retest counts without a provider, account or model call."""
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from paired_statistics import paired_binary

ARMS = ("feedback", "feedback_open_commands", "grounding_only", "raw_actor")
COUNTS = ("calls", "commands", "rejected", "unrecognized", "complete_probe_pairs", "both_probes_passed")
FIELDS = {"goal", "task_id", "depth", "variant", "status", "success", *COUNTS}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    checks = 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise ValueError(message)

    def equal(actual, expected, label):
        if isinstance(expected, dict):
            check(isinstance(actual, dict) and set(actual) == set(expected), label)
            for key in expected:
                equal(actual[key], expected[key], f"{label}.{key}")
        elif isinstance(expected, list):
            check(isinstance(actual, list) and len(actual) == len(expected), label)
            for i, value in enumerate(expected):
                equal(actual[i], value, f"{label}[{i}]")
        elif isinstance(expected, float):
            check(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)
        else:
            check(type(actual) is type(expected) and actual == expected, label)

    release = read(root / "release_manifest.json")
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.name != "release_manifest.json"}
    check(actual_files == set(release["files"]), "Public file list changed")
    for name, expected in release["files"].items():
        relative = Path(name)
        path = (root / relative).resolve()
        check(not relative.is_absolute() and ".." not in relative.parts and path.is_relative_to(root), "Unsafe release path")
        check(not (root / relative).is_symlink() and re.fullmatch(r"[a-f0-9]{64}", expected) is not None, "Invalid file identity")
        check(digest(path) == expected, f"Hash differs: {name}")

    record = read(root / "source_record.json")
    package = hashlib.sha256()
    for name in sorted(record["frozen_source_files"]):
        path = root / "frozen_runner/src/crmn_agent" / name
        check(digest(path) == record["frozen_source_files"][name], f"Frozen source differs: {name}")
        package.update(name.encode())
        package.update(path.read_bytes())
    check(package.hexdigest() == record["executed_source_sha256"], "Executed package identity differs")
    check(digest(root / "runner.patch") == record["runner_patch_sha256"], "Recorded patch differs")

    inputs = root / "frozen_inputs"
    manifest = read(inputs / "manifest.json")
    tasks = manifest["tasks"]
    check(len(tasks) == len({t["key"] for t in tasks}) == len({t["task_id"] for t in tasks}) == 40, "Expected forty distinct goals")
    check(digest(inputs / "manifest.json") == record["recorded_manifest_sha256"], "Recorded manifest differs")
    by_key = {t["key"]: t for t in tasks}
    for task in tasks:
        check(task["benchmark"] == "textcraft" and sorted(task["variants"]) == sorted(ARMS), "Invalid arm design")
        check(re.fullmatch(r"[a-f0-9]{64}", task["task_id"]) is not None and re.fullmatch(r"[a-f0-9]{64}", task["initial_inputs_sha256"]) is not None, "Invalid goal/input digest")
    audit = read(inputs / "pilot_audit.json")
    check(digest(inputs / "pilot_audit.json") == record["public_redacted_audit_sha256"], "Public digest audit differs")
    check(audit["manifest_sha256"] == record["recorded_manifest_sha256"], "Digest audit has another manifest")
    excluded = set()
    for source in audit["sources"]:
        path = inputs / source["path"]
        check(digest(path) == source["sha256"], "Digest evidence hash differs")
        value = read(path)
        if source["role"] == "prior_main":
            ids = {t["task_id"] for t in value["tasks"] if t["benchmark"] == "textcraft"}
            check(len(ids) == 8, "Expected eight prior primary goals")
        else:
            check(value["role"] == source["role"] and value["visibility"].startswith("goal-digests-only"), "Unexpected public evidence scope")
            ids = {t["task_id"] for t in value["rows"]}
        excluded.update(ids)
    check(excluded == set(audit["excluded_task_ids"]) and len(excluded) == 18, "Public exclusion set differs")
    check(not excluded.intersection(t["task_id"] for t in tasks), "Selected goals overlap declared exclusions")

    with (root / "source_data/episodes.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        check(set(reader.fieldnames) == FIELDS, "Unexpected numeric CSV fields")
        rows = list(reader)
    check(len(rows) == 160, "Expected 160 episode records")
    expected_pairs = {(t["key"], arm) for t in tasks for arm in ARMS}
    check({(r["goal"], r["variant"]) for r in rows} == expected_pairs, "Episode identities differ or are duplicated")
    for row in rows:
        task = by_key[row["goal"]]
        check(row["task_id"] == task["task_id"] and int(row["depth"]) == task["depth"], "Episode goal binding differs")
        check(row["success"] in {"True", "False"}, "Invalid success flag")
        row["success"] = row["success"] == "True"
        for name in COUNTS:
            row[name] = int(row[name])
        check(row["status"] in {"success", "episode_compute_cap"} and row["success"] == (row["status"] == "success"), "Invalid observed endpoint")
        check(0 < row["calls"] <= 32 and row["commands"] <= row["calls"], "Invalid episode budget")
        check(0 <= row["unrecognized"] <= row["rejected"] <= row["commands"], "Invalid rejection counts")
        check(0 <= row["both_probes_passed"] <= row["complete_probe_pairs"] <= row["commands"], "Invalid probe counts")
        if row["variant"] in {"grounding_only", "raw_actor"}:
            check(row["complete_probe_pairs"] == 0, "Ablation unexpectedly uses probes")

    summary = read(root / "source_data/summary.json")
    closed = read(root / "source_data/closure.json")
    check(summary["paid_api"] is False and closed["paid_api_calls"] == 0, "Paid-API declaration differs")
    check(closed["status"] == "closed" and closed["stop_after"] == 0, "Recorded campaign is not closed")
    check(closed["executed_source_sha256"] == record["executed_source_sha256"], "Closure source identity differs")
    equal(summary["statuses"], dict(Counter(r["status"] for r in rows)), "statuses")
    equal(closed["episode_status_counts"], summary["statuses"], "closure.statuses")
    check(summary["episodes"] == closed["completed_episodes"] == 160 and summary["goals"] == closed["goal_count"] == 40, "Wrong sample units")
    check(sum(r["calls"] for r in rows) == summary["physical_starts"] == closed["reserved_physical_launches"] == 2724, "Reported call counters differ")
    check(closed["authorized_launches"] == 5120 and closed["unspent_launches"] == 2396, "Recorded allowance differs")
    check(summary["manifest_sha256"] == record["recorded_manifest_sha256"] and summary["audit_sha256"] == record["recorded_private_audit_sha256"], "Historical identities differ")

    def rate(numerator, denominator):
        return {"count": numerator, "total": denominator, "rate": numerator / denominator}

    for arm in ARMS:
        group = [r for r in rows if r["variant"] == arm]
        totals = {name: sum(r[name] for r in group) for name in COUNTS}
        calculated = {"success": rate(sum(r["success"] for r in group), len(group)),
                      "calls": totals["calls"], "mean_calls": totals["calls"] / len(group),
                      "commands": totals["commands"], "rejected": rate(totals["rejected"], totals["commands"]),
                      "unrecognized": rate(totals["unrecognized"], totals["commands"]),
                      "complete_probe_pairs": totals["complete_probe_pairs"], "both_probes_passed": totals["both_probes_passed"]}
        equal(summary["per_arm"][arm], calculated, f"per_arm.{arm}")
        equal(closed["episode_calls_by_variant"][arm], totals["calls"], f"closure.calls.{arm}")
    outcomes = {(r["goal"], r["variant"]): r["success"] for r in rows}
    for other in ("feedback", "grounding_only", "raw_actor"):
        actual = paired_binary([outcomes[(t["key"], "feedback_open_commands")] for t in tasks],
                               [outcomes[(t["key"], other)] for t in tasks])
        equal(summary["paired_new_instruction_vs"][other], actual, f"paired.{other}")
    return {"passed": True, "checks": checks, "goals": 40, "episodes": 160,
            "recorded_calls": 2724, "new_model_calls": 0,
            "boundary": "Numeric consistency and published digests only; private coverage completeness, raw responses and server receipts are not reverified."}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
