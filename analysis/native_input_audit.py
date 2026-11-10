"""Read-only initial-input fidelity audit; never rewrite the selection manifest.

Only hashes, IDs and counts are exported. Matching seed/goal does not imply an
identical TextCraft recipe sheet. Missing initial frames remain unverified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

RECIPE = re.compile(r"^craft\s+(\d+)\s+(.+?)\s+using\s+(.+)$")
INGREDIENT = re.compile(r"^(\d+)\s+(.+)$")


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def json_sha(value):
    return sha(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def recipe_fingerprint(background):
    recipes, unparsed = [], []
    for line in background.splitlines():
        line = line.strip()
        if not line.startswith("craft "):
            continue
        match = RECIPE.fullmatch(line)
        if match is None:
            unparsed.append(sha(line))
            continue
        ingredients = [
            INGREDIENT.fullmatch(part.strip()) for part in match[3].split(",")
        ]
        if any(part is None for part in ingredients):
            unparsed.append(sha(line))
            continue
        recipes.append(
            (
                int(match[1]),
                " ".join(match[2].split()),
                sorted(
                    (int(part[1]), " ".join(part[2].split())) for part in ingredients
                ),
            )
        )
    return {
        "runtime_recipe_line_n": len(recipes) + len(unparsed),
        "runtime_unparsed_recipe_line_n": len(unparsed),
        "runtime_recipe_multiset_sha256": json_sha(
            {"recipes": sorted(recipes), "unparsed_line_hashes": sorted(unparsed)}
        )
        if recipes or unparsed
        else None,
    }


def compare(actual, expected):
    return actual == expected if actual is not None and expected is not None else None


def analyze(native):
    native = Path(native).resolve()
    plans = list(native.rglob("run_plan.json"))
    if len(plans) != 1:
        raise ValueError(
            "Select one native root containing exactly one frozen run_plan.json"
        )
    plan = json.loads(plans[0].read_text())
    if plan.get("schema_version") != "feedback-native-run-plan-v1":
        raise ValueError("Unexpected native plan schema")
    if not plan.get("tasks") or not plan.get("variants"):
        raise ValueError("Frozen native plan has no task/variant allocation")
    labels = [
        part for part in native.parts if part.startswith(("main", "pilot", "heldout"))
    ]
    cohort = labels[-1] if labels else "unknown"
    phase = (
        "main"
        if cohort.startswith(("main", "heldout"))
        else "pilot"
        if cohort.startswith("pilot")
        else "unknown"
    )
    rows, conflicts = [], []
    for task in plan["tasks"]:
        block = []
        for variant in plan["variants"]:
            row = {
                "phase": phase,
                "cohort": cohort,
                "key": task["key"],
                "benchmark": task["benchmark"],
                "variant": variant,
                "seed": task["seed"],
                "model": plan["model"],
                "source_sha256": plan["source_sha256"],
                "selection_task_id": task.get("task_id"),
                "selection_initial_observation_sha256": task.get(
                    "initial_observation_sha256"
                ),
                "selection_background_sha256": task.get("public_background_sha256"),
                "selection_mission_sha256": sha(task["mission"])
                if isinstance(task.get("mission"), str)
                else None,
                "runtime_task_id": None,
                "runtime_initial_observation_sha256": None,
                "runtime_background_sha256": None,
                "runtime_mission_sha256": None,
                "runtime_actions_sha256": None,
                "runtime_frame_sha256": None,
                "runtime_recipe_multiset_sha256": None,
                "runtime_recipe_line_n": None,
                "runtime_unparsed_recipe_line_n": None,
                "episode_started": False,
                "episode_finished": None,
                "episode_status": None,
                "runtime_background_constant_across_recorded_steps": None,
                "runtime_mission_constant_across_recorded_steps": None,
                "audit_input_status": "episode_not_started",
            }
            matches = list(native.glob(f"**/episodes/{task['key']}--{variant}.json"))
            if len(matches) > 1:
                raise ValueError("Conflicting duplicate native episode paths")
            if matches:
                row["episode_started"] = True
                try:
                    episode = json.loads(matches[0].read_text())
                except ValueError:
                    row["audit_input_status"] = "snapshot_read_incomplete"
                    block.append(row)
                    continue
                row.update(
                    episode_started=True,
                    episode_finished=episode.get("finished"),
                    episode_status=episode.get("status"),
                    runtime_task_id=episode.get("task_id"),
                )
                if any(
                    episode.get(k) != expected
                    for k, expected in (
                        ("key", task["key"]),
                        ("benchmark", task["benchmark"]),
                        ("variant", variant),
                        ("seed", task["seed"]),
                        ("model", plan["model"]),
                        ("source_sha256", plan["source_sha256"]),
                    )
                ):
                    raise ValueError(
                        "Native episode identity differs from its frozen plan"
                    )
                turns = episode.get("trajectory", [])
                row["audit_input_status"] = "initial_frame_not_recorded"
                if (
                    turns
                    and turns[0].get("step") == 0
                    and all(
                        isinstance(turns[0].get(k), str)
                        for k in ("observation", "mission", "background")
                    )
                    and isinstance(turns[0].get("available_actions"), list)
                ):
                    first = turns[0]
                    row.update(
                        runtime_initial_observation_sha256=sha(first["observation"]),
                        runtime_background_sha256=sha(first["background"]),
                        runtime_mission_sha256=sha(first["mission"]),
                        runtime_actions_sha256=json_sha(first["available_actions"]),
                        runtime_frame_sha256=json_sha(
                            {
                                k: first[k]
                                for k in (
                                    "observation",
                                    "mission",
                                    "background",
                                    "available_actions",
                                )
                            }
                        ),
                        runtime_background_constant_across_recorded_steps=all(
                            t.get("background") == first["background"] for t in turns
                        ),
                        runtime_mission_constant_across_recorded_steps=all(
                            t.get("mission") == first["mission"] for t in turns
                        ),
                        audit_input_status="initial_frame_available",
                    )
                    if task["benchmark"] == "textcraft":
                        row.update(recipe_fingerprint(first["background"]))
            for name in (
                "task_id",
                "initial_observation_sha256",
                "background_sha256",
                "mission_sha256",
            ):
                runtime_name = "runtime_" + name
                row[name.replace("_sha256", "") + "_matches_selection"] = compare(
                    row.get(runtime_name), row.get("selection_" + name)
                )
            block.append(row)
        known = [r for r in block if r["runtime_frame_sha256"] is not None]
        agreement = (
            len({(r["runtime_frame_sha256"], r["runtime_task_id"]) for r in known}) == 1
            if len(known) >= 2
            else None
        )
        if agreement is False:
            conflicts.append(task["key"])
        for row in block:
            row.update(
                block_recorded_input_n=len(known),
                block_planned_variants_n=len(block),
                within_block_recorded_inputs_agree=agreement,
                within_block_all_planned_inputs_verified=agreement
                if len(known) == len(block)
                else None,
            )
        rows.extend(block)
    summary = {
        "schema_version": "native-input-fidelity-v1",
        "planned_episode_n": len(rows),
        "planned_task_n": len(plan["tasks"]),
        "fully_verified_block_n": len(
            {
                r["key"]
                for r in rows
                if r["within_block_all_planned_inputs_verified"] is True
            }
        ),
        "recorded_initial_input_n": sum(
            r["runtime_frame_sha256"] is not None for r in rows
        ),
        "within_block_conflicting_keys": conflicts,
        "selection_background_mismatch_n": sum(
            r.get("background_matches_selection") is False for r in rows
        ),
        "selection_goal_mismatch_n": sum(
            r.get("mission_matches_selection") is False for r in rows
        ),
        "limits": [
            "Selection hashes are preserved separately; runtime hashes do not overwrite the frozen plan.",
            "Agreement covers available recorded initial inputs, not missing/unstarted variants.",
            "Collector PIDs and full hidden environment state were not recorded in episode files.",
            "Recipe fingerprints normalize command ingredient/line order; they are not a proof of identical hidden physics.",
        ],
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    for output in (args.out, args.summary):
        if output is not None and output.resolve().is_relative_to(
            args.native.resolve()
        ):
            raise ValueError(
                "Derived audit outputs must be outside the private native input root"
            )
    rows, summary = analyze(args.native)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as stream:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if summary["within_block_conflicting_keys"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
