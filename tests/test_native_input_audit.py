"""Synthetic initial-input audits; no environment, model or actual main data."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "native_input_audit.py"
SPEC = importlib.util.spec_from_file_location("native_input_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, variants=("raw_actor", "feedback")):
    root = tmp_path / "main-synthetic" / "native"
    mission = "PRIVATE_MISSION"
    task = {
        "key": "fixture-task",
        "benchmark": "textcraft",
        "task_id": audit.sha(mission),
        "seed": 42,
        "mission": mission,
        "initial_observation_sha256": audit.sha("PRIVATE_OBSERVATION"),
        "public_background_sha256": "f" * 64,
    }
    plan = {
        "schema_version": "feedback-native-run-plan-v1",
        "tasks": [task],
        "variants": list(variants),
        "model": "fixture-model",
        "source_sha256": "a" * 64,
    }
    write(root / "run_plan.json", plan)
    return root, task, plan


def episode(root, task, plan, variant, background, turns=True):
    value = {
        "key": task["key"],
        "benchmark": task["benchmark"],
        "task_id": task["task_id"],
        "variant": variant,
        "seed": task["seed"],
        "model": plan["model"],
        "source_sha256": plan["source_sha256"],
        "status": "success",
        "finished": True,
        "trajectory": [
            {
                "step": 0,
                "mission": task["mission"],
                "observation": "PRIVATE_OBSERVATION",
                "background": background,
                "available_actions": [],
            }
        ]
        if turns
        else [],
    }
    write(root / task["key"] / "episodes" / f"{task['key']}--{variant}.json", value)


def test_actual_within_block_identity_is_separate_from_selection_hash(tmp_path):
    root, task, plan = fixture(tmp_path)
    before = (root / "run_plan.json").read_bytes()
    for variant in plan["variants"]:
        episode(root, task, plan, variant, "craft 1 PRIVATE_RECIPE using 2 clay")
    rows, summary = audit.analyze(root)
    assert all(r["background_matches_selection"] is False for r in rows)
    assert all(r["mission_matches_selection"] is True for r in rows)
    assert all(r["within_block_all_planned_inputs_verified"] is True for r in rows)
    assert summary["fully_verified_block_n"] == 1
    assert not summary["within_block_conflicting_keys"]
    assert before == (root / "run_plan.json").read_bytes()
    assert "PRIVATE_" not in json.dumps(rows)


def test_recipe_semantics_hash_ignores_only_line_and_ingredient_order():
    first = "craft 1 brick using 2 clay, 1 water\ncraft 1 glass using 1 sand"
    reordered = "craft 1 glass using 1 sand\ncraft 1 brick using 1 water, 2 clay"
    different = "craft 1 glass using 1 sand\ncraft 1 brick using 1 water, 3 clay"
    assert audit.sha(first) != audit.sha(reordered)
    assert audit.recipe_fingerprint(first) == audit.recipe_fingerprint(reordered)
    assert audit.recipe_fingerprint(first) != audit.recipe_fingerprint(different)


def test_different_background_bytes_are_flagged_even_when_recipe_set_is_same(tmp_path):
    root, task, plan = fixture(tmp_path)
    episode(
        root,
        task,
        plan,
        "raw_actor",
        "craft 1 brick using 2 clay\ncraft 1 glass using 1 sand",
    )
    episode(
        root,
        task,
        plan,
        "feedback",
        "craft 1 glass using 1 sand\ncraft 1 brick using 2 clay",
    )
    rows, summary = audit.analyze(root)
    assert all(r["within_block_recorded_inputs_agree"] is False for r in rows)
    assert rows[0]["runtime_recipe_multiset_sha256"] == rows[1]["runtime_recipe_multiset_sha256"]
    assert summary["within_block_conflicting_keys"] == [task["key"]]


def test_unstarted_or_zero_action_variant_is_unverified_not_false(tmp_path):
    root, task, plan = fixture(tmp_path, ("raw_actor", "feedback", "stateact"))
    episode(root, task, plan, "raw_actor", "craft 1 brick using 2 clay")
    episode(root, task, plan, "feedback", "craft 1 brick using 2 clay")
    rows, summary = audit.analyze(root)
    missing = next(r for r in rows if r["variant"] == "stateact")
    assert missing["episode_started"] is False
    assert missing["runtime_background_sha256"] is None
    assert missing["background_matches_selection"] is None
    assert missing["within_block_recorded_inputs_agree"] is True
    assert missing["within_block_all_planned_inputs_verified"] is None
    assert summary["fully_verified_block_n"] == 0
    episode(root, task, plan, "stateact", "unused", turns=False)
    rows, _ = audit.analyze(root)
    missing = next(r for r in rows if r["variant"] == "stateact")
    assert missing["episode_started"] is True
    assert missing["runtime_initial_observation_sha256"] is None


def test_identity_drift_is_not_silently_assigned_to_the_planned_case(tmp_path):
    root, task, plan = fixture(tmp_path)
    episode(root, task, plan, "raw_actor", "craft 1 brick using 2 clay")
    path = root / task["key"] / "episodes" / f"{task['key']}--raw_actor.json"
    value = json.loads(path.read_text())
    value["seed"] = 99
    write(path, value)
    with pytest.raises(ValueError, match="identity differs"):
        audit.analyze(root)
