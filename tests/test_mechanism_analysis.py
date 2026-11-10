"""Accounting and observational invariants; no model calls in these fixtures."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "analyze_mechanism", Path(__file__).resolve().parents[1] / "scripts" / "analyze_mechanism.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def episode(variant="feedback", **changes):
    return {
        "variant": variant,
        "path": "fixture",
        "split": "heldout",
        "stage": "v2_corrected_protocol",
        "key": "a",
        "task_id": "a",
        "seed": 1,
        "model": "test",
        "benchmark": "textcraft",
        "status": "success",
        "finished": True,
        "success": True,
        "protocol": "test-only",
        "config": {
            "horizons": {"textcraft": 39},
            "episode_call_cap": 64,
            "provider": {"reasoning_effort": "low", "backend": "codex"},
        },
        "trajectory": [],
        "usage": {"calls": 3},
        **changes,
    }


def test_empty_fraction_is_not_zero_accuracy():
    assert module.fraction(0, 0) == {"count": 0, "n": 0, "rate": None}
    assert module.fraction(1, 2)["rate"] == 0.5


def test_diagnostic_coverage_is_only_observable_measured_steps():
    value = episode(
        "diagnostic",
        trajectory=[
            {
                "diagnostics": {
                    "probe_count": 0,
                    "reason": "no_public_measurable_slots",
                    "diagnosis": "unresolved",
                }
            },
            {
                "diagnostics": {
                    "probe_count": 2,
                    "full_audit": True,
                    "qualified_decoders": {"literal": True, "relational": False},
                    "diagnosis": "decoder_failure",
                }
            },
        ],
    )
    result = module.summary_group([value], "diagnostic")
    assert result["measured_steps"]["rate"] == 0.5
    assert result["no_public_slots_steps"]["count"] == 1
    assert result["qualified_roles"]["literal"] == {"count": 1, "n": 1, "rate": 1}
    assert result["qualified_roles"]["relational"]["rate"] == 0
    assert "semantic truth" in result["measurement_limit"]


def test_healthy_quote_profile_does_not_turn_failed_task_into_success():
    profile = {"observed_pairs": 1, "beneficial_correction": 1, "harmful_corruption": 0}
    failed = episode(
        success=False,
        status="episode_compute_cap",
        trajectory=[
            {"diagnostics": {"profile": profile, "critic_requested_revision": True}},
            {"diagnostics": {"profile": profile, "critic_requested_revision": False}},
        ],
    )
    result = module.summary_group([failed], "feedback")
    assert result["successes"] == 0
    assert result["beneficial_correction"]["rate"] == 1
    assert result["profile_passed_but_task_failed"] == {"count": 1, "n": 1, "rate": 1}
    assert result["episodes_with_healthy_profile"] == 1
    assert result["critic_requested_revision"]["rate"] == 0.5


def accounting_fixture(tmp_path):
    manifest = tmp_path / "manifest.json"
    write(manifest, {"tasks": [{"key": "a", "mission": "fixture-only"}]})
    run = {"config": {"manifest": str(manifest), "variants": ["feedback"]}}
    write(tmp_path / "run.json", run)
    write(tmp_path / "episodes/a--feedback.json", episode())
    records = [
        {
            "model": "test",
            "input": {"mission": "fixture-only"},
            "subscription_launch": 101 + i,
            "replayed": i < 2,
        }
        for i in range(3)
    ]
    (tmp_path / "model_calls.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    return run


def test_replayed_calls_are_logical_cost_not_new_physical_launches(tmp_path):
    run = accounting_fixture(tmp_path)
    result = module.run_call_accounting(tmp_path / "run.json", run)
    value = result["mapped_episodes"][str(tmp_path / "episodes/a--feedback.json")]
    assert value["logical_calls"] == 3
    assert value["replayed_logical_calls"] == 2
    assert value["new_physical_launches"] == 1
    assert value["historical_launch_ids"] == [101, 102]
    assert not result["warnings"]


def test_relocated_call_accounting_resolves_included_manifest_by_authentic_hash(tmp_path):
    run = accounting_fixture(tmp_path)
    local_manifest = tmp_path / "manifest.json"
    run["config"]["manifest"] = "/nonexistent-original-machine/manifest.json"
    run["manifest_sha256"] = module.digest(local_manifest)
    metadata = tmp_path / "run.json"
    write(metadata, run)
    before = metadata.read_bytes()
    result = module.run_call_accounting(metadata, run)
    assert not result["warnings"]
    assert result["resolved_manifest"] == str(local_manifest)
    assert len(result["mapped_episodes"]) == 1
    assert metadata.read_bytes() == before
    run["manifest_sha256"] = "0" * 64
    rejected = module.run_call_accounting(metadata, run)
    assert rejected["mapped_episodes"] == {} and rejected["warnings"]


def test_inconsistent_order_fails_closed_without_partial_call_claim(tmp_path):
    run = accounting_fixture(tmp_path)
    manifest = Path(run["config"]["manifest"])
    write(manifest, {"tasks": [{"key": "missing"}, {"key": "a", "mission": "fixture-only"}]})
    result = module.run_call_accounting(tmp_path / "run.json", run)
    assert result["mapped_episodes"] == {}
    assert result["warnings"]


def test_cost_comparison_matches_conditions_and_uses_logical_cost():
    ours = episode(usage={"calls": 15})
    baseline = episode("stateact", usage={"calls": 5})
    other = episode("reflact_paper", model="other-model", usage={"calls": 1})
    pairs = module.cost_pairs([ours, baseline, other], "feedback")
    assert len(pairs) == 1
    assert pairs[0]["logical_call_ratio"] == 3
    assert pairs[0]["candidate_success"] is True


def test_optional_raw_actor_cost_pair_is_exploratory_and_only_present_with_data():
    ours = episode(usage={"calls": 15})
    assert module.cost_pairs([ours], "feedback") == []
    raw = episode("raw_actor", usage={"calls": 5})
    pair = module.cost_pairs([ours, raw], "feedback")[0]
    assert pair["logical_call_ratio"] == 3
    assert pair["comparison_kind"] == "exploratory_ablation"
    assert pair["baseline_label"] == "Исходный актор · абляция"
    assert module.BASELINES == {"stateact", "reflact_paper"}


def test_missing_path_is_reported_and_output_describes_selected_runs(tmp_path):
    output = tmp_path / "appendix.md"
    result = module.analyze([tmp_path / "missing"], "feedback", output)
    assert result["warnings"]
    data = json.loads(output.with_suffix(".json").read_text())
    assert data["status"] == "selected_runs_analysis"
    assert data["groups"] == []
    assert data["absolute_sota_claim"] is False
    assert output.is_file()


def test_finished_censoring_is_excluded_but_compute_cap_is_evaluable():
    assert not module.eligible(episode(status="campaign_budget_censored", success=False))
    assert not module.eligible(episode(finished=False))
    assert module.eligible(episode(status="episode_compute_cap", success=False))


def test_failure_examples_are_capped_and_only_use_registered_failed_traces():
    values = [episode(key=str(i), success=False, status="action_horizon") for i in range(6)]
    result = module.failure_examples(values, "feedback", {})
    assert len(result) == 3
    assert all(value["last_public_turns"] == [] for value in result)
