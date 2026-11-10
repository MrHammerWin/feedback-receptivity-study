"""Statistical/coverage invariants; synthetic fixtures are never research results."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_report", Path(__file__).resolve().parents[1] / "scripts" / "build_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def episode(key="a", variant="diagnostic", **changes):
    return {
        "key": key,
        "task_id": key,
        "seed": 7,
        "benchmark": "textcraft",
        "variant": variant,
        "model": "unit-test-model",
        "protocol": "synthetic-fixture",
        "finished": True,
        "status": "success",
        "success": True,
        "steps": 2,
        "trajectory": [],
        "usage": {"calls": 3},
        "source_sha256": "fixture",
        **changes,
    }


def run_metadata(path, tasks=("a", "b"), variants=("diagnostic",), **changes):
    manifest = path / "tasks.json"
    write(manifest, {"tasks": [{"key": key, "benchmark": "textcraft"} for key in tasks]})
    value = {
        "config": {
            "manifest": str(manifest),
            "variants": list(variants),
            "horizons": {"textcraft": 39},
            "episode_call_cap": 64,
            "provider": {"backend": "codex", "reasoning_effort": "low"},
            **changes,
        }
    }
    write(path / "run.json", value)
    return value


def normalized(tmp_path, values):
    root = tmp_path / "heldout"
    run_metadata(root, variants=tuple({v["variant"] for v in values}))
    for i, value in enumerate(values):
        write(root / "episodes" / f"{i}.json", value)
    return report.load_episodes([root], "diagnostic")[0]


def test_wilson_empty_is_undefined_and_boundary_is_not_degenerate():
    assert report.wilson(0, 0) == (None, None)
    low, high = report.wilson(0, 10)
    assert low == pytest.approx(0)
    assert high == pytest.approx(0.2775328, abs=1e-6)
    low, high = report.wilson(10, 10)
    assert low == pytest.approx(0.7224672, abs=1e-6)
    assert high == pytest.approx(1)
    with pytest.raises(ValueError):
        report.wilson(2, 1)


def test_exact_mcnemar_needs_six_unopposed_discordants_for_nominal_significance():
    assert report.exact_mcnemar(5, 0) == pytest.approx(0.0625)
    assert report.exact_mcnemar(6, 0) == pytest.approx(0.03125)
    assert report.exact_mcnemar(0, 6) == pytest.approx(0.03125)
    assert report.exact_mcnemar(4, 4) == pytest.approx(1)
    assert report.exact_mcnemar(0, 0) == 1


def test_holm_is_monotone_and_restores_original_order():
    assert report.holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    assert report.holm_adjust([]) == []
    assert report.holm_adjust([1, 1]) == [1, 1]


def test_censoring_is_not_failure_but_episode_compute_cap_is(tmp_path):
    rows = normalized(
        tmp_path,
        [
            episode("done"),
            episode("cap", status="episode_compute_cap", success=False),
            episode("quota", status="campaign_budget_censored", success=False),
            episode("inflight", status="running", finished=False, success=False),
            episode("invalid", status="model_error", success=False),
        ],
    )
    summary = report.summarize(rows)[0]
    assert summary["records"] == 5
    assert summary["n"] == 3
    assert summary["successes"] == 1
    assert summary["excluded"] == 2
    assert summary["compute_cap_failures"] == 1
    assert summary["model_error_failures"] == 1
    assert summary["calls_total_all_records"] == 15
    assert summary["calls_per_solved_episode"] == 9


def test_pairing_does_not_cross_models_or_pilot_split(tmp_path):
    root = tmp_path
    for split in ["pilot", "heldout"]:
        r = root / split
        run_metadata(r, variants=("diagnostic", "stateact"))
        write(r / "episodes" / "ours.json", episode("a", success=split == "heldout"))
        write(r / "episodes" / "theirs.json", episode("a", variant="stateact", success=False))
    write(
        root / "heldout" / "episodes" / "other_model.json",
        episode("b", variant="stateact", model="different-model"),
    )
    rows = report.load_episodes([root], "diagnostic")[0]
    results = [r for r in report.paired_comparisons(rows, "diagnostic") if r["paired_n"]]
    assert len(results) == 2
    assert {r["split"]: r["delta_pp"] for r in results} == {"pilot": 0, "heldout": 100}
    assert all(r["paired_n"] == 1 for r in results)


def test_protocol_changes_cannot_make_silent_pairs(tmp_path):
    roots = [tmp_path / "heldout" / "ours", tmp_path / "heldout" / "theirs"]
    for root, method, cap in zip(roots, ["diagnostic", "stateact"], [64, 20], strict=True):
        run_metadata(root, variants=(method,), episode_call_cap=cap)
        write(root / "episodes" / "a.json", episode(variant=method))
    rows = report.load_episodes([tmp_path], "diagnostic")[0]
    assert all(r["paired_n"] == 0 for r in report.paired_comparisons(rows, "diagnostic"))


def test_identical_copied_baseline_is_counted_once(tmp_path):
    for location in ["one", "two"]:
        root = tmp_path / "heldout" / location
        run_metadata(root, variants=("stateact",))
        write(root / "episodes" / "a.json", episode(variant="stateact"))
    rows, warnings, *_ = report.load_episodes([tmp_path], "diagnostic")
    assert len(rows) == 1
    assert len(rows[0]["aliases"]) == 1
    assert not warnings


def test_conflicting_duplicate_is_excluded_not_averaged(tmp_path):
    for location, success in [("one", True), ("two", False)]:
        root = tmp_path / "heldout" / location
        run_metadata(root)
        write(root / "episodes" / "a.json", episode(success=success))
    rows, warnings, *_ = report.load_episodes([tmp_path], "diagnostic")
    assert rows == []
    assert "Конфликтующие" in warnings[0]


def test_coverage_counts_tasks_even_if_no_episode_started(tmp_path):
    root = tmp_path / "heldout" / "diagnostic"
    run_metadata(root, tasks=("a", "b"))
    rows, _, _, runs = report.load_episodes([root], "diagnostic")
    coverage, warnings, _ = report.expected_coverage(runs, rows)
    assert len(coverage) == 2
    assert not any(r["record_present"] for r in coverage)
    assert not warnings


def test_diagnostic_unknown_labels_are_not_scored_as_errors():
    result = report.battery_summary(
        [
            {"condition": "clean", "diagnosis_correct": True, "ablation_correct": False},
            {"condition": "ecological", "diagnosis_correct": None, "ablation_correct": None},
        ]
    )
    assert len(result) == 4  # Per-condition rows and the same cases' overall summary.
    assert all(row["condition"] != "ecological" for row in result)
    assert {r["method"]: r["rate"] for r in result} == {"diagnostic": 1, "literal_only_ablation": 0}
    assert all(r["n"] == 1 for r in result)


def test_feedback_battery_uses_conditional_denominators():
    result = report.battery_summary(
        [
            {"mode": "controlled", "before_correct": False, "after_correct": True},
            {"mode": "controlled", "before_correct": False, "after_correct": False},
            {"mode": "controlled", "before_correct": True, "after_correct": False},
        ]
    )
    by_metric = {r["metric"]: r for r in result}
    assert by_metric["source_fidelity_accuracy"]["rate"] == pytest.approx(1 / 3)
    assert by_metric["bad_to_good"]["n"] == 2
    assert by_metric["bad_to_good"]["rate"] == 0.5
    assert by_metric["good_to_bad"]["n"] == 1
    assert by_metric["good_to_bad"]["rate"] == 1


def test_battery_json_and_jsonl_exports_do_not_double_count(tmp_path):
    value = {"case_id": "one", "mode": "controlled", "before_correct": False, "after_correct": True}
    p = tmp_path / "battery.json"
    write(p, {"rows": [value], "model": "fixture"})
    q = tmp_path / "records.jsonl"
    q.write_text(json.dumps(value) + "\n")
    rows, *_ = report.load_battery([p, q])
    assert len(rows) == 1


def test_battery_pilot_and_heldout_are_never_pooled():
    rows = [
        {"battery_split": split, "mode": "controlled", "after_correct": correct}
        for split, correct in [("pilot", True), ("heldout", False)]
    ]
    result = report.battery_summary(rows)
    assert {(r["split"], r["rate"]) for r in result} == {("pilot", 1), ("heldout", 0)}


def test_redaction_retains_usage_but_not_credential_values():
    assert report.redact({"provider": {"api_key": "do-not-copy"}, "input_tokens": 12}) == {
        "provider": {"api_key": "[REDACTED]"},
        "input_tokens": 12,
    }


def test_clustered_battery_does_not_report_spurious_independent_case_intervals():
    result = report.battery_summary(
        [
            {"state_id": "same-state", "mode": "controlled", "after_correct": True},
            {"state_id": "same-state", "mode": "controlled", "after_correct": False},
        ]
    )[0]
    assert result["n"] == 2
    assert result["state_n"] == 1
    assert result["ci_low"] is None and result["ci_high"] is None


def test_condition_specific_state_ids_share_one_provenance_source(tmp_path):
    source = "The cabinet 1 is closed."
    cases = []
    rows = []
    for i, condition in enumerate(["clean", "representation_loss"]):
        cases.append(
            {
                "case_id": str(i),
                "state_id": "physical-state--" + condition,
                "source": source,
                "provenance": {"benchmark": "alfworld", "task_id": "same-task", "step": 2},
            }
        )
        rows.append(
            {
                "case_id": str(i),
                "state_id": "physical-state--" + condition,
                "condition": condition,
                "diagnosis_correct": i == 0,
                "ablation_correct": False,
            }
        )
    write(tmp_path / "manifest.json", {"cases": cases})
    (tmp_path / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    loaded, _, warnings = report.load_battery([tmp_path])
    assert not warnings
    assert len({row["original_state_id"] for row in loaded}) == 2
    assert len({row["canonical_source_state_id"] for row in loaded}) == 1
    overall = next(
        row
        for row in report.battery_summary(loaded)
        if row["method"] == "diagnostic" and row["condition"] == "all_synthetic"
    )
    assert overall["n"] == 2
    assert overall["state_n"] == 1
    assert overall["underlying_task_n"] == 1
    assert overall["ci_low"] is None and overall["ci_high"] is None
    inventory = {row["scope"]: row for row in report.battery_inventory(loaded)}
    assert inventory["synthetic_scored"]["source_states"] == 1
    assert inventory["synthetic_scored"]["underlying_tasks"] == 1


def test_diagnostic_state_ids_without_provenance_do_not_claim_independence():
    result = report.battery_summary(
        [{"state_id": "condition-specific-label", "condition": "clean", "diagnosis_correct": True}]
    )
    assert all(row["state_n"] is None for row in result)


def test_feedback_results_json_is_discovered_from_battery_directory(tmp_path):
    write(
        tmp_path / "results.json",
        {
            "model": "fixture",
            "rows": [
                {
                    "case_id": "feedback",
                    "state_id": "correct-physical-source",
                    "task_id": "one-task",
                    "mode": "controlled",
                    "before_correct": True,
                    "after_correct": True,
                }
            ],
        },
    )
    rows, _, warnings = report.load_battery([tmp_path])
    assert not warnings
    assert len(rows) == 1
    inventory = report.battery_inventory(rows)[0]
    assert inventory["source_states"] == 1 and inventory["underlying_tasks"] == 1


def test_optional_actor_is_absent_without_rows_and_has_no_significance_test(tmp_path):
    rows = normalized(
        tmp_path,
        [
            episode(),
            episode(variant="stateact", success=False),
            episode(variant="reflact_paper", success=False),
        ],
    )
    original = report.paired_comparisons(rows, "diagnostic")
    assert all(row["baseline"] != "raw_actor" for row in original)
    actor = dict(rows[0], variant="raw_actor", success=False, calls=1)
    extended = report.paired_comparisons([*rows, actor], "diagnostic")
    ablation = next(row for row in extended if row["baseline"] == "raw_actor")
    assert ablation["paired_n"] == 1
    assert ablation["comparison_kind"] == "exploratory_ablation"
    assert ablation["baseline_label"] == "Исходный актор · абляция"
    assert ablation["p_exact"] is None and ablation["p_holm"] is None
    assert [(r["baseline"], r["p_holm"]) for r in original] == [
        (r["baseline"], r["p_holm"]) for r in extended if r["baseline"] in report.BASELINES
    ]
    assert report.BASELINES == ("stateact", "reflact_paper")


@pytest.mark.parametrize("with_actor", [False, True])
def test_success_figure_renders_with_optional_actor_and_without_it(tmp_path, with_actor):
    pytest.importorskip("matplotlib")
    values = [episode(), episode(variant="stateact"), episode(variant="reflact_paper")]
    if with_actor:
        values.append(episode(variant="raw_actor", success=False))
    rows = normalized(tmp_path, values)
    output = tmp_path / "figures"
    output.mkdir()
    plots = report.make_plots(
        output, report.summarize(rows), report.paired_comparisons(rows, "diagnostic"), [], [], "diagnostic"
    )
    assert plots[0][0] == "success"
    assert (output / "success.png").stat().st_size > 1000
    assert (output / "success.svg").stat().st_size > 1000


def test_relocated_frozen_manifest_requires_matching_hash_and_preserves_raw_run(tmp_path):
    campaign = tmp_path / "campaign"
    manifest = campaign / "heldout_manifest.json"
    write(manifest, {"tasks": [{"key": "a", "benchmark": "textcraft"}]})
    path = campaign / "heldout_v2" / "diagnostic" / "run.json"
    run = {
        "config": {
            "manifest": "/nonexistent-original-machine/heldout_manifest.json",
            "variants": ["diagnostic"],
        },
        "manifest_sha256": report.sha256(manifest),
    }
    write(path, run)
    before = path.read_bytes()
    resolved, provenance = report.resolve_frozen_manifest(path, run)
    assert resolved == manifest
    assert provenance["manifest_resolution"] == "ancestor_basename_and_sha256"
    coverage, warnings, sources = report.expected_coverage({str(path): run}, [])
    assert len(coverage) == 1 and not warnings
    assert sources[0]["sha256"] == run["manifest_sha256"]
    assert path.read_bytes() == before


def test_manifest_basename_alone_never_authorizes_fallback(tmp_path):
    manifest = tmp_path / "heldout_manifest.json"
    write(manifest, {"tasks": [{"key": "wrong-task"}]})
    path = tmp_path / "nested" / "run.json"
    run = {
        "config": {"manifest": "/nonexistent-original-machine/heldout_manifest.json"},
        "manifest_sha256": "0" * 64,
    }
    write(path, run)
    with pytest.raises(FileNotFoundError, match="SHA-256"):
        report.resolve_frozen_manifest(path, run)
    run.pop("manifest_sha256")
    with pytest.raises(FileNotFoundError, match="manifest_sha256"):
        report.resolve_frozen_manifest(path, run)


def test_battery_folder_metadata_supplies_model_but_path_does_not_guess_split(tmp_path):
    root = tmp_path / "heldout" / "diagnostic"
    write(root / "run_metadata.json", {"model": "actual-model"})
    write(
        root / "manifest.json",
        {
            "cases": [
                {
                    "case_id": "one",
                    "source": "You have 1 apple.",
                    "provenance": {"benchmark": "fixture", "task_id": "task", "step": 1},
                }
            ]
        },
    )
    value = {"case_id": "one", "state_id": "one-condition", "diagnosis_correct": True, "condition": "clean"}
    (root / "records.jsonl").write_text(json.dumps(value) + "\n")
    rows, provenance, warnings = report.load_battery([root])
    assert not warnings
    assert rows[0]["battery_model"] == "actual-model"
    assert rows[0]["battery_split"] == "unknown"
    assert any(Path(item["path"]).name == "run_metadata.json" for item in provenance)


def test_transformed_battery_preserves_explicit_heldout_derived_scope(tmp_path):
    write(
        tmp_path / "results.json",
        {
            "split": "heldout-derived",
            "model": "fixture",
            "rows": [{"case_id": "one", "state_id": "one", "mode": "controlled", "after_correct": True}],
        },
    )
    rows, _, warnings = report.load_battery([tmp_path])
    assert not warnings
    assert rows[0]["battery_split"] == "heldout-derived"
    assert report.battery_split_metadata({"source_split": "heldout"}) == "heldout-derived"
    assert report.battery_split_metadata({"split": "test"}) != "heldout"
    assert report.battery_split_metadata({"task_id": "valid_unseen/some-task"}) == "unknown"
