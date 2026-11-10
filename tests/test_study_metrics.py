"""Offline analysis fixtures, never research model outputs."""

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import build_metrics as metrics
import paired_statistics as stats


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def qa_row(
    question="q1",
    condition="false_advice",
    mode="controlled",
    before=True,
    after=True,
    **extra,
):
    return {
        "record_id": f"{question}:{condition}:{mode}",
        "question_id": question,
        "source_dataset": "fixture-topic",
        "condition": condition,
        "mode": mode,
        "study_split": "test",
        "model": "fixture-model",
        "source_identity": "frozen",
        "repeat_id": 0,
        "attempt": 1,
        "status": "completed",
        "finished": True,
        "before_correct": before,
        "after_correct": after,
        "draft": {"reflection": "fixture", "action": "A"},
        "final": {"reflection": "fixture", "action": "A" if after else "B"},
        "diagnostics": {
            "intervention": "critique",
            "profile": {
                "observed_pairs": 1,
                "beneficial_correction": 1,
                "harmful_corruption": 0,
            },
        },
        "usage": {"calls": 1, "input_tokens": 100, "output_tokens": 10, "seconds": 1},
        "total_policy_usage": {
            "calls": 2,
            "input_tokens": 200,
            "output_tokens": 20,
            "seconds": 2,
        },
        "new_launch_ids": [],
        "reused_launch_ids": [],
        "request_sha256": [],
        **extra,
    }


def qa_file(root, rows, logs=()):
    path = root / "main-v2" / "qa"
    write(path / "run_metadata.json", {"frozen": {"reasoning_effort": "low"}})
    (path / "outcomes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (path / "model_calls.jsonl").write_text(
        "\n".join(json.dumps(r) for r in logs) + "\n"
    )
    return path


def episode(calls, status, steps=(), success=False, **extra):
    return {
        "usage": {"calls": calls},
        "status": status,
        "trajectory": list(steps),
        "success": success,
        "finished": True,
        **extra,
    }


def test_cp_zero_and_all_success_and_empty_boundaries():
    assert stats.clopper_pearson(0, 0) == [None, None]
    low, high = stats.clopper_pearson(0, 10)
    assert low == 0 and high == pytest.approx(1 - 0.025**0.1)
    low, high = stats.clopper_pearson(10, 10)
    assert low == pytest.approx(0.025**0.1) and high == 1


def test_conservative_paired_interval_has_uncertainty_with_all_ties():
    result = stats.paired_binary([True] * 120, [True] * 120)
    assert result["difference"] == 0
    assert result["interval95"][0] < 0 < result["interval95"][1]
    assert result["mcnemar_exact_two_sided_p"] == 1
    assert stats.paired_binary([True] * 6, [False] * 6)[
        "mcnemar_exact_two_sided_p"
    ] == pytest.approx(0.03125)
    assert metrics.pair_stats([], [])["mcnemar_exact_two_sided_p"] is None


def test_cp_matches_scipy_beta_quantiles_if_available():
    beta = pytest.importorskip("scipy.stats").beta
    for n in [10, 120]:
        for k in [1, n // 2, n - 1]:
            lo, hi = stats.clopper_pearson(k, n)
            assert lo == pytest.approx(beta.ppf(0.025, k, n - k + 1), abs=1e-12)
            assert hi == pytest.approx(beta.ppf(0.975, k + 1, n - k), abs=1e-12)


def test_action_crossing_call_budget_does_not_count_early_success():
    row = episode(66, "success", [{"success": True, "end_logical_calls": 66}], True)
    assert metrics.native_success(row, 64) is False
    assert metrics.native_success(row, 128) is True


def test_censoring_is_per_cutoff_and_infrastructure_has_one_unobserved_call():
    assert metrics.native_success(episode(90, "campaign_budget_censored"), 64) is False
    assert metrics.native_success(episode(90, "campaign_budget_censored"), 128) is None
    assert metrics.native_success(episode(50, "campaign_budget_censored"), 64) is None
    assert metrics.native_success(episode(64, "infrastructure_censored"), 64) is None
    assert metrics.native_success(episode(64, "infrastructure_censored"), 32) is False
    assert metrics.native_success(episode(64, "campaign_budget_censored"), 64) is False


def test_terminal_model_error_is_failure_not_censored_and_running_prefix_can_be_known():
    assert metrics.native_success(episode(12, "model_error"), 64) is False
    assert metrics.native_success(episode(32, "episode_compute_cap"), 64) is None
    running = {
        "status": "running",
        "finished": False,
        "trajectory": [{"end_logical_calls": 32, "success": False}],
    }
    assert metrics.native_success(running, 16) is False
    assert metrics.native_success(running, 64) is None
    with pytest.raises(ValueError):
        metrics.native_success(running, 0)


def test_old_transport_model_error_not_scored_as_semantic_failure():
    row = episode(
        20,
        "model_error",
        [{"end_logical_calls": 12, "success": False}],
        error="Codex failed (exit1)",
    )
    assert metrics.native_success(row, 16, allow_legacy_transport=True) is None
    assert metrics.native_success(row, 8, allow_legacy_transport=True) is False
    assert metrics.native_success(row, 16) is False


def test_qa_two_conditions_do_not_double_count_initial_drafts(tmp_path):
    values = [
        qa_row(condition=c, mode=m)
        for c in metrics.CONDITIONS
        for m in metrics.QA_MODES
    ]
    path = qa_file(tmp_path, values)
    rows = metrics.load_qa([path], [], [])
    result, _ = metrics.qa_metrics(rows)
    assert result["qa_draft_summary"][0]["valid_initial_n"] == 1
    assert result["qa_draft_summary"][0]["initial_correct_n"] == 1
    primary = [r for r in result["qa_pairs"] if r["comparison_kind"] == "primary"]
    assert len(primary) == 1 and primary[0]["n"] == 1
    assert all(row["n"] == 1 for row in result["qa_summary"])


def test_qa_provider_outage_separate_from_mechanism_error(tmp_path):
    row = qa_row(
        status="model_error",
        after=False,
        new_launch_ids=[7],
        request_sha256=["private-prompt"],
    )
    log = {
        "prompt_sha256": "private-prompt",
        "subscription_launch": 7,
        "error_type": "BackendTransportError",
        "output": "",
        "backend_usage": None,
        "usage_delta": {"input_tokens": 0, "output_tokens": 0},
    }
    path = qa_file(tmp_path, [row], [log])
    rows = metrics.load_qa([path], [], [])
    result, _ = metrics.qa_metrics(rows)
    summary = result["qa_summary"][0]
    assert summary["n"] == 0 and summary["accuracy"] is None
    assert summary["n_transport_contaminated"] == 1 and summary["operational_n"] == 1


def test_empty_semantic_error_is_not_transport_in_v2(tmp_path):
    row = qa_row(status="model_error", after=False, new_launch_ids=[7])
    log = {
        "subscription_launch": 7,
        "error_type": "ModelOutputError",
        "output": "",
        "backend_usage": None,
        "usage_delta": {"input_tokens": 0, "output_tokens": 0},
    }
    assert not metrics.transport_log(log)
    assert metrics.transport_log(log, allow_legacy=True)
    path = qa_file(tmp_path, [row], [log])
    result, _ = metrics.qa_metrics(metrics.load_qa([path], [], []))
    assert result["qa_summary"][0]["n"] == 1
    assert result["qa_summary"][0]["n_transport_contaminated"] == 0


def test_legacy_empty_transport_heuristic_only_in_audited_v1_cohort(tmp_path):
    row = qa_row(status="model_error", after=False, new_launch_ids=[7])
    log = {"subscription_launch": 7, "error_type": "ModelOutputError", "output": ""}
    current = qa_file(tmp_path, [row], [log])
    legacy = current.parent.parent / "pilot" / "qa"
    legacy.parent.mkdir(parents=True)
    current.rename(legacy)
    result, _ = metrics.qa_metrics(metrics.load_qa([legacy], [], []))
    assert result["qa_summary"][0]["n"] == 0
    assert result["qa_summary"][0]["n_transport_contaminated"] == 1


def test_invalid_json_after_real_response_is_model_failure(tmp_path):
    row = qa_row(
        status="model_error",
        after=False,
        new_launch_ids=[7],
        request_sha256=["format-prompt"],
    )
    log = {
        "prompt_sha256": "format-prompt",
        "subscription_launch": 7,
        "error_type": "ModelOutputError",
        "output": "not-json",
        "backend_usage": {"input_tokens": 50, "output_tokens": 4},
    }
    path = qa_file(tmp_path, [row], [log])
    result, _ = metrics.qa_metrics(metrics.load_qa([path], [], []))
    assert result["qa_summary"][0]["n"] == 1
    assert result["qa_summary"][0]["model_error_n"] == 1
    assert result["qa_summary"][0]["accuracy"] == 0


def test_latest_finished_attempt_and_cross_cohort_separation(tmp_path):
    first = qa_row()
    path = qa_file(
        tmp_path,
        [
            first,
            {
                **first,
                "attempt": 2,
                "finished": False,
                "status": "campaign_budget_censored",
            },
        ],
    )
    rows = metrics.load_qa([path], [], [])
    assert len(rows) == 1 and rows[0]["attempt"] == 1 and rows[0]["_eligible"]
    assert metrics.cohort("results/pilot/qa/outcomes.jsonl") != metrics.cohort(
        "results/pilot-v2/qa/outcomes.jsonl"
    )


def test_repeated_terminal_answers_are_audited_not_cherry_picked(tmp_path):
    first = qa_row()
    path = qa_file(tmp_path, [first, {**first, "attempt": 2, "after_correct": False}])
    warnings = []
    rows = metrics.load_qa([path], warnings, [])
    assert warnings and not rows[0]["_eligible"]


def test_numeric_export_never_contains_raw_qa_or_evidence(tmp_path):
    row = qa_row()
    row["critique"] = {
        "issue": "PRIVATE_RETRIEVED_EVIDENCE",
        "evidence": "PRIVATE_QUESTION",
    }
    row["diagnostics"]["profile"].update(
        {
            "source_quote": "PRIVATE_SOURCE_QUOTE",
            "corrected_quote": "PRIVATE_SOURCE_QUOTE",
            "preserved_quote": "PRIVATE_SOURCE_QUOTE",
        }
    )
    path = qa_file(tmp_path, [row])
    out = tmp_path / "export"
    data = metrics.build([path], [], out)
    assert data["has_main_data"]
    for file in out.iterdir():
        content = file.read_text()
        assert (
            "PRIVATE_RETRIEVED_EVIDENCE" not in content
            and "PRIVATE_QUESTION" not in content
            and "PRIVATE_SOURCE_QUOTE" not in content
        )


def telemetry_record(
    calls=1, inputs=100, outputs=10, seconds=1, backend=True, cached=0
):
    return {
        "usage_delta": {
            "calls": calls,
            "input_tokens": inputs,
            "output_tokens": outputs,
            "seconds": seconds,
        },
        "backend_usage": {
            "input_tokens": inputs,
            "output_tokens": outputs,
            "cached_input_tokens": cached,
        }
        if backend
        else None,
    }


def test_partial_backend_tokens_do_not_erase_known_calls_and_time():
    records = [
        telemetry_record(cached=20),
        telemetry_record(inputs=0, outputs=0, seconds=2, backend=False),
    ]
    unit = metrics.cost_unit(
        {"calls": 2, "input_tokens": 100, "output_tokens": 10, "seconds": 3},
        records,
        transport=True,
    )
    result = metrics.aggregate_cost([unit])
    assert result["n_complete_token_cost"] == result["n_cost_known"] == 0
    assert result["n_logical_calls_known"] == result["n_inference_time_known"] == 1
    assert result["n_transport_unknown"] == 1
    assert result["logical_calls_total"] == 2
    assert result["inference_seconds_total"] == 3
    assert result["input_tokens_lower_bound_total"] == 100
    assert result["cached_input_tokens_lower_bound_total"] == 20
    assert not result["token_cost_complete"]


def test_missing_telemetry_remains_unknown_and_unique_segment_is_required():
    assert metrics.aggregate_cost([metrics.cost_unit({})])["input_tokens_total"] is None
    record = telemetry_record()
    assert metrics.usage_segment(record["usage_delta"], [record]) == [record]
    assert metrics.usage_segment(record["usage_delta"], [record, record]) is None
    complete = metrics.cost_unit(record["usage_delta"], [record])
    assert complete["complete_token_telemetry"] is True
    assert complete["cached_input_tokens_lower_bound"] == 0
    mismatched = {**record, "backend_usage": {"input_tokens": 999, "output_tokens": 10}}
    assert (
        metrics.cost_unit(record["usage_delta"], [mismatched])[
            "complete_token_telemetry"
        ]
        is False
    )


def test_qa_costs_include_transport_attempt_but_keep_backend_gap_visible(tmp_path):
    draft = {"calls": 1, "input_tokens": 50, "output_tokens": 5, "seconds": 1}
    completed = qa_row(
        shared_draft_usage=draft,
        usage={"calls": 1, "input_tokens": 100, "output_tokens": 10, "seconds": 2},
        total_policy_usage={
            "calls": 2,
            "input_tokens": 150,
            "output_tokens": 15,
            "seconds": 3,
        },
    )
    censored = qa_row(
        question="q2",
        status="infrastructure_censored",
        after_correct=None,
        shared_draft_usage=draft,
        usage={"calls": 1, "input_tokens": 0, "output_tokens": 0, "seconds": 4},
        total_policy_usage={
            "calls": 2,
            "input_tokens": 50,
            "output_tokens": 5,
            "seconds": 5,
        },
    )
    logs = [
        telemetry_record(inputs=50, outputs=5, seconds=1),
        telemetry_record(seconds=2, cached=20),
        telemetry_record(inputs=0, outputs=0, seconds=4, backend=False),
    ]
    path = qa_file(tmp_path, [completed, censored], logs)
    result, numeric = metrics.qa_metrics(metrics.load_qa([path], [], []))
    summary = result["qa_summary"][0]
    assert summary["n"] == 1 and summary["cost_n_started"] == 2
    assert summary["n_complete_token_cost"] == 1
    assert summary["n_transport_unknown"] == 1
    assert summary["logical_calls_total"] == 4
    assert summary["input_tokens_lower_bound_total"] == 200
    assert all("cost_inference_seconds" in r for r in numeric)


def test_repeated_crafting_feedback_is_not_named_a_semantic_loop(tmp_path):
    root = tmp_path / "main-v2" / "native" / "textcraft"
    turns = [
        {
            "observation": "Crafted bricks",
            "next_observation": "Crafted bricks",
            "action": "craft 1 bricks",
            "end_logical_calls": index + 1,
        }
        for index in range(3)
    ]
    value = episode(
        3,
        "action_horizon",
        turns,
        key="t1",
        benchmark="textcraft",
        variant="raw_actor",
        model="fixture",
        task_id="t1",
        source_sha256="source",
        seed=1,
    )
    value["usage"] = {
        "calls": 3,
        "input_tokens": 300,
        "output_tokens": 30,
        "seconds": 3,
    }
    write(root / "episodes" / "t1.json", value)
    (root / "model_calls.jsonl").write_text(
        "\n".join(json.dumps(telemetry_record()) for _ in range(3))
    )
    result, numeric = metrics.native_metrics([root], [], [])
    cost = result["native_costs"][0]
    assert cost["three_repeated_actions_same_feedback_text"] == 1
    assert "episodes_with_stagnant_three_repeat_loop" not in cost
    assert cost["n_complete_token_cost"] == 1
    assert result["native_episode_costs"][0]["cached_input_tokens_lower_bound"] == 0
    assert numeric[0]["cost_input_tokens_lower_bound"] == 300


def test_proxy_transfer_uses_unconditional_critique_not_gated_final(tmp_path):
    controlled = qa_row(mode="controlled", after=True)
    always = qa_row(mode="always_critique", after=False)
    path = qa_file(tmp_path, [controlled, always])
    result, _ = metrics.qa_metrics(metrics.load_qa([path], [], []))
    row = next(
        r
        for r in result["qa_proxy_transfer"]
        if r["predictor"] == "beneficial_flag" and r["flag"] is True
    )
    assert row["endpoint"] == "critique_corruption"
    assert row["n"] == row["events"] == 1
    assert row["rate"] == 1
    assert row["downstream_policy"] == "always_critique"


def test_cutoff_beyond_frozen_maximum_is_not_a_comparative_point(tmp_path):
    root = tmp_path / "main-v2" / "native" / "textcraft"
    manifest = root / "tasks.json"
    write(manifest, {"tasks": [{"key": "t1", "benchmark": "textcraft"}]})
    write(
        root / "run.json",
        {
            "config": {
                "manifest": str(manifest),
                "episode_call_cap": 64,
                "variants": ["raw_actor"],
            }
        },
    )
    value = episode(
        3,
        "success",
        [{"success": True, "end_logical_calls": 3}],
        True,
        key="t1",
        task_id="t1",
        benchmark="textcraft",
        variant="raw_actor",
        model="fixture",
        source_sha256="source",
        seed=1,
    )
    write(root / "episodes" / "t1.json", value)
    result, numeric = metrics.native_metrics([root], [], [])
    at64 = next(r for r in result["native_summary"] if r["cutoff"] == 64)
    at128 = next(r for r in result["native_summary"] if r["cutoff"] == 128)
    assert at64["within_requested_budget"] and at64["success_rate"] == 1
    assert at128["within_requested_budget"] is False
    assert at128["success_rate"] is None and at128["n_observed"] == 0
    assert at128["budget_point_status"] == "not_requested"
    inferred = next(r for r in numeric if r["cutoff"] == 128)
    assert inferred["observed"] is True and inferred["comparative_observed"] is False


def test_native_plan_preserves_all_twenty_unstarted_blocks(tmp_path):
    root = tmp_path / "main-v5" / "native"
    tasks = [
        {"key": f"{b}-{i}", "benchmark": b}
        for b, n in (("alfworld", 12), ("textcraft", 8))
        for i in range(n)
    ]
    plan = {
        "schema_version": "feedback-native-run-plan-v1",
        "tasks": tasks,
        "variants": [
            "feedback",
            "grounding_only",
            "raw_actor",
            "stateact",
            "reflact_paper",
        ],
        "model": "fixture",
        "source_sha256": "source",
        "reasoning_effort": "low",
        "horizons": {"alfworld": 100, "textcraft": 100},
        "episode_call_cap": 64,
        "environment_protocols": {"alfworld": "public-alf", "textcraft": "public-tc"},
    }
    write(root / "run_plan.json", plan)
    result, numeric = metrics.native_metrics([root], [], [])
    assert not numeric
    assert len(result["native_summary"]) == 2 * 5 * 4
    for row in result["native_summary"]:
        assert row["n_total"] == (12 if row["benchmark"] == "alfworld" else 8)
        assert row["n_started"] == 0 and row["success_rate"] is None
        assert row["reasoning_effort"] == "low"
    manifest = root / "tasks.json"
    write(manifest, {"tasks": tasks})
    run = root / "task0"
    write(
        run / "run.json",
        {
            "config": {
                "manifest": str(manifest),
                "task_keys": ["alfworld-0"],
                "variants": plan["variants"],
                "horizons": plan["horizons"],
                "episode_call_cap": 64,
                "provider": {"reasoning_effort": "low"},
            }
        },
    )
    value = episode(
        3,
        "success",
        [{"success": True, "end_logical_calls": 3}],
        True,
        key="alfworld-0",
        task_id="a0",
        benchmark="alfworld",
        variant="raw_actor",
        model="fixture",
        source_sha256="source",
        protocol="public-alf",
        seed=1,
    )
    write(run / "episodes" / "a0.json", value)
    result, _ = metrics.native_metrics([root], [], [])
    observed = [
        r
        for r in result["native_summary"]
        if r["benchmark"] == "alfworld"
        and r["variant"] == "raw_actor"
        and r["cutoff"] == 64
    ]
    assert len(observed) == 1
    assert observed[0]["n_total"] == 12 and observed[0]["n_started"] == 1
    assert observed[0]["n_missing"] == 11


def test_typographic_sensitivity_preserves_runtime_and_missing_labels():
    profile = {
        "source_quote": "A Patient's fever: 38.0°C",
        "corrected_quote": "a patient's FEVER: 38.0°c!",
        "preserved_quote": None,
        "beneficial_correction": 0,
        "harmful_corruption": None,
        "observed_pairs": 0,
    }
    original = json.dumps(profile, sort_keys=True)
    rows = metrics.quote_format_flags(profile)
    assert rows[0]["correction"] is False and rows[1]["correction"] is False
    assert rows[2]["correction"] is True
    assert all(r["corruption"] is None for r in rows)
    assert json.dumps(profile, sort_keys=True) == original
    result = metrics.summarize_format_units(rows, ())
    punct = next(
        r for r in result if r["normalization"] == "casefold_punctuation_removed"
    )
    assert punct["correction_runtime_disagreement_n"] == 1
    assert punct["corruption_missing_n"] == 1
    assert "source_quote" not in json.dumps(result)


def test_repeat_coverage_uses_bound_invocation_not_full_test_split(tmp_path):
    manifest = {"splits": {"test": [{"id": "q1"}, {"id": "q2"}]}}
    sha = hashlib.sha256(
        json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    manifest_path = tmp_path / "qa_manifest.json"
    write(manifest_path, manifest)
    path = qa_file(tmp_path, [qa_row(repeat_id=1, manifest_sha256=sha)])
    write(
        path / "run_metadata.json",
        {
            "frozen": {
                "model": "fixture-model",
                "source_identity": "frozen",
                "repeat_id": 1,
                "split": "test",
                "manifest_sha256": sha,
                "reasoning_effort": "low",
            }
        },
    )
    write(
        path / "invocation_selection.json",
        {
            "split": "test",
            "partition": 0,
            "parts": 1,
            "repeat": 1,
            "question_ids": ["q1"],
        },
    )
    data = metrics.build([path], [], tmp_path / "export", manifest_path)
    row = next(
        r
        for r in data["qa_coverage"]
        if r["mode"] == "controlled" and r["condition"] == "false_advice"
    )
    assert row["planned_full_split"] == 2
    assert row["planned_requested_questions"] == row["planned_denominator"] == 1
    assert row["missing_requested"] == 0
    assert row["scope"] == "bound invocation ID union"
    (path / "outcomes.jsonl").write_text("")
    empty = metrics.build([path], [], tmp_path / "export-empty", manifest_path)
    assert len(empty["qa_coverage"]) == 10
    assert all(r["missing_requested"] == 1 for r in empty["qa_coverage"])


def test_secondary_amendment_preserves_pre_frozen_primary_qa_metrics(tmp_path):
    snapshot = Path(metrics.__file__).with_name(
        "build_metrics.pre_coverage_format_v3.py.snapshot"
    )
    loader = importlib.machinery.SourceFileLoader(
        "pre_coverage_analysis", str(snapshot)
    )
    previous = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(loader.name, loader)
    )
    loader.exec_module(previous)
    values = [
        qa_row(
            question=q,
            condition=c,
            mode=m,
            before=q == "q1",
            after=(q == "q1" if m == "controlled" else q == "q2"),
        )
        for q in ("q1", "q2")
        for c in metrics.CONDITIONS
        for m in metrics.QA_MODES
    ]
    path = qa_file(tmp_path, values)
    rows = metrics.load_qa([path], [], [])
    current, _ = metrics.qa_metrics(rows)
    old, _ = previous.qa_metrics(rows)
    for name in (
        "qa_summary",
        "qa_pairs",
        "qa_proxy_transfer",
        "qa_profiles",
        "qa_coupling",
        "qa_draft_summary",
        "qa_routes",
    ):
        assert current[name] == old[name]
