"""Synthetic public numeric fixtures; no experimental results or model calls."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_scored_data.py"
SPEC = importlib.util.spec_from_file_location("verify_scored_data", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
KERNEL_SPEC = importlib.util.spec_from_file_location(
    "fixture_paired_statistics", SCRIPT.parent.parent / "analysis" / "paired_statistics.py"
)
kernel = importlib.util.module_from_spec(KERNEL_SPEC)
KERNEL_SPEC.loader.exec_module(kernel)


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row)) or ["no_data"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path):
    base = {
        "phase": "main",
        "cohort": "main-synthetic",
        "model": "fixture-model",
        "source_identity": "fixture-source",
        "repeat_id": 0,
        "reasoning_effort": "low",
    }
    qa = []
    for mode, outcomes in (("controlled", (True, False, False)), ("grounding_only", (False, True, False))):
        for index, after in enumerate(outcomes):
            qa.append(
                {
                    **base,
                    "question_id": f"fixture-q{index}",
                    "condition": "false_advice",
                    "mode": mode,
                    "mechanism_eligible": True,
                    "finished": True,
                    "transport_contaminated": False,
                    "status": "completed",
                    "before_correct": index < 2,
                    "after_correct": after,
                    "shared_draft_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                    "shared_critique_sha256": hashlib.sha256(f"advice-{index}".encode()).hexdigest(),
                }
            )
    qa_summary = [
        {
            **base,
            "mode": mode,
            "condition": "false_advice",
            "n": 3,
            "n_questions": 3,
            "n_records": 3,
            "successes": 1,
            "initially_correct_n": 2,
            "retained_n": 1,
            "initially_wrong_n": 1,
            "repaired_n": 0,
            "accuracy": 1 / 3,
            "retention": 0.5,
            "repair": 0.0,
            "interval95": kernel.clopper_pearson(1, 3),
            "retention_interval95": kernel.clopper_pearson(1, 2),
            "repair_interval95": kernel.clopper_pearson(0, 1),
        }
        for mode in ("controlled", "grounding_only")
    ]
    bound = 2 * math.sqrt(0.9875) - 1
    pair = {
        **base,
        "condition": "false_advice",
        "endpoint": "retention",
        "a": "controlled",
        "b": "grounding_only",
        "comparison_kind": "primary",
        "n": 2,
        "a_only": 1,
        "b_only": 1,
        "difference": 0.0,
        "interval95": [-bound, bound],
        "mcnemar_exact_two_sided_p": 1.0,
    }
    native_group = {
        "phase": base["phase"],
        "cohort": base["cohort"],
        "model": base["model"],
        "benchmark": "fixture-benchmark",
        "source_sha256": "fixture-source",
        "protocol_id": "fixture-protocol",
        "variant": "raw_actor",
        "repeat_id": 0,
        "reasoning_effort": "low",
        "configured_max_calls": 64,
    }
    native, native_summary = [], []
    for cutoff, outcomes in (
        (16, (True, False, False)),
        (64, (True, False, None)),
        (128, (True, None, None)),
    ):
        requested = cutoff <= 64
        for index, success in enumerate(outcomes):
            native.append(
                {
                    **native_group,
                    "key": f"fixture-task-{index}",
                    "task_id": f"fixture-task-{index}",
                    "seed": index,
                    "cutoff": cutoff,
                    "observed": success is not None,
                    "comparative_observed": requested and success is not None,
                    "within_requested_budget": requested,
                    "success": success,
                }
            )
        n = sum(v is not None for v in outcomes) if requested else 0
        native_summary.append(
            {
                **native_group,
                "cutoff": cutoff,
                "n_total": 3,
                "n_started": 3,
                "n_observed": n,
                "n_success": 1 if requested else 0,
                "n_missing": 3 - n,
                "success_rate": 1 / n if n else None,
                "interval95": kernel.clopper_pearson(1, n) if n else [None, None],
            }
        )
    accuracy_pair = {
        **base,
        "condition": "false_advice",
        "endpoint": "accuracy",
        "a": "controlled",
        "b": "grounding_only",
        "comparison_kind": "exploratory",
        **kernel.paired_binary([True, False, False], [False, True, False]),
    }
    stats = {
        "qa_summary": qa_summary,
        "qa_pairs": [pair, accuracy_pair],
        "native_summary": native_summary,
        "native_pairs": [],
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    return qa, native, stats


def save(root, qa, native, stats):
    write_csv(root / "qa_scored_numeric.csv", qa)
    write_csv(root / "native_scored_numeric.csv", native)
    (root / "aggregated_stats.json").write_text(json.dumps(stats))


def test_rebuilds_primary_and_respects_not_requested_native_success(tmp_path):
    qa, native, stats = fixture(tmp_path)
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert result["passed"], result["errors"]
    assert result["primary_pair_groups_checked"] == 1
    assert result["checks"] > 40
    assert not result["warnings"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("n", 3),
        ("a_only", 2),
        ("b_only", 0),
        ("difference", 0.5),
        ("interval95", [0, 0]),
        ("mcnemar_exact_two_sided_p", 0.001),
    ],
)
def test_detects_tampered_primary_statistics(tmp_path, field, value):
    qa, native, stats = fixture(tmp_path)
    stats["qa_pairs"][0][field] = value
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert not result["passed"]
    assert any("paired retention " + field in e for e in result["errors"])


@pytest.mark.parametrize("conflicting", [False, True])
def test_rejects_duplicate_or_conflicting_qa_units(tmp_path, conflicting):
    qa, native, stats = fixture(tmp_path)
    qa.append({**qa[0], "after_correct": False} if conflicting else dict(qa[0]))
    save(tmp_path, qa, native, stats)
    assert "duplicate" in " ".join(verifier.verify(tmp_path)["errors"]).lower()


def test_blank_eligible_outcome_is_missing_not_false(tmp_path):
    qa, native, stats = fixture(tmp_path)
    qa[0]["after_correct"] = None
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert not result["passed"]
    assert "missing" in " ".join(result["errors"]).lower()


def test_missing_native_result_cannot_be_scored_as_failure(tmp_path):
    qa, native, stats = fixture(tmp_path)
    row = next(r for r in native if r["cutoff"] == 64 and r["success"] is None)
    row["comparative_observed"] = True
    stats["native_summary"][1].update(n_observed=3, n_missing=0, success_rate=1 / 3)
    save(tmp_path, qa, native, stats)
    assert not verifier.verify(tmp_path)["passed"]


def test_known_requested_native_failure_cannot_be_dropped(tmp_path):
    qa, native, stats = fixture(tmp_path)
    row = next(r for r in native if r["cutoff"] == 64 and r["success"] is False)
    row["comparative_observed"] = False
    stats["native_summary"][1].update(n_observed=1, n_missing=2, success_rate=1.0)
    save(tmp_path, qa, native, stats)
    assert not verifier.verify(tmp_path)["passed"]


def test_multiple_models_are_checked_separately_and_pooling_fails(tmp_path):
    qa, native, stats = fixture(tmp_path)
    qa.extend({**row, "model": "second-fixture-model"} for row in list(qa))
    stats["qa_summary"].extend({**row, "model": "second-fixture-model"} for row in list(stats["qa_summary"]))
    stats["qa_pairs"].extend({**row, "model": "second-fixture-model"} for row in list(stats["qa_pairs"]))
    save(tmp_path, qa, native, stats)
    assert verifier.verify(tmp_path)["passed"]
    stats["qa_summary"] = stats["qa_summary"][:2]
    stats["qa_summary"][0]["n"] = 6
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert not result["passed"]
    assert any("model/source" in e for e in result["errors"])


@pytest.mark.parametrize("field", ["shared_draft_sha256", "shared_critique_sha256"])
def test_optional_hashes_bind_the_frozen_primary_pair(tmp_path, field):
    qa, native, stats = fixture(tmp_path)
    qa[3][field] = "different-fixture-digest"
    save(tmp_path, qa, native, stats)
    assert not verifier.verify(tmp_path)["passed"]


def test_optional_cost_sums_preserve_partial_and_unknown_telemetry(tmp_path):
    qa, native, stats = fixture(tmp_path)
    for i, row in enumerate(qa[:3]):
        row.update(
            cost_logical_calls=(1, 2, None)[i],
            cost_input_tokens_lower_bound=(100, None, None)[i],
            cost_complete_token_telemetry=False,
            cost_transport_unknown=i == 1,
        )
    stats["qa_summary"][0].update(
        logical_calls_total=3,
        input_tokens_lower_bound_total=100,
        n_logical_calls_known=2,
        n_complete_token_cost=0,
        n_transport_unknown=1,
    )
    save(tmp_path, qa, native, stats)
    assert verifier.verify(tmp_path)["passed"]
    stats["qa_summary"][0]["input_tokens_lower_bound_total"] = 0
    save(tmp_path, qa, native, stats)
    assert not verifier.verify(tmp_path)["passed"]


def test_empty_sources_are_explicit_and_cli_never_rewrites_inputs(tmp_path, capsys):
    stats = {"qa_summary": [], "qa_pairs": [], "native_summary": [], "native_pairs": []}
    save(tmp_path, [], [], stats)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert verifier.main(["--source-data", str(tmp_path)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert verifier.verify(tmp_path)["warnings"]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_missing_file_or_grouping_field_fails_clearly(tmp_path):
    assert not verifier.verify(tmp_path)["passed"]
    qa, native, stats = fixture(tmp_path)
    qa[0]["model"] = ""
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert not result["passed"]
    assert "pooling" in " ".join(result["errors"])


def choice_rows(rows):
    for row in rows:
        row.update(
            n_choices=4,
            reference_choice_index=0,
            advice_choice_index=2,
            initial_choice_index=0 if row["before_correct"] else 1,
            final_choice_index=0 if row["after_correct"] else 1,
            initial_response_present=True,
            final_response_present=True,
            question_sha256=hashlib.sha256(row["question_id"].encode()).hexdigest(),
            row_sha256=hashlib.sha256(("row:" + row["question_id"]).encode()).hexdigest(),
        )
    return rows


def references(rows):
    return {
        r["question_id"]: {
            "reference_choice_index": 0,
            "alternative_choice_index": 2,
            "n_choices": 4,
            "question_sha256": r["question_sha256"],
            "row_sha256": r["row_sha256"],
            "source_dataset": r.get("source_dataset"),
        }
        for r in rows
    }


def test_numeric_choices_rescore_all_flags_without_raw_questions(tmp_path):
    qa, native, stats = fixture(tmp_path)
    choice_rows(qa)
    save(tmp_path, qa, native, stats)
    report = verifier.verify(tmp_path)
    assert report["passed"], report["errors"]
    assert report["choice_index_rows_checked"] == len(qa)
    assert report["upstream_qa_reference_file_verified"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_choice_index", 1),
        ("final_choice_index", 1),
        ("reference_choice_index", 1),
        ("advice_choice_index", 0),
        ("final_choice_index", 4),
    ],
)
def test_numeric_choice_or_advice_tampering_fails(tmp_path, field, value):
    qa, native, stats = fixture(tmp_path)
    choice_rows(qa)
    qa[0][field] = value
    save(tmp_path, qa, native, stats)
    assert not verifier.verify(tmp_path)["passed"]


def test_downloaded_pinned_reference_binds_id_hash_target_and_advice(tmp_path, monkeypatch):
    qa, native, stats = fixture(tmp_path)
    choice_rows(qa)
    reference = references(qa)
    monkeypatch.setattr(verifier, "load_qa_reference_data", lambda path: reference)
    save(tmp_path, qa, native, stats)
    report = verifier.verify(tmp_path, qa_data=tmp_path / "synthetic-loader-fixture.parquet")
    assert report["passed"] and report["upstream_qa_reference_file_verified"]
    qa[0]["question_sha256"] = "f" * 64
    save(tmp_path, qa, native, stats)
    report = verifier.verify(tmp_path, qa_data=tmp_path / "synthetic-loader-fixture.parquet")
    assert not report["passed"]
    assert any("upstream question_sha256" in e for e in report["errors"])


@pytest.mark.parametrize(
    "status,before,after,present",
    [
        ("model_error", None, False, False),
        ("completed", None, False, True),
        ("infrastructure_censored", None, None, False),
        ("infrastructure_censored", False, None, True),
    ],
)
def test_null_indices_do_not_conflate_invalid_responses_and_censoring(status, before, after, present):
    row = {
        "question_id": "fixture",
        "condition": "false_advice",
        "n_choices": 4,
        "reference_choice_index": 0,
        "advice_choice_index": 2,
        "initial_choice_index": None,
        "final_choice_index": None,
        "initial_response_present": present,
        "final_response_present": False,
        "before_correct": before,
        "after_correct": after,
        "finished": True,
        "status": status,
    }
    checks = verifier.Checks()
    assert verifier.verify_choice_indices([row], checks) == 1
    assert not checks.errors


def test_actual_qa_file_requires_exact_pin_before_optional_parquet_import(tmp_path):
    path = tmp_path / "not-the-pinned-dataset.parquet"
    path.write_bytes(b"synthetic invalid file")
    with pytest.raises(verifier.InvalidSourceData, match="fixed source SHA256"):
        verifier.load_qa_reference_data(path)


def full_native_fixture(tmp_path):
    qa, _, stats = fixture(tmp_path)
    rows, summaries, pairs = [], [], []
    for benchmark, state_losses, refl_losses in (("alfworld", 6, 5), ("textcraft", 7, 8)):
        base = {
            "phase": "main",
            "cohort": "main-synthetic",
            "benchmark": benchmark,
            "model": "fixture-model",
            "source_sha256": "fixture-source",
            "protocol_id": "fixture-" + benchmark,
            "repeat_id": 0,
            "reasoning_effort": "low",
            "configured_max_calls": 64,
        }
        outcomes = {
            "feedback": [True] * 8,
            "raw_actor": [True] * 8,
            "grounding_only": [False] + [True] * 7,
            "stateact": [i >= state_losses for i in range(8)],
            "reflact_paper": [i >= refl_losses for i in range(8)],
        }
        for cutoff in (16, 32, 64, 128):
            requested = cutoff <= 64
            for method, values in outcomes.items():
                for i, success in enumerate(values):
                    rows.append(
                        {
                            **base,
                            "variant": method,
                            "cutoff": cutoff,
                            "key": f"{benchmark}-t{i}",
                            "task_id": f"{benchmark}-t{i}",
                            "seed": i,
                            "observed": True,
                            "comparative_observed": requested,
                            "within_requested_budget": requested,
                            "success": success,
                        }
                    )
                n, k = (8, sum(values)) if requested else (0, 0)
                summaries.append(
                    {
                        **base,
                        "variant": method,
                        "cutoff": cutoff,
                        "n_total": 8,
                        "n_started": 8,
                        "n_observed": n,
                        "n_success": k,
                        "n_missing": 8 - n,
                        "success_rate": k / n if n else None,
                        "interval95": kernel.clopper_pearson(k, n),
                    }
                )
            if requested:
                for other in ("grounding_only", "raw_actor", "stateact", "reflact_paper"):
                    # Mirror the published schema: effort/cap are encoded by
                    # protocol ID, not duplicated in each native pair record.
                    pair_base = {k: base[k] for k in verifier.NATIVE_PAIR_GROUP}
                    pairs.append(
                        {
                            **pair_base,
                            "cutoff": cutoff,
                            "a": "feedback",
                            "b": other,
                            "comparison_kind": "secondary_published"
                            if other in {"stateact", "reflact_paper"}
                            else "ablation",
                            **kernel.paired_binary(outcomes["feedback"], outcomes[other]),
                        }
                    )
    family = [p for p in pairs if p["cutoff"] == 64 and p["comparison_kind"] == "secondary_published"]
    previous = 0.0
    for rank, row in enumerate(sorted(family, key=lambda p: p["mcnemar_exact_two_sided_p"])):
        previous = max(previous, min(1.0, (4 - rank) * row["mcnemar_exact_two_sided_p"]))
        row["p_holm_secondary64"] = previous
    stats.update(native_summary=summaries, native_pairs=pairs)
    return qa, rows, stats


@pytest.mark.parametrize(
    "section,field",
    [
        ("qa_summary", "interval95"),
        ("qa_summary", "retention_interval95"),
        ("qa_summary", "repair_interval95"),
        ("native_summary", "interval95"),
    ],
)
def test_wrong_marginal_interval_cannot_pass(tmp_path, section, field):
    qa, native, stats = fixture(tmp_path)
    stats[section][0][field] = [-2, 2]
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert not result["passed"]
    assert any(field in error for error in result["errors"])


def test_all_native_pairs_and_nontrivial_four_test_holm_family_are_checked(tmp_path):
    qa, native, stats = full_native_fixture(tmp_path)
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert result["passed"], result["errors"]
    assert result["native_pair_groups_checked"] == 24
    assert result["native_holm_families"][0]["complete"]
    adjusted = sorted(p["p_holm_secondary64"] for p in stats["native_pairs"] if "p_holm_secondary64" in p)
    assert adjusted == [0.03125, 0.046875, 0.0625, 0.0625]


@pytest.mark.parametrize(
    "field,value",
    [
        ("n", 7),
        ("a_only", 0),
        ("b_only", 1),
        ("difference", -0.5),
        ("interval95", [0, 0]),
        ("mcnemar_exact_two_sided_p", 0.99),
        ("p_holm_secondary64", 0.123),
    ],
)
def test_native_pair_and_holm_mutations_fail(tmp_path, field, value):
    qa, native, stats = full_native_fixture(tmp_path)
    row = next(p for p in stats["native_pairs"] if p["cutoff"] == 64 and p["b"] == "stateact")
    row[field] = value
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert not result["passed"]
    assert any("native" in e for e in result["errors"])


def test_missing_holm_field_and_wholly_dropped_contrast_cannot_shrink_planned_family(tmp_path):
    qa, native, stats = full_native_fixture(tmp_path)
    pair = next(p for p in stats["native_pairs"] if p["cutoff"] == 64 and p["b"] == "stateact")
    del pair["p_holm_secondary64"]
    save(tmp_path, qa, native, stats)
    assert not verifier.verify(tmp_path)["passed"]
    qa, native, stats = full_native_fixture(tmp_path)
    native = [r for r in native if not (r["benchmark"] == "alfworld" and r["variant"] == "stateact")]
    stats["native_summary"] = [
        r
        for r in stats["native_summary"]
        if not (r["benchmark"] == "alfworld" and r["variant"] == "stateact")
    ]
    stats["native_pairs"] = [
        r for r in stats["native_pairs"] if not (r["benchmark"] == "alfworld" and r["b"] == "stateact")
    ]
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert not result["passed"]
    assert any("cannot verify planned4" in e for e in result["errors"])


def test_inferred_128_success_cannot_become_a_pair_and_protocol_settings_cannot_pool(tmp_path):
    qa, native, stats = full_native_fixture(tmp_path)
    stats["native_pairs"].append({**stats["native_pairs"][0], "cutoff": 128})
    save(tmp_path, qa, native, stats)
    assert not verifier.verify(tmp_path)["passed"]
    qa, native, stats = full_native_fixture(tmp_path)
    native.append({**native[0], "reasoning_effort": "high"})
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert not result["passed"]
    assert any("Ambiguous native protocol" in e for e in result["errors"])


def test_nonprimary_qa_pair_and_missing_marginal_interval_are_not_skipped(tmp_path):
    qa, native, stats = fixture(tmp_path)
    stats["qa_pairs"][1]["mcnemar_exact_two_sided_p"] = 0.001
    del stats["qa_summary"][0]["retention_interval95"]
    save(tmp_path, qa, native, stats)
    result = verifier.verify(tmp_path)
    assert not result["passed"]
    assert any("paired accuracy" in e for e in result["errors"])
    assert any("retention_interval95" in e for e in result["errors"])
