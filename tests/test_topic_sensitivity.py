"""Synthetic topic sensitivity fixtures only; no experimental data or model calls."""

import copy
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "analysis/topic_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("topic_sensitivity", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def pair(question, topic, a=True, b=False, **changes):
    common = {
        "phase": "main", "cohort": "synthetic", "model": "fixture-model",
        "source_identity": "fixture-source", "repeat_id": 0, "reasoning_effort": "low",
        "question_id": question, "source_dataset": topic, "condition": "false_advice",
        "mechanism_eligible": True, "finished": True, "transport_contaminated": False,
        "status": "completed", "before_correct": True,
        "shared_draft_sha256": "draft-" + question, "shared_critique_sha256": "advice-" + question,
        **changes,
    }
    return [{**common, "mode": mode, "after_correct": value}
            for mode, value in (("controlled", a), ("grounding_only", b))]


def summary(rows, expected_topics=2, **kwargs):
    return analysis.analyze(rows, expected_topics=expected_topics, **kwargs)["groups"][0]


def test_bootstrap_resamples_whole_unequal_clusters_without_resampling_questions():
    draws = analysis.cluster_draws({"large": [1, 1, 1], "small": [-1]}, replicates=1000)
    # AA, AB/BA, BB are the only possible topic-cluster draws. Individual-row
    # resampling can produce other values and would fail this support check.
    assert set(draws) == {-1, .5, 1}


def test_question_weighted_point_exact_reference_and_leave_topic_out():
    rows = sum([pair(f"q{i}", "large") for i in range(3)], []) + pair("q3", "small", False, True)
    result = summary(rows)
    assert result["primary_exact"] == analysis.PAIRED.paired_binary([True, True, True, False],
                                                                   [False, False, False, True])
    assert result["primary_exact"]["difference"] == .5
    assert result["topic_pair_n"] == {"large": 3, "small": 1}
    assert result["leave_one_topic_out"]["difference_range"] == [-1, 1]
    assert result["cluster_bootstrap"]["interval95"] == [-1, 1]
    assert result["cluster_bootstrap"]["computed_replicates"] == 10000


def test_wrong_initial_missing_policy_and_ineligible_are_excluded_separately():
    rows = pair("keep", "one") + pair("wrong", "two", before_correct=False)
    rows += pair("unpaired", "two")[:1]
    missing = pair("missing", "two", mechanism_eligible=False, finished=False,
                   status="campaign_budget_censored")
    missing[0]["after_correct"] = None
    rows += missing
    result = summary(rows)
    assert result["candidate_question_n"] == 4
    assert result["primary_exact"]["n"] == 1
    assert result["initially_wrong_pair_n"] == 1
    assert result["missing_policy_pair_n"] == 1
    assert result["ineligible_pair_n"] == 1
    assert result["topic_pair_n"] == {"one": 1, "two": 0}
    # Empty eligible clusters are still sampled; zero denominators are not
    # silently replaced by a redraw or dropped to claim a full interval.
    assert result["cluster_bootstrap"]["status"] == "undefined_zero_denominator_draw"
    assert result["cluster_bootstrap"]["zero_denominator_replicates"] > 0
    assert result["cluster_bootstrap"]["interval95"] == [None, None]


def test_conditions_and_policies_do_not_multiply_question_count():
    rows = pair("q1", "one") + pair("q2", "two")
    additions = []
    for row in rows:
        other = {**row, "condition": "correct_advice", "shared_critique_sha256": "other-" + row["question_id"]}
        additions.append(other)
        if row["mode"] == "controlled":
            additions.append({**other, "mode": "facts_only"})
    result = summary(rows + additions)
    assert result["candidate_question_n"] == 2
    assert result["primary_exact"]["n"] == 2


@pytest.mark.parametrize("field,value", [("model", "other-model"), ("source_identity", "other-source"),
                                         ("repeat_id", 1), ("reasoning_effort", "high"),
                                         ("phase", "pilot"), ("cohort", "other-cohort")])
def test_model_source_repeat_effort_and_phase_are_never_pooled(field, value):
    rows = pair("q1", "one") + pair("q2", "two")
    others = [{**r, field: value} for r in rows]
    results = analysis.analyze(rows + others, expected_topics=2)["groups"]
    assert len(results) == 2
    assert all(row["primary_exact"]["n"] == 2 for row in results)


def test_all_ties_bootstrap_zero_is_flagged_and_keeps_nonzero_exact_interval():
    rows = pair("q1", "one", True, True) + pair("q2", "two", False, False)
    result = summary(rows)
    assert result["cluster_bootstrap"]["interval95"] == [0, 0]
    assert result["cluster_bootstrap"]["degenerate"] is True
    low, high = result["primary_exact"]["interval95"]
    assert low < 0 < high
    assert any("not evidence of equivalence" in warning for warning in result["warnings"])


def test_seed_is_deterministic_and_input_order_does_not_change_bootstrap():
    rows = pair("q1", "one") + pair("q2", "two", False, True) + pair("q3", "three", True, True)
    first = summary(rows, expected_topics=3, seed=20261110, replicates=73)
    second = summary(list(reversed(rows)), expected_topics=3, seed=20261110, replicates=73)
    assert first == second
    assert analysis.cluster_draws({"a": [1], "b": [-1]}, 40, 1) != analysis.cluster_draws({"a": [1], "b": [-1]}, 40, 2)


@pytest.mark.parametrize("field,value", [("before_correct", False), ("source_dataset", "other-topic"),
                                         ("shared_draft_sha256", "other-draft"),
                                         ("shared_critique_sha256", "other-advice"),
                                         ("question_sha256", "other-question")])
def test_conflicting_shared_metadata_fails_closed(field, value):
    rows = pair("q1", "one")
    if field == "question_sha256":
        rows[0][field] = "question-digest"
    rows[1][field] = value
    with pytest.raises(analysis.InvalidSourceData, match="Conflicting shared question metadata"):
        summary(rows, expected_topics=1)


def test_identical_duplicate_cannot_increase_n():
    rows = pair("q1", "one")
    with pytest.raises(analysis.InvalidSourceData, match="Duplicate/conflicting"):
        summary(rows + [copy.deepcopy(rows[0])], expected_topics=1)


def test_missing_topics_and_zero_pairs_are_explicitly_undefined():
    result = summary(pair("q1", "one"), expected_topics=20)
    assert result["topic_n"] == 1
    assert result["cluster_bootstrap"]["status"] == "undefined_topic_coverage"
    assert result["cluster_bootstrap"]["interval95"] == [None, None]
    result = summary(pair("q1", "one", before_correct=False), expected_topics=1)
    assert result["primary_exact"]["n"] == 0
    assert result["primary_exact"]["difference"] is None
    assert result["primary_exact"]["mcnemar_exact_two_sided_p"] is None
    assert result["cluster_bootstrap"]["status"] == "undefined_no_eligible_pairs"


def test_absent_topic_does_not_remove_pair_from_exact_primary():
    result = summary(pair("q1", None), expected_topics=1)
    assert result["primary_exact"]["n"] == 1
    assert result["missing_topic_pair_n"] == 1
    assert result["cluster_bootstrap"]["status"] == "undefined_missing_topic"


def test_unknown_initial_correctness_is_not_imputed_from_other_policy():
    rows = pair("q1", "one")
    rows[0]["before_correct"] = None
    result = summary(rows, expected_topics=1)
    assert result["primary_exact"]["n"] == 0
    assert result["initial_correctness_missing_pair_n"] == 1


def test_absent_digests_are_disclosed_without_imputing_identity():
    rows = pair("q1", "one")
    for row in rows:
        row.pop("shared_draft_sha256")
        row.pop("shared_critique_sha256")
    result = summary(rows, expected_topics=1)
    assert result["primary_exact"]["n"] == 1
    assert result["draft_digest_verified_pair_n"] == 0
    assert result["advice_digest_verified_pair_n"] == 0
    assert any("digests are absent" in warning for warning in result["warnings"])


def test_eligible_row_with_missing_outcome_is_not_a_false_answer():
    rows = pair("q1", "one")
    rows[0]["after_correct"] = None
    with pytest.raises(analysis.InvalidSourceData, match="missing/censored"):
        summary(rows, expected_topics=1)


def test_empty_input_is_undefined_not_a_zero_effect():
    result = analysis.analyze([])
    assert result["status"] == "undefined_no_input_rows"
    assert result["groups"] == []


def test_cli_reads_numeric_csv_without_altering_source(tmp_path):
    source, output = tmp_path / "synthetic.csv", tmp_path / "report.json"
    rows = pair("q1", "one") + pair("q2", "two")
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = source.read_bytes()
    run = subprocess.run([sys.executable, str(SCRIPT), str(source), "--out", str(output),
                          "--expected-topics", "2", "--replicates", "101"],
                         text=True, capture_output=True, check=True)
    assert run.stdout == ""
    assert source.read_bytes() == before
    assert json.loads(output.read_text())["groups"][0]["primary_exact"]["n"] == 2


def test_cli_cannot_overwrite_numeric_source(tmp_path):
    source = tmp_path / "synthetic.csv"
    source.write_text("no_data\n")
    before = source.read_bytes()
    result = subprocess.run([sys.executable, str(SCRIPT), str(source), "--out", str(source)],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "cannot overwrite" in result.stderr
    assert source.read_bytes() == before


@pytest.mark.parametrize("kwargs", [{"replicates": 0}, {"seed": True}, {"expected_topics": 0}])
def test_invalid_settings_fail_even_for_empty_input(kwargs):
    with pytest.raises(ValueError):
        analysis.analyze([], **kwargs)
