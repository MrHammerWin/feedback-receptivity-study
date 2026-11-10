"""Transformed diagnostic tests are separate from official task-success tests."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from crmn_agent.feedback_battery import BatteryManifest, build_battery, run_battery
from crmn_agent.feedback_control import QuoteRevision
from crmn_agent.providers import BudgetExceeded


def write_episode(tmp_path, *, name="episode.json", observation="The box is closed.", success=True):
    path = tmp_path / name
    path.write_text(json.dumps({
        "benchmark": "fixtureworld", "task_id": "box-task", "seed": 17,
        "variant": "published-scaffold-fixture", "success": success,
        "trajectory": [{
            "step": 0, "observation": observation, "mission": "Open the box.",
            "available_actions": ["open box"], "action": "open box",
            "next_observation": "FUTURE_INFORMATION_MUST_NOT_LEAK", "reward": int(success),
            "hidden_gold": "EVALUATOR_ONLY_SECRET_MARKER",
        }],
    }))
    return path


class SourceFidelityProvider:
    """A scripted rule, not evidence about a model's actual capabilities."""

    def __init__(self, *, follow_critique=False, fail_call=None):
        self.calls = []
        self.follow_critique = follow_critique
        self.fail_call = fail_call

    def complete(self, schema, instruction, payload, max_tokens=None):
        self.calls.append(deepcopy(payload))
        if len(self.calls) == self.fail_call:
            raise BudgetExceeded("scripted exhausted authorization")
        if schema is QuoteRevision:
            return schema(revised_quote=payload["source_fragment"])
        if self.follow_critique and "admitted_critique" in payload:
            action = payload["admitted_critique"]["suggested_action"]
        else:
            action = next(candidate for candidate in payload["available_actions"]
                          if candidate in payload["observation"])
        return schema(reflection="Select the observed quotation.", action=action)


def test_builder_freezes_all_four_conditions_without_future_or_hidden_fields(tmp_path):
    episode = write_episode(tmp_path)
    manifest = build_battery([episode], max_states=1, steps=(0,))
    assert len(manifest.cases) == 4
    assert len({case.state_id for case in manifest.cases}) == 1
    assert {(case.evaluation_only.draft_condition, case.evaluation_only.feedback_condition)
            for case in manifest.cases} == {
        ("correct", "helpful"), ("correct", "harmful"), ("wrong", "helpful"), ("wrong", "harmful"),
    }
    assert all(case.available_actions == manifest.cases[0].available_actions for case in manifest.cases)
    frozen = manifest.model_dump_json()
    assert "FUTURE_INFORMATION_MUST_NOT_LEAK" not in frozen
    assert "EVALUATOR_ONLY_SECRET_MARKER" not in frozen
    assert "hidden_gold" not in frozen
    assert manifest.protocol == "transformed_source_fidelity_not_stock_benchmark"


def test_selecting_states_does_not_filter_on_episode_success(tmp_path):
    first = write_episode(tmp_path, name="first.json", success=False)
    second = write_episode(tmp_path, name="second.json", success=True)
    a = build_battery([first], max_states=1, steps=(0,))
    b = build_battery([second], max_states=1, steps=(0,))
    assert [case.state_id for case in a.cases] == [case.state_id for case in b.cases]
    combined = build_battery([first, second], max_states=2, steps=(0,))
    assert len(combined.cases) == 4  # Same public decision state is not counted twice.


def test_alternative_must_not_also_occur_in_public_observation(tmp_path):
    episode = write_episode(tmp_path, observation="The box is closed. The box is open.")
    manifest = build_battery([episode], max_states=1, steps=(0,))
    for case in manifest.cases:
        matches = [candidate for candidate in case.available_actions if candidate in case.observation]
        assert matches == [case.evaluation_only.expected_action]


def test_runner_uses_actual_gate_and_reports_cost_and_scorer_labels_separately(tmp_path):
    manifest = build_battery([write_episode(tmp_path)], max_states=1, steps=(0,))
    provider = SourceFidelityProvider()
    output = tmp_path / "battery_results.json"
    frozen_before = manifest.model_dump_json()
    report = run_battery(manifest, provider, output_path=output)
    assert report["status"] == "completed"
    assert len(report["rows"]) == 12
    assert len(provider.calls) == 20  # 4 * (3 controlled + 1 critique + 1 facts).
    assert report["usage_total"]["calls"] == 20
    assert manifest.model_dump_json() == frozen_before
    for mode, summary in report["summaries"].items():
        assert summary["n"] == 4 and summary["state_n"] == 1
        assert summary["accuracy"] == 1
        assert summary["bad_to_good"] == {"n": 2, "count": 2, "rate": 1.0}
        assert summary["good_to_bad"] == {"n": 2, "count": 0, "rate": 0.0}
        assert summary["usage"]["calls"] == (12 if mode == "controlled" else 4)
    assert json.loads(output.read_text())["status"] == "completed"
    assert len(output.with_suffix(".records.jsonl").read_text().splitlines()) == 12
    forbidden = {"evaluation_only", "expected_action", "draft_condition", "feedback_condition",
                 "case_id", "state_id", "source_episode_sha256"}

    def assert_no_labels(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                assert_no_labels(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_labels(child)

    for payload in provider.calls:
        assert_no_labels(payload)


def test_passing_local_controls_does_not_magically_prove_safe_task_revision(tmp_path):
    manifest = build_battery([write_episode(tmp_path)], max_states=1, steps=(0,))
    provider = SourceFidelityProvider(follow_critique=True)
    report = run_battery(manifest, provider)
    # Healthy copy-controls can coexist with harmful acceptance in the live revision.
    # The battery must record this transfer failure instead of editing the final action.
    controlled = report["summaries"]["controlled"]
    assert controlled["accuracy"] == 0.5
    assert controlled["bad_to_good"]["rate"] == 0.5
    assert controlled["good_to_bad"]["rate"] == 0.5
    for row in report["rows"]:
        if row["mode"] == "controlled":
            assert row["profile"]["beneficial_correction"] == 1
            assert row["profile"]["harmful_corruption"] == 0
    assert report["summaries"]["facts_only"]["accuracy"] == 1


def test_budget_interruption_keeps_attempted_calls_and_no_automatic_retry(tmp_path):
    manifest = build_battery([write_episode(tmp_path)], max_states=1, steps=(0,))
    provider = SourceFidelityProvider(fail_call=2)
    output = tmp_path / "interrupted.json"
    with pytest.raises(BudgetExceeded):
        run_battery(manifest, provider, output_path=output)
    checkpoint = json.loads(output.read_text())
    assert checkpoint["status"] == "interrupted_BudgetExceeded"
    assert checkpoint["usage_total"]["calls"] == 2
    assert checkpoint["rows"] == []
    assert len(provider.calls) == 2
    with pytest.raises(FileExistsError, match="rerun"):
        run_battery(manifest, provider, output_path=output)
    assert len(provider.calls) == 2


def test_frozen_data_is_not_overwritten_and_bad_labels_fail_before_calls(tmp_path):
    frozen = tmp_path / "frozen.json"
    manifest = build_battery([write_episode(tmp_path)], max_states=1, steps=(0,), output_path=frozen)
    with pytest.raises(FileExistsError):
        build_battery([write_episode(tmp_path)], max_states=1, steps=(0,), output_path=frozen)
    tampered = manifest.model_dump()
    tampered["cases"][0]["evaluation_only"]["expected_action"] = "an invented hidden answer"
    provider = SourceFidelityProvider()
    with pytest.raises(ValueError, match="public observation"):
        run_battery(tampered, provider)
    assert provider.calls == []
    tampered = manifest.model_dump()
    tampered["cases"][0]["hidden_gold"] = "secret"
    with pytest.raises(ValidationError):
        BatteryManifest.model_validate(tampered)


def test_incomplete_crossed_conditions_are_not_silently_selected_for_success(tmp_path):
    manifest = build_battery([write_episode(tmp_path)], max_states=1, steps=(0,))
    edited = manifest.model_dump()
    edited["cases"] = edited["cases"][:3]
    provider = SourceFidelityProvider()
    with pytest.raises(ValueError, match="all four"):
        run_battery(edited, provider)
    assert provider.calls == []


@pytest.mark.parametrize("kwargs", [{"max_states": 0}, {"steps": (-1,)}, {"steps": (0, 0)}])
def test_selection_policy_must_be_explicit_and_valid(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build_battery([write_episode(tmp_path)], **kwargs)


def test_empty_or_unanchored_states_cannot_generate_fake_measurement(tmp_path):
    with pytest.raises(ValueError, match="No selected public"):
        build_battery([write_episode(tmp_path, observation="")], max_states=1, steps=(0,))
