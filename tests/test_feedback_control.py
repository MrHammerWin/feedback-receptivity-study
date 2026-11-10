"""Behavioral boundary tests: no real model calls or hidden task answers."""

from copy import deepcopy

import pytest

from crmn_agent.feedback_control import (
    FeedbackAgent,
    FeedbackCritique,
    QuoteRevision,
    ReceptivityProfile,
    canonical_quote,
    mutate_quote,
    select_intervention,
)
from crmn_agent.providers import BudgetExceeded, EpisodeBudgetExceeded, ModelOutputError
from crmn_agent.schemas import Decision


class ScriptedProvider:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def complete(self, schema, purpose, payload, max_tokens=None):
        self.calls.append((schema, purpose, deepcopy(payload), max_tokens))
        if not self.answers:
            raise AssertionError("Unexpected model call")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            answer = answer(payload)
        return schema.model_validate(answer)


OBSERVATION = "The box is closed. The key is on the table."
MISSION = "Open the box."
ACTIONS = ["take key", "open box"]


def draft(action="open box"):
    return {"reflection": "Consider the observed task.", "action": action}


def critique(*, revision=True, grounded=True):
    return {
        "needs_revision": revision,
        "issue": "Check the state of the box.",
        "evidence": "The box is closed." if grounded else "The hidden key is green.",
        "suggested_action": "take key",
    }


def preserve_source(payload):
    return {"revised_quote": payload["source_fragment"]}


def retain_saved(payload):
    return {"revised_quote": payload["saved_quote"]}


def follow_advice(payload):
    return {"revised_quote": payload["critique"].split("Replace it with: ", 1)[1]}


def scripted_agent(*probe_answers, final="take key", **kwargs):
    provider = ScriptedProvider(draft(), critique(), *probe_answers, draft(final))
    return FeedbackAgent(provider, **kwargs), provider


@pytest.mark.parametrize(
    ("correction", "corruption", "complete", "grounded", "expected"),
    [
        (1, 0, 1, True, "critique"),
        (1, 1, 1, True, "rebuild"),
        (0, 0, 1, True, "facts_only"),
        (1, 0, 1, False, "facts_only"),
        (None, 0, 0, True, "unresolved_rebuild"),
        (1, None, 0, True, "unresolved_rebuild"),
        (None, 0, 1, True, "unresolved_rebuild"),
    ],
)
def test_gate_separates_helpfulness_harm_and_missing_measurement(
    correction, corruption, complete, grounded, expected,
):
    profile = ReceptivityProfile(correction, corruption, complete, "source", "mutated")
    assert select_intervention(profile, grounded)[0] == expected


def test_healthy_controls_admit_critique_and_execute_only_final_action():
    agent, provider = scripted_agent(preserve_source, preserve_source)
    turn = agent.act(OBSERVATION, MISSION, ACTIONS, 0)
    assert turn.action == "take key"
    assert turn.diagnostics["intervention"] == "critique"
    assert turn.diagnostics["profile"]["beneficial_correction"] == 1
    assert turn.diagnostics["profile"]["harmful_corruption"] == 0
    assert len(provider.calls) == 5
    assert [call[0] for call in provider.calls] == [
        Decision, FeedbackCritique, QuoteRevision, QuoteRevision, Decision,
    ]
    assert "admitted_critique" in provider.calls[-1][2]
    assert agent.history == [{"observation": OBSERVATION, "action": "take key"}]
    # Both microtasks use identical instructions and keys, with no disclosed gold label.
    correction, preservation = provider.calls[2:4]
    assert correction[1] == preservation[1]
    assert correction[2].keys() == preservation[2].keys()
    assert set(correction[2]) == {
        "source_observation", "source_fragment", "saved_quote", "critique",
    }
    assert correction[2]["saved_quote"] != preservation[2]["saved_quote"]
    assert "source_fragment" not in provider.calls[-1][2]


def test_suggestible_model_rebuilds_without_critic_or_contaminated_draft():
    agent, provider = scripted_agent(follow_advice, follow_advice)
    turn = agent.act(OBSERVATION, MISSION, ACTIONS, 0)
    assert turn.diagnostics["profile"]["beneficial_correction"] == 1
    assert turn.diagnostics["profile"]["harmful_corruption"] == 1
    assert turn.diagnostics["intervention"] == "rebuild"
    final_payload = provider.calls[-1][2]
    assert set(final_payload) == {
        "mission", "observation", "available_actions", "history", "background",
    }
    assert agent.history == [{"observation": OBSERVATION, "action": "take key"}]


def test_stubborn_model_uses_observed_facts_without_passing_critic():
    agent, provider = scripted_agent(retain_saved, retain_saved)
    turn = agent.act(OBSERVATION, MISSION, ACTIONS, 0)
    assert turn.diagnostics["profile"]["beneficial_correction"] == 0
    assert turn.diagnostics["profile"]["harmful_corruption"] == 0
    assert turn.diagnostics["intervention"] == "facts_only"
    payload = provider.calls[-1][2]
    assert "draft" in payload and "observed_facts" in payload
    assert "admitted_critique" not in payload and "critique" not in payload
    assert all(fact in OBSERVATION for fact in payload["observed_facts"])


def test_good_controls_do_not_certify_ungrounded_critic():
    provider = ScriptedProvider(
        draft(), critique(grounded=False), preserve_source, preserve_source, draft("take key"),
    )
    turn = FeedbackAgent(provider).act(OBSERVATION, MISSION, ACTIONS, 0)
    assert turn.diagnostics["intervention"] == "facts_only"
    assert turn.diagnostics["critic_evidence_present"] is False
    assert "admitted_critique" not in provider.calls[-1][2]


def test_no_anchor_is_missing_measurement_and_uses_raw_context_only():
    provider = ScriptedProvider(draft(), critique(), draft("take key"))
    turn = FeedbackAgent(provider).act("", MISSION, ACTIONS, 0)
    assert len(provider.calls) == 3
    assert turn.diagnostics["intervention"] == "unresolved_rebuild"
    assert turn.diagnostics["profile"]["observed_pairs"] == 0
    assert turn.diagnostics["profile"]["errors"] == ("no_observed_anchor",)
    assert "draft" not in provider.calls[-1][2]


def test_approved_draft_has_no_probes_and_zero_reward_is_not_an_action_error():
    provider = ScriptedProvider(draft(), critique(revision=False))
    agent = FeedbackAgent(provider)
    agent.feedback(False, False)
    turn = agent.act(OBSERVATION, MISSION, ACTIONS, 0)
    assert turn.action == "open box"
    assert len(provider.calls) == 2
    assert turn.diagnostics["intervention"] == "keep_draft"
    assert turn.diagnostics["profile"] is None
    assert agent.last_feedback == {"task_success": False, "terminated": False}


def test_always_probe_policy_is_chosen_before_outcomes():
    provider = ScriptedProvider(
        draft(), critique(revision=False), preserve_source, preserve_source, draft("take key"),
    )
    turn = FeedbackAgent(provider, probe_policy="always").act(OBSERVATION, MISSION, ACTIONS, 0)
    assert len(provider.calls) == 5
    assert turn.diagnostics["probe_policy"] == "always"
    assert turn.diagnostics["intervention"] == "critique"


def test_invalid_probe_does_not_become_a_failed_or_successful_control():
    agent, provider = scripted_agent(ModelOutputError("invalid probe"), preserve_source)
    turn = agent.act(OBSERVATION, MISSION, ACTIONS, 0)
    assert len(provider.calls) == 5
    profile = turn.diagnostics["profile"]
    assert profile["beneficial_correction"] is None
    assert profile["harmful_corruption"] == 0
    assert profile["observed_pairs"] == 0
    assert profile["errors"] == ("correction:invalid_model_output",)
    assert turn.diagnostics["intervention"] == "unresolved_rebuild"


@pytest.mark.parametrize("exception", [BudgetExceeded("global cap"), EpisodeBudgetExceeded("episode cap")])
def test_budget_error_is_not_swallowed_or_followed_by_more_model_calls(exception):
    provider = ScriptedProvider(draft(), critique(), exception)
    agent = FeedbackAgent(provider)
    with pytest.raises(type(exception)):
        agent.act(OBSERVATION, MISSION, ACTIONS, 0)
    assert len(provider.calls) == 3
    assert agent.history == []


@pytest.mark.parametrize("command", ["unlock unknown safe", "", "  invalid command  "])
def test_illegal_final_action_returns_exact_command_for_native_feedback(command):
    agent, provider = scripted_agent(preserve_source, preserve_source, final=command)
    turn = agent.act(OBSERVATION, MISSION, ACTIONS, 0)
    assert turn.action == command
    assert turn.diagnostics["action_in_public_list"] is False
    assert len(provider.calls) == 5
    assert agent.history == [{"observation": OBSERVATION, "action": command}]


def test_illegal_draft_forces_rebuild_even_with_healthy_controls():
    provider = ScriptedProvider(
        draft("teleport"), critique(revision=False), preserve_source, preserve_source, draft("take key"),
    )
    turn = FeedbackAgent(provider).act(OBSERVATION, MISSION, ACTIONS, 0)
    assert turn.diagnostics["intervention"] == "rebuild"
    assert turn.diagnostics["gate_reason"] == "draft_action_unavailable"
    assert "draft" not in provider.calls[-1][2]


def test_odd_step_counterbalances_controls_without_reversing_measurements():
    agent, provider = scripted_agent(follow_advice, follow_advice)
    turn = agent.act(OBSERVATION, MISSION, ACTIONS, 1)
    profile = turn.diagnostics["profile"]
    assert provider.calls[2][2]["saved_quote"] == profile["source_quote"]
    assert provider.calls[3][2]["saved_quote"] != profile["source_quote"]
    assert profile["beneficial_correction"] == 1
    assert profile["harmful_corruption"] == 1


def test_profiles_are_not_reused_across_changed_public_contexts():
    provider = ScriptedProvider(
        draft(), critique(), preserve_source, preserve_source, draft("take key"),
        draft(), critique(), preserve_source, preserve_source, draft("open box"),
    )
    agent = FeedbackAgent(provider)
    first = agent.act(OBSERVATION, MISSION, ACTIONS, 0)
    changed = "The box is open. The key is in your hand."
    second = agent.act(changed, MISSION, ACTIONS, 0)
    assert len(provider.calls) == 10
    assert first.diagnostics["profile"]["source_quote"] != second.diagnostics["profile"]["source_quote"]
    assert provider.calls[7][2]["source_observation"] == changed
    assert len(agent.history) == 2
    assert all("source_fragment" not in item for item in agent.history)


def test_public_measurement_api_takes_no_environment_action_or_history():
    provider = ScriptedProvider(preserve_source, preserve_source)
    agent = FeedbackAgent(provider)
    profile = agent.measure_receptivity(OBSERVATION, MISSION)
    assert profile.beneficial_correction == 1
    assert profile.harmful_corruption == 0
    assert len(provider.calls) == 2
    assert agent.history == []


def test_frozen_proposal_api_reuses_the_gate_without_generating_new_critic():
    provider = ScriptedProvider(preserve_source, preserve_source, draft("take key"))
    agent = FeedbackAgent(provider, probe_policy="gated")
    turn = agent.revise_proposal(
        OBSERVATION, MISSION, ACTIONS, 0,
        Decision.model_validate(draft()), FeedbackCritique.model_validate(critique()),
    )
    assert [call[0] for call in provider.calls] == [QuoteRevision, QuoteRevision, Decision]
    assert turn.action == "take key"
    assert turn.diagnostics["intervention"] == "critique"
    assert agent.history == [{"observation": OBSERVATION, "action": "take key"}]


def test_no_probe_ablation_exposes_the_compute_difference():
    provider = ScriptedProvider(draft("take key"))
    agent = FeedbackAgent(provider, probe_policy="never", gate_mode="always_critique")
    turn = agent.revise_proposal(
        OBSERVATION, MISSION, ACTIONS, 0,
        Decision.model_validate(draft()), FeedbackCritique.model_validate(critique()),
    )
    assert len(provider.calls) == 1
    assert turn.diagnostics["profile"] is None
    assert turn.diagnostics["intervention"] == "critique"


@pytest.mark.parametrize("mode, expected", [
    ("always_critique", "critique"), ("always_rebuild", "rebuild"), ("facts_only", "facts_only"),
])
def test_ablation_route_is_preconfigured_not_selected_from_probe_score(mode, expected):
    agent, provider = scripted_agent(follow_advice, follow_advice, gate_mode=mode, probe_policy="always")
    turn = agent.act(OBSERVATION, MISSION, ACTIONS, 0)
    assert turn.diagnostics["intervention"] == expected
    assert turn.diagnostics["gate_reason"] == "preconfigured_ablation_override"
    assert len(provider.calls) == 5


@pytest.mark.parametrize("kwargs", [
    {"probe_policy": "after_test"}, {"gate_mode": "oracle"}, {"history_turns": -1},
    {"probe_policy": "never", "gate_mode": "controlled"},
])
def test_runtime_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        FeedbackAgent(ScriptedProvider(), **kwargs)


def test_quote_controls_test_literal_fidelity_not_semantic_truth():
    assert mutate_quote("The box is closed.") == "The box is open."
    assert mutate_quote("leftover apples") == "NOT: leftover apples"
    assert canonical_quote("The box\n is  closed.") == "The box is closed."
    assert canonical_quote("The box is closed.") != canonical_quote("The container is closed.")
