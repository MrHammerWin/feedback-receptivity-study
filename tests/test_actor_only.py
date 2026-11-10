"""Verify actor identity and absence of hidden repairs without any model calls."""

import json
from copy import deepcopy

import pytest

from crmn_agent.ablations import DraftOnlyAgent
from crmn_agent.feedback_control import FeedbackAgent
from crmn_agent.providers import BudgetExceeded
from crmn_agent.schemas import Decision


class StopAfterDraft(RuntimeError):
    pass


class RecordingProvider:
    def __init__(self, action="inventory", *, failure=None):
        self.action = action
        self.failure = failure
        self.calls = []

    def complete(self, schema, purpose, payload, cap):
        self.calls.append((schema, purpose, deepcopy(payload), cap))
        if self.failure is not None:
            raise self.failure
        if len(self.calls) > 1:
            raise StopAfterDraft("The comparison needs only the feedback actor's first call")
        return schema(reflection="Use the current public observation.", action=self.action)


@pytest.mark.parametrize("history_turns", [0, 1, 100])
def test_initial_request_exactly_matches_feedback_actor_schema_purpose_payload_order_and_cap(history_turns):
    raw_provider, full_provider = RecordingProvider(), RecordingProvider()
    actor = DraftOnlyAgent(raw_provider, history_turns=history_turns)
    full = FeedbackAgent(full_provider, history_turns=history_turns)
    past = [{"observation": f"public past observation {i}", "action": f"command {i}"} for i in range(120)]
    actor.history = deepcopy(past)
    full.history = deepcopy(past)
    arguments = ("Current public observation: закрытый ящик.", "Inspect the box.",
                 ["inventory", "open box"], 120, "Shared public rules.")
    actor.act(*arguments)
    with pytest.raises(StopAfterDraft):
        full.act(*arguments)
    raw, initial = raw_provider.calls[0], full_provider.calls[0]
    assert raw[0] is initial[0] is Decision
    assert raw[1] == initial[1] and raw[3] == initial[3] == 256
    # Serialized equality checks insertion order as well as content and Unicode handling.
    assert json.dumps(raw[2]) == json.dumps(initial[2])
    assert list(raw[2]) == ["mission", "observation", "available_actions", "history", "background"]
    assert len(raw[2]["history"]) == min(120, history_turns)
    assert len(raw_provider.calls) == 1


@pytest.mark.parametrize("action", ["  unavailable command  ", "", "open unknown safe"])
def test_action_is_passed_exactly_to_native_environment_without_replacement(action):
    provider = RecordingProvider(action)
    agent = DraftOnlyAgent(provider)
    turn = agent.act("Public room observation.", "Find the key.", ["inventory"], 0)
    assert turn.action == action
    assert agent.history == [{"observation": "Public room observation.", "action": action}]
    assert turn.diagnostics["action_not_admissible"] is True
    assert len(provider.calls) == 1


def test_unconstrained_commands_do_not_trigger_an_oracle_action_choice():
    provider = RecordingProvider("craft 1 planks using 1 log")
    agent = DraftOnlyAgent(provider)
    turn = agent.act("Initial inventory is empty.", "Craft planks.", [], 0)
    assert turn.action == "craft 1 planks using 1 log"
    assert turn.diagnostics["action_not_admissible"] is False
    assert set(provider.calls[0][2]) == {
        "mission", "observation", "available_actions", "history", "background",
    }


def test_feedback_is_a_noop_and_cannot_inject_reward_or_hidden_labels_into_actor_context():
    provider = RecordingProvider()
    agent = DraftOnlyAgent(provider)
    state_before = deepcopy(agent.history)
    agent.feedback(False, False)
    agent.feedback(True, True)
    assert agent.history == state_before
    agent.act("Public observation.", "Goal.", ["inventory"], 0)
    assert "success" not in provider.calls[0][2]
    assert "last_feedback" not in provider.calls[0][2]
    assert "terminated" not in provider.calls[0][2]
    assert agent.history == [{"observation": "Public observation.", "action": "inventory"}]


def test_provider_failure_is_not_retried_or_recorded_as_an_executed_action():
    provider = RecordingProvider(failure=BudgetExceeded("Scripted budget boundary"))
    agent = DraftOnlyAgent(provider)
    with pytest.raises(BudgetExceeded):
        agent.act("Public observation.", "Goal.", ["inventory"], 0)
    assert len(provider.calls) == 1
    assert agent.history == []


def test_default_history_matches_predeclared_actor_ablation_and_invalid_window_is_rejected():
    assert DraftOnlyAgent(RecordingProvider()).history_turns == 100
    with pytest.raises(ValueError):
        DraftOnlyAgent(RecordingProvider(), history_turns=-1)
