"""Invalid commands must reach native feedback without replacement or episode abort."""

from collections import deque

import pytest

from crmn_agent.feedback_control import FeedbackAgent, FeedbackCritique
from crmn_agent.schemas import Decision


class ScriptedProvider:
    def __init__(self, values):
        self.values = deque(values)
        self.calls = []

    def complete(self, schema, instruction, payload, max_tokens=None):
        self.calls.append((instruction, payload))
        return schema.model_validate(self.values.popleft())


class NativeFeedbackEnv:
    """A unit-test stand-in: the environment, rather than the agent, rejects syntax."""
    def __init__(self):
        self.commands = []

    def step(self, command):
        self.commands.append(command)
        return "Nothing happens." if command != "look" else "You look around."


def quiet_critique():
    return FeedbackCritique(needs_revision=False, issue="", evidence="", suggested_action="")


def test_native_invalid_feedback_enters_history_and_next_decision_without_oracle_repair():
    provider = ScriptedProvider([
        Decision(reflection="Draft", action="look"), quiet_critique(),
        Decision(reflection="Try a command", action="go sideways"),
        Decision(reflection="Draft", action="look"), quiet_critique(),
        Decision(reflection="Use the feedback", action="look"),
    ])
    agent = FeedbackAgent(provider, probe_policy="never", gate_mode="always_rebuild")
    env = NativeFeedbackEnv()
    first = agent.act("You are indoors.", "Explore", ["look"], 0)
    observed = env.step(first.action)
    agent.feedback(False, False)
    second = agent.act(observed, "Explore", ["look"], 1)
    env.step(second.action)
    assert env.commands == ["go sideways", "look"]
    assert first.diagnostics["action_in_public_list"] is False
    assert second.diagnostics["action_in_public_list"] is True
    assert provider.calls[-1][1]["observation"] == "Nothing happens."
    assert provider.calls[-1][1]["history"] == [
        {"observation": "You are indoors.", "action": "go sideways"},
    ]
    assert len(agent.history) == 2


@pytest.mark.parametrize("command", ["unknown command", ""])
def test_no_public_action_list_is_explicitly_unassessed(command):
    provider = ScriptedProvider([
        Decision(reflection="Draft", action="look"), quiet_critique(),
        Decision(reflection="Try", action=command),
    ])
    turn = FeedbackAgent(provider, probe_policy="never", gate_mode="always_rebuild").act(
        "You are indoors.", "Explore", [], 0,
    )
    assert turn.action == command
    assert turn.diagnostics["action_in_public_list"] is None
