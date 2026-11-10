"""Synthetic control-flow demo, not an LLM run or an empirical benchmark result."""

from crmn_agent.feedback_control import FeedbackAgent, FeedbackCritique, QuoteRevision
from crmn_agent.schemas import Decision


class ScriptedProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, schema, instruction, payload, cap):
        self.calls += 1
        if schema is QuoteRevision:
            return schema(revised_quote=payload["source_fragment"])
        if schema is FeedbackCritique:
            return schema(needs_revision=True, issue="The goal needs a direct action.",
                          evidence="The box is closed.", suggested_action="open box")
        return Decision(reflection="Synthetic demonstration.", action="look" if self.calls == 1 else "open box")


def run():
    provider = ScriptedProvider()
    agent = FeedbackAgent(provider)
    turn = agent.act("The box is closed.", "Open the box.", ["look", "open box"], 0)
    return {"action": turn.action, "scripted_calls": provider.calls,
            "intervention": turn.diagnostics["intervention"], "is_real_llm": False}


if __name__ == "__main__":
    print(run())
