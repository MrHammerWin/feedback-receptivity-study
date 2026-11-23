"""Component ablations; these are not additional published SOTA baselines."""

from .agents import Turn
from .providers import Provider
from .schemas import Decision


class DraftOnlyAgent:
    """Execute the feedback agent's identical first proposal with one model call.

    Purpose, payload order, history truncation, schema and output cap match the
    feedback actor. No critic or probe is invoked. Commands pass unchanged to the
    native environment, including commands absent from its admissible list.
    """

    def __init__(self, provider: Provider, history_turns: int = 100):
        if history_turns < 0:
            raise ValueError("history_turns must be nonnegative")
        self.provider = provider
        self.history_turns = history_turns
        self.history: list[dict] = []

    def feedback(self, success: bool, terminated: bool):
        pass

    def act(self, observation: str, mission: str, actions: list[str], step: int, background: str = "") -> Turn:
        history = self.history[-self.history_turns:] if self.history_turns else []
        context = {
            "mission": mission,
            "observation": observation,
            "available_actions": actions,
            "history": history,
            "background": background,
        }
        decision = self.provider.complete(
            Decision,
            "Reason briefly about the observed task and propose ONE useful next action. "
            "Respect the supplied commands and rules.",
            context,
            256,
        )
        self.history.append({"observation": observation, "action": decision.action})
        return Turn(decision.action, {
            "mechanism": "actor_only_ablation",
            "intervention": "none",
            "decision": decision.model_dump(),
            "action_not_admissible": bool(actions and decision.action not in actions),
            "comparison_scope": "same initial feedback actor without metacognitive control, not SOTA",
        })
