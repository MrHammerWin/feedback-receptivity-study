import json
from dataclasses import dataclass, field

from .controller import Prior, cycle_error, entropy, features, prediction_error, responsibilities
from .grounding import observation_anchors
from .providers import ModelOutputError, Provider
from .schemas import Audit, Decision, Reconstruction, Representation


@dataclass
class Turn:
    action: str
    diagnostics: dict = field(default_factory=dict)


class Agent:
    def __init__(self, provider: Provider, variant: str, prior: Prior | None = None,
                 history_turns: int = 8):
        self.provider = provider
        self.variant = variant
        self.prior = prior or Prior()
        self.history_turns = history_turns
        self.history = []
        self.recent_actions = []
        self.prediction_ema = [0.0, 0.0]
        self.reward_ema = [0.0, 0.0]
        self.pending_outcome = None
        self.last_reward_error = None
        self.previous = None

    def feedback(self, success: bool, terminated: bool):
        """Binary task reward is public; it is never used to choose an expert action."""
        if self.previous:
            self.pending_outcome = (float(success), terminated)
            if terminated:
                branch = self.previous['branch']
                delta = float(success) - self.previous['value']
                self.last_reward_error = delta
                self.reward_ema[branch] = 0.7 * self.reward_ema[branch] + 0.3 * abs(delta)
                self.pending_outcome = None

    def act(self, observation: str, mission: str, actions: list[str], step: int, background: str = '') -> Turn:
        context = {
            "mission": mission, "observation": observation, "available_actions": actions,
            "history": self.history[-self.history_turns:],
            "background": background,
        }
        if self.variant in {"react", "reflact", "naive"}:
            instruction = {
                "react": "Briefly reason about the next useful action, then act.",
                "reflact": ("Reflect on the current observed state, what the goal requires, and the gap "
                            "between them. Revise failed assumptions before choosing the next action."),
                "naive": "Choose the next action directly. Leave reflection empty.",
            }[self.variant]
            decision = self.provider.complete(Decision, instruction, context, 192)
            turn = Turn(decision.action, {"reflection": decision.reflection})
        else:
            turn = self._controlled(context, observation, actions, step)
        if actions and turn.action not in actions:
            raise ModelOutputError("Model selected an unavailable action; no oracle replacement was made")
        self.history.append({"observation": observation, "action": turn.action})
        self.recent_actions.append(turn.action)
        return turn

    def _controlled(self, context: dict, observation: str, actions: list[str], step: int) -> Turn:
        # Only the EXECUTED branch's predictions are evaluated against the next real observation.
        if self.previous and self.previous["predictions"] and self.variant != "no_prediction":
            audit = self.provider.complete(
                Audit,
                "Check the predictions for the last executed action against the new observation. "
                "Return unknown when the partial view cannot decide. For supported/contradicted, "
                "copy a literal quotation from the NEW observation as evidence.",
                {"action": self.previous["action"], "predictions": self.previous["predictions"],
                 "new_observation": observation}, 384,
            )
            error, count = prediction_error(audit, observation, self.previous["predictions"])
            if count:
                branch = self.previous["branch"]
                self.prediction_ema[branch] = 0.7 * self.prediction_ema[branch] + 0.3 * error

        anchors = observation_anchors(observation, context['mission'])

        kinds = ["object_map", "subgoal_plan"]
        if self.variant == "map_only":
            kinds = ["object_map"]
        if self.variant == "plan_only":
            kinds = ["subgoal_plan"]
        candidates, errors, cycles = [], [], []
        for kind in kinds:
            representation = self.provider.complete(
                Representation,
                f"Encode this task as a compact {kind}. Keep fine-grained facts and coarse subgoals. "
                "Copy canonical fact keys and values exactly. Use at most 4 facts, 4 short subgoals "
                "and 2 observable predictions. Propose ONE action, respecting the game's rules. "
                "value estimates usefulness toward the goal, not confidence alone.",
                {**context, "current_facts": [a.model_dump(exclude={"evidence"}) for a in anchors]}, 768,
            )
            if actions and representation.action not in actions:
                continue
            if self.variant == "no_cycle":
                mismatch = 0.0
            else:
                # Critical information barrier: no raw observation, history or anchors reach decoder.
                reconstruction = self.provider.complete(
                    Reconstruction,
                    "Decode only the supplied representation into canonical facts. Copy keys and values "
                    "exactly when retained. Omitted information cannot be recovered by guessing.",
                    {"representation": {
                        "summary": representation.summary[:800],
                        "facts": [f.model_dump() for f in representation.facts[:4]],
                        "subgoals": representation.subgoals[:4],
                    }}, 256,
                )
                mismatch = cycle_error(anchors, reconstruction)
            branch = 0 if kind == "object_map" else 1
            cycles.append(mismatch)
            errors.append(mismatch + (0 if self.variant == "no_prediction" else self.prediction_ema[branch]))
            candidates.append((branch, representation))
        if not candidates:
            raise ModelOutputError("Neither representation supplied a legal action")
        if self.previous and self.pending_outcome:
            reward, terminated = self.pending_outcome
            old_branch = self.previous['branch']
            next_value = next((r.value for i, r in candidates if i == old_branch), 0.0)
            delta = reward + (0.0 if terminated else 0.95 * next_value) - self.previous['value']
            self.last_reward_error = delta
            self.reward_ema[old_branch] = 0.7 * self.reward_ema[old_branch] + 0.3 * abs(delta)
            self.pending_outcome = None
        if self.variant != 'no_reward':
            errors = [error + self.reward_ema[branch]
                      for error, (branch, _) in zip(errors, candidates, strict=True)]
        x = features(observation, step, self.recent_actions)
        prior = self.prior.predict(x)
        weights = responsibilities(errors, [prior[i] for i, _ in candidates],
                                   [r.value for _, r in candidates])
        selected = max(range(len(weights)), key=weights.__getitem__)
        branch, representation = candidates[selected]
        self.previous = {"branch": branch, "action": representation.action,
                         "predictions": representation.expected[:2], "value": representation.value}
        full_weights = [0.0, 0.0]
        for (index, _), weight in zip(candidates, weights, strict=True):
            full_weights[index] = weight
        return Turn(representation.action, {
            "features": x.tolist(), "responsibilities": full_weights,
            "entropy": entropy(weights), "cycle_errors": cycles,
            "prediction_ema": list(self.prediction_ema), "selected_branch": branch,
            "reward_error_ema": list(self.reward_ema), "last_td_error": self.last_reward_error,
            "anchors": [a.model_dump() for a in anchors],
            "representations": [json.loads(r.model_dump_json()) for _, r in candidates],
        })


VARIANTS = ("naive", "react", "reflact", "crmn", "no_cycle", "no_prediction", "no_reward", "map_only", "plan_only")
