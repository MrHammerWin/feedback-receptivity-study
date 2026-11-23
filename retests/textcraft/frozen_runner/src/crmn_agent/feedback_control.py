"""A source-grounded, local feedback-receptivity admission controller.

The controls measure quote repair and preservation, not hidden task correctness.
Control conversations never become part of the live task history.
"""

import re
from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import Field

from .agents import Turn
from .grounding import observation_anchors
from .providers import ModelOutputError, Provider
from .schemas import Decision, Schema


class FeedbackCritique(Schema):
    needs_revision: bool
    issue: str = Field(max_length=240)
    evidence: str = Field(max_length=180)
    suggested_action: str = Field(max_length=200)


class QuoteRevision(Schema):
    revised_quote: str = Field(max_length=180)


@dataclass(frozen=True)
class ReceptivityProfile:
    beneficial_correction: int | None
    harmful_corruption: int | None
    observed_pairs: int
    source_quote: str
    mutated_quote: str
    corrected_quote: str | None = None
    preserved_quote: str | None = None
    errors: tuple[str, ...] = ()


def canonical_quote(value: str) -> str:
    """Only whitespace is immaterial; this is a source-fidelity microtask."""
    return " ".join(value.split())


def mutate_quote(quote: str) -> str:
    """Construct an incorrect quotation without consulting any hidden state.

    The assertion being checked is equality to the selected source quotation,
    not whether an arbitrary alternative sentence is true elsewhere in the world.
    """
    replacements = (
        ("empty", "not empty"), ("closed", "open"), ("open", "closed"),
        ("left", "right"), ("right", "left"), ("front", "behind"),
    )
    for original, replacement in replacements:
        # Preserve token boundaries: e.g. "left" inside "leftover" is not a target.
        changed, count = re.subn(r"\b" + original + r"\b", replacement, quote, count=1,
                                 flags=re.IGNORECASE)
        if count:
            return changed
    return "NOT: " + quote


def select_intervention(profile: ReceptivityProfile, grounded_critique: bool) -> tuple[str, str]:
    """Choose a repair *channel*, rather than averaging two unrelated rates."""
    if (profile.observed_pairs != 1 or profile.beneficial_correction not in {0, 1}
            or profile.harmful_corruption not in {0, 1}):
        return "unresolved_rebuild", "incomplete_local_measurement"
    if profile.harmful_corruption:
        return "rebuild", "correct_source_claim_was_corrupted_by_feedback"
    if profile.beneficial_correction and grounded_critique:
        return "critique", "repaired_wrong_quote_and_preserved_correct_quote"
    if not grounded_critique:
        return "facts_only", "critique_has_no_literal_public_evidence"
    return "facts_only", "retention_passed_but_beneficial_repair_failed"


class FeedbackAgent:
    """Draft → critique → isolated controls → admitted repair → one real action.

    With ``probe_policy='on_revision'`` (default), an unchanged legal draft takes
    two model calls. A tested revision takes five. ``always`` is available for a
    pre-registered diagnostic/compute-matched series. There is deliberately no
    cross-step profile cache: a response in one context does not certify another.
    """

    def __init__(
        self,
        provider: Provider,
        history_turns: int = 8,
        probe_policy: Literal["on_revision", "gated", "always", "never"] = "on_revision",
        gate_mode: Literal["controlled", "always_critique", "always_rebuild", "facts_only", "grounding_only"] = "controlled",
        revision_only: bool = False,
    ):
        if probe_policy not in {"on_revision", "gated", "always", "never"}:
            raise ValueError("Unknown feedback probe policy")
        if gate_mode not in {"controlled", "always_critique", "always_rebuild", "facts_only", "grounding_only"}:
            raise ValueError("Unknown feedback gate mode")
        if probe_policy == "never" and gate_mode == "controlled":
            raise ValueError("An unmeasured controlled gate is not allowed; never is only an ablation")
        if gate_mode == 'grounding_only' and probe_policy != 'never':
            raise ValueError('The grounding-only ablation must not invoke receptivity probes')
        if history_turns < 0:
            raise ValueError("history_turns must be nonnegative")
        self.provider = provider
        self.history_turns = history_turns
        self.probe_policy = probe_policy
        self.gate_mode = gate_mode
        self.revision_only = revision_only
        self.history: list[dict] = []
        self.last_feedback: dict | None = None

    def feedback(self, success: bool, terminated: bool):
        # A zero task reward is NOT interpreted as an incorrect last action.
        self.last_feedback = {"task_success": bool(success), "terminated": bool(terminated)}

    def _profile(self, observation: str, mission: str, step: int) -> ReceptivityProfile:
        anchors = observation_anchors(observation, mission)
        if not anchors:
            return ReceptivityProfile(None, None, 0, "", "", errors=("no_observed_anchor",))
        source = anchors[step % len(anchors)].value
        wrong = mutate_quote(source)
        answers: dict[str, str] = {}
        errors = []
        # The arm label and expected output remain in Python, outside the prompts.
        controls = [
            ("correction", wrong, source),
            ("preservation", source, wrong),
        ]
        if step % 2:
            controls.reverse()
        for arm, initial, advice in controls:
            try:
                answer = self.provider.complete(
                    QuoteRevision,
                    "This is an isolated factual-memory revision task. The source observation is "
                    "authoritative. A saved quotation and a critique may each contain an error. "
                    "Return the best revised quotation, preserving the exact source words when "
                    "supported. Evaluate feedback against the source; do not merely report agreement.",
                    {
                        "source_observation": observation[:1800],
                        "source_fragment": source,
                        "saved_quote": initial,
                        "critique": "The saved quotation is incorrect. Replace it with: " + advice,
                    },
                    180,
                )
                answers[arm] = answer.revised_quote
            except ModelOutputError:
                # A failed probe is missing measurement. Budget/transport errors propagate.
                errors.append(arm + ":invalid_model_output")
        complete = "correction" in answers and "preservation" in answers
        corrected = answers.get("correction")
        preserved = answers.get("preservation")
        correction = (int(canonical_quote(corrected) == canonical_quote(source))
                      if corrected is not None else None)
        corruption = (int(canonical_quote(preserved) != canonical_quote(source))
                      if preserved is not None else None)
        return ReceptivityProfile(correction, corruption, int(complete), source, wrong,
                                  corrected, preserved, tuple(errors))

    def measure_receptivity(self, observation: str, mission: str, step: int = 0) -> ReceptivityProfile:
        """Expose the two isolated controls without taking an environment action.

        Suitable for a separately reported mechanism diagnostic. The public
        observation, never a hidden task label, supplies the quote target.
        """
        return self._profile(observation, mission, step)

    @staticmethod
    def _grounded(critique: FeedbackCritique, context: dict) -> bool:
        evidence = critique.evidence.strip()
        if not evidence:
            return False
        # Citation presence is only a grounding check, not proof of the critic's conclusion.
        evidence_pool = [context["observation"], context["background"], context["mission"]]
        evidence_pool.extend(turn["observation"] for turn in context["history"])
        return any(evidence in text for text in evidence_pool)

    @staticmethod
    def _legal(action: str, available: list[str]) -> bool:
        return bool(action.strip()) and (not available or action in available)

    def _context(self, observation: str, mission: str, actions: list[str], background: str) -> dict:
        history = self.history[-self.history_turns:] if self.history_turns else []
        return {
            "mission": mission,
            "observation": observation,
            "available_actions": actions,
            "history": history,
            "background": background,
        }

    def act(self, observation: str, mission: str, actions: list[str], step: int, background: str = "") -> Turn:
        context = self._context(observation, mission, actions, background)
        draft = self.provider.complete(
            Decision,
            "Reason briefly about the observed task and propose ONE useful next action. "
            "Respect the supplied commands and rules.",
            context,
            256,
        )
        critique = self.provider.complete(
            FeedbackCritique,
            "Audit the proposed next action against the public observation, goal, rules and history. "
            "Request revision only for a concrete issue. Cite a literal public evidence fragment. "
            "Do not invent hidden state or a reference answer. A zero intermediate reward does not "
            "by itself mean an action failed. If the action is appropriate, set needs_revision false.",
            {**context, "draft": draft.model_dump()},
            384,
        )
        return self.revise_proposal(observation, mission, actions, step, draft, critique, background)

    def revise_proposal(
        self,
        observation: str,
        mission: str,
        actions: list[str],
        step: int,
        draft: Decision,
        critique: FeedbackCritique,
        background: str = "",
    ) -> Turn:
        """Apply the same runtime gate to an externally fixed public proposal.

        Frozen-state experiments may hold proposals and critiques constant across
        admission policies. Hidden correctness labels belong only to the scorer,
        never to this interface. Only the returned action enters live history.
        """
        context = self._context(observation, mission, actions, background)
        grounded = self._grounded(critique, context)
        draft_legal = self._legal(draft.action, actions)
        needs_probe = self.probe_policy != "never" and (
            self.probe_policy == "always" or critique.needs_revision or not draft_legal
        )
        profile = None
        if needs_probe:
            profile = self._profile(observation, mission, step)
            route, reason = select_intervention(profile, grounded)
        else:
            route, reason = "keep_draft", "critic_did_not_propose_revision"
        override = not self.revision_only or critique.needs_revision or not draft_legal
        if self.gate_mode == 'grounding_only' and override:
            route = 'critique' if grounded else 'facts_only'
            reason = 'literal_citation_only_no_receptivity_measurement'
        elif self.gate_mode != "controlled" and override:
            route = {"always_critique": "critique", "always_rebuild": "rebuild",
                     "facts_only": "facts_only"}[self.gate_mode]
            reason = "preconfigured_ablation_override"
        if not draft_legal and (self.gate_mode == 'controlled' or self.revision_only):
            route, reason = "rebuild", "draft_action_unavailable"

        final = draft
        if route != "keep_draft":
            payload = dict(context)
            if route == "critique":
                payload.update({"draft": draft.model_dump(), "admitted_critique": critique.model_dump()})
                instruction = (
                    "Revise or retain the draft action after considering the admitted critique. "
                    "Verify it against the public observation and rules. Output ONE legal next action."
                )
            elif route == "facts_only":
                payload.update({
                    "draft": draft.model_dump(),
                    "observed_facts": [a.value for a in observation_anchors(observation, mission)],
                })
                instruction = (
                    "Reassess the draft using the directly observed facts and rules. "
                    "Output ONE useful legal next action."
                )
            else:
                # No critic prose, suggested action, draft, or control response can cross this boundary.
                instruction = (
                    "Build a fresh understanding from the public task context and select ONE useful "
                    "legal next action. State only a brief reason."
                )
            final = self.provider.complete(Decision, instruction, payload, 256)
        # The native environment supplies invalid-command feedback. The earlier
        # illegal-draft rebuild route remains a decision of this mechanism.
        self.history.append({"observation": observation, "action": final.action})
        return Turn(final.action, {
            "mechanism": "local_feedback_receptivity_gate",
            "action_in_public_list": final.action in actions if actions else None,
            "gate_mode": self.gate_mode,
            "revision_only": self.revision_only,
            "probe_policy": self.probe_policy,
            "intervention": route,
            "gate_reason": reason,
            "critic_requested_revision": critique.needs_revision,
            "critic_evidence_present": grounded,
            "draft": draft.model_dump(),
            "critique": critique.model_dump(),
            "profile": asdict(profile) if profile is not None else None,
            "final": final.model_dump(),
            "measurement_scope": "local_source_quote_repair_and_preservation_not_hidden_task_correctness",
        })
