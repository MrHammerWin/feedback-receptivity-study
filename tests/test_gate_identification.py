"""Distinguish receptivity-based routing from literal-citation filtering."""
from copy import deepcopy

import pytest

from crmn_agent.feedback_control import FeedbackAgent, FeedbackCritique, QuoteRevision
from crmn_agent.schemas import Decision


class SourceFaithful:
    def __init__(self):
        self.calls = []

    def complete(self, schema, purpose, payload, cap):
        self.calls.append((schema, purpose, deepcopy(payload), cap))
        if schema is QuoteRevision:
            return schema(revised_quote=payload['source_fragment'])
        return schema(reflection='Follow the public source.', action='open box')


@pytest.mark.parametrize('citation', ['The box is closed.', 'Absent quotation'])
def test_healthy_gate_and_grounding_only_produce_identical_final_requests(citation):
    left, right = SourceFaithful(), SourceFaithful()
    controlled = FeedbackAgent(left, revision_only=True)
    grounding = FeedbackAgent(right, gate_mode='grounding_only', probe_policy='never', revision_only=True)
    args = ('The box is closed.', 'Open the box.', ['open box'], 0,
            Decision(reflection='Inspect.', action='open box'),
            FeedbackCritique(needs_revision=True, issue='Revise the action.', evidence=citation,
                             suggested_action='open box'))
    a = controlled.revise_proposal(*args)
    b = grounding.revise_proposal(*args)
    assert a.diagnostics['intervention'] == b.diagnostics['intervention']
    assert left.calls[-1] == right.calls[-1]
    assert len(left.calls) == 3 and len(right.calls) == 1


@pytest.mark.parametrize('mode', ['always_critique', 'facts_only', 'grounding_only'])
def test_on_revision_ablation_keeps_approved_legal_draft_without_extra_calls(mode):
    p = SourceFaithful()
    agent = FeedbackAgent(p, gate_mode=mode, probe_policy='never', revision_only=True)
    turn = agent.revise_proposal('The box is closed.', 'Open it.', ['open box'], 0,
        Decision(reflection='Open it.', action='open box'),
        FeedbackCritique(needs_revision=False, issue='', evidence='', suggested_action=''))
    assert turn.action == 'open box' and not p.calls


@pytest.mark.parametrize('mode', ['controlled', 'grounding_only', 'always_critique', 'facts_only'])
def test_common_invalid_draft_guard_is_shared_across_study_policies(mode):
    p = SourceFaithful()
    agent = FeedbackAgent(p, gate_mode=mode, probe_policy='on_revision' if mode=='controlled' else 'never',
                          revision_only=True)
    turn = agent.revise_proposal('The box is closed.', 'Open it.', ['open box'], 0,
        Decision(reflection='Invalid proposal.', action='teleport'),
        FeedbackCritique(needs_revision=True, issue='Unsupported.', evidence='The box is closed.',
                         suggested_action='teleport'))
    assert turn.diagnostics['intervention'] == 'rebuild'
    assert 'draft' not in p.calls[-1][2] and 'admitted_critique' not in p.calls[-1][2]
