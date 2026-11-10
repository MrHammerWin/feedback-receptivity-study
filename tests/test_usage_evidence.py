import pytest

from crmn_agent.codex_provider import completion_usage
from crmn_agent.providers import BudgetExceeded, ModelOutputError, Provider
from crmn_agent.schemas import Decision


@pytest.mark.parametrize('usage', [None, {}, {'input_tokens': 1, 'output_tokens': 1},
                                    {'input_tokens': True, 'output_tokens': 1, 'cached_input_tokens': 0},
                                    {'input_tokens': 1, 'output_tokens': -1, 'cached_input_tokens': 0},
                                    {'input_tokens': 1, 'output_tokens': 1, 'cached_input_tokens': 2}])
def test_missing_or_inconsistent_telemetry_is_not_zero_cost(usage):
    with pytest.raises(ModelOutputError, match='usage|telemetry|Cached'):
        completion_usage([{'type': 'turn.completed', 'usage': usage}])


def test_cached_tokens_are_a_subset_of_input():
    assert completion_usage([{'type': 'turn.completed', 'usage': {
        'input_tokens': 100, 'cached_input_tokens': 80, 'output_tokens': 2}}]) == {
        'input_tokens': 100, 'cached_input_tokens': 80, 'output_tokens': 2}


def test_denied_reservation_is_not_a_logical_inference():
    class Denied(Provider):
        def _generate(self, *args):
            raise BudgetExceeded('No launch was authorized')
    provider = Denied({'backend': 'codex'})
    with pytest.raises(BudgetExceeded):
        provider.complete(Decision, 'choose', {})
    assert provider.usage.calls == 0
