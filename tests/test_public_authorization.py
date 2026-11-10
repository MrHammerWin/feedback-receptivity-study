import json
from types import SimpleNamespace

import pytest

from crmn_agent import codex_provider
from crmn_agent.codex_provider import AUTHORIZED_CAMPAIGNS, CodexProvider, reserve_call
from crmn_agent.providers import BudgetExceeded, Provider
from crmn_agent.public_authorization import authorize, campaign_maximum
from crmn_agent.schemas import Decision


def test_new_fifty_call_owner_campaign_is_portable_and_does_not_reset(tmp_path):
    config = authorize(tmp_path / 'deployment', 'local-demo', 50, 'owner-selected-model', True)
    value = json.loads(config.read_text())['provider']
    ledger = config.with_name('allowance.json')
    assert value['allow_subscription'] is True and value['allow_paid'] is False
    assert value['authorized_campaigns'] == {'local-demo': 50}
    for expected in range(1, 51):
        assert reserve_call(ledger, 50, 'local-demo', authorized_campaigns=value['authorized_campaigns']) == expected
    with pytest.raises(BudgetExceeded, match='exhausted'):
        reserve_call(ledger, 50, 'local-demo', authorized_campaigns=value['authorized_campaigns'])
    with pytest.raises(FileExistsError):
        authorize(config.parent, 'local-demo', 50, 'owner-selected-model', True)
    assert len(ledger.with_suffix('.attempts.jsonl').read_text().splitlines()) == 50


def test_missing_consent_and_unknown_campaign_make_no_reservation(tmp_path):
    with pytest.raises(BudgetExceeded, match='consent'):
        authorize(tmp_path / 'deployment', 'local-demo', 50, 'fixture', False)
    assert not (tmp_path / 'deployment').exists()
    ledger = tmp_path / 'allowance.json'
    ledger.write_text(json.dumps({'campaign': 'local-demo', 'limit': 50}))
    with pytest.raises(BudgetExceeded, match='Unrecognized'):
        reserve_call(ledger, 50, 'local-demo')
    assert not ledger.with_suffix('.attempts.jsonl').exists()


@pytest.mark.parametrize('extra', [None, {'local-demo': 50}])
def test_recorded_ceilings_are_unchanged(extra):
    for name, maximum in AUTHORIZED_CAMPAIGNS.items():
        assert campaign_maximum(name, AUTHORIZED_CAMPAIGNS, extra) == maximum
        with pytest.raises(BudgetExceeded, match='cannot be overridden'):
            campaign_maximum(name, AUTHORIZED_CAMPAIGNS, {name: maximum + 1})


@pytest.mark.parametrize('extra', [{}, [], {'a': True}, {'a': 0}, {'a': -1}, {'bad/name': 50}])
def test_invalid_portable_authorization_is_rejected(extra):
    with pytest.raises(BudgetExceeded):
        campaign_maximum('a', AUTHORIZED_CAMPAIGNS, extra)


def test_ledger_identity_and_smaller_runtime_cap_remain_enforced(tmp_path):
    ledger = tmp_path / 'allowance.json'
    ledger.write_text(json.dumps({'campaign': 'local-demo', 'limit': 50}))
    assert reserve_call(ledger, 1, 'local-demo', authorized_campaigns={'local-demo': 50}) == 1
    with pytest.raises(BudgetExceeded, match='exhausted'):
        reserve_call(ledger, 1, 'local-demo', authorized_campaigns={'local-demo': 50})
    with pytest.raises(BudgetExceeded, match='Wrong campaign'):
        reserve_call(ledger, 50, 'another-demo', authorized_campaigns={'another-demo': 50})


def test_provider_passes_explicit_portable_registry_to_real_reservation_guard(tmp_path, monkeypatch):
    config_path = authorize(tmp_path / 'deployment', 'local-demo', 50, 'fixture', True)
    config = json.loads(config_path.read_text())['provider']
    provider = object.__new__(CodexProvider)
    Provider.__init__(provider, config)
    provider.root = tmp_path / 'isolated'
    provider.root.mkdir()
    provider.ledger = config_path.with_name('allowance.json')
    provider.process_env = {}

    def fake_process(arguments, **kwargs):
        output = arguments[arguments.index('--output-last-message') + 1]
        from pathlib import Path
        Path(output).write_text('{"reflection":"fixture","action":"inventory"}')
        event = {'type': 'turn.completed', 'usage': {
            'input_tokens': 20, 'output_tokens': 5, 'cached_input_tokens': 0}}
        return SimpleNamespace(returncode=0, stdout=json.dumps(event), stderr='')

    monkeypatch.setattr(codex_provider.subprocess, 'run', fake_process)
    result = provider.complete(Decision, 'Synthetic decision.', {'observation': 'fixture'}, 100)
    assert result.action == 'inventory'
    records = provider.ledger.with_suffix('.attempts.jsonl').read_text().splitlines()
    assert len(records) == 1 and json.loads(records[0])['campaign'] == 'local-demo'
    assert provider.usage.calls == 1
