import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crmn_agent import codex_provider
from crmn_agent.codex_provider import AUTHORIZED_CAMPAIGNS, CodexProvider, reserve_call
from crmn_agent.providers import BudgetExceeded, Provider
from crmn_agent.public_authorization import RESOURCE_PROFILES, authorize, campaign_maximum
from crmn_agent.schemas import Decision


def test_new_fifty_call_owner_campaign_is_portable_and_does_not_reset(tmp_path):
    config = authorize(tmp_path / 'deployment', 'local-demo', 50, 'owner-selected-model', True)
    value = json.loads(config.read_text())['provider']
    ledger = config.with_name('allowance.json')
    assert value['allow_subscription'] is True and value['allow_paid'] is False
    assert value['authorized_campaigns'] == {'local-demo': 50}
    assert value['max_calls'] == 1000 and value['max_total_tokens'] == 1_000_000
    assert value['resource_profile'] == 'smoke'
    for expected in range(1, 51):
        assert reserve_call(ledger, 50, 'local-demo', authorized_campaigns=value['authorized_campaigns']) == expected
    with pytest.raises(BudgetExceeded, match='exhausted'):
        reserve_call(ledger, 50, 'local-demo', authorized_campaigns=value['authorized_campaigns'])
    with pytest.raises(FileExistsError):
        authorize(config.parent, 'local-demo', 50, 'owner-selected-model', True)
    assert len(ledger.with_suffix('.attempts.jsonl').read_text().splitlines()) == 50


def test_recorded_resource_profile_keeps_the_separate_owner_launch_allowance(tmp_path):
    config = authorize(tmp_path / 'replica', 'local-replica', 2, 'fixture', True,
                       resource_profile='recorded-study')
    value = json.loads(config.read_text())['provider']
    ledger = config.with_name('allowance.json')
    assert {name: value[name] for name in ('max_calls', 'max_total_tokens')} == {
        'max_calls': 20_000, 'max_total_tokens': 200_000_000}
    assert value['subscription_max_launches'] == 2
    assert json.loads(ledger.read_text())['limit'] == 2
    for expected in (1, 2):
        assert reserve_call(ledger, 2, 'local-replica', authorized_campaigns=value['authorized_campaigns']) == expected
    with pytest.raises(BudgetExceeded, match='exhausted'):
        reserve_call(ledger, 2, 'local-replica', authorized_campaigns=value['authorized_campaigns'])
    assert len(ledger.with_suffix('.attempts.jsonl').read_text().splitlines()) == 2


def test_recorded_profile_can_cover_full_qa_logical_calls_without_inference(tmp_path):
    config = authorize(tmp_path / 'qa', 'local-qa', 1560, 'fixture', True,
                       resource_profile='recorded-study')
    value = json.loads(config.read_text())['provider']
    provider = Provider(value)
    # The 120-question protocol has 1,560 logical calls. Validate its process
    # guards directly: this does not construct CodexProvider or launch anything.
    provider.usage.calls = 1560
    provider.usage.input_tokens = 23_000_000
    provider._check_budget(100_000, 500)
    assert not config.with_name('allowance.attempts.jsonl').exists()
    assert value['max_calls'] == RESOURCE_PROFILES['recorded-study']['max_calls']


def test_unknown_resource_profile_creates_no_allowance(tmp_path):
    with pytest.raises(ValueError, match='resource profile'):
        authorize(tmp_path / 'invalid', 'local-invalid', 50, 'fixture', True,
                  resource_profile='unknown')
    assert not (tmp_path / 'invalid').exists()


def test_recorded_profile_cli_creates_configuration_without_launching_a_model(tmp_path):
    destination = tmp_path / 'cli-profile'
    root = Path(__file__).parents[1]
    subprocess.run([
        sys.executable, '-m', 'crmn_agent.public_authorization',
        '--directory', str(destination), '--campaign', 'fixture-cli-profile',
        '--max-launches', '2', '--model', 'fixture-no-model-access',
        '--resource-profile', 'recorded-study', '--i-authorize-subscription-use',
    ], env={'PYTHONPATH': str(root / 'src')}, capture_output=True, text=True, check=True, timeout=15)
    provider = json.loads((destination / 'provider.json').read_text())['provider']
    assert provider['max_calls'] == 20_000 and provider['max_total_tokens'] == 200_000_000
    assert provider['subscription_max_launches'] == 2
    assert json.loads((destination / 'allowance.json').read_text())['limit'] == 2
    assert not (destination / 'allowance.attempts.jsonl').exists()


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
