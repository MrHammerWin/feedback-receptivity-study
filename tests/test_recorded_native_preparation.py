"""Synthetic environments only; no upstream source text or model access."""

import copy
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts/prepare_recorded_native.py'
    spec = importlib.util.spec_from_file_location('recorded_native_preparation', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    target = module()
    monkeypatch.setattr(target, 'runtime_identity', lambda: {'python_version': 'synthetic', 'packages': {}})
    # Synthetic preparation tests vary environment values inside this process.
    # Actual startup-state enforcement is tested in fresh subprocesses below.
    monkeypatch.setattr(target, 'sys', SimpleNamespace(flags=SimpleNamespace(hash_randomization=0)))
    assets = tmp_path / 'official-fixture'
    assets.mkdir()
    game = assets / 'synthetic-game.tw-pddl'
    game.write_text('synthetic game fixture, not upstream data')
    tasks = []
    for benchmark, index in [('alfworld', 0), ('textcraft', 0), ('alfworld', 1), ('textcraft', 1)]:
        mission = f'synthetic mission {benchmark} {index}'
        text = f'synthetic public observation {benchmark} {index}'
        background = f'synthetic recipes {index}' if benchmark == 'textcraft' else ''
        row = {'key': f'{benchmark}-main-{index:02}', 'benchmark': benchmark, 'seed': index,
               'task_id': sha(mission) if benchmark == 'textcraft' else f'synthetic-{index}',
               'initial_observation_sha256': sha(text), 'mission_sha256': sha(mission)}
        row['game_sha256' if benchmark == 'alfworld' else 'public_background_sha256'] = (
            sha(game.read_text()) if benchmark == 'alfworld' else sha(background))
        tasks.append(row)
    plan = {'schema_version': 'feedback-native-run-plan-v1', 'model': 'synthetic-model',
            'source_sha256': 'a' * 64, 'reasoning_effort': 'low', 'variants': target.VARIANTS,
            'horizons': target.HORIZONS, 'episode_call_cap': 64, 'tasks': tasks}
    provider = {'provider': {'backend': 'codex', 'model': 'synthetic-model', 'allow_subscription': False,
                             'allow_paid': False, 'subscription_ledger': str(tmp_path / 'not-created.json'),
                             **target.RESOURCE_PROFILES['recorded-study']}}
    plan_path, provider_path = tmp_path / 'plan.json', tmp_path / 'provider.json'
    plan_path.write_text(json.dumps(plan))
    provider_path.write_text(json.dumps(provider))
    closes = []

    def factory(benchmark, task, seed, config):
        assert task == 'default' and config['split'] == 'test'
        mission = f'synthetic mission {benchmark} {seed}'
        observed = SimpleNamespace(text=f'synthetic public observation {benchmark} {seed}', mission=mission,
                                   background=f'synthetic recipes {seed}' if benchmark == 'textcraft' else '')
        return SimpleNamespace(task_id=sha(mission) if benchmark == 'textcraft' else f'synthetic-{seed}',
                               game=game, reset=lambda: observed, close=lambda: closes.append((benchmark, seed)))
    return target, plan, provider, plan_path, provider_path, assets, factory, closes


def test_all_recorded_inputs_verified_before_config_creation_and_cyclic_order_preserved(fixture, tmp_path):
    target, plan, provider, plan_path, provider_path, assets, factory, closes = fixture
    output = tmp_path / 'local-prepared'
    record = target.prepare(plan_path, provider_path, output, assets, factory)
    assert record['new_model_calls'] == 0 and len(closes) == 4
    assert record['per_process_provider_guards'] == {'max_calls': 20_000, 'max_total_tokens': 200_000_000}
    assert not Path(provider['provider']['subscription_ledger']).exists()
    assert len(record['tasks']) == 4 and all(row['verified'] for row in record['tasks'])
    for i, task in enumerate(plan['tasks']):
        config = json.loads((output / 'configs' / (task['key'] + '.json')).read_text())
        shift = i // 2
        assert config['variants'] == target.VARIANTS[shift:] + target.VARIANTS[:shift]
        assert config['episode_call_cap'] == 64 and config['horizons'] == target.HORIZONS
        assert config['task_keys'] == [task['key']]
        assert config['agent_options']['feedback']['revision_only'] is True
        assert config['agent_options']['feedback']['history_turns'] == 100
        assert config['provider']['allow_subscription'] is False
        assert config['provider']['allow_paid'] is False
        assert config['provider']['max_calls'] == 20_000
        assert config['provider']['max_total_tokens'] == 200_000_000
    manifest = json.loads((output / 'native_manifest.json').read_text())
    assert manifest['tasks'][0]['mission'].startswith('synthetic mission')
    with pytest.raises(FileExistsError):
        target.prepare(plan_path, provider_path, output, assets, factory)


@pytest.mark.parametrize('field,task_index', [
    ('task_id', 0), ('initial_observation_sha256', 0), ('game_sha256', 0),
    ('public_background_sha256', 1), ('mission_sha256', 1),
])
def test_mismatched_last_task_inputs_leave_no_partial_configuration(fixture, tmp_path, field, task_index):
    target, plan, _, plan_path, provider_path, assets, factory, closes = fixture
    # Alter a later matching environment type so earlier tasks have already been checked.
    task_index += 2
    plan['tasks'][task_index][field] = 'b' * 64
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='differs'):
        target.prepare(plan_path, provider_path, output, assets, factory)
    assert not output.exists() and closes


@pytest.mark.parametrize('change', ['cap', 'horizon', 'variants', 'duplicate', 'unsafe_key', 'private_plan'])
def test_protocol_changes_rejected_before_any_environment_or_auth_access(fixture, tmp_path, change):
    target, plan, _, plan_path, provider_path, assets, _, _ = fixture
    changed = copy.deepcopy(plan)
    if change == 'cap':
        changed['episode_call_cap'] = 65
    elif change == 'horizon':
        changed['horizons']['alfworld'] = 50
    elif change == 'variants':
        changed['variants'] = changed['variants'][:-1]
    elif change == 'duplicate':
        changed['tasks'].append(changed['tasks'][0])
    elif change == 'unsafe_key':
        changed['tasks'][0]['key'] = '../unsafe'
    else:
        changed['tasks'][0]['mission'] = 'private raw source text'
    plan_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        target.prepare(plan_path, provider_path, tmp_path / 'not-created', assets,
                       lambda *args: pytest.fail('Validation must precede environment construction'))


def test_model_mismatch_is_separate_experiment_and_does_not_grant_consent(fixture, tmp_path):
    target, _, provider, plan_path, provider_path, assets, _, _ = fixture
    provider['provider']['model'] = 'different-model'
    provider_path.write_text(json.dumps(provider))
    with pytest.raises(ValueError, match='differ'):
        target.prepare(plan_path, provider_path, tmp_path / 'not-created', assets,
                       lambda *args: pytest.fail('No environment should start'))


@pytest.mark.parametrize('field,value', [
    ('max_calls', None), ('max_calls', 1000), ('max_total_tokens', None), ('max_total_tokens', 1_000_000),
])
def test_replica_rejects_additional_process_censoring_before_environment_or_auth_access(
        fixture, tmp_path, field, value):
    target, _, provider, plan_path, provider_path, assets, _, _ = fixture
    if value is None:
        provider['provider'].pop(field)
    else:
        provider['provider'][field] = value
    provider_path.write_text(json.dumps(provider))
    output = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='resource-profile recorded-study'):
        target.prepare(plan_path, provider_path, output, assets,
                       lambda *args: pytest.fail('Guard validation must precede environment construction'))
    assert not output.exists()
    assert not Path(provider['provider']['subscription_ledger']).exists()


def test_explicit_fresh_mode_preserves_goal_but_discloses_changed_recipe_sheet(fixture, tmp_path, monkeypatch):
    target, plan, _, plan_path, provider_path, assets, factory, _ = fixture
    plan['tasks'][1]['public_background_sha256'] = 'b' * 64
    plan_path.write_text(json.dumps(plan))
    output = tmp_path / 'fresh'
    monkeypatch.delenv('PYTHONHASHSEED', raising=False)
    with pytest.raises(ValueError, match='PYTHONHASHSEED=0'):
        target.prepare(plan_path, provider_path, output, assets, factory, fresh_replication=True)
    assert not output.exists()
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    result = target.prepare(plan_path, provider_path, output, assets, factory, fresh_replication=True)
    assert result['fresh_replication'] is True
    assert result['preparation_matches_recorded_selection_backgrounds'] is False
    assert result['tasks'][1]['matches_recorded_selection_background'] is False
    local = json.loads((output / 'native_manifest.json').read_text())['tasks'][1]
    assert local['task_id'] == plan['tasks'][1]['task_id']
    assert local['recorded_selection_background_sha256'] == 'b' * 64
    assert local['public_background_sha256'] == sha('synthetic recipes 0')


def load_runner(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('recorded_native_runner', scripts / 'run_recorded_native.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    preparation = importlib.import_module('prepare_recorded_native')
    monkeypatch.setattr(preparation, 'sys', SimpleNamespace(flags=SimpleNamespace(hash_randomization=0)))
    monkeypatch.setattr(runner, 'runtime_identity', lambda: {'python_version': 'synthetic', 'packages': {}})
    return runner


def test_explicit_runner_rechecks_inputs_before_delegating_to_inference(fixture, tmp_path, monkeypatch):
    target, _, _, plan_path, provider_path, assets, factory, _ = fixture
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    output = tmp_path / 'replica'
    target.prepare(plan_path, provider_path, output, assets, factory, fresh_replication=True)
    runner = load_runner(monkeypatch)
    launched = []
    runner.run(output / 'preparation.json', 'textcraft-main-00', environment_factory=factory,
               experiment_runner=lambda config: launched.append(config))
    assert launched == [output / 'configs/textcraft-main-00.json']


@pytest.mark.parametrize('failure', ['hashseed', 'runtime', 'config', 'manifest', 'input'])
def test_runner_guard_failure_never_delegates_to_inference(fixture, tmp_path, monkeypatch, failure):
    target, _, _, plan_path, provider_path, assets, factory, _ = fixture
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    output = tmp_path / 'replica'
    target.prepare(plan_path, provider_path, output, assets, factory, fresh_replication=True)
    runner = load_runner(monkeypatch)
    if failure == 'hashseed':
        monkeypatch.setenv('PYTHONHASHSEED', '1')
    elif failure == 'runtime':
        monkeypatch.setattr(runner, 'runtime_identity', lambda: {'python_version': 'changed'})
    elif failure == 'config':
        path = output / 'configs/textcraft-main-00.json'
        path.write_text(path.read_text() + ' ')
    elif failure == 'manifest':
        path = output / 'native_manifest.json'
        path.write_text(path.read_text() + ' ')
    else:
        original = factory

        def changed(*args):
            env = original(*args)
            reset = env.reset

            def changed_reset():
                result = reset()
                result.background += ' modified'
                return result
            env.reset = changed_reset
            return env
        factory = changed
    with pytest.raises(ValueError):
        runner.run(output / 'preparation.json', 'textcraft-main-00', environment_factory=factory,
                   experiment_runner=lambda config: pytest.fail('No inference may start after a guard failure'))


@pytest.mark.parametrize('startup_seed,late_seed,accepted', [
    ('0', None, True), ('1', None, False), (None, None, False),
    ('1', '0', False), (None, '0', False), ('0', '1', False),
])
def test_hashseed_guard_checks_actual_subprocess_startup(startup_seed, late_seed, accepted):
    root = Path(__file__).parents[1]
    code = (
        'import os, sys\n'
        f'sys.path.insert(0, {str(root / "scripts")!r})\n'
        'from prepare_recorded_native import require_hashseed_zero\n'
    )
    if late_seed is not None:
        code += f'os.environ["PYTHONHASHSEED"] = {late_seed!r}\n'
    code += (
        'try:\n'
        '    require_hashseed_zero()\n'
        'except ValueError:\n'
        '    sys.exit(23)\n'
    )
    env = {'PYTHONPATH': str(root / 'src')}
    if startup_seed is not None:
        env['PYTHONHASHSEED'] = startup_seed
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, timeout=15)
    assert result.returncode == (0 if accepted else 23), result.stderr.decode()


@pytest.mark.parametrize('implementation,flag', [('PyPy', 0), ('CPython', None)])
def test_hashseed_guard_fails_closed_for_unknown_runtime(monkeypatch, implementation, flag):
    target = module()
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    monkeypatch.setattr(target.platform, 'python_implementation', lambda: implementation)
    flags = SimpleNamespace() if flag is None else SimpleNamespace(hash_randomization=flag)
    monkeypatch.setattr(target, 'sys', SimpleNamespace(flags=flags))
    with pytest.raises(ValueError, match='before starting CPython'):
        target.require_hashseed_zero()
