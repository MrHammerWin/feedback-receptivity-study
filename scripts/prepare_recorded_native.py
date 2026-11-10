"""Reconstruct recorded native tasks locally without running a model.

Reads a sanitized native_run_plan.json, a user-supplied provider configuration
and official locally installed environments/assets. Verifies every task before
writing any config. The default demands the recorded input hashes. Explicit
fresh mode permits new TextCraft distractors for the same recorded goals and
records the mismatch; it never claims original study-input reproduction.
Horizons, call caps and agent policies retain the recorded protocol.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import platform
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from crmn_agent.environments import make_env
from crmn_agent.public_authorization import RESOURCE_PROFILES

VARIANTS = ['raw_actor', 'stateact', 'reflact_paper', 'grounding_only', 'feedback']
HORIZONS = {'alfworld': 100, 'textcraft': 100}
SHA256 = re.compile(r'^[0-9a-f]{64}$')


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_hash(value: str) -> str:
    return digest(value.encode('utf-8'))


def encoded(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def require_hashseed_zero():
    """Require CPython's actual startup setting, not a later environment edit."""
    if (os.environ.get('PYTHONHASHSEED') != '0'
            or platform.python_implementation() != 'CPython'
            or getattr(sys.flags, 'hash_randomization', None) != 0):
        raise ValueError('Set PYTHONHASHSEED=0 before starting CPython; no model was called')


def runtime_identity():
    """Lock a fresh replica to its own Python, source, dependency and data state."""
    from crmn_agent.experiment import fingerprint

    packages = {name: importlib.metadata.version(name) for name in ('textcraft', 'alfworld', 'textworld')}
    if packages['textcraft'] != '0.0.3' or packages['alfworld'] != '0.4.2':
        raise ValueError('Install the recorded TextCraft 0.0.3 and ALFWorld 0.4.2 versions')
    directory = Path(str(importlib.resources.files('textcraft') / 'data'))
    state = hashlib.sha256()
    # Preserve the local enumeration order as well as bytes: upstream uses
    # unsorted directory iteration and set iteration to construct recipe sheets.
    for path in directory.rglob('*'):
        if path.is_file():
            state.update(str(path.relative_to(directory)).encode('utf-8'))
            state.update(path.read_bytes())
    return {'python_version': platform.python_version(), 'python_implementation': platform.python_implementation(),
            'packages': packages, 'textcraft_data_content_and_order_sha256': state.hexdigest(),
            'public_source_sha256': fingerprint()}


def validate_plan(plan: dict):
    if plan.get('schema_version') != 'feedback-native-run-plan-v1':
        raise ValueError('Unsupported recorded native plan')
    if plan.get('variants') != VARIANTS or plan.get('horizons') != HORIZONS or plan.get('episode_call_cap') != 64:
        raise ValueError('Recorded protocol must retain five variants, 100-action horizons and a 64-call cap')
    if plan.get('reasoning_effort') != 'low' or not isinstance(plan.get('model'), str):
        raise ValueError('Recorded model and reasoning settings are missing or changed')
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Recorded plan must contain tasks')
    keys, identities = set(), set()
    for task in tasks:
        key, benchmark = task.get('key', ''), task.get('benchmark')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', key) or key in keys:
            raise ValueError('Task keys must be unique safe filenames')
        if benchmark not in HORIZONS or type(task.get('seed')) is not int or task['seed'] < 0:
            raise ValueError('Unknown environment or invalid recorded seed')
        if not isinstance(task.get('task_id'), str) or not task['task_id']:
            raise ValueError('Recorded task identity is missing')
        identity = (benchmark, task['task_id'])
        if identity in identities:
            raise ValueError('Repeated task identities would change the study unit')
        keys.add(key)
        identities.add(identity)
        required = ['initial_observation_sha256']
        required.append('game_sha256' if benchmark == 'alfworld' else 'public_background_sha256')
        for name in required:
            if not isinstance(task.get(name), str) or SHA256.fullmatch(task[name]) is None:
                raise ValueError('A required task-source hash is missing: ' + name)
        if {'config_file', 'mission', 'initial_observation'} & task.keys():
            raise ValueError('Use the sanitized public plan, not the private plan with local paths or source text')
    return tasks


def provider_settings(path: Path, plan: dict) -> dict:
    """Read configuration only; never follow credential, ledger or checkpoint paths."""
    if path.name in {'auth.json', '.env'}:
        raise ValueError('Supply a provider configuration, never an account credential file')
    value = json.loads(path.read_text(encoding='utf-8'))
    provider = dict(value.get('provider', value))
    if provider.get('backend') != 'codex' or provider.get('allow_paid', False):
        raise ValueError('Recorded study reproduction uses the explicitly authorized subscription backend')
    if provider.get('model') != plan['model'] or provider.get('reasoning_effort', 'low') != 'low':
        raise ValueError('Model/reasoning settings differ from the recorded study; report a separate model experiment')
    recorded_guards = RESOURCE_PROFILES['recorded-study']
    if any(provider.get(name) != expected for name, expected in recorded_guards.items()):
        raise ValueError('Per-process provider guards differ from the recorded study. Create a new owner-authorized '
                         'configuration with --resource-profile recorded-study; this does not raise or reset '
                         'the persistent launch allowance')
    # Do not enable access, increase a limit, create an allowance or inspect it.
    # Relative deployment paths are interpreted in the caller's current directory.
    for key in ('subscription_ledger', 'subscription_checkpoint', 'isolated_dir', 'study_cache_dir'):
        if provider.get(key):
            provider[key] = str(Path(provider[key]).resolve())
    return provider


def reconstruct(task: dict, alfworld_data: Path, environment_factory, fresh_replication=False):
    benchmark = task['benchmark']
    config = {'split': 'test', 'alfworld_data': str(alfworld_data)}
    environment = environment_factory(benchmark, 'default', task['seed'], config)
    try:
        observed = environment.reset()
        if environment.task_id != task['task_id']:
            raise ValueError('Recorded task ID differs after reset: ' + task['key'])
        if text_hash(observed.text) != task['initial_observation_sha256']:
            raise ValueError('Initial public observation hash differs: ' + task['key'])
        if task.get('mission_sha256') and text_hash(observed.mission) != task['mission_sha256']:
            raise ValueError('Recorded mission hash differs: ' + task['key'])
        background_hash = text_hash(observed.background)
        background_matches = True
        if benchmark == 'textcraft':
            # In the adapter, TextCraft task_id is exactly the mission SHA256.
            if text_hash(observed.mission) != task['task_id']:
                raise ValueError('TextCraft mission identity differs: ' + task['key'])
            background_matches = background_hash == task['public_background_sha256']
            if not background_matches and not fresh_replication:
                raise ValueError('Public recipe/rule background hash differs: ' + task['key'])
        else:
            game = Path(environment.game).resolve()
            if not game.is_relative_to(alfworld_data) or digest(game.read_bytes()) != task['game_sha256']:
                raise ValueError('Official local game asset hash differs: ' + task['key'])
            if observed.background:
                raise ValueError('Recorded ALFWorld adapter had no additional background text')
        # Source text stays local. It is reconstructed from installed upstream
        # environments, never redistributed inside the public plan or fixtures.
        local_task = {**task, 'mission': observed.mission, 'mission_sha256': text_hash(observed.mission),
                      'initial_observation': observed.text}
        if benchmark == 'textcraft' and fresh_replication:
            local_task.update(recorded_selection_background_sha256=task['public_background_sha256'],
                              public_background_sha256=background_hash)
        return local_task, {
            'key': task['key'], 'benchmark': benchmark, 'seed': task['seed'], 'task_id': environment.task_id,
            'initial_observation_sha256': text_hash(observed.text), 'mission_sha256': text_hash(observed.mission),
            'public_background_sha256': background_hash, 'verified': True,
            'matches_recorded_selection_background': background_matches,
        }
    finally:
        environment.close()


def prepare(plan_path: Path, provider_path: Path, output: Path, alfworld_data: Path,
            environment_factory=make_env, fresh_replication=False):
    """Validate all tasks, then atomically create a fresh local run directory."""
    plan_path, provider_path = Path(plan_path), Path(provider_path)
    output, alfworld_data = Path(output).resolve(), Path(alfworld_data).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refusing to overwrite an existing preparation or experiment')
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    tasks = validate_plan(plan)
    provider = provider_settings(provider_path, plan)
    if fresh_replication:
        require_hashseed_zero()
    identity = runtime_identity()
    verified, verification = [], []
    for task in tasks:
        entry, evidence = reconstruct(task, alfworld_data, environment_factory, fresh_replication)
        verified.append(entry)
        verification.append(evidence)
    # No config is written until every official task/input hash has passed.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prepare-native-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'tree'
        stage.mkdir()
        (stage / 'configs').mkdir()
        manifest_path = output / 'native_manifest.json'
        manifest_bytes = encoded({
            'schema_version': 'locally-reconstructed-recorded-native-v1', 'tasks': verified,
            'recorded_public_plan_sha256': digest(plan_bytes),
        })
        (stage / 'native_manifest.json').write_bytes(manifest_bytes)
        order = Counter()
        commands, configurations = [], []
        for task in tasks:
            key, benchmark = task['key'], task['benchmark']
            shift = order[benchmark] % len(VARIANTS)
            order[benchmark] += 1
            variants = VARIANTS[shift:] + VARIANTS[:shift]
            config = {
                'provider': {**provider, 'study_cache_dir': str(output / 'cache')},
                'split': 'test', 'manifest': str(manifest_path), 'alfworld_data': str(alfworld_data),
                'results_dir': str(output / 'runs' / key), 'task_keys': [key], 'variants': variants,
                'horizons': HORIZONS, 'episode_call_cap': 64, 'resume_from_cache': True,
                'agent_options': {
                    'feedback': {'history_turns': 100, 'probe_policy': 'on_revision',
                                 'gate_mode': 'controlled', 'revision_only': True},
                    'grounding_only': {'history_turns': 100}, 'raw_actor': {'history_turns': 100},
                },
            }
            data = encoded(config)
            (stage / 'configs' / (key + '.json')).write_bytes(data)
            relative = 'configs/' + key + '.json'
            configurations.append({'key': key, 'benchmark': benchmark, 'path': relative,
                                   'sha256': digest(data), 'variants': variants})
            commands.append({'environment': {'PYTHONHASHSEED': '0'},
                             'argv': ['python', 'scripts/run_recorded_native.py', '--preparation',
                                      str(output / 'preparation.json'), '--task-key', key]})
        record = {
            'schema_version': 'recorded-native-local-preparation-v1', 'new_model_calls': 0,
            'allowances_or_authentication_modified': False, 'paid_api': False,
            'recorded_public_plan_sha256': digest(plan_bytes),
            'local_manifest_sha256': digest(manifest_bytes),
            'recorded_private_plan_sha256': plan.get('original_plan_sha256'),
            'recorded_source_sha256': plan.get('source_sha256'),
            'model': provider['model'], 'reasoning_effort': provider.get('reasoning_effort', 'low'),
            'per_process_provider_guards': {name: provider[name] for name in RESOURCE_PROFILES['recorded-study']},
            'fresh_replication': bool(fresh_replication), 'runtime_identity': identity,
            'preparation_matches_recorded_selection_backgrounds': all(
                evidence['matches_recorded_selection_background'] for evidence in verification),
            'tasks': verification, 'configurations': configurations, 'commands': commands,
            'execution_notice': 'Commands consume independently authorized subscription resources. '
                                'No automatic launching, retries, allowance creation or extension.',
            'comparability': 'Same recorded selected tasks and inference policies; local paths, private manifests '
                             'and public transport adaptation change file/source hashes. Fresh mode is goal-matched '
                             'replication with newly realized distractors, not exact study-input reproduction. '
                             'Execution rechecks the locally prepared inputs before inference; responses are not deterministic.',
        }
        (stage / 'preparation.json').write_bytes(encoded(record))
        stage.rename(output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--provider-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--alfworld-data', type=Path, default=Path('data/alfworld'))
    parser.add_argument('--fresh-replication', action='store_true',
                        help='Explicitly permit newly realized TextCraft distractors for the same recorded goal; '
                             'requires PYTHONHASHSEED=0 and does not claim original-input reproduction')
    args = parser.parse_args()
    result = prepare(args.plan, args.provider_config, args.out, args.alfworld_data,
                     fresh_replication=args.fresh_replication)
    print(json.dumps({'verified_tasks': len(result['tasks']), 'new_model_calls': 0,
                      'preparation': str(args.out / 'preparation.json')}))


if __name__ == '__main__':
    main()
