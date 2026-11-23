"""Published TextCraft baselines on the 40 frozen retest goals."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from .asset_download import resolve_assets
from .environments import OPEN_COMMANDS_NOTE, make_env
from .providers import (
    BackendTransportError,
    BudgetExceeded,
    EpisodeBudgetExceeded,
    ModelOutputError,
)
from .public_authorization import RESOURCE_PROFILES
from .published_baselines import PublishedAgent


VARIANTS = ('stateact', 'reflact_paper')
RETEST_VARIANTS = ('feedback', 'feedback_open_commands', 'grounding_only', 'raw_actor')
CAMPAIGN = 'textcraft-published-baselines'
LIMIT = 2560
EPISODE_CAP = 32
MODEL = 'gpt-5.6-luna'
RETEST_MANIFEST_SHA256 = 'f34c5c929e10b27a0ad6bc7976497803f6b6696c3e6e9ccaed5fa0e9e635fc18'
RETEST_AUDIT_SHA256 = '52edee2371638810daf9d3bdf75a53cc9cdcd381f00eabd19d99946d2934b144'
PUBLISHED_PLAN_SHA256 = 'fea3c727a5b6fe439f9455f09d5f34f76cd600544caef6dd82d381d032de798e'
PILOT_ROLES = {'prior_main', 'pilot_2026_11_08', 'engineering_2026_11_10', 'early_crmn_reconciled'}
SHA256 = re.compile(r'[0-9a-f]{64}')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def initial_inputs_sha256(env, obs):
    inputs = [obs.text, obs.mission, obs.actions, env.recipe_text]
    return sha256(json.dumps(inputs, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def validate_design(tasks, variants=VARIANTS):
    if not isinstance(tasks, list) or len(tasks) != 40:
        raise ValueError('Exactly forty distinct TextCraft goals are required')
    keys, ids = set(), set()
    for task in tasks:
        key, task_id = task['key'], task['task_id']
        if (task['benchmark'] != 'textcraft' or type(task['seed']) is not int or task['seed'] < 0
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', key)
                or key in keys or task_id in ids or not SHA256.fullmatch(task_id)
                or not SHA256.fullmatch(task['initial_inputs_sha256'])
                or sorted(task['variants']) != sorted(variants)):
            raise ValueError('Invalid TextCraft task, duplicate goal/key or unapproved variant order')
        keys.add(key)
        ids.add(task_id)
    return ids


def verified_pilot_ids(data, role):
    if data.get('role') != role or data.get('coverage_verified') is not True or not data.get('rows'):
        raise ValueError('Pilot export lacks verified source coverage')
    ids = set()
    for row in data['rows']:
        if not row.get('phase') or not row.get('source_file'):
            raise ValueError('Every pilot goal must reference its source record')
        old_id = row.get('task_id')
        if old_id:
            if not SHA256.fullmatch(old_id):
                raise ValueError('Invalid pilot goal digest')
            ids.add(old_id)
        item = row.get('item_id', '').removeprefix('minecraft:')
        if row.get('mission'):
            match = re.fullmatch(r'craft ([a-z0-9 _]+)\.?', row['mission'].strip().lower())
            if not match:
                raise ValueError('Pilot goal must identify the crafted item')
            from_mission = '_'.join(match[1].replace('_', ' ').split())
            if item and item != from_mission:
                raise ValueError('Pilot goal and crafted item disagree')
            item = from_mission
        if item:
            if not re.fullmatch(r'[a-z0-9_]+', item):
                raise ValueError('Invalid pilot goal item')
            ids.add(sha256(('craft ' + item.replace('_', ' ') + '.').encode('utf-8')))
        if not (old_id or item):
            raise ValueError('A seed is not enough to identify a pilot goal')
    return ids


def verify_pilot_audit(audit_path, manifest_hash, selected_ids):
    path = Path(audit_path).resolve(strict=True)
    audit = read_json(path)
    if audit.get('status') != 'complete' or audit.get('manifest_sha256') != manifest_hash:
        raise ValueError('Pilot reconciliation is incomplete or for a different manifest')
    ids = audit['excluded_task_ids']
    if not isinstance(ids, list) or not all(isinstance(i, str) and SHA256.fullmatch(i) for i in ids):
        raise ValueError('Invalid pilot exclusion list')
    excluded = set(ids)
    if selected_ids & excluded:
        raise ValueError('Selected goals overlap prior runs')
    seen, derived = set(), set()
    for source in audit['sources']:
        role, relative = source['role'], Path(source['path'])
        if role not in PILOT_ROLES or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid role or relative path for pilot evidence')
        actual = (path.parent / relative).resolve(strict=True)
        if not actual.is_relative_to(path.parent):
            raise ValueError('Pilot evidence path must remain inside the audit directory')
        content = actual.read_bytes()
        if not content or sha256(content) != source['sha256']:
            raise ValueError('Pilot evidence changed or is empty')
        if role == 'prior_main':
            if source['sha256'] != PUBLISHED_PLAN_SHA256:
                raise ValueError('Use the pinned published Native task plan as prior_main evidence')
            ids = {row['task_id'] for row in json.loads(content)['tasks'] if row['benchmark'] == 'textcraft'}
            if len(ids) != 8 or not ids <= excluded:
                raise ValueError('Some prior primary TextCraft tasks are missing from the exclusions')
        else:
            ids = verified_pilot_ids(json.loads(content), role)
        derived.update(ids)
        seen.add(role)
    if seen != PILOT_ROLES:
        raise ValueError('The pilot, engineering, early CRMN and published main goals must be reconciled')
    if derived != excluded:
        raise ValueError('The exclusion list does not match the audited pilot and published sources')
    return sha256(path.read_bytes())


def load_study(manifest_path, audit_path):
    content = Path(manifest_path).read_bytes()
    manifest_hash = sha256(content)
    if manifest_hash != RETEST_MANIFEST_SHA256:
        raise ValueError('The approved frozen 40-goal manifest is required')
    original = json.loads(content)['tasks']
    ids = validate_design(original, RETEST_VARIANTS)
    audit_hash = verify_pilot_audit(audit_path, manifest_hash, ids)
    if audit_hash != RETEST_AUDIT_SHA256:
        raise ValueError('The approved frozen pilot audit is required')
    counts = {2: 0, 3: 0}
    tasks = []
    # Preserve the original goal order and alternate the first baseline within each depth.
    for row in original:
        depth = row['depth']
        order = VARIANTS if counts[depth] % 2 == 0 else VARIANTS[::-1]
        counts[depth] += 1
        tasks.append({**row, 'variants': list(order)})
    if counts != {2: 20, 3: 20} or validate_design(tasks) != ids:
        raise ValueError('The paired baseline design must retain all forty approved goals')
    return tasks, manifest_hash, audit_hash


def check_provider(config):
    provider = config['provider']
    expected = {
        'backend': 'codex', 'model': MODEL, 'reasoning_effort': 'low',
        'allow_subscription': True, 'allow_paid': False, 'subscription_campaign': CAMPAIGN,
        'subscription_max_launches': LIMIT, 'authorized_campaigns': {CAMPAIGN: LIMIT},
        'stop_on_transport_error': True, 'resource_profile': 'recorded-study',
        **RESOURCE_PROFILES['recorded-study'],
    }
    if any(type(provider.get(k)) is not type(v) or provider[k] != v for k, v in expected.items()):
        raise ValueError('The separate TextCraft subscription-only authorization is required')
    if provider.keys() - (expected.keys() | {
        'subscription_ledger', 'subscription_checkpoint', 'subscription_min_interval_seconds',
        'isolated_dir', 'study_cache_dir', 'timeout_seconds',
    }):
        raise ValueError('Unrecognized provider setting')
    ledger = Path(provider['subscription_ledger']).resolve(strict=True)
    checkpoint = Path(provider['subscription_checkpoint']).resolve(strict=True)
    output = Path(config['results_dir']).resolve()
    if (not output.is_relative_to(ledger.parent) or output == ledger.parent
            or checkpoint.parent != ledger.parent or checkpoint == ledger):
        raise ValueError('Results and checkpoint must be inside the separate TextCraft authorization directory')
    allowance = read_json(ledger)
    if (allowance.get('campaign') != CAMPAIGN or allowance.get('limit') != LIMIT
            or allowance.get('paid_api') is not False):
        raise ValueError('Wrong allowance ledger for the TextCraft retest')
    stop = read_json(checkpoint).get('stop_after')
    if type(stop) is not int or not 0 < stop <= LIMIT:
        raise ValueError('The new campaign checkpoint is closed or invalid')
    return {**provider, 'max_calls_per_episode': EPISODE_CAP}, output


def start_env(task, variant):
    if variant not in VARIANTS:
        raise ValueError('Only StateAct and ReflAct can run in this campaign')
    return make_env('textcraft', 'default', task['seed'], {'textcraft_open_commands_note': True})


def verified_observation(env, task, variant):
    obs = env.reset()
    if (env.task_id != task['task_id'] or env.task_id != sha256(obs.mission.encode('utf-8'))
            or initial_inputs_sha256(env, obs) != task['initial_inputs_sha256']):
        raise ValueError(f'Frozen TextCraft inputs differ for {task["key"]} / {variant}')
    if not obs.background.startswith(OPEN_COMMANDS_NOTE):
        raise ValueError('Both baselines must receive the shared TextCraft command note')
    return obs, obs.background.removeprefix(OPEN_COMMANDS_NOTE)


def preflight_tasks(tasks):
    # Compare every arm before authenticating a provider, then recheck its own reset before agent.act.
    baselines = {}
    for task in tasks:
        for variant in task['variants']:
            env = start_env(task, variant)
            try:
                _, background = verified_observation(env, task, variant)
                background_hash = sha256(background.encode('utf-8'))
                if baselines.setdefault(task['key'], background_hash) != background_hash:
                    raise ValueError('TextCraft backgrounds differ beyond the tested note')
            finally:
                env.close()
    return baselines


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def fingerprint():
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.rglob('*')):
        if p.is_file() and p.suffix in {'.py', '.txt', '.json'} and '__pycache__' not in p.parts:
            digest.update(str(p.relative_to(Path(__file__).parent)).encode())
            digest.update(p.read_bytes())
    return digest.hexdigest()


def run(config_path):
    config_path = Path(config_path)
    config = read_json(config_path)
    if set(config) != {'provider', 'manifest', 'pilot_audit', 'results_dir'}:
        raise ValueError('Expected a frozen TextCraft configuration without overrides')
    if (os.environ.get('PYTHONHASHSEED') != '0' or sys.implementation.name != 'cpython'
            or sys.flags.hash_randomization != 0):
        raise ValueError('Set PYTHONHASHSEED=0 before starting CPython')
    tasks, manifest_hash, audit_hash = load_study(config['manifest'], config['pilot_audit'])
    design_hash = sha256(json.dumps(tasks, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    provider_config, output = check_provider(config)
    packages = {name: importlib.metadata.version(name) for name in ('pydantic', 'numpy', 'textcraft', 'gymnasium')}
    if packages['textcraft'] != '0.0.3':
        raise ValueError('TextCraft 0.0.3 is required for the retest')
    resolve_assets()
    bases = preflight_tasks(tasks)
    source_hash = fingerprint()
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / 'run.json'
    config_hash = sha256(config_path.read_bytes())
    identity = {'config_sha256': config_hash, 'source_sha256': source_hash, 'manifest_sha256': manifest_hash,
                'baseline_design_sha256': design_hash,
                'pilot_audit_sha256': audit_hash, 'packages': packages, 'python_version': sys.version,
                'initial_base_backgrounds': bases}
    if record_path.exists():
        previous = read_json(record_path)
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError('Cannot mix runs after changing code, configuration, inputs or pilot audit')
    else:
        if any(output.iterdir()):
            raise ValueError('A new, empty results directory is required')
        dump(record_path, {'created_at': datetime.now(UTC).isoformat(), 'config': config,
                           **identity, 'paid_api': False, 'episode_call_cap': EPISODE_CAP,
                           'variants': VARIANTS, 'comparison': 'Published scaffold adapters on the frozen TextCraft goals'})
    snapshot = output / 'source_snapshot.zip'
    if not snapshot.exists():
        with ZipFile(snapshot, 'w') as zipped:
            for source in sorted(Path(__file__).parent.rglob('*')):
                if source.is_file() and '__pycache__' not in source.parts and source.suffix in {'.py', '.txt', '.json'}:
                    zipped.write(source, 'crmn_agent/' + str(source.relative_to(Path(__file__).parent)))
    from .codex_provider import CodexProvider
    provider = CodexProvider(provider_config, output / 'model_calls.jsonl')
    for task in tasks:
        for variant in task['variants']:
            episode_file = output / 'episodes' / f"{task['key']}--{variant}.json"
            if episode_file.exists():
                previous = read_json(episode_file)
                if previous.get('finished') and previous.get('status') not in {
                    'campaign_budget_censored', 'infrastructure_censored'
                }:
                    continue
                raise ValueError('Interrupted episode must be explicitly adjudicated, not silently repeated')
            env = start_env(task, variant)
            started = time.monotonic()
            before = asdict(provider.usage)
            provider.begin_episode()
            try:
                obs, background = verified_observation(env, task, variant)
                if sha256(background.encode('utf-8')) != bases[task['key']]:
                    raise ValueError('The actual episode background differs from its verified starting frame')
                agent = PublishedAgent(provider, variant, task['benchmark'], task['task_id'])
                episode = {'key': task['key'], 'benchmark': task['benchmark'], 'seed': task['seed'],
                           'task_id': env.task_id, 'variant': variant, 'protocol': env.protocol,
                           'initial_inputs_sha256': task['initial_inputs_sha256'],
                           'initial_base_background_sha256': bases[task['key']],
                           'model': provider.model, 'source_sha256': source_hash,
                           'success': False, 'status': 'running', 'finished': False, 'trajectory': []}
                dump(episode_file, episode)
                for step in range(100):
                    used = provider.usage.calls - before['calls']
                    if used >= EPISODE_CAP:
                        episode['status'] = 'episode_compute_cap'
                        break
                    try:
                        step_before = asdict(provider.usage)
                        turn = agent.act(obs.text, obs.mission, obs.actions, step, obs.background)
                        nxt = env.step(turn.action)
                        agent.feedback(nxt.success, nxt.terminated)
                        episode['trajectory'].append({
                            'step': step, 'observation': obs.text, 'mission': obs.mission,
                            'background': obs.background, 'available_actions': obs.actions,
                            'action': turn.action, 'next_observation': nxt.text,
                            'reward': nxt.reward, 'terminated': nxt.terminated,
                            'success': bool(nxt.success),
                            'diagnostics': turn.diagnostics,
                            'usage': {k: v - step_before[k] for k, v in asdict(provider.usage).items()},
                            'end_logical_calls': provider.usage.calls - before['calls'],
                        })
                        episode['success'] = bool(nxt.success)
                        obs = nxt
                        dump(episode_file, episode)
                        if obs.terminated:
                            episode['status'] = 'success' if obs.success else 'terminated'
                            break
                    except EpisodeBudgetExceeded:
                        episode['status'] = 'episode_compute_cap'
                        break
                    except BudgetExceeded:
                        episode['status'] = 'campaign_budget_censored'
                        break
                    except ModelOutputError as exc:
                        episode['status'] = 'model_error'
                        episode['error'] = str(exc)
                        break
                    except BackendTransportError as exc:
                        episode['status'] = 'infrastructure_censored'
                        episode['error'] = str(exc)
                        break
                else:
                    episode['status'] = 'action_horizon'
                episode['steps'] = len(episode['trajectory'])
                episode['usage'] = {k: v - before[k] for k, v in asdict(provider.usage).items()}
                completed_calls = sum(t.get('usage', {}).get('calls', 0) for t in episode['trajectory'])
                episode['unexecuted_turn_calls'] = episode['usage']['calls'] - completed_calls
                episode['execution_wall_seconds'] = time.monotonic() - started
                episode['historical_replay_seconds'] = 0.0
                episode['wall_seconds'] = episode['execution_wall_seconds']
                episode['response_reuse'] = False
                episode['finished'] = True
                dump(episode_file, episode)
                print(json.dumps({k: episode[k] for k in ('key', 'variant', 'success', 'status', 'steps', 'usage')}), flush=True)
                if episode['status'] in {'campaign_budget_censored', 'infrastructure_censored'}:
                    raise RuntimeError(f'TextCraft retest stopped: {episode["status"]}')
            finally:
                env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
