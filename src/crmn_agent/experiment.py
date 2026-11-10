"""Frozen-task evaluation with per-step checkpoints and source fingerprints."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from .environments import make_env
from .providers import (
    BackendTransportError,
    BudgetExceeded,
    EpisodeBudgetExceeded,
    ModelOutputError,
    make_provider,
)
from .published_baselines import PublishedAgent


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


def build_agent(provider, variant, benchmark, task_id, options):
    if variant == 'raw_actor':
        from .ablations import DraftOnlyAgent
        return DraftOnlyAgent(provider, **options)
    if variant in {'stateact', 'reflact_paper'}:
        return PublishedAgent(provider, variant, benchmark, task_id)
    if variant == 'feedback':
        from .feedback_control import FeedbackAgent
        return FeedbackAgent(provider, **options)
    if variant in {'grounding_only', 'critic_only', 'facts_only'}:
        from .feedback_control import FeedbackAgent
        mode = {'grounding_only': 'grounding_only', 'critic_only': 'always_critique',
                'facts_only': 'facts_only'}[variant]
        return FeedbackAgent(provider, **{'probe_policy': 'never', 'gate_mode': mode,
                                         'revision_only': True, **options})
    raise ValueError('Unknown experiment variant')


def run(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    if config['provider']['backend'] != 'codex':
        raise ValueError('This campaign permits only the authorized subscription backend')
    frozen = json.loads(Path(config['manifest']).read_text())
    source_hash = fingerprint()
    output = Path(config['results_dir'])
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / 'run.json'
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(Path(config['manifest']).read_bytes()).hexdigest()
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous['config_sha256'] != config_hash or previous['source_sha256'] != source_hash:
            raise ValueError('Cannot mix runs after changing code or configuration')
        if previous['manifest_sha256'] != manifest_hash:
            raise ValueError('Cannot resume after changing the frozen task manifest')
    else:
        packages = {}
        for name in ('pydantic', 'numpy', 'textcraft', 'alfworld', 'textworld'):
            packages[name] = importlib.metadata.version(name)
        dump(record_path, {'created_at': datetime.now(UTC).isoformat(), 'config': config,
                           'config_sha256': config_hash, 'source_sha256': source_hash,
                           'manifest_sha256': manifest_hash,
                           'packages': packages, 'paid_api': False,
                           'comparison': 'published scaffolds on shared model, not absolute SOTA leaderboard'})
    provider_config = {**config['provider'], 'max_calls_per_episode': config.get('episode_call_cap', 1000)}
    snapshot = output / 'source_snapshot.zip'
    if not snapshot.exists():
        with ZipFile(snapshot, 'w') as zipped:
            for source in sorted(Path(__file__).parent.rglob('*')):
                if source.is_file() and '__pycache__' not in source.parts and source.suffix in {'.py', '.txt', '.json'}:
                    zipped.write(source, 'crmn_agent/' + str(source.relative_to(Path(__file__).parent)))
    scoped = bool(provider_config.get('study_cache_dir'))
    if scoped and config.get('replay_prefixes'):
        raise ValueError('Scoped study cache cannot be combined with legacy replay')
    if scoped:
        from .study_provider import ScopedCodexProvider
        provider_config['study_source_identity'] = hashlib.sha256(
            (source_hash + manifest_hash + config_hash).encode()).hexdigest()
        provider_config['task_kind'] = 'native'
        provider = ScopedCodexProvider(provider_config, output / 'model_calls.jsonl')
    elif config.get('replay_prefixes'):
        from .replay_provider import ReplayCodexProvider
        provider = ReplayCodexProvider(provider_config, output / 'model_calls.jsonl')
    else:
        provider = make_provider(provider_config, output / 'model_calls.jsonl')
    for task in frozen['tasks']:
        if task['key'] not in config.get('task_keys', [t['key'] for t in frozen['tasks']]):
            continue
        for variant in config['variants']:
            episode_file = output / 'episodes' / f"{task['key']}--{variant}.json"
            if episode_file.exists():
                previous = json.loads(episode_file.read_text())
                resumable = previous.get('status') == 'campaign_budget_censored'
                if previous.get('finished') and not (resumable and scoped and config.get('resume_from_cache')):
                    continue
                if not (scoped and config.get('resume_from_cache') and resumable):
                    raise ValueError('Interrupted episode must be explicitly adjudicated, not silently repeated')
                archive = output / 'partial_checkpoints' / (episode_file.stem + f'--{time.time_ns()}.json')
                dump(archive, previous)
            env = make_env(task['benchmark'], 'default', task['seed'], config)
            started = time.monotonic()
            before = asdict(provider.usage)
            replay_seconds_before = getattr(provider, 'replayed_seconds', 0.0)
            if config.get('replay_prefixes'):
                key = task['key'] + '--' + variant
                prefix = config['replay_prefixes'].get(key)
                provider.set_replay_prefix(Path(prefix) if prefix else None)
            provider.begin_episode()
            try:
                obs = env.reset()
                if env.task_id != task['task_id']:
                    raise ValueError('Frozen task identity changed')
                agent = build_agent(provider, variant, task['benchmark'], env.task_id,
                                    config.get('agent_options', {}).get(variant, {}))
                episode = {'key': task['key'], 'benchmark': task['benchmark'], 'seed': task['seed'],
                           'task_id': env.task_id, 'variant': variant, 'protocol': env.protocol,
                           'model': provider.model, 'source_sha256': source_hash,
                           'success': False, 'status': 'running', 'finished': False, 'trajectory': []}
                dump(episode_file, episode)
                for step in range(config['horizons'][task['benchmark']]):
                    used = provider.usage.calls - before['calls']
                    if used >= config.get('episode_call_cap', 1000):
                        episode['status'] = 'episode_compute_cap'
                        break
                    try:
                        step_before = asdict(provider.usage)
                        if scoped:
                            provider.begin_scope(f"{task['key']}--{variant}--step-{step}", 'sequence')
                        turn = agent.act(obs.text, obs.mission, obs.actions, step, obs.background)
                        if scoped:
                            provider.assert_scope_exhausted()
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
                if config.get('replay_prefixes'):
                    provider.assert_replay_exhausted()
                episode['usage'] = {k: v - before[k] for k, v in asdict(provider.usage).items()}
                completed_calls = sum(t.get('usage', {}).get('calls', 0) for t in episode['trajectory'])
                episode['unexecuted_turn_calls'] = episode['usage']['calls'] - completed_calls
                episode['execution_wall_seconds'] = time.monotonic() - started
                episode['historical_replay_seconds'] = getattr(provider, 'replayed_seconds', 0.0) - replay_seconds_before
                episode['wall_seconds'] = episode['execution_wall_seconds'] + episode['historical_replay_seconds']
                episode['response_reuse'] = bool(config.get('replay_prefixes') or scoped)
                episode['finished'] = True
                dump(episode_file, episode)
                print(json.dumps({k: episode[k] for k in ('key', 'variant', 'success', 'status', 'steps', 'usage')}), flush=True)
                if episode['status'] in {'campaign_budget_censored', 'infrastructure_censored'}:
                    return
            finally:
                env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    run(args.config)


if __name__ == '__main__':
    main()
