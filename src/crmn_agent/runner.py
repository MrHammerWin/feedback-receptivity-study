import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

from .agents import VARIANTS, Agent
from .controller import Prior
from .environments import make_env, tasks_for
from .providers import BudgetExceeded, EpisodeBudgetExceeded, ModelOutputError, make_provider


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def validate_config(config: dict):
    if config.get('provider', {}).get('backend') not in {'mlx', 'openai', 'codex'}:
        raise ValueError('An explicit supported model backend is required')
    if config.get('split') not in {'train', 'validation', 'test'}:
        raise ValueError('Explicit train, validation, or test split is required')
    if not config.get('seeds') or len(config['seeds']) != len(set(config['seeds'])):
        raise ValueError('A nonempty set of unique seeds is required')
    if any(v not in VARIANTS for v in config.get('variants', [])):
        raise ValueError('Unknown agent variant')
    if not config.get('variants') or config.get('max_steps', 0) <= 0:
        raise ValueError('Variants and a positive episode horizon are required')
    if any(v <= 0 for v in config.get('episode_steps', {}).values()):
        raise ValueError('Per-benchmark step limits must be positive')
    if config.get('max_tokens_per_episode') is not None and config['max_tokens_per_episode'] <= 0:
        raise ValueError('Per-episode token limit must be positive')
    if any(key in config['provider'] for key in ['api_key', 'token', 'password', 'secret', 'authorization']):
        raise ValueError('Never put secrets in experiment configuration; use protected environment variables')
    if config['provider']['backend'] == 'codex' and not config['provider'].get('allow_subscription'):
        raise BudgetExceeded('Subscription launches require explicit authorization')
    if config['split'] == 'test' and not config.get('frozen_manifest'):
        raise ValueError('Final testing requires a frozen task manifest, not a seed-only split')
    if config['provider']['backend'] == 'openai':
        if not config['provider'].get('allow_paid') or config['provider'].get('max_usd', 0) <= 0:
            raise BudgetExceeded('Paid calls disabled. No network request was made.')


def run(config_path: Path, resume=False):
    config = json.loads(config_path.read_text())
    validate_config(config)
    out = Path(config['results_dir'])
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    prior_digest = (hashlib.sha256(Path(config['prior_path']).read_bytes()).hexdigest()
                    if config.get('prior_path') else None)
    manifest_path = out / 'run.json'
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not resume:
            raise ValueError('Results directory already contains a run; use --resume or a new directory')
        if previous['config_sha256'] != digest or previous['code_sha256'] != code_hash():
            raise ValueError('Resume refused: configuration or source changed')
        if previous.get('prior_sha256') != prior_digest:
            raise ValueError('Resume refused: trained prior checkpoint changed')
    elif resume:
        raise ValueError('Nothing to resume')
    packages = {}
    for name in ['pydantic', 'numpy', 'openai', 'mlx-lm', 'outlines', 'textcraft', 'minigrid', 'alfworld', 'textworld']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = previous or {
        'created_at_utc': datetime.now(UTC).isoformat(), 'config': config,
        'config_sha256': digest, 'code_sha256': code_hash(),
        'python': sys.version, 'platform': platform.platform(), 'packages': packages,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'prior_sha256': prior_digest,
        'is_real_llm': True, 'study_status': 'engineering_pilot_not_SOTA_evidence',
    }
    write_json(manifest_path, manifest)
    snapshot = out / 'source_snapshot.zip'
    if not snapshot.exists():
        with ZipFile(snapshot, 'w') as zipped:
            for source in sorted(Path(__file__).parent.glob('*.py')):
                zipped.write(source, arcname='crmn_agent/' + source.name)
    episode_path = out / 'episodes.jsonl'
    done_keys = set()
    if episode_path.exists():
        done_keys = {(r['benchmark'], r['task'], r['seed'], r['variant']) for r in
                     [json.loads(line) for line in episode_path.read_text().splitlines() if line.strip()]}
    provider = make_provider(config['provider'], out / 'model_calls.jsonl')
    if previous:
        # Restore GLOBAL usage, including calls in a partially completed episode.
        calls = out / 'model_calls.jsonl'
        if calls.exists():
            records = [json.loads(s) for s in calls.read_text().splitlines() if s.strip()]
            if records:
                for key, value in records[-1]['usage_cumulative'].items():
                    setattr(provider.usage, key, value)
    prior = Prior.load(Path(config['prior_path'])) if config.get('prior_path') else Prior()
    frozen = None
    if config.get('frozen_manifest'):
        frozen = json.loads(Path(config['frozen_manifest']).read_text())
        if frozen['split'] != config['split']:
            raise ValueError('Frozen task manifest split mismatch')
    for benchmark in config['benchmarks']:
        for task in tasks_for(benchmark, config):
            for seed in config['seeds']:
                for variant in config['variants']:
                    key = (benchmark, task, seed, variant)
                    if key in done_keys:
                        continue
                    # Reset and construct an isolated environment for EVERY paired variant.
                    env = make_env(benchmark, task, seed, config)
                    try:
                        obs = env.reset()
                        if frozen:
                            expected = frozen['tasks'][f'{benchmark}/{task}/{seed}']
                            if expected != env.task_id:
                                raise ValueError('Task identity changed since the manifest was frozen')
                        agent = Agent(provider, variant, prior, config.get('history_turns', 8))
                        before = asdict(provider.usage)
                        provider.begin_episode(config.get('max_tokens_per_episode'))
                        started = time.monotonic()
                        status, error_type, success, steps = 'horizon', None, False, 0
                        trajectory = []
                        horizon = config.get('episode_steps', {}).get(benchmark, config['max_steps'])
                        for step in range(horizon):
                            try:
                                turn = agent.act(obs.text, obs.mission, obs.actions, step, obs.background)
                                new_obs = env.step(turn.action)
                                agent.feedback(new_obs.success, new_obs.terminated)
                                trajectory.append({
                                    'step': step, 'observation': obs.text, 'mission': obs.mission,
                                    'background': obs.background,
                                    'action': turn.action, 'next_observation': new_obs.text,
                                    'reward': new_obs.reward, 'done': new_obs.terminated,
                                    'diagnostics': turn.diagnostics,
                                })
                                obs = new_obs
                                steps += 1
                                success = obs.success
                                if obs.terminated:
                                    status = 'success' if success else 'terminated'
                                    break
                            except EpisodeBudgetExceeded:
                                status = 'episode_budget'
                                break
                            except BudgetExceeded:
                                status = 'budget'
                                break
                            except ModelOutputError as exc:
                                status, error_type = 'error', type(exc).__name__
                                break
                        usage = {k: v - before[k] for k, v in asdict(provider.usage).items()}
                        record = {
                            'benchmark': benchmark, 'task': task, 'task_id': env.task_id,
                            'protocol': env.protocol, 'seed': seed, 'variant': variant,
                            'split': config['split'], 'success': bool(success), 'steps': steps,
                            'status': status, 'error_type': error_type, 'usage': usage,
                            'wall_seconds': time.monotonic() - started, 'model': provider.model,
                            'is_real_llm': provider.is_real_llm, 'trajectory': trajectory,
                        }
                        with episode_path.open('a', encoding='utf-8') as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                            handle.flush()
                        print(f'{benchmark}/{task}/{seed}/{variant}: {status}, steps={steps}, '
                              f'calls={usage["calls"]}', flush=True)
                        if status == 'budget':
                            manifest['finished_status'] = 'budget_exhausted'
                            write_json(manifest_path, manifest)
                            return episode_path
                    finally:
                        env.close()
    manifest['finished_status'] = 'completed'
    manifest['total_usage'] = asdict(provider.usage)
    write_json(manifest_path, manifest)
    return episode_path


def freeze_tasks(config_path: Path, output: Path):
    config = json.loads(config_path.read_text())
    tasks = {}
    for benchmark in config['benchmarks']:
        for task in tasks_for(benchmark, config):
            for seed in config['seeds']:
                env = make_env(benchmark, task, seed, config)
                try:
                    env.reset()
                    tasks[f'{benchmark}/{task}/{seed}'] = env.task_id
                finally:
                    env.close()
    write_json(output, {'split': config['split'], 'tasks': tasks,
                        'warning': 'Validate task ID disjointness against ALL other split manifests'})

