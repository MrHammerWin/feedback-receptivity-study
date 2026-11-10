"""Freeze one locally generated public task for a small explicitly funded smoke run."""

import argparse
import json
from pathlib import Path

from crmn_agent.environments import make_env


def prepare(provider_config: Path, output: Path, benchmark: str, seed: int):
    """Environment setup only; this function never authenticates or calls a model."""
    if benchmark not in {'textcraft', 'alfworld'}:
        raise ValueError('Choose textcraft or alfworld')
    output = Path(output)
    manifest = output.with_suffix('.manifest.json')
    if output.exists() or manifest.exists():
        raise FileExistsError('Refusing to overwrite an existing smoke configuration or task manifest')
    provider = json.loads(Path(provider_config).read_text())['provider']
    config = {'provider': provider, 'split': 'test', 'manifest': str(manifest),
              'results_dir': str(output.parent / (output.stem + '-private-results')),
              'alfworld_data': 'data/alfworld', 'episode_call_cap': 50,
              'horizons': {'textcraft': 50, 'alfworld': 50},
              'variants': ['raw_actor', 'feedback']}
    env = make_env(benchmark, 'default', seed, config)
    try:
        env.reset()
        task = {'key': f'{benchmark}-local-smoke-{seed}', 'benchmark': benchmark,
                'seed': seed, 'task_id': env.task_id}
    finally:
        env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'tasks': [task], 'purpose': 'local smoke; not a powered comparison'}, indent=2))
    output.write_text(json.dumps(config, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider-config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--benchmark', choices=['textcraft', 'alfworld'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(prepare(args.provider_config, args.out, args.benchmark, args.seed))
