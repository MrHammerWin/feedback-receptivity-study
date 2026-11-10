import argparse
import json
import sys
from pathlib import Path

from .analysis import summarize
from .controller import Prior
from .environments import make_env, tasks_for
from .runner import freeze_tasks, run


def main():
    parser = argparse.ArgumentParser(description='CRMN-inspired agent experiments')
    commands = parser.add_subparsers(dest='command', required=True)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--config', type=Path, required=True)
    evaluate.add_argument('--resume', action='store_true')
    analyze = commands.add_parser('analyze')
    analyze.add_argument('episodes', type=Path)
    doctor = commands.add_parser('doctor')
    doctor.add_argument('--config', type=Path, default=Path('configs/local_smoke.json'))
    freeze = commands.add_parser('freeze-tasks')
    freeze.add_argument('--config', type=Path, required=True)
    freeze.add_argument('--output', type=Path, required=True)
    fit = commands.add_parser('fit-prior')
    fit.add_argument('--episodes', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'evaluate':
            path = run(args.config, args.resume)
            print(json.dumps(summarize(path), indent=2))
        elif args.command == 'analyze':
            print(json.dumps(summarize(args.episodes), indent=2))
        elif args.command == 'freeze-tasks':
            freeze_tasks(args.config, args.output)
        elif args.command == 'fit-prior':
            rows = []
            for line in args.episodes.read_text().splitlines():
                episode = json.loads(line)
                if episode['split'] != 'train':
                    raise ValueError('Prior fitting refuses non-training episodes')
                for turn in episode['trajectory']:
                    diagnostics = turn['diagnostics']
                    if 'responsibilities' in diagnostics:
                        rows.append({**diagnostics, 'split': 'train'})
            prior = Prior()
            prior.fit(rows)
            prior.save(args.output)
            print(f'Trained prior on {len(rows)} TRAIN steps')
        elif args.command == 'doctor':
            config = json.loads(args.config.read_text())
            status = {}
            for benchmark in config['benchmarks']:
                try:
                    env = make_env(benchmark, tasks_for(benchmark, config)[0], config['seeds'][0], config)
                    try:
                        obs = env.reset()
                        assert obs.text and obs.mission
                        status[benchmark] = {'ready': True, 'task_id': env.task_id,
                                             'protocol': env.protocol}
                    finally:
                        env.close()
                except Exception as exc:
                    status[benchmark] = {'ready': False, 'error_type': type(exc).__name__}
            print(json.dumps(status, indent=2))
            if not all(v['ready'] for v in status.values()):
                return 1
    except Exception as exc:
        # Avoid accidentally printing credentials or signed URLs from third-party exceptions.
        print(f'{type(exc).__name__}: operation failed. See project setup and budget instructions.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
