"""Episode-level, paired statistics. No treating correlated steps as independent samples."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0, centre - spread), min(1, centre + spread)


def paired_bootstrap(a: list[dict], b: list[dict], seed=1729, samples=10000) -> dict:
    def index(rows):
        result = {}
        for r in rows:
            key = (r['benchmark'], r['task'], r['seed'])
            if key in result:
                raise ValueError("Duplicate episodes cannot be silently pooled")
            result[key] = r
        return result
    aa, bb = index(a), index(b)
    keys = sorted(aa.keys() & bb.keys())
    if not keys:
        raise ValueError("No paired episodes")
    differences = np.asarray([int(aa[k]['success']) - int(bb[k]['success']) for k in keys])
    rng = np.random.default_rng(seed)
    # Bounded memory even for large episode collections.
    means = np.asarray([rng.choice(differences, len(keys), replace=True).mean() for _ in range(samples)])
    interval = np.quantile(means, [0.025, 0.975]).tolist()
    return {"paired_n": len(keys), "a_only": len(aa.keys() - bb.keys()),
            "b_only": len(bb.keys() - aa.keys()), "difference": float(differences.mean()),
            "ci95": interval, "bootstrap_seed": seed, "samples": samples}


def summarize(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row['benchmark'], row['variant'])].append(row)
    summaries = []
    for (benchmark, variant), episodes in sorted(groups.items()):
        success = sum(r['success'] for r in episodes)
        lo, hi = wilson(success, len(episodes))
        summaries.append({
            "benchmark": benchmark, "variant": variant, "episodes": len(episodes),
            "success_rate": success / len(episodes), "success_ci_low": lo, "success_ci_high": hi,
            "mean_steps": sum(r['steps'] for r in episodes) / len(episodes),
            "input_tokens": sum(r['usage']['input_tokens'] for r in episodes),
            "output_tokens": sum(r['usage']['output_tokens'] for r in episodes),
            "model_calls": sum(r['usage']['calls'] for r in episodes),
            "error_episodes": sum(r['status'] == 'error' for r in episodes),
            "budget_episodes": sum(r['status'] in {'budget', 'episode_budget'} for r in episodes),
            "estimated_usd": sum(r['usage']['estimated_usd'] for r in episodes),
        })
    output = path.parent / 'summary.csv'
    if summaries:
        with output.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    paired = {}
    for (benchmark, variant), episodes in groups.items():
        if variant == 'crmn':
            for control in ['react', 'reflact', 'no_cycle', 'no_prediction', 'no_reward', 'map_only', 'plan_only']:
                if (benchmark, control) in groups:
                    paired[f'{benchmark}:crmn-minus-{control}'] = paired_bootstrap(
                        episodes, groups[(benchmark, control)])
    (path.parent / 'paired_comparisons.json').write_text(json.dumps(paired, indent=2))
    return summaries

