"""Check exported scientific files and exact-response provenance without any model calls."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    evidence = (root / 'results/campaign-2026-11-08').resolve()
    hashes = json.loads((evidence / 'export_checksums.json').read_text())
    for relative, expected in hashes.items():
        path = (evidence / relative).resolve()
        if not path.is_relative_to(evidence) or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Scientific evidence changed or is missing: {relative}')
    identities, replayed = {}, 0
    for log in sorted(evidence.rglob('model_calls.jsonl')):
        for line in log.read_text().splitlines():
            row = json.loads(line)
            launch = row.get('subscription_launch')
            if launch is None:
                continue
            identity = (row['model'], row['prompt_sha256'], row['output'], row.get('backend_usage'))
            if launch in identities and identities[launch] != identity:
                raise ValueError('Conflicting original/replayed response identity')
            identities[launch] = identity
            replayed += bool(row.get('replayed'))
    prefix_count = 0
    for path in (evidence / 'replay').rglob('*.json'):
        prefix = json.loads(path.read_text())
        if prefix.get('schema_version') != 'codex-replay-prefix-v1':
            continue
        source = (path.parent / prefix['source_log']).resolve()
        if not source.is_relative_to(evidence) or digest(source) != prefix['source_log_hash']:
            raise ValueError('Historical prefix source is missing or modified')
        for row in prefix['rows']:
            original = identities[row['subscription_launch']]
            if original[:3] != (prefix['model'], row['prompt_sha256'], row['output']):
                raise ValueError('Prefix differs from its original response')
        prefix_count += 1
    summary = json.loads((evidence / 'campaign_summary.json').read_text())
    return {'files_verified': len(hashes), 'prefixes_verified': prefix_count,
            'unique_launch_ids_represented_in_this_repo': len(identities),
            'replayed_rows_in_this_repo': replayed,
            'whole_two_project_campaign_launches': summary['physical_launches_used'],
            'new_model_calls': 0,
            'scope': 'This repository contains its candidate plus shared baselines; not every call of the other candidate.'}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2))
