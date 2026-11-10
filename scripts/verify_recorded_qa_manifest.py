"""Verify a locally rebuilt full QA manifest against released ID/hash metadata."""

import argparse
import hashlib
import json
from pathlib import Path


def verify(recorded_path: Path, local_path: Path):
    recorded = json.loads(Path(recorded_path).read_text())
    local = json.loads(Path(local_path).read_text())
    payload = json.dumps(local, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(payload).hexdigest() != recorded.get('original_manifest_sha256'):
        raise ValueError('Locally rebuilt QA manifest differs from the recorded canonical hash')
    if local.get('splits') != recorded.get('splits'):
        raise ValueError('Locally rebuilt QA question IDs or source hashes differ')
    return {'verified': True, 'dev_n': len(local['splits']['dev']),
            'test_n': len(local['splits']['test']), 'new_model_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recorded', type=Path, required=True)
    parser.add_argument('--local', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.recorded, arguments.local)))
