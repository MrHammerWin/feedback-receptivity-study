"""Manual local authorization file; never overwrites or resets an allowance."""
import argparse
import json
import os
from pathlib import Path

CAMPAIGN = 'two-candidates-2026-11-08'


def create(path: Path, launches: int):
    if not 0 < launches <= 1000:
        raise ValueError('Choose an explicitly authorized limit from 1 to 1000')
    if path.exists() or path.with_suffix('.attempts.jsonl').exists():
        raise FileExistsError('Refusing to reset any existing authorization or journal')
    # This coursework campaign has an immutable maximum of 1000. A smaller
    # requested cap belongs in subscription_max_launches in EACH run config.
    if launches != 1000:
        raise ValueError('This campaign has a 1000 maximum; set a smaller runtime cap in configuration')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump({'campaign': CAMPAIGN, 'limit': launches,
                   'authorization': 'Manually created by the subscription owner.',
                   'paid_api': False}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--authorized-launches', type=int, required=True)
    arguments = parser.parse_args()
    create(arguments.path, arguments.authorized_launches)
    print('Local campaign created. Point BOTH projects to this same file. No model was called.')
