"""Explicit inference entry point for an independently authorized local replica.

Checks local preparation hashes and the locked Python/dependency/data identity
before delegating to the existing experiment runner. Unlike the prepare script,
this command DOES consume the owner's subscription allowance when executed.
"""

import argparse
import json
from pathlib import Path

from prepare_recorded_native import digest, reconstruct, require_hashseed_zero, runtime_identity


def run(preparation_path, task_key, *, environment_factory=None, experiment_runner=None):
    require_hashseed_zero()
    preparation_path = Path(preparation_path).resolve()
    record = json.loads(preparation_path.read_text())
    if record.get('schema_version') != 'recorded-native-local-preparation-v1':
        raise ValueError('Unsupported local preparation')
    if record.get('runtime_identity') != runtime_identity():
        raise ValueError('Python, public source, dependencies or local dataset state changed; no model was called')
    matches = [row for row in record['configurations'] if row['key'] == task_key]
    if len(matches) != 1:
        raise ValueError('Select one recorded task key')
    relative = Path(matches[0]['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Configuration path must stay inside the local preparation')
    config_path = preparation_path.parent / relative
    if digest(config_path.read_bytes()) != matches[0]['sha256']:
        raise ValueError('Local task configuration changed; no model was called')
    config = json.loads(config_path.read_text())
    manifest = json.loads(Path(config['manifest']).read_text())
    if digest(Path(config['manifest']).read_bytes()) != record['local_manifest_sha256']:
        raise ValueError('Local task manifest changed; no model was called')
    selected = [row for row in manifest['tasks'] if row['key'] == task_key]
    if len(selected) != 1:
        raise ValueError('The selected local task identity is ambiguous')
    if environment_factory is None:
        from crmn_agent.environments import make_env
        environment_factory = make_env
    # Strictly recheck the freshly prepared public inputs in this execution
    # process before even constructing a model provider. False permits no drift.
    reconstruct(selected[0], Path(config['alfworld_data']).resolve(), environment_factory, False)
    if experiment_runner is None:
        from crmn_agent.experiment import run as experiment_runner
    return experiment_runner(config_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--task-key', required=True)
    arguments = parser.parse_args()
    run(arguments.preparation, arguments.task_key)
