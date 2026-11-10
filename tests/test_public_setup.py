import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def script(name):
    path = Path(__file__).parents[1] / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name.removesuffix('.py'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_data_mismatch_never_writes_a_cache(tmp_path, monkeypatch):
    module = script('fetch_qa_data.py')
    monkeypatch.setattr(module, 'PARQUET_SHA256', hashlib.sha256(b'synthetic fixture').hexdigest())
    output = tmp_path / 'data' / 'fixture.parquet'
    with pytest.raises(ValueError, match='differs'):
        module.prepare(output, fetch=lambda: b'wrong fixture')
    assert not output.exists()
    assert module.prepare(output, fetch=lambda: b'synthetic fixture') == output
    assert output.read_bytes() == b'synthetic fixture'
    module.prepare(output, fetch=lambda: pytest.fail('An existing verified file should not be fetched again'))
    output.write_bytes(b'modified')
    with pytest.raises(ValueError, match='not overwritten'):
        module.prepare(output, fetch=lambda: pytest.fail('No network fallback'))


def test_task_smoke_freezes_identity_without_calling_models(tmp_path, monkeypatch):
    module = script('prepare_smoke.py')
    closed = []

    class Environment:
        task_id = 'synthetic-identity'

        def reset(self):
            return None

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, 'make_env', lambda *args: Environment())
    config = tmp_path / 'provider.json'
    config.write_text(json.dumps({'provider': {'backend': 'codex', 'model': 'fixture'}}))
    output = tmp_path / 'native.json'
    module.prepare(config, output, 'textcraft', 3)
    manifest = json.loads(output.with_suffix('.manifest.json').read_text())
    assert manifest['tasks'][0]['task_id'] == 'synthetic-identity'
    assert manifest['tasks'][0]['seed'] == 3
    assert closed == [True]
    assert json.loads(output.read_text())['variants'] == ['raw_actor', 'feedback']
    with pytest.raises(FileExistsError):
        module.prepare(config, output, 'textcraft', 3)
