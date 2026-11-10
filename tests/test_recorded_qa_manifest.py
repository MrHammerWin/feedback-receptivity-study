import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def fixture(tmp_path):
    path = Path(__file__).parents[1] / 'scripts/verify_recorded_qa_manifest.py'
    spec = importlib.util.spec_from_file_location('verify_recorded_qa_manifest', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    local = {'schema_version': 'synthetic-manifest',
             'splits': {'dev': [{'id': 'synthetic-dev', 'row_sha256': 'a' * 64}],
                        'test': [{'id': 'synthetic-test', 'row_sha256': 'b' * 64}]}}
    payload = json.dumps(local, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    recorded = {'schema_version': 'synthetic-public-manifest',
                'original_manifest_sha256': hashlib.sha256(payload).hexdigest(), 'splits': local['splits']}
    local_path, recorded_path = tmp_path / 'local.json', tmp_path / 'recorded.json'
    local_path.write_text(json.dumps(local, indent=7))
    recorded_path.write_text(json.dumps(recorded))
    return module, local, recorded, local_path, recorded_path


def test_qa_manifest_comparison_uses_canonical_json_not_whitespace(tmp_path):
    module, _, _, local_path, recorded_path = fixture(tmp_path)
    assert module.verify(recorded_path, local_path) == {
        'verified': True, 'dev_n': 1, 'test_n': 1, 'new_model_calls': 0}


def test_changed_local_question_or_metadata_is_rejected(tmp_path):
    module, local, _, local_path, recorded_path = fixture(tmp_path)
    local['splits']['test'][0]['id'] = 'different-question'
    local_path.write_text(json.dumps(local))
    with pytest.raises(ValueError, match='canonical hash'):
        module.verify(recorded_path, local_path)


def test_public_id_list_tampering_is_rejected_even_with_original_hash(tmp_path):
    module, _, recorded, local_path, recorded_path = fixture(tmp_path)
    recorded['splits']['test'] = []
    recorded_path.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match='question IDs'):
        module.verify(recorded_path, local_path)
