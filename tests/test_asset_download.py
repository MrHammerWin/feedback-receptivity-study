import ast
import hashlib
import json

import pytest

from crmn_agent import asset_download as assets


def test_literal_extraction_does_not_import_or_execute_upstream_code(tmp_path):
    marker = tmp_path / "not-created"
    source = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('bad')\nVALUE = 'literal'\n"
    assert assets.literal_assignments(source) == {"VALUE": "literal"}
    assert not marker.exists()


def test_static_expression_evaluator_rejects_arbitrary_calls():
    expression = ast.parse("__import__('os').system('untrusted')", mode="eval").body
    with pytest.raises(assets.AssetError, match="whitelist"):
        assets._static_text(expression, {})


def test_stateact_extraction_uses_only_pinned_literals(monkeypatch):
    runner = (
        b"BASE_PROMPT1 = 'fixture prefix'\nTASK_MESSAGE = 'fixture task'\nINSTRUCTIONS = ''\n"
        b"HINTS = ''\nOPENING_MARK = ''\nCLOSING_MARK = ''\n"
        b"def generate_prompt_from_example():\n"
        b"    raw_prompt = f'{base_prompt}|{examples}|{task_message}'\n"
    )
    keys = ["goal", "locations_visited", "current_location", "current_inventory", "thought", "action"]
    example = ["fixture observation", json.dumps(dict.fromkeys(keys, "fixture value"))]
    demonstration = ("put_v4_base_1 = " + repr(example) + "\nput_v4_base_0 = " + repr(example)).encode()
    craft = (
        b"atomic_exec_prompt = 'fixture prefix'\nenvironment_context = 'fixture commands'\n"
        b"atomic_examples_stateact = {'one': 'fixture demo'}\nstateact_trajectory = 'fixture trajectory'\n"
        b"if AGENT_TYPE == 'other':\n    pass\n"
        b"elif AGENT_TYPE == 'stateact':\n"
        b"    atomic_exec_prompt += '|'+ '\\n'.join(atomic_examples_stateact[k] for k in atomic_examples_stateact.keys())\n"
        b"    atomic_exec_prompt += stateact_trajectory\n"
        b"atomic_exec_prompt += '|done'\n"
    )
    sources = {"runner.py": runner, "put.py": demonstration, "craft.py": craft}
    meta = {"stateact": {"source_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in sources.items()},
                          "alfworld_runner": "runner.py", "textcraft_runner": "craft.py",
                          "keys": keys, "families": ["put"], "alfworld_sources": {"put": "put.py"},
                          "prompt_ids": [1, 0]}}
    monkeypatch.setattr(assets, "catalogue", lambda: meta)
    result = assets.extract_stateact(sources)
    assert result["stateact_textcraft_system.txt"] == b"fixture prefix|fixture demofixture trajectory|done"
    assert result["stateact_textcraft_environment_context.txt"] == b"fixture commands"
    sources["runner.py"] += b"\nraise RuntimeError('must not execute')"
    with pytest.raises(assets.AssetError, match="digest"):
        assets.extract_stateact(sources)


def test_network_source_allowlist_rejects_other_hosts_without_request():
    with pytest.raises(assets.AssetError, match="URL"):
        assets.fetch_public("https://example.invalid/source")


def test_prepare_verifies_all_outputs_before_writing_cache(monkeypatch, tmp_path):
    generated = {"fixture_a.txt": b"one", "fixture_b.txt": b"two"}
    meta = {"stateact": {"source_sha256": {"source.py": "unused"}, "raw_prefix": "https://fixture/"},
            "reflact": {"url": "https://fixture/paper"},
            "assets": {k: hashlib.sha256(v).hexdigest() for k, v in generated.items()}}
    monkeypatch.setattr(assets, "catalogue", lambda: meta)
    monkeypatch.setattr(assets, "extract_stateact", lambda source: {"fixture_a.txt": b"one"})
    monkeypatch.setattr(assets, "extract_reflact", lambda source: {"fixture_b.txt": b"wrong"})
    destination = tmp_path / "cache"
    with pytest.raises(assets.AssetError, match="differs"):
        assets.prepare_assets(destination, fetch=lambda url: b"fixture")
    assert not destination.exists()
    monkeypatch.setattr(assets, "extract_reflact", lambda source: {"fixture_b.txt": b"two"})
    assert assets.prepare_assets(destination, fetch=lambda url: b"fixture") == destination
    assert (destination / "fixture_b.txt").read_bytes() == b"two"
