"""Synthetic public prompt fixtures; upstream demonstrations are not vendored."""

import hashlib
import json

import pytest

from crmn_agent.asset_download import AssetError, catalogue, resolve_assets, validate_asset_bytes
from crmn_agent.published_baselines import PublishedAgent, ReflActReply, TextCraftStateActReply


class Scripted:
    def __init__(self):
        self.calls = []

    def complete(self, schema, instruction, payload, cap):
        self.calls.append((schema, instruction, json.loads(json.dumps(payload))))
        if schema is ReflActReply:
            return schema(reflection="Synthetic reflection.", action="inventory")
        return schema(goal="fixture", inventory="empty", think="Inspect.", action="inventory")


@pytest.fixture
def assets(tmp_path):
    for name in catalogue()["assets"]:
        (tmp_path / name).write_text("Synthetic unit-test prompt.\n")
    return tmp_path


def test_stateact_native_transport_and_full_history_without_upstream_prompts(assets):
    provider = Scripted()
    agent = PublishedAgent(provider, "stateact", "textcraft", "fixture", assets=assets)
    for step in range(12):
        agent.act(f"observation {step}", "goal", [], step)
    assert provider.calls[0][0] is TextCraftStateActReply
    assert len(provider.calls[-1][2]["full_history"]) == 11


def test_reflact_transport_is_a_paper_based_reproduction(assets):
    provider = Scripted()
    agent = PublishedAgent(provider, "reflact_paper", "alfworld", "pick_and_place_simple", assets=assets)
    agent.act("public fixture", "goal", [], 0)
    assert provider.calls[0][0] is ReflActReply
    assert set(ReflActReply.model_fields) == {"reflection", "action"}
    assert "WORKED EXAMPLE" in provider.calls[0][1]


def test_missing_default_cache_fails_without_a_download(monkeypatch, tmp_path):
    monkeypatch.setattr("crmn_agent.asset_download.default_cache", lambda: tmp_path)
    with pytest.raises(AssetError, match="asset_download"):
        resolve_assets()


def test_catalogue_has_only_pins_and_digests_not_prompt_text():
    value = catalogue()
    assert len(value["stateact"]["commit"]) == 40
    assert len(value["assets"]) == 12
    assert all(len(digest) == 64 for digest in value["assets"].values())
    assert all("system.txt" in name or name.endswith(".txt") for name in value["assets"])


def test_modified_asset_is_never_accepted_as_the_recorded_baseline():
    name = next(iter(catalogue()["assets"]))
    assert hashlib.sha256(b"modified").hexdigest() != catalogue()["assets"][name]
    with pytest.raises(AssetError, match="differs"):
        validate_asset_bytes({name: b"modified"})
