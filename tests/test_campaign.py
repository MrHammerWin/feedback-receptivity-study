"""Offline campaign accounting tests; subprocesses are always mocked."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from crmn_agent import codex_provider, experiment
from crmn_agent.codex_provider import CAMPAIGN, CodexProvider, reserve_call
from crmn_agent.environments import Observation
from crmn_agent.feedback_control import FeedbackAgent, FeedbackCritique, QuoteRevision
from crmn_agent.providers import (
    BackendTransportError,
    BudgetExceeded,
    EpisodeBudgetExceeded,
    Provider,
    Usage,
)
from crmn_agent.schemas import Decision


def ledger(tmp_path, campaign=CAMPAIGN, limit=1000, used=0):
    path = tmp_path / "authorized_campaign.json"
    path.write_text(json.dumps({"campaign": campaign, "limit": limit}))
    if used:
        path.with_suffix(".attempts.jsonl").write_text("".join(
            json.dumps({"launch": i, "campaign": campaign}) + "\n" for i in range(1, used + 1)
        ))
    return path


def test_simultaneous_shared_ledger_reservations_never_exceed_authorization(tmp_path):
    path = ledger(tmp_path, used=975)

    def attempt(_):
        try:
            return reserve_call(path)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        outcomes = list(executor.map(attempt, range(64)))
    accepted = sorted(value for value in outcomes if value is not None)
    assert accepted == list(range(976, 1001))
    records = [json.loads(line) for line in path.with_suffix(".attempts.jsonl").read_text().splitlines()]
    assert len(records) == 1000
    assert [row["launch"] for row in records] == list(range(1, 1001))


def test_missing_ledger_is_never_created_implicitly(tmp_path):
    path = tmp_path / "not_authorized.json"
    with pytest.raises(BudgetExceeded, match="separately authorized"):
        reserve_call(path)
    assert not path.exists()
    assert not path.with_suffix(".attempts.jsonl").exists()


@pytest.mark.parametrize("campaign,limit", [("old-campaign", 1000), (CAMPAIGN, 30), (CAMPAIGN, 1001)])
def test_old_campaign_and_unrecognized_authorization_are_rejected(tmp_path, campaign, limit):
    path = ledger(tmp_path, campaign=campaign, limit=limit)
    with pytest.raises(BudgetExceeded, match="Wrong campaign"):
        reserve_call(path)
    assert not path.with_suffix(".attempts.jsonl").exists()


@pytest.mark.parametrize("limit", [-1, 0, 1001])
def test_runtime_limit_cannot_expand_the_approved_allowance(tmp_path, limit):
    with pytest.raises(ValueError, match="at most 1000"):
        reserve_call(ledger(tmp_path), authorized_limit=limit)


def test_stricter_runtime_cap_and_failed_attempts_remain_reserved(tmp_path):
    path = ledger(tmp_path)
    assert reserve_call(path, authorized_limit=1) == 1
    # The journal is authoritative: abandoning a reserved attempt is not a refund.
    with pytest.raises(BudgetExceeded, match="exhausted"):
        reserve_call(path, authorized_limit=1)
    assert reserve_call(path, authorized_limit=2) == 2


def test_subprocess_failure_consumes_one_shared_launch_and_has_no_implicit_retry(tmp_path, monkeypatch):
    path = ledger(tmp_path)
    generation_calls = []

    def fake_run(arguments, **kwargs):
        if arguments == ["codex", "login", "status"]:
            return SimpleNamespace(returncode=0, stdout="Logged in using ChatGPT", stderr="")
        generation_calls.append(arguments)
        return SimpleNamespace(returncode=1, stdout="", stderr="suppressed failure detail")

    monkeypatch.setattr(codex_provider.shutil, "which", lambda _: "/mock/codex")
    monkeypatch.setattr(codex_provider.subprocess, "run", fake_run)
    config = {"backend": "codex", "allow_subscription": True, "model": "offline-test",
              "subscription_ledger": str(path), "isolated_dir": str(tmp_path / "isolated")}
    first = CodexProvider(config)
    second = CodexProvider(config)
    for provider in (first, second):
        with pytest.raises(BackendTransportError, match="unclassified_backend_failure"):
            provider.complete(Decision, "offline test", {}, 30)
    assert len(generation_calls) == 2
    assert first.last_launch == 1 and second.last_launch == 2
    assert len(path.with_suffix(".attempts.jsonl").read_text().splitlines()) == 2


class FakeEnvironment:
    protocol = "offline-no-model-fixture"

    def __init__(self, task_id="task-a"):
        self.task_id = task_id
        self.closed = False

    def reset(self):
        return Observation("A box is closed.", "Open the box.", ["open box"])

    def step(self, action):
        assert action == "open box"
        return Observation("A box is open.", "Open the box.", reward=1, terminated=True, success=True)

    def close(self):
        self.closed = True


def campaign_config(tmp_path):
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps({"tasks": [
        {"key": "task-a", "benchmark": "offline", "seed": 7, "task_id": "task-a"},
    ]}))
    config = {"provider": {"backend": "codex"}, "manifest": str(manifest),
              "results_dir": str(tmp_path / "results"), "variants": ["stateact"],
              "horizons": {"offline": 2}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, manifest, config


def mock_campaign(monkeypatch, environment):
    monkeypatch.setattr(experiment, "fingerprint", lambda: "fixed-source-hash")
    monkeypatch.setattr(experiment.importlib.metadata, "version", lambda _: "offline-fixture")
    monkeypatch.setattr(experiment, "make_env", lambda *args: environment)
    provider = SimpleNamespace(model="offline-test", usage=Usage(), begin_episode=lambda: None)
    monkeypatch.setattr(experiment, "make_provider", lambda *args: provider)
    agent = SimpleNamespace(
        act=lambda *args: SimpleNamespace(action="open box", diagnostics={}),
        feedback=lambda *args: None,
    )
    monkeypatch.setattr(experiment, "build_agent", lambda *args: agent)
    return provider


def test_frozen_task_identity_change_is_rejected_and_environment_closed(tmp_path, monkeypatch):
    config_path, _, _ = campaign_config(tmp_path)
    environment = FakeEnvironment("unexpected-task")
    mock_campaign(monkeypatch, environment)
    with pytest.raises(ValueError, match="Frozen task identity changed"):
        experiment.run(config_path)
    assert environment.closed
    assert not list((tmp_path / "results" / "episodes").glob("*.json"))


def test_run_metadata_does_not_claim_absolute_sota(tmp_path, monkeypatch):
    config_path, _, _ = campaign_config(tmp_path)
    mock_campaign(monkeypatch, FakeEnvironment())
    experiment.run(config_path)
    metadata = json.loads((tmp_path / "results" / "run.json").read_text())
    assert metadata["paid_api"] is False
    assert "not absolute SOTA" in metadata["comparison"]
    assert metadata["manifest_sha256"]


def test_resume_rejects_a_modified_manifest_at_the_same_path(tmp_path, monkeypatch):
    config_path, manifest_path, _ = campaign_config(tmp_path)
    mock_campaign(monkeypatch, FakeEnvironment())
    experiment.run(config_path)
    manifest_path.write_text(json.dumps({"tasks": []}))
    with pytest.raises(ValueError, match="manifest|Manifest|changing code or configuration"):
        experiment.run(config_path)


def test_old_campaign_retains_its_original_limit(tmp_path):
    path = ledger(tmp_path, used=999)
    assert reserve_call(path) == 1000
    with pytest.raises(BudgetExceeded):
        reserve_call(path)


class CountingProvider(Provider):
    def __init__(self):
        super().__init__({"backend": "offline-test", "max_calls_per_episode": 3})
        self.generated = 0

    def _generate(self, system, user, schema, cap):
        self.generated += 1
        payload = json.loads(user)
        if schema is FeedbackCritique:
            answer = {"needs_revision": True, "issue": "Check the box.",
                      "evidence": "The box is closed.", "suggested_action": "open box"}
        elif schema is QuoteRevision:
            answer = {"revised_quote": payload["source_fragment"]}
        else:
            answer = {"reflection": "Check the observed state.", "action": "open box"}
        return json.dumps(answer), 10, 10


def test_per_episode_call_cap_stops_inside_multicall_agent_before_extra_launch():
    provider = CountingProvider()
    provider.begin_episode()
    agent = FeedbackAgent(provider)
    with pytest.raises(EpisodeBudgetExceeded, match="Per-episode call limit"):
        agent.act("The box is closed.", "Open the box.", ["open box"], 0)
    assert provider.generated == 3 and provider.usage.calls == 3
    assert agent.history == []
    provider.begin_episode()
    provider.complete(Decision, "A new episode has its own allowance", {}, 40)
    assert provider.generated == 4
