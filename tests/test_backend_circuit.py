"""Global backend circuit checks with temporary ledgers and mocked subprocesses."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from crmn_agent import codex_provider
from crmn_agent.codex_provider import CodexProvider, failure_category, reserve_call
from crmn_agent.providers import BackendTransportError, BudgetExceeded, ModelOutputError, Provider
from crmn_agent.schemas import Decision

CAMPAIGN = "feedback-nature-2026-11-10"
PRIVATE_SENTINEL = "SYNTHETIC_PRIVATE_FAILURE_MARKER"


@pytest.fixture
def circuit(tmp_path, monkeypatch):
    ledger = tmp_path / "fake-authorized-campaign.json"
    ledger.write_text(json.dumps({"campaign": CAMPAIGN, "limit": 5000}))
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"stop_after": 500}))
    state = {"launches": 0, "mode": "exit", "stderr": "network reset " + PRIVATE_SENTINEL}

    def fake_run(arguments, **kwargs):
        state["launches"] += 1
        if state["mode"] == "timeout":
            raise subprocess.TimeoutExpired(["codex", PRIVATE_SENTINEL], 1)
        if state["mode"] == "spawn_error":
            raise OSError(PRIVATE_SENTINEL)
        if state["mode"] == "non_object_events":
            return SimpleNamespace(returncode=1, stdout="null\n[]\n", stderr=state["stderr"])
        if state["mode"] == "null_item":
            return SimpleNamespace(returncode=1, stdout='{"type":"item.completed","item":null}\n',
                                   stderr=state["stderr"])
        if state["mode"] == "exit_with_unrelated_numbers":
            stdout = json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 1429, "cached_input_tokens": 0, "output_tokens": 3,
            }})
            return SimpleNamespace(returncode=1, stdout=stdout, stderr=PRIVATE_SENTINEL)
        if state["mode"] == "exit":
            return SimpleNamespace(returncode=1, stdout="", stderr=state["stderr"])
        response_path = Path(arguments[arguments.index("--output-last-message") + 1])
        response_path.write_text("not JSON" if state["mode"] == "invalid_json" else
                                 '{"reflection":"Public response.","action":"inventory"}')
        usage = {} if state["mode"] == "invalid_usage" else {
            "input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20,
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps({"type": "turn.completed", "usage": usage}),
                               stderr="")

    monkeypatch.setattr(codex_provider.subprocess, "run", fake_run)

    def make(name):
        # Bypass login discovery and environment copying. Exercise the real
        # generation/reservation code against a wholly synthetic local ledger.
        provider = object.__new__(CodexProvider)
        Provider.__init__(provider, {
            "backend": "codex", "model": "offline-test-model", "subscription_campaign": CAMPAIGN,
            "subscription_max_launches": 5000, "subscription_checkpoint": str(checkpoint),
            "stop_on_transport_error": True,
        }, tmp_path / (name + ".jsonl"))
        provider.ledger = ledger
        provider.root = tmp_path / (name + "-isolated")
        provider.root.mkdir()
        provider.process_env = {}
        return provider

    return make, state, ledger, checkpoint


@pytest.mark.parametrize("mode,category", [
    ("exit", "network"), ("timeout", "timeout"), ("spawn_error", "local_execution"),
    ("non_object_events", "network"), ("null_item", "network"),
    ("invalid_usage", "invalid_usage_telemetry"),
    ("exit_with_unrelated_numbers", "unclassified_backend_failure"),
])
def test_first_transport_failure_closes_shared_gate_and_second_provider_launches_nothing(circuit, mode, category):
    make, state, ledger, checkpoint = circuit
    state["mode"] = mode
    first, second = make("first"), make("second")
    with pytest.raises(BackendTransportError, match=category):
        first.complete(Decision, "A public toy request", {"observation": "Public input"}, 100)
    assert json.loads(checkpoint.read_text())["stop_after"] == 0
    assert first.usage.calls == 1 and first.usage.invalid_outputs == 0
    first_log = json.loads(first.call_log.read_text())
    assert first_log["backend_usage"] is None  # Unknown is not evidence of zero token use.
    assert first_log["backend_failure"]["category"] == category
    assert PRIVATE_SENTINEL not in first.call_log.read_text()
    with pytest.raises(BudgetExceeded):
        second.complete(Decision, "Another public request", {"observation": "Public input"}, 100)
    assert state["launches"] == 1
    assert second.usage.calls == 0 and second.last_launch is None
    reservations = ledger.with_suffix(".attempts.jsonl").read_text().splitlines()
    assert len(reservations) == 1  # No refund for the failed actual attempt.


def test_valid_transport_with_invalid_model_json_does_not_open_infrastructure_circuit(circuit):
    make, state, ledger, checkpoint = circuit
    state["mode"] = "invalid_json"
    provider = make("invalid-model-output")
    with pytest.raises(ModelOutputError):
        provider.complete(Decision, "A public toy request", {}, 100)
    assert provider.usage.invalid_outputs == 1
    assert provider.last_backend_usage["input_tokens"] == 100
    assert json.loads(checkpoint.read_text())["stop_after"] == 500
    state["mode"] = "success"
    other_case = make("another-case")
    assert other_case.complete(Decision, "A different case", {}, 100).action == "inventory"
    assert state["launches"] == 2
    assert len(ledger.with_suffix(".attempts.jsonl").read_text().splitlines()) == 2


def test_failure_classifier_returns_only_fixed_categories_not_matched_private_text():
    assert failure_category("HTTP 429 " + PRIVATE_SENTINEL) == "rate_limit"
    assert failure_category("input_tokens:1429 " + PRIVATE_SENTINEL) == "unclassified_backend_failure"
    assert failure_category("unfamiliar failure " + PRIVATE_SENTINEL) == "unclassified_backend_failure"


def test_spacing_rechecks_stop_after_wait_before_any_new_reservation(circuit, monkeypatch):
    _, _, ledger, checkpoint = circuit
    reserve_call(ledger, 5000, CAMPAIGN, checkpoint)
    record = json.loads(ledger.with_suffix(".attempts.jsonl").read_text())
    last = datetime.fromisoformat(record["reserved_at"]).timestamp()
    monkeypatch.setattr(codex_provider.time, "time", lambda: last)
    waited = []

    def stop_during_wait(seconds):
        waited.append(seconds)
        checkpoint.write_text(json.dumps({"stop_after": 0, "reason": "another worker failed"}))

    monkeypatch.setattr(codex_provider.time, "sleep", stop_during_wait)
    with pytest.raises(BudgetExceeded, match="stopped while waiting"):
        reserve_call(ledger, 5000, CAMPAIGN, checkpoint, 3)
    assert waited == [3]
    assert len(ledger.with_suffix(".attempts.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("spacing", [True, -1, 61, float("nan"), float("inf")])
def test_invalid_launch_spacing_is_rejected_without_consuming_authorization(circuit, spacing):
    _, _, ledger, checkpoint = circuit
    with pytest.raises(ValueError):
        reserve_call(ledger, 5000, CAMPAIGN, checkpoint, spacing)
    assert not ledger.with_suffix(".attempts.jsonl").exists()
