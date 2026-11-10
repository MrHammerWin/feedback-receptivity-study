"""Offline fixtures only: no subscription, model, network or dataset download."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from crmn_agent import qa_study as qa
from crmn_agent.codex_provider import CodexProvider
from crmn_agent.feedback_control import FeedbackCritique
from crmn_agent.providers import BackendTransportError, BudgetExceeded, Provider
from crmn_agent.study_provider import ScopedCodexProvider


def fixture_rows():
    return [
        {
            "id": f"topic{topic}_{i}",
            "question": f"Which option matches fixture topic {topic}, case {i}?",
            "choices": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
            "answer": "A",
            "alternative_supported_answer": "B",
            "has_counterevidence": True,
            "evidence": "Untrusted retrieved material. " * 20,
            "source_dataset": f"topic{topic}",
        }
        for topic in range(2)
        for i in range(3)
    ]


def bundle(rows):
    return qa.build_manifests(rows, dev_per_topic=1, test_per_topic=2, expected_topics=2)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    state = {
        "calls": [],
        "fresh_limit": None,
        "invalid_at": None,
        "crash_at": None,
        "invalid_repairs": False,
        "transport_at": None,
    }
    monkeypatch.setattr(qa, "source_fingerprint", lambda: "offline-frozen-source")

    def initialize(self, config, call_log=None):
        Provider.__init__(self, config, call_log)

    def generate(self, system, user, schema, cap):
        if state["fresh_limit"] is not None and len(state["calls"]) >= state["fresh_limit"]:
            raise BudgetExceeded("Offline simulated physical checkpoint")
        if state["crash_at"] == len(state["calls"]) + 1:
            raise RuntimeError("Offline unknown hard crash")
        payload = json.loads(user)
        state["calls"].append({"system": system, "payload": payload, "schema": schema.__name__})
        self.last_launch = len(state["calls"])
        self.last_backend_usage = {"input_tokens": 100, "output_tokens": 20}
        if state["transport_at"] == self.last_launch:
            self.last_backend_usage = None
            raise BackendTransportError("Offline transport failure, no model response")
        if state["invalid_at"] == self.last_launch or (
            state["invalid_repairs"] and ("admitted_critique" in payload or "observed_facts" in payload)
        ):
            return "invalid-json", 100, 20
        if schema.__name__ == "QuoteRevision":
            result = {"revised_quote": payload["source_fragment"]}
        else:
            action = payload.get("admitted_critique", {}).get("suggested_action", "A")
            result = {"reflection": "Offline fixture only.", "action": action}
        return json.dumps(result), 100, 20

    monkeypatch.setattr(CodexProvider, "__init__", initialize)
    monkeypatch.setattr(CodexProvider, "_generate", generate)

    def make(manifest, output, **settings):
        return ScopedCodexProvider(
            {
                "backend": "codex",
                "model": "offline-fixture",
                "reasoning_effort": "low",
                "task_kind": "qa",
                "study_source_identity": qa.protocol_identity(manifest),
                "study_cache_dir": str(tmp_path / "cache"),
                "max_calls": 10000,
                "subscription_campaign": "offline-test-only",
                **settings,
            },
            Path(output) / "model_calls.jsonl",
        )

    return make, state


def outcomes(output):
    return [json.loads(line) for line in (Path(output) / "outcomes.jsonl").read_text().splitlines()]


def test_manifest_is_deterministic_disjoint_and_contains_no_dataset_text():
    rows = fixture_rows()
    one = bundle(rows)
    assert one == bundle(list(reversed(rows)))
    assert len(one["splits"]["dev"]) == 2 and len(one["splits"]["test"]) == 4
    dev = {x["question_sha256"] for x in one["splits"]["dev"]}
    test = {x["question_sha256"] for x in one["splits"]["test"]}
    assert not dev & test
    text = json.dumps(one)
    assert '"question":' not in text and '"answer":' not in text and '"evidence":' not in text


def test_duplicate_questions_and_invalid_advice_are_removed_before_model_outcomes():
    rows = fixture_rows()
    duplicate = {**rows[0], "id": "zzz-duplicate"}
    invalid = {**rows[0], "id": "zzz-invalid", "alternative_supported_answer": "A"}
    result = bundle([*rows, duplicate, invalid])
    assert {x["reason"] for x in result["excluded"]} == {
        "duplicate_id_or_question_hash",
        "invalid_mcq_or_advice",
    }


def test_full_external_evidence_does_not_change_native_schema_or_public_question():
    row = fixture_rows()[0]
    advice = qa.external_advice(row, "false_advice")
    assert len(advice.issue) > 240 and row["evidence"] in advice.issue
    assert row["evidence"] not in qa.observation(row)
    assert advice.evidence in row["question"] and len(advice.evidence) <= 180
    assert FeedbackCritique.model_fields["issue"].metadata[0].max_length == 240
    with pytest.raises(ValueError):
        qa.ExternalFeedbackCritique(
            needs_revision=True, issue="x" * 12001, evidence="q", suggested_action="A"
        )


def test_shared_draft_and_identical_repairs_are_coupled_without_gold_leakage(backend, tmp_path):
    make, state = backend
    rows, output = fixture_rows(), tmp_path / "run"
    manifest = bundle(rows)
    result = qa.run_study(make(manifest, output), rows, manifest, output, limit=1)
    assert result["status"] == "completed"
    records = outcomes(output)
    assert len(records) == 10 and all(r["finished"] for r in records)
    assert len(state["calls"]) == 9  # draft + 2*(two probes + critic repair + facts repair)
    assert len({json.dumps(r["draft"], sort_keys=True) for r in records}) == 1
    for condition in qa.CONDITIONS:
        shared = [
            r
            for r in records
            if r["condition"] == condition
            and r["mode"] in {"controlled", "grounding_only", "always_critique"}
        ]
        assert len({r["final"]["action"] for r in shared}) == 1
        assert sum(r["new_physical_launches"] for r in shared) == 3  # probes + ONE repair
    assert all(r["evaluation_only"]["gold_option"] == "A" for r in records)
    assert all(r["total_policy_usage"]["calls"] >= 1 for r in records)
    for request in state["calls"]:
        payload = request["payload"]
        assert not {"gold_option", "evaluation_only", "condition", "correct_answer", "is_correct"} & set(
            payload
        )
        assert payload.get("history", []) == []
        if request["schema"] == "QuoteRevision":
            assert "draft" not in payload and "initial_draft" not in payload
        if "admitted_critique" not in payload:
            assert "Untrusted retrieved material" not in json.dumps(payload)


def test_completed_resume_does_not_generate_or_duplicate_outcomes(backend, tmp_path):
    make, state = backend
    rows, output = fixture_rows(), tmp_path / "run"
    manifest = bundle(rows)
    qa.run_study(make(manifest, output), rows, manifest, output, limit=1)
    fresh = len(state["calls"])
    qa.run_study(make(manifest, output), rows, manifest, output, limit=1, resume=True)
    assert len(state["calls"]) == fresh and len(outcomes(output)) == 10


def test_budget_checkpoint_resumes_exact_cached_work_without_retrying_completed_arms(backend, tmp_path):
    make, state = backend
    rows, output = fixture_rows(), tmp_path / "run"
    manifest = bundle(rows)
    state["fresh_limit"] = 4
    stopped = qa.run_study(make(manifest, output), rows, manifest, output, limit=1)
    assert stopped["status"] == "campaign_budget_censored"
    state["fresh_limit"] = None
    finished = qa.run_study(make(manifest, output), rows, manifest, output, limit=1, resume=True)
    assert finished["status"] == "completed" and len(state["calls"]) == 9
    records = outcomes(output)
    assert sum(r["finished"] for r in records) == 10
    assert any(r["status"] == "campaign_budget_censored" for r in records)


def test_finished_draft_model_error_is_not_retried(backend, tmp_path):
    make, state = backend
    rows, output = fixture_rows(), tmp_path / "run"
    manifest = bundle(rows)
    state["invalid_at"] = 1
    qa.run_study(make(manifest, output), rows, manifest, output, limit=1)
    assert len(state["calls"]) == 1
    assert all(r["status"] == "draft_model_error" for r in outcomes(output))
    qa.run_study(make(manifest, output), rows, manifest, output, limit=1, resume=True)
    assert len(state["calls"]) == 1


def test_finished_repair_model_errors_are_not_retried_after_backend_recovers(backend, tmp_path):
    make, state = backend
    rows, output = fixture_rows(), tmp_path / "run"
    manifest = bundle(rows)
    state["invalid_repairs"] = True
    qa.run_study(make(manifest, output), rows, manifest, output, limit=1)
    assert any(row["status"] == "model_error" for row in outcomes(output))
    count = len(state["calls"])
    state["invalid_repairs"] = False
    qa.run_study(make(manifest, output), rows, manifest, output, limit=1, resume=True)
    assert len(state["calls"]) == count


def test_changed_execution_config_is_rejected_but_mutable_ledger_contents_are_not_read(backend, tmp_path):
    make, state = backend
    rows, output = fixture_rows(), tmp_path / "run"
    manifest = bundle(rows)
    ledger = tmp_path / "fake-ledger.json"
    ledger.write_text('{"used": 0}')
    settings = {"subscription_ledger": str(ledger)}
    qa.run_study(make(manifest, output, **settings), rows, manifest, output, limit=1)
    count = len(state["calls"])
    ledger.write_text('{"used": 999}')
    qa.run_study(make(manifest, output, **settings), rows, manifest, output, limit=1, resume=True)
    assert len(state["calls"]) == count
    provider = make(manifest, output, **settings)
    provider.config["timeout_seconds"] = 999
    count = len(state["calls"])
    with pytest.raises(ValueError, match="unchanged source"):
        qa.run_study(provider, rows, manifest, output, limit=1, resume=True)
    assert len(state["calls"]) == count


def test_unknown_running_crash_requires_adjudication(backend, tmp_path):
    make, state = backend
    rows, output = fixture_rows(), tmp_path / "run"
    manifest = bundle(rows)
    state["crash_at"] = 1
    with pytest.raises(RuntimeError, match="hard crash"):
        qa.run_study(make(manifest, output), rows, manifest, output, limit=1)
    state["crash_at"] = None
    with pytest.raises(ValueError, match="adjudicated"):
        qa.run_study(make(manifest, output), rows, manifest, output, limit=1, resume=True)
    assert not state["calls"]


def test_row_or_source_identity_change_is_rejected_before_new_call(backend, tmp_path, monkeypatch):
    make, state = backend
    rows, output = fixture_rows(), tmp_path / "run"
    manifest = bundle(rows)
    qa.run_study(make(manifest, output), rows, manifest, output, limit=1)
    changed = deepcopy(rows)
    changed[0]["answer"] = "B"
    count = len(state["calls"])
    with pytest.raises(ValueError, match="identity changed"):
        qa.run_study(make(manifest, output), changed, manifest, output, limit=1, resume=True)
    monkeypatch.setattr(qa, "source_fingerprint", lambda: "different-source")
    with pytest.raises(ValueError, match="unchanged source"):
        qa.run_study(make(manifest, output), rows, manifest, output, limit=1, resume=True)
    assert len(state["calls"]) == count


def test_pilot_subset_can_extend_only_within_frozen_split(backend, tmp_path):
    make, state = backend
    rows, output = fixture_rows(), tmp_path / "run"
    manifest = bundle(rows)
    qa.run_study(make(manifest, output), rows, manifest, output, limit=1)
    qa.run_study(make(manifest, output), rows, manifest, output, limit=2, resume=True)
    assert len(state["calls"]) == 18 and len(outcomes(output)) == 20
    with pytest.raises(ValueError, match="outside"):
        qa.run_study(
            make(manifest, output), rows, manifest, output, question_ids=["not-a-frozen-ID"], resume=True
        )


def test_pinned_dataset_hash_fails_before_optional_reader_import(tmp_path):
    path = tmp_path / "not-the-pinned-data.parquet"
    path.write_bytes(b"offline invalid data")
    with pytest.raises(ValueError, match="pinned source hash"):
        qa.load_local_wafer(path)


@pytest.mark.parametrize("failed_launch", [1, 2])
def test_transport_failure_halts_without_scoring_or_other_question_calls(backend, tmp_path, failed_launch):
    make, state = backend
    rows, output = fixture_rows(), tmp_path / "run"
    manifest = bundle(rows)
    state["transport_at"] = failed_launch
    result = qa.run_study(make(manifest, output), rows, manifest, output, limit=2)
    assert result["status"] == "infrastructure_censored"
    assert result["completed_questions"] == 0
    assert len(state["calls"]) == failed_launch
    assert (output / "HALTED.json").is_file()
    if (output / "outcomes.jsonl").exists():
        interrupted = [r for r in outcomes(output) if r["status"] == "infrastructure_censored"]
        assert len(interrupted) == 1
        assert interrupted[0]["after_correct"] is None
        assert interrupted[0]["evaluation_only"]["final_correct"] is None
        assert all(r["question_id"] == manifest["splits"]["dev"][0]["id"] for r in outcomes(output))
    state["transport_at"] = None
    with pytest.raises(ValueError, match="adjudication"):
        qa.run_study(make(manifest, output), rows, manifest, output, limit=2, resume=True)
    assert len(state["calls"]) == failed_launch
