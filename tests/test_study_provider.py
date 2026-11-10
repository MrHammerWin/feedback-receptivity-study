"""All study-cache tests replace backend initialization and generation offline."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from crmn_agent.codex_provider import CodexProvider
from crmn_agent.providers import BudgetExceeded, EpisodeBudgetExceeded, ModelOutputError, Provider
from crmn_agent.schemas import Decision
from crmn_agent.study_provider import ScopedCodexProvider, StudyCacheError


class OtherDecision(Decision):
    pass


@pytest.fixture
def backend(tmp_path, monkeypatch):
    state = {"calls": [], "outputs": [], "delay": 0}

    def fake_init(self, config, call_log=None):
        Provider.__init__(self, config, call_log)

    def fake_generate(self, system, user, schema, cap):
        state["calls"].append((system, user, schema, cap))
        self.last_launch = len(state["calls"])
        self.last_backend_usage = {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20}
        if state["delay"]:
            time.sleep(state["delay"])
        output = state["outputs"].pop(0) if state["outputs"] else json.dumps({
            "reflection": f"Frozen backend draw {len(state['calls'])}.", "action": "inventory",
        })
        if isinstance(output, Exception):
            raise output
        return output, 100, 20

    monkeypatch.setattr(CodexProvider, "__init__", fake_init)
    monkeypatch.setattr(CodexProvider, "_generate", fake_generate)

    def make(**settings):
        return ScopedCodexProvider({
            "backend": "codex", "model": "offline-test-model", "reasoning_effort": "low",
            "task_kind": "qa", "study_source_identity": "frozen-protocol-and-source-sha",
            "study_cache_dir": str(tmp_path / "cache"),
            "subscription_campaign": "feedback-nature-2026-11-10", **settings,
        })

    return make, state


def call(provider, *, system="system", user="user", schema=Decision):
    return provider._generate(system, user, schema, 256)


def test_sequence_resume_reuses_exact_prefix_then_delegates_only_after_exhaustion(backend):
    make, state = backend
    first = make()
    first.begin_scope("qa:case-1:condition-A:policy-feedback:step-0")
    one = call(first)
    two = call(first, user="second request")
    assert len(state["calls"]) == 2 and not first.last_replayed
    resumed = make()
    resumed.begin_scope("qa:case-1:condition-A:policy-feedback:step-0")
    assert resumed.replay_remaining == 2
    assert call(resumed) == one
    assert resumed.last_replayed and resumed.last_launch == 1
    assert resumed.last_original_campaign == "feedback-nature-2026-11-10"
    assert resumed.last_backend_usage["cached_input_tokens"] == 80
    assert resumed.last_replay_seconds > 0
    assert call(resumed, user="second request") == two
    resumed.assert_scope_exhausted()
    replayed_seconds = resumed.replayed_seconds
    assert replayed_seconds > 0
    call(resumed, user="new continuation")
    assert len(state["calls"]) == 3
    assert resumed.last_replayed is False and resumed.last_replay_seconds == 0
    assert resumed.replayed_seconds == replayed_seconds


@pytest.mark.parametrize("changed_request", [
    {"system": "changed system"}, {"user": "changed user"}, {"schema": OtherDecision},
])
def test_sequence_mismatch_fails_closed_without_expensive_fallback(backend, changed_request):
    make, state = backend
    original = make()
    original.begin_scope("episode-step")
    call(original)
    resumed = make()
    resumed.begin_scope("episode-step")
    with pytest.raises(StudyCacheError, match="prefix request mismatch"):
        call(resumed, **changed_request)
    assert len(state["calls"]) == 1
    assert resumed.replay_remaining == 1


@pytest.mark.parametrize("settings", [
    {"model": "another-model"}, {"reasoning_effort": "high"},
    {"task_kind": "alfworld"}, {"study_source_identity": "new-source-sha"},
])
def test_metadata_rejects_changed_model_reasoning_task_or_source(backend, settings):
    make, state = backend
    original = make()
    original.begin_scope("frozen-scope")
    call(original)
    with pytest.raises(StudyCacheError, match="identity differs"):
        make(**settings).begin_scope("frozen-scope")
    assert len(state["calls"]) == 1


def test_sequence_cannot_discard_unread_prefix_when_ending_or_switching(backend):
    make, _ = backend
    original = make()
    original.begin_scope("first-scope")
    call(original)
    call(original, user="second request")
    resumed = make()
    resumed.begin_scope("first-scope")
    call(resumed)
    with pytest.raises(StudyCacheError, match="Unconsumed"):
        resumed.assert_scope_exhausted()
    with pytest.raises(StudyCacheError, match="Unconsumed"):
        resumed.begin_scope("next-scope")
    call(resumed, user="second request")
    resumed.begin_scope("next-scope")


def test_shared_qa_returns_identical_draw_across_policies_inside_same_case_condition(backend):
    make, state = backend
    first, second = make(), make()
    for provider in (first, second):
        provider.begin_scope("qa:case-1:condition-helpful", mode="shared")
    one, two = call(first), call(second)
    assert one == two and len(state["calls"]) == 1
    assert second.last_replayed and second.last_launch == first.last_launch
    call(second, user="policy-specific final request")
    assert len(state["calls"]) == 2 and not second.last_replayed


@pytest.mark.parametrize("second_scope", ["qa:other-case:helpful", "qa:case-1:harmful"])
def test_shared_cache_never_crosses_distinct_case_or_condition_scope(backend, second_scope):
    make, state = backend
    first = make()
    first.begin_scope("qa:case-1:helpful", mode="shared")
    original = call(first)
    other = make()
    other.begin_scope(second_scope, mode="shared")
    fresh = call(other)
    assert original != fresh and len(state["calls"]) == 2


def test_identical_concatenation_with_different_system_user_boundaries_is_not_a_shared_match(backend):
    make, state = backend
    provider = make()
    provider.begin_scope("qa:case:condition", mode="shared")
    call(provider, system="ab", user="c")
    call(provider, system="a", user="bc")
    assert len(state["calls"]) == 2


@pytest.mark.parametrize("kind", ["alfworld", "textcraft"])
def test_shared_mode_is_forbidden_for_native_interactive_tasks(backend, kind):
    make, state = backend
    with pytest.raises(ValueError, match="task_kind='qa'"):
        make(task_kind=kind).begin_scope("case-without-QA", mode="shared")
    assert state["calls"] == []


def test_mode_cannot_silently_reinterpret_an_existing_scope(backend):
    make, state = backend
    first = make()
    first.begin_scope("qa:case:condition", mode="sequence")
    call(first)
    with pytest.raises(StudyCacheError, match="mode or frozen"):
        make().begin_scope("qa:case:condition", mode="shared")
    assert len(state["calls"]) == 1


def test_invalid_json_is_persisted_and_replayed_without_drawing_a_replacement(backend):
    make, state = backend
    state["outputs"] = ["not JSON"]
    original = make()
    original.begin_scope("qa:case:invalid-output")
    with pytest.raises(ModelOutputError):
        original.complete(Decision, "same purpose", {"observation": "public"}, 256)
    resumed = make()
    resumed.begin_scope("qa:case:invalid-output")
    with pytest.raises(ModelOutputError):
        resumed.complete(Decision, "same purpose", {"observation": "public"}, 256)
    assert resumed.last_replayed
    assert original.usage.invalid_outputs == resumed.usage.invalid_outputs == 1
    assert len(state["calls"]) == 1


@pytest.mark.parametrize("failure", [ModelOutputError("scripted transport failure"), BudgetExceeded("checkpoint")])
def test_delegated_failure_propagates_once_and_does_not_cache_a_response(backend, failure):
    make, state = backend
    state["outputs"] = [failure]
    provider = make()
    provider.begin_scope("episode:failed-attempt")
    with pytest.raises(type(failure)):
        call(provider)
    assert len(state["calls"]) == 1
    records = json.loads((provider._scope_dir / "records.json").read_text())
    assert records["entries"] == []
    assert provider.last_replayed is False


def test_cached_requests_still_consume_logical_call_token_and_historical_time_budgets(backend):
    make, state = backend
    original = make()
    original.begin_scope("qa:budget:helpful", mode="shared")
    original.complete(Decision, "same purpose", {"observation": "public"}, 256)
    provider = make(max_calls_per_episode=1)
    provider.begin_scope("qa:budget:helpful", mode="shared")
    provider.begin_episode()
    provider.complete(Decision, "same purpose", {"observation": "public"}, 256)
    assert provider.usage.calls == 1
    assert provider.usage.input_tokens == 100 and provider.usage.output_tokens == 20
    assert provider.usage.seconds >= provider.last_replay_seconds > 0
    with pytest.raises(EpisodeBudgetExceeded):
        provider.complete(Decision, "new request", {}, 256)
    assert len(state["calls"]) == 1


def test_persistent_shared_lock_prevents_duplicate_concurrent_miss_generation(backend):
    make, state = backend
    state["delay"] = 0.03
    first, second = make(), make()
    for provider in (first, second):
        provider.begin_scope("qa:concurrent-case:helpful", mode="shared")
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(call, [first, second]))
    assert results[0] == results[1]
    assert len(state["calls"]) == 1
    assert sum(p.last_replayed for p in (first, second)) == 1


def test_missing_or_malformed_records_do_not_turn_into_expensive_cache_misses(backend):
    make, state = backend
    original = make()
    original.begin_scope("qa:broken-cache:condition", mode="shared")
    call(original)
    records = original._scope_dir / "records.json"
    records.unlink()
    with pytest.raises(StudyCacheError, match="Incomplete"):
        make().begin_scope("qa:broken-cache:condition", mode="shared")
    records.write_text("not JSON")
    with pytest.raises(StudyCacheError, match="Malformed"):
        make().begin_scope("qa:broken-cache:condition", mode="shared")
    assert len(state["calls"]) == 1


def test_scope_ids_are_hashed_and_do_not_create_paths_or_enter_model_requests(backend, tmp_path):
    make, state = backend
    provider = make(subscription_ledger=str(tmp_path / "not-created-ledger.json"))
    scope = "../../qa:private-case-label:condition-gold"
    provider.begin_scope(scope, mode="shared")
    call(provider)
    assert provider._scope_dir.parent == provider.cache_root
    assert len(provider._scope_dir.name) == 64
    assert scope not in state["calls"][0][0] and scope not in state["calls"][0][1]
    assert not (tmp_path / "not-created-ledger.json").exists()


def test_scope_required_before_requests_and_identity_must_stay_frozen(backend):
    make, state = backend
    provider = make()
    with pytest.raises(StudyCacheError, match="begin_scope"):
        call(provider)
    provider.begin_scope("qa:identity:condition", mode="shared")
    provider.config["reasoning_effort"] = "high"
    with pytest.raises(StudyCacheError, match="identity changed"):
        call(provider)
    assert state["calls"] == []


@pytest.mark.parametrize("settings", [{"study_source_identity": ""}, {"task_kind": ""}])
def test_missing_study_identity_is_rejected_without_authentication_or_generation(backend, settings):
    make, state = backend
    with pytest.raises(ValueError, match="required"):
        make(**settings)
    assert state["calls"] == []


def test_cache_keeps_raw_output_provenance_and_no_pending_atomic_files(backend):
    make, state = backend
    raw = '{ "reflection": "Keep exact bytes.", "action": "inventory" }\n'
    state["outputs"] = [raw]
    provider = make()
    provider.begin_scope("qa:raw:condition", mode="shared")
    result = call(provider)
    assert result[0] == raw
    cached = json.loads((provider._scope_dir / "records.json").read_text())["entries"][0]
    assert cached["output"] == raw
    assert cached["subscription_launch"] == 1
    assert cached["original_campaign"] == "feedback-nature-2026-11-10"
    assert cached["backend_usage"] == provider.last_backend_usage
    assert cached["inference_seconds"] >= 0
    assert not list(Path(provider._scope_dir).glob(".pending-*"))
