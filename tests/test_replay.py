"""Exact-prefix accounting tests with no subprocesses, models or credentials."""

import hashlib
import json
from pathlib import Path

import pytest

from crmn_agent.codex_provider import CodexProvider
from crmn_agent.providers import EpisodeBudgetExceeded, Provider
from crmn_agent.replay_provider import (
    ReplayCodexProvider,
    ReplayPrefixError,
    prepare_replay_prefixes,
)
from crmn_agent.schemas import Decision

RAW = '{"reflection":"Cached observed decision.","action":"inventory"}'


def row(system="system", user="user", *, launch=101, inputs=10, outputs=3, seconds=0.25):
    return {
        "output": RAW, "prompt_sha256": hashlib.sha256((system + user).encode()).hexdigest(),
        "input_tokens": inputs, "output_tokens": outputs,
        "inference_seconds": seconds,
        "backend_usage": {"input_tokens": inputs, "output_tokens": outputs, "cached_input_tokens": 2},
        "subscription_launch": launch,
    }


def write_prefix(tmp_path, *, rows=None, **changes):
    path = tmp_path / "prefix.json"
    path.write_text(json.dumps({
        "model": "test-model", "reasoning_effort": "low", "source_log_hash": "a" * 64,
        "rows": [row()] if rows is None else rows, **changes,
    }))
    return path


@pytest.fixture
def provider_factory(monkeypatch):
    fresh_launches = []

    def fake_init(self, config, call_log=None):
        Provider.__init__(self, config, call_log)

    def fake_generate(self, system, user, schema, cap):
        fresh_launches.append((system, user))
        self.last_launch = 1000 + len(fresh_launches)
        self.last_backend_usage = {"input_tokens": 100, "output_tokens": 50}
        return RAW, 100, 50

    monkeypatch.setattr(CodexProvider, "__init__", fake_init)
    monkeypatch.setattr(CodexProvider, "_generate", fake_generate)

    def make(**settings):
        return ReplayCodexProvider({"backend": "codex", "model": "test-model",
                                    "reasoning_effort": "low", **settings})

    return make, fresh_launches


def test_exact_prefix_reuses_raw_response_and_original_launch_with_no_new_reservation(tmp_path, provider_factory):
    make, fresh_launches = provider_factory
    provider = make()
    original = row()
    original["output"] = RAW + "\n"
    provider.set_replay_prefix(write_prefix(tmp_path, rows=[original]))
    output, inputs, outputs = provider._generate("system", "user", Decision, 100)
    assert (output, inputs, outputs) == (RAW + "\n", 10, 3)
    assert provider.last_replayed is True
    assert provider.last_launch == 101
    assert provider.last_backend_usage["cached_input_tokens"] == 2
    assert provider.last_replay_seconds == 0.25 and provider.replayed_seconds == 0.25
    assert provider.replay_remaining == 0
    assert fresh_launches == []
    provider.assert_replay_exhausted()
    provider._generate("new system", "new user", Decision, 100)
    assert len(fresh_launches) == 1
    assert provider.last_replayed is False
    assert provider.last_launch == 1001
    assert provider.last_replay_seconds == 0 and provider.replayed_seconds == 0.25


def test_prompt_mismatch_stops_without_fallback_consumption_or_new_launch(tmp_path, provider_factory):
    make, fresh_launches = provider_factory
    provider = make()
    provider.set_replay_prefix(write_prefix(tmp_path))
    with pytest.raises(ReplayPrefixError, match="prompt mismatch"):
        provider._generate("system", "different user", Decision, 100)
    assert fresh_launches == []
    assert provider.replay_remaining == 1
    assert provider.last_replayed is False and provider.last_launch is None


@pytest.mark.parametrize("changes,message", [
    ({"model": "another-model"}, "model differs"),
    ({"reasoning_effort": "high"}, "reasoning setting"),
    ({"schema_version": "unknown-v9"}, "Unsupported"),
])
def test_incompatible_metadata_fails_before_any_call(tmp_path, provider_factory, changes, message):
    make, fresh_launches = provider_factory
    provider = make()
    with pytest.raises(ReplayPrefixError, match=message):
        provider.set_replay_prefix(write_prefix(tmp_path, **changes))
    assert fresh_launches == []


def test_original_source_hash_is_checked_when_source_file_is_recorded(tmp_path, provider_factory):
    make, fresh_launches = provider_factory
    source = tmp_path / "calls.jsonl"
    source.write_text("frozen historical log\n")
    prefix = write_prefix(tmp_path, source_log=source.name,
                          source_log_hash=hashlib.sha256(source.read_bytes()).hexdigest())
    source.write_text("modified history\n")
    with pytest.raises(ReplayPrefixError, match="hash changed"):
        make().set_replay_prefix(prefix)
    assert fresh_launches == []


def test_unread_rows_cannot_be_discarded_by_switching_tasks_or_ending_early(tmp_path, provider_factory):
    make, _ = provider_factory
    provider = make()
    prefix = write_prefix(tmp_path, rows=[row(), row(launch=102)])
    provider.set_replay_prefix(prefix)
    provider._generate("system", "user", Decision, 100)
    with pytest.raises(ReplayPrefixError, match="unconsumed"):
        provider.assert_replay_exhausted()
    with pytest.raises(ReplayPrefixError, match="unconsumed"):
        provider.set_replay_prefix(None)
    with pytest.raises(ReplayPrefixError, match="unconsumed"):
        provider.set_replay_prefix(prefix)
    provider._generate("system", "user", Decision, 100)
    provider.set_replay_prefix(None)
    assert provider.replay_remaining == 0


class PromptCaptureProvider(Provider):
    def _generate(self, system, user, schema, cap):
        self.prompt = (system, user)
        return RAW, 10, 3


def test_inherited_logical_call_and_token_budgets_include_cached_inference(tmp_path, provider_factory):
    make, fresh_launches = provider_factory
    capture = PromptCaptureProvider({})
    capture.complete(Decision, "same request", {"observation": "public"}, 100)
    historical = row(*capture.prompt)
    provider = make(max_calls_per_episode=2)
    provider.call_log = tmp_path / "replayed_calls.jsonl"
    provider.set_replay_prefix(write_prefix(tmp_path, rows=[historical, historical]))
    provider.begin_episode()
    for _ in range(2):
        provider.complete(Decision, "same request", {"observation": "public"}, 100)
    assert provider.usage.calls == 2
    assert provider.usage.input_tokens == 20 and provider.usage.output_tokens == 6
    assert provider.usage.seconds >= 0.5
    logged = [json.loads(line) for line in provider.call_log.read_text().splitlines()]
    assert all(entry["replayed"] and entry["historical_inference_seconds"] == 0.25 for entry in logged)
    assert all(entry["subscription_launch"] == 101 for entry in logged)
    with pytest.raises(EpisodeBudgetExceeded):
        provider.complete(Decision, "new request", {}, 100)
    assert fresh_launches == []
    provider.begin_episode()
    provider.complete(Decision, "new request", {}, 100)
    assert len(fresh_launches) == 1
    assert provider.usage.calls == 3
    assert provider.usage.input_tokens == 120 and provider.usage.output_tokens == 56


def make_oldrun(tmp_path):
    old = tmp_path / "old"
    (old / "episodes").mkdir(parents=True)
    (old / "run.json").write_text(json.dumps({"config": {
        "provider": {"model": "test-model", "reasoning_effort": "low"}, "variants": ["feedback"],
    }}))
    records = []
    cumulative_inputs = cumulative_outputs = 0
    cumulative_seconds = 0.0
    for index, (inputs, outputs) in enumerate(((10, 3), (5, 2), (7, 4)), 1):
        cached = row(launch=100 + index, inputs=inputs, outputs=outputs)
        cumulative_inputs += inputs
        cumulative_outputs += outputs
        cumulative_seconds += 0.25
        records.append({
            "output": cached["output"], "prompt_sha256": cached["prompt_sha256"],
            "backend_usage": cached["backend_usage"], "subscription_launch": cached["subscription_launch"],
            "model": "test-model", "error_type": None,
            "usage_cumulative": {"calls": index, "input_tokens": cumulative_inputs,
                                 "output_tokens": cumulative_outputs, "seconds": cumulative_seconds},
        })
    (old / "model_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for key, calls, inputs, outputs in (("first", 2, 15, 5), ("second", 1, 7, 4)):
        (old / "episodes" / f"{key}--feedback.json").write_text(json.dumps({
            "key": key, "variant": "feedback", "model": "test-model", "finished": True,
            "status": "success", "usage": {"calls": calls, "input_tokens": inputs, "output_tokens": outputs,
                                             "seconds": calls * 0.25},
        }))
    return old, records


def test_builder_reconstructs_adjacent_slices_and_usage_deltas_without_calls(tmp_path):
    old, _ = make_oldrun(tmp_path)
    output = tmp_path / "prefixes"
    mapping = prepare_replay_prefixes(old, [{"key": "first"}, {"key": "second"}], output)
    assert list(mapping) == ["first--feedback", "second--feedback"]
    first = json.loads(Path(mapping["first--feedback"]).read_text())
    second = json.loads(Path(mapping["second--feedback"]).read_text())
    assert [r["input_tokens"] for r in first["rows"]] == [10, 5]
    assert [r["output_tokens"] for r in first["rows"]] == [3, 2]
    assert [r["inference_seconds"] for r in first["rows"]] == [0.25, 0.25]
    assert [r["subscription_launch"] for r in second["rows"]] == [103]
    assert first["source_log_hash"] == hashlib.sha256((old / "model_calls.jsonl").read_bytes()).hexdigest()
    assert json.loads((output / "index.json").read_text())["logical_calls"] == 3
    with pytest.raises(ReplayPrefixError, match="overwrite"):
        prepare_replay_prefixes(old, ["first", "second"], output)


@pytest.mark.parametrize("change,message", [
    ("provider_error", "provider error"), ("counter_reset", "reset or skip"),
    ("invalid_json", "valid JSON"), ("backend_usage", "backend usage"),
    ("model", "model identities"), ("nested_replay", "nested replay"),
])
def test_builder_rejects_ambiguous_or_failed_historical_calls(tmp_path, change, message):
    old, records = make_oldrun(tmp_path)
    if change == "provider_error":
        records[1]["error_type"] = "ModelOutputError"
    elif change == "counter_reset":
        records[1]["usage_cumulative"]["calls"] = 1
    elif change == "invalid_json":
        records[1]["output"] = "not JSON"
    elif change == "backend_usage":
        records[1]["backend_usage"]["input_tokens"] = 999
    elif change == "model":
        records[1]["model"] = "changed-model"
    else:
        records[1]["replayed"] = True
    (old / "model_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    output = tmp_path / "prefixes"
    with pytest.raises(ReplayPrefixError, match=message):
        prepare_replay_prefixes(old, ["first", "second"], output)
    assert not output.exists()


@pytest.mark.parametrize("change,message", [
    ("unfinished", "unfinished"), ("usage_mismatch", "token usage"),
    ("missing_predecessor", "missing historical predecessor"), ("unassigned_calls", "outside completed"),
])
def test_builder_requires_complete_unambiguous_episode_partition(tmp_path, change, message):
    old, _ = make_oldrun(tmp_path)
    first = old / "episodes/first--feedback.json"
    data = json.loads(first.read_text())
    keys = ["first", "second"]
    if change == "unfinished":
        data["finished"] = False
        first.write_text(json.dumps(data))
    elif change == "usage_mismatch":
        data["usage"]["input_tokens"] += 1
        first.write_text(json.dumps(data))
    elif change == "missing_predecessor":
        first.unlink()
    else:
        keys = ["first"]
    with pytest.raises(ReplayPrefixError, match=message):
        prepare_replay_prefixes(old, keys, tmp_path / "prefixes")


def test_valid_output_rejected_by_agent_action_policy_is_not_a_failed_provider_request(tmp_path):
    old, _ = make_oldrun(tmp_path)
    first = old / "episodes/first--feedback.json"
    data = json.loads(first.read_text())
    data["status"] = "model_error"
    data["error"] = "Unavailable final action was rejected after valid model output"
    first.write_text(json.dumps(data))
    mapping = prepare_replay_prefixes(old, ["first", "second"], tmp_path / "prefixes")
    assert len(json.loads(Path(mapping["first--feedback"]).read_text())["rows"]) == 2
