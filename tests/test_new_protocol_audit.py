"""Independent offline checks for the prospective protocol; no model backends."""

import json
from types import SimpleNamespace

import pytest

from crmn_agent import experiment
from crmn_agent.ablations import DraftOnlyAgent
from crmn_agent.codex_provider import completion_usage
from crmn_agent.environments import Observation
from crmn_agent.feedback_control import FeedbackAgent, FeedbackCritique
from crmn_agent.providers import BudgetExceeded, EpisodeBudgetExceeded, ModelOutputError, Provider, Usage
from crmn_agent.published_baselines import PublishedAgent
from crmn_agent.schemas import Decision


@pytest.mark.parametrize("usage", [
    None, {}, {"input_tokens": 1, "output_tokens": 1},
    {"input_tokens": True, "output_tokens": 1, "cached_input_tokens": 0},
    {"input_tokens": -1, "output_tokens": 1, "cached_input_tokens": 0},
    {"input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 2},
])
def test_missing_or_invalid_telemetry_cannot_become_free_zero_token_generation(usage):
    with pytest.raises(ModelOutputError):
        completion_usage([{"type": "turn.completed", "usage": usage}])


def test_cached_input_is_a_subset_not_an_extra_input_token_charge():
    value = completion_usage([
        {"type": "turn.completed", "usage": {
            "input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20,
        }},
        {"type": "turn.completed", "usage": {
            "input_tokens": 50, "cached_input_tokens": 5, "output_tokens": 3,
        }},
    ])
    assert value == {"input_tokens": 150, "cached_input_tokens": 85, "output_tokens": 23}


class RejectingProvider(Provider):
    def __init__(self, *, after_launch=False, call_log=None):
        super().__init__({"backend": "codex"}, call_log)
        self.after_launch = after_launch

    def _generate(self, system, user, schema, cap):
        if self.after_launch:
            self.last_launch = 42
        raise BudgetExceeded("Scripted checkpoint refusal")


@pytest.mark.parametrize("after_launch,expected_calls", [(False, 0), (True, 1)])
def test_checkpoint_before_launch_is_uncharged_but_postlaunch_attempt_is_retained(tmp_path, after_launch, expected_calls):
    path = tmp_path / "calls.jsonl"
    provider = RejectingProvider(after_launch=after_launch, call_log=path)
    with pytest.raises(BudgetExceeded):
        provider.complete(Decision, "toy", {}, 30)
    assert provider.usage.calls == expected_calls
    logged = json.loads(path.read_text())
    assert logged["logical_calls"] == expected_calls
    assert logged["usage_delta"]["calls"] == expected_calls


class ActionProvider:
    def __init__(self, action="inventory"):
        self.action, self.calls = action, []

    def complete(self, schema, purpose, payload, cap):
        self.calls.append((purpose, payload))
        return schema(reflection="A public action proposal.", action=self.action)


def test_all_final_transports_preserve_raw_commands_for_native_environment_feedback(tmp_path):
    action = " > inventory "
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "reflact_core_instruction.txt").write_text("Reflect on current progress and choose an action.")
    (assets / "reflact_alfworld_instruction.txt").write_text("An unused transport fixture.")
    raw = DraftOnlyAgent(ActionProvider(action)).act("Public.", "Goal.", ["inventory"], 0)
    published = PublishedAgent(ActionProvider(action), "reflact_paper", "textcraft", "toy", assets=assets)
    baseline = published.act("Public.", "Goal.", ["inventory"], 0)
    controlled = FeedbackAgent(ActionProvider(action), probe_policy="never", gate_mode="facts_only",
                               revision_only=True).revise_proposal(
        "Public.", "Goal.", ["inventory"], 0,
        Decision(reflection="Original proposal.", action="inventory"),
        FeedbackCritique(needs_revision=True, issue="Reconsider.", evidence="Public.", suggested_action="inventory"),
    )
    assert raw.action == baseline.action == controlled.action == action


@pytest.mark.parametrize("mode", ["always_critique", "facts_only", "grounding_only"])
def test_revision_only_control_keeps_approved_legal_draft_without_spending_final_call(mode):
    provider = ActionProvider("unused generated alternative")
    agent = FeedbackAgent(provider, probe_policy="never", gate_mode=mode, revision_only=True)
    draft = Decision(reflection="Public draft.", action="inventory")
    critic = FeedbackCritique(needs_revision=False, issue="No issue.", evidence="Public.", suggested_action="inventory")
    turn = agent.revise_proposal("Public.", "Goal.", ["inventory"], 0, draft, critic)
    assert turn.action == draft.action
    assert turn.diagnostics["intervention"] == "keep_draft"
    assert turn.diagnostics["revision_only"] is True
    assert provider.calls == []


class ToyEnvironment:
    task_id = "toy-task"
    protocol = "independent-offline-fixture"

    def __init__(self, *, terminal=True, success=True):
        self.terminal, self.success = terminal, success
        self.resets = self.steps = 0

    def reset(self):
        self.resets += 1
        return Observation("Public current room.", "Public goal.", ["inventory"])

    def step(self, action):
        self.steps += 1
        assert action == "inventory"
        return Observation("Native feedback.", "Public goal.", reward=int(self.success),
                           terminated=self.terminal, success=self.success)

    def close(self):
        pass


def setup_runner(tmp_path, monkeypatch, *, terminal=True, success=True, interrupt_second=False):
    from crmn_agent import study_provider

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [
        {"key": "toy-case", "benchmark": "textcraft", "seed": 1, "task_id": "toy-task"},
    ]}))
    output = tmp_path / "output"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "provider": {"backend": "codex", "study_cache_dir": str(tmp_path / "cache")},
        "manifest": str(manifest), "results_dir": str(output), "variants": ["feedback"],
        "horizons": {"textcraft": 2}, "episode_call_cap": 3, "resume_from_cache": True,
    }))
    environment = ToyEnvironment(terminal=terminal, success=success)
    instances = []

    class FakeScoped:
        def __init__(self, config, call_log):
            self.model, self.usage, self.replayed_seconds = "offline", Usage(), 0
            self.scopes = []
            instances.append(self)

        def begin_episode(self):
            pass

        def begin_scope(self, name, mode):
            self.scopes.append((name, mode))

        def assert_scope_exhausted(self):
            pass

    def build_agent(provider, *args):
        def act(observation, mission, actions, step, background):
            if step and interrupt_second:
                provider.usage.calls += 1
                raise EpisodeBudgetExceeded("Partial final turn")
            provider.usage.calls += 2
            provider.usage.input_tokens += 100
            provider.usage.output_tokens += 20
            return SimpleNamespace(action="inventory", diagnostics={"reflection": "Task Completed"})
        return SimpleNamespace(act=act, feedback=lambda *args: None)

    monkeypatch.setattr(study_provider, "ScopedCodexProvider", FakeScoped)
    monkeypatch.setattr(experiment, "make_env", lambda *args: environment)
    monkeypatch.setattr(experiment, "build_agent", build_agent)
    monkeypatch.setattr(experiment, "fingerprint", lambda: "frozen-source")
    monkeypatch.setattr(experiment.importlib.metadata, "version", lambda _: "mocked-offline-version")
    episode = output / "episodes/toy-case--feedback.json"
    return config_path, episode, environment, instances


def test_scoped_runner_records_step_calls_and_uses_environment_success_only(tmp_path, monkeypatch):
    config, episode, _, instances = setup_runner(tmp_path, monkeypatch, success=False)
    experiment.run(config)
    value = json.loads(episode.read_text())
    assert value["success"] is False and value["status"] == "terminated"
    assert value["trajectory"][0]["success"] is False
    assert value["trajectory"][0]["usage"]["calls"] == 2
    assert value["trajectory"][0]["end_logical_calls"] == 2
    assert value["unexecuted_turn_calls"] == 0
    assert instances[0].scopes == [("toy-case--feedback--step-0", "sequence")]


def test_unexecuted_control_calls_are_not_mistaken_for_executed_actions(tmp_path, monkeypatch):
    config, episode, environment, _ = setup_runner(tmp_path, monkeypatch, terminal=False,
                                                   success=False, interrupt_second=True)
    experiment.run(config)
    value = json.loads(episode.read_text())
    assert value["status"] == "episode_compute_cap"
    assert value["usage"]["calls"] == 3
    assert value["steps"] == environment.steps == 1
    assert value["unexecuted_turn_calls"] == 1


def test_crashed_running_checkpoint_is_not_automatically_retried(tmp_path, monkeypatch):
    config, episode, environment, _ = setup_runner(tmp_path, monkeypatch)
    experiment.run(config)
    previous = json.loads(episode.read_text())
    previous.update(status="running", finished=False)
    episode.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="explicitly adjudicated"):
        experiment.run(config)
    assert environment.resets == 1


def test_finished_model_error_is_never_automatically_retried_on_resume(tmp_path, monkeypatch):
    config, episode, environment, _ = setup_runner(tmp_path, monkeypatch)
    experiment.run(config)
    previous = json.loads(episode.read_text())
    previous.update(status="model_error", success=False, finished=True)
    episode.write_text(json.dumps(previous))
    experiment.run(config)
    assert environment.resets == 1
    assert json.loads(episode.read_text())["status"] == "model_error"


def test_only_explicit_budget_checkpoint_resume_archives_previous_state(tmp_path, monkeypatch):
    config, episode, environment, _ = setup_runner(tmp_path, monkeypatch)
    experiment.run(config)
    previous = json.loads(episode.read_text())
    previous.update(status="campaign_budget_censored", success=False, finished=True)
    episode.write_text(json.dumps(previous))
    experiment.run(config)
    assert environment.resets == 2
    assert json.loads(episode.read_text())["status"] == "success"
    assert len(list((episode.parent.parent / "partial_checkpoints").glob("*.json"))) == 1
