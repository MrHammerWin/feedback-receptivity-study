"""Evaluation state stays inside adapters; the agent receives text and public legal actions only."""
import hashlib
import importlib.resources
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Observation:
    text: str
    mission: str
    actions: list[str] = field(default_factory=list)
    reward: float = 0.0
    terminated: bool = False
    success: bool = False
    background: str = ''


class TextCraftEnv:
    protocol = 'textcraft-pypi-0.0.3; official ADaPT task split not asserted'

    def __init__(self, task: str, seed: int, config: dict):
        from textcraft import TextCraft
        # PyPI 0.0.3 has a context-manager default instead of a real data path.
        data = str(importlib.resources.files('textcraft') / 'data')
        self.factory = TextCraft
        self.data_path = data
        self.env = None
        self.seed = seed
        self.recipe_text = ''

    def reset(self) -> Observation:
        if self.env is not None:
            self.env.close()
        # The upstream crafting tree mutates recipe order. A fresh instance is needed per reset.
        self.env = self.factory(minecraft_dir=self.data_path)
        text, _ = self.env.reset(seed=self.seed)
        self.recipe_text = text
        self.background = self._text('')
        self.mission = text.split('Goal:')[-1].strip()
        self.task_id = hashlib.sha256(self.mission.encode()).hexdigest()
        return Observation('Initial inventory is empty.', self.mission, background=self.background)

    def _text(self, feedback: str) -> str:
        # The initial recipe sheet is available to EVERY variant, even after history truncation.
        rules = ('Commands: get <positive integer> <base item>; craft <quantity> <item> using '
                 '<quantity> <ingredient>, ...; inventory. Craftable items cannot be obtained with get. '
                 'One craft action executes the recipe ONCE. Use the exact recipe quantities shown. '
                 'Replace generic recipe ingredients with concrete inventory items, for example '
                 'use oak planks instead of the generic tag planks. '
                 'To make several batches, repeat the recipe command; changing its leading number '
                 'does not repeat the recipe.')
        return f'{rules}\n{self.recipe_text}\nLatest observation: {feedback}'

    def step(self, action: str) -> Observation:
        text, reward, done, truncated, _ = self.env.step(action)
        return Observation(text, self.mission, reward=float(reward),
                           terminated=bool(done or truncated), success=bool(reward > 0),
                           background=self.background)

    def close(self):
        if self.env is not None:
            self.env.close()


BABY_TASKS = ('goto', 'pickup', 'open', 'putnext', 'pick_up_seq_go_to')
BABY_ACTIONS = ('turn left', 'turn right', 'go forward', 'pick up', 'drop', 'toggle')


class BabyAITextEnv:
    protocol = 'BALROG MixedTrainLocal tasks, num_dists=0, native textual descriptions'

    def __init__(self, task: str, seed: int, config: dict):
        import gymnasium as gym
        import minigrid  # noqa: F401 -- registers the BALROG fork environments
        if 'BabyAI-MixedTrainLocal-v0' not in gym.registry:
            minigrid.register_minigrid_envs()
        if config.get('babyai_protocol') != 'balrog_mixed':
            raise ValueError('Only the explicitly named balrog_mixed protocol is implemented')
        if 'BabyAI-MixedTrainLocal-v0' not in gym.registry:
            raise RuntimeError('Install the pinned BALROG MiniGrid fork; standard MiniGrid is not equivalent')
        self.env = None
        self.seed = seed
        self.task = task
        self.effective_seed = seed
        self._gym = gym

    def reset(self) -> Observation:
        from gymnasium.utils import seeding
        from minigrid.envs.babyai.mixed_seq_levels import Level_MixedTrainLocal

        class SeededMixed(Level_MixedTrainLocal):
            def __init__(self, constructor_seed):
                # Upstream samples task type BEFORE reset(). Seed that constructor RNG as well.
                self._np_random, self._np_random_seed = seeding.np_random(constructor_seed)
                super().__init__(num_dists=0)

        # Task rejection sampling is deterministic and bounded, and does not inspect hidden geometry.
        for attempt in range(500):
            if self.env is not None:
                self.env.close()
            self.effective_seed = self.seed + 104729 * attempt
            self.env = SeededMixed(self.effective_seed)
            obs, info = self.env.reset(seed=self.effective_seed)
            kind = self.env.unwrapped.action_kinds[0].replace(' ', '_')
            if kind == self.task:
                break
        else:
            raise RuntimeError('Unable to generate requested BabyAI task in 500 attempts')
        self.mission = obs['mission']
        self.task_id = f'{self.task}:{self.effective_seed}'
        text = self._text(info)
        return Observation(text, self.mission, list(BABY_ACTIONS), background=self.rules)

    def _text(self, info: dict) -> str:
        if 'descriptions' not in info:
            raise RuntimeError('Missing official partial-observation text; refuse a full-state fallback')
        rules = ('You see a partial egocentric view. Left/right are relative to current facing. '
                 'To pick up or toggle, the object must be one step in front. '
                 'go forward moves one cell; turn left/right rotates without moving. '
                 'drop puts the carried item in the empty cell in front. '
                 'An object outside the current view is unknown, not absent.')
        self.rules = rules
        return '\n'.join(info['descriptions'])

    def step(self, action: str) -> Observation:
        _, reward, terminated, truncated, info = self.env.step(BABY_ACTIONS.index(action))
        text = self._text(info)
        return Observation(text, self.mission, list(BABY_ACTIONS), float(reward),
                           bool(terminated or truncated), bool(reward > 0), self.rules)

    def close(self):
        if self.env is not None:
            self.env.close()


class ALFWorldEnv:
    protocol = 'ALFWorld 0.4.2 textworld PDDL, public admissible commands, no expert/facts access'

    def __init__(self, task: str, seed: int, config: dict):
        path = Path(config.get('alfworld_data', 'data/alfworld')).resolve()
        os.environ['ALFWORLD_DATA'] = str(path)
        split = {'train': 'train', 'validation': 'valid_seen', 'test': 'valid_unseen'}[config['split']]
        games = sorted(p for p in path.rglob('*.tw-pddl') if split in p.parts)
        if not games:
            raise RuntimeError('ALFWorld data missing: run scripts/prepare_alfworld.py')
        self.game = games[seed % len(games)]
        self.task_id = str(self.game.relative_to(path))
        self.env = None

    def reset(self) -> Observation:
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
        self.env = AlfredDemangler(textworld.start(
            str(self.game), request_infos=textworld.EnvInfos(won=True, admissible_commands=True)))
        self.env.load(str(self.game))
        state = self.env.reset()
        self.mission = state.get('objective') or state.feedback
        return self._observe(state)

    def _observe(self, state, reward=0, done=False) -> Observation:
        return Observation(state.feedback, self.mission, list(state.admissible_commands or []),
                           float(reward), bool(done), bool(state.won))

    def step(self, action: str) -> Observation:
        state, reward, done = self.env.step(action)
        return self._observe(state, reward, done)

    def close(self):
        if self.env is not None:
            self.env.close()


def tasks_for(benchmark: str, config: dict | None = None) -> list[str]:
    native = list(BABY_TASKS) if benchmark == 'babyai_text' else ['default']
    selected = (config or {}).get('tasks', {}).get(benchmark, native)
    if not selected or len(selected) != len(set(selected)) or any(task not in native for task in selected):
        raise ValueError('Invalid explicit task subset')
    return selected


def make_env(benchmark: str, task: str, seed: int, config: dict):
    classes = {'textcraft': TextCraftEnv, 'babyai_text': BabyAITextEnv, 'alfworld': ALFWorldEnv}
    if benchmark not in classes:
        raise ValueError(f'Unsupported benchmark: {benchmark}')
    return classes[benchmark](task, seed, config)
