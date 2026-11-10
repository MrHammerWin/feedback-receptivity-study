"""Attributed published scaffolds on the same subscription model.

StateAct prompts are extracted from its pinned official code; ReflAct is a
paper-based reproduction. JSON transport, public legal commands and common
environment adapters are adaptations, not the authors' original evaluation.
"""
from pydantic import Field

from .agents import Turn
from .asset_download import resolve_assets
from .schemas import Schema


class StateActReply(Schema):
    goal: str = Field(max_length=1000)
    locations_visited: str = Field(max_length=1500)
    current_location: str = Field(max_length=500)
    current_inventory: str = Field(max_length=1200)
    thought: str = Field(max_length=1600)
    action: str = Field(max_length=240)


class ReflActReply(Schema):
    reflection: str = Field(max_length=2200)
    action: str = Field(max_length=240)


class TextCraftStateActReply(Schema):
    goal: str = Field(max_length=1000)
    inventory: str = Field(max_length=1500)
    think: str = Field(max_length=1600)
    action: str = Field(max_length=240)


def alfworld_family(task_id: str) -> str:
    for key, family in [('pick_clean', 'clean'), ('pick_heat', 'heat'), ('pick_cool', 'cool'),
                        ('pick_two', 'puttwo'), ('look_at_obj', 'examine')]:
        if key in task_id:
            return family
    return 'put'


class PublishedAgent:
    def __init__(self, provider, variant, benchmark, task_id, assets=None):
        if variant not in {'stateact', 'reflact_paper'}:
            raise ValueError('Unknown published baseline')
        self.provider, self.variant = provider, variant
        self.benchmark, self.history = benchmark, []
        assets = resolve_assets(assets)
        if variant == 'stateact':
            name = f'stateact_alfworld_{alfworld_family(task_id)}_system.txt' if benchmark == 'alfworld' else 'stateact_textcraft_system.txt'
            self.instruction = (assets / name).read_text()
            self.schema = StateActReply if benchmark == 'alfworld' else TextCraftStateActReply
        else:
            self.instruction = (assets / 'reflact_alfworld_instruction.txt').read_text()
            if benchmark == 'alfworld':
                example = 'clean' if alfworld_family(task_id) == 'clean' else 'put'
                self.instruction += '\nWORKED EXAMPLE:\n' + (assets / f'reflact_alfworld_{example}_example.txt').read_text()
            else:
                self.instruction = (assets / 'reflact_core_instruction.txt').read_text()
            self.schema = ReflActReply

    def act(self, observation, mission, actions, step, background=''):
        reply = self.provider.complete(
            self.schema,
            self.instruction + '\nUse the JSON transport schema. Choose exactly ONE executable game action. '
            'Do not claim task completion instead of acting. Preserve full reasoning/state fields as needed.',
            {'mission': mission, 'observation': observation, 'background': background,
             'available_actions': actions, 'full_history': self.history}, 1400,
        )
        self.history.append({'observation': observation, 'response': reply.model_dump()})
        return Turn(reply.action, reply.model_dump())

    def feedback(self, success, terminated):
        pass
