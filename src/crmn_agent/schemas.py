from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


ShortText = Annotated[str, Field(max_length=120)]


class Fact(Schema):
    key: str = Field(max_length=48)
    value: ShortText


class Anchor(Schema):
    key: str = Field(max_length=48)
    value: ShortText
    evidence: str = Field(max_length=200)


class Grounding(Schema):
    facts: list[Anchor] = Field(max_length=4)


class Representation(Schema):
    summary: str = Field(max_length=240)
    facts: list[Fact] = Field(max_length=4)
    subgoals: list[ShortText] = Field(max_length=4)
    action: str = Field(max_length=200)
    expected: list[ShortText] = Field(max_length=2)
    value: float = Field(ge=0, le=1)


class Reconstruction(Schema):
    facts: list[Fact] = Field(max_length=4)


class Verdict(Schema):
    prediction: ShortText
    status: Literal["supported", "contradicted", "unknown"]
    evidence: str = Field(max_length=200)


class Audit(Schema):
    verdicts: list[Verdict] = Field(max_length=2)


class Decision(Schema):
    reflection: str = Field(max_length=120)
    action: str = Field(max_length=200)

