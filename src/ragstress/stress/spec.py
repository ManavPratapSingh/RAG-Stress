"""Stress condition model (SPEC §6.1, FR-STRESS-1)."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DistractorType = Literal[
    "same_attr_other_entity",
    "same_entity_other_attr",
    "near_miss_entity",
    "random_irrelevant",
]

DEFAULT_DISTRACTOR_MIX: dict[str, float] = {
    "same_attr_other_entity": 0.4,
    "same_entity_other_attr": 0.3,
    "near_miss_entity": 0.2,
    "random_irrelevant": 0.1,
}


class StressCondition(BaseModel):
    k_noise: int = 0
    position: Literal["top", "middle", "bottom", "random"] = "random"
    conflict: Literal["none", "dated", "authority", "unresolvable"] = "none"
    answerable: bool = True
    distractor_mix: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_DISTRACTOR_MIX))
    mode: Literal["A", "B"] = "A"
    k_retrieve: int | None = None

    @model_validator(mode="after")
    def validate_condition(self) -> "StressCondition":
        if not self.answerable and self.conflict != "none":
            raise ValueError("answerable=False with conflict != 'none' is invalid")
        return self

    @property
    def item_kind(self) -> str:
        """Derived item kind per SPEC §6.1."""
        if not self.answerable:
            return "unanswerable"
        if self.conflict == "none":
            return "answerable_plain"
        return self.conflict
