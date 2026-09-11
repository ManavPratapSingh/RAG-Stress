"""Suite and item schemas (SPEC §9.1, FR-SUITE-1)."""

from typing import Literal

from pydantic import BaseModel

from ragstress.stress.spec import StressCondition


class SuiteItem(BaseModel):
    """A single evaluation benchmark item."""

    item_id: str
    split: Literal["dev", "test"]
    group: str
    question_id: str
    seed: int
    condition: StressCondition
