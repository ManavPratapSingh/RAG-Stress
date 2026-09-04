"""Stress engine package."""

from ragstress.stress.builder import Context, build_context
from ragstress.stress.conflict import build_conflict_twin
from ragstress.stress.distractors import (
    create_near_miss_name,
    generate_distractor_passages,
)
from ragstress.stress.position import assign_opaque_doc_ids, order_passages
from ragstress.stress.spec import DEFAULT_DISTRACTOR_MIX, StressCondition

__all__ = [
    "DEFAULT_DISTRACTOR_MIX",
    "Context",
    "StressCondition",
    "assign_opaque_doc_ids",
    "build_conflict_twin",
    "build_context",
    "create_near_miss_name",
    "generate_distractor_passages",
    "order_passages",
]
