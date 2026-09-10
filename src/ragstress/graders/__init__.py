"""Deterministic grading pipeline and failure taxonomy (SPEC §8)."""

from ragstress.graders.citations import CitationGrade, grade_citations
from ragstress.graders.conflict import ConflictGrade, grade_conflict
from ragstress.graders.correctness import CorrectnessGrade, grade_correctness
from ragstress.graders.item_grader import GradedItem, grade_item
from ragstress.graders.metrics import AggregateMetrics, compute_aggregate_metrics
from ragstress.graders.normalize import normalize_value
from ragstress.graders.provenance import determine_provenance
from ragstress.graders.taxonomy import classify_failure

__all__ = [
    "AggregateMetrics",
    "CitationGrade",
    "ConflictGrade",
    "CorrectnessGrade",
    "GradedItem",
    "classify_failure",
    "compute_aggregate_metrics",
    "determine_provenance",
    "grade_citations",
    "grade_conflict",
    "grade_correctness",
    "grade_item",
    "normalize_value",
]
