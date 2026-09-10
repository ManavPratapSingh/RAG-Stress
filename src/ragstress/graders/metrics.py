"""Metrics aggregation across slices of graded items (SPEC §8.6, FR-GRADE-6)."""

from pydantic import BaseModel

from ragstress.graders.item_grader import GradedItem


class AggregateMetrics(BaseModel):
    """Aggregate benchmark metrics across a slice of items."""

    total_items: int
    accuracy: float
    strict_accuracy: float
    hallucination_rate: float
    hallucination_rate_answered: float | None
    distractor_adoption_rate: float
    conflict_loser_adoption_rate: float
    false_abstain_rate: float | None
    format_error_rate: float
    abstain_recall: float | None
    abstain_precision: float | None
    leakage_rate: float | None
    unneeded_conflict_flag_rate: float | None
    mean_citation_precision: float | None
    mean_citation_recall: float
    citation_exact_rate: float
    fabricated_citation_rate: float


def compute_aggregate_metrics(items: list[GradedItem]) -> AggregateMetrics:
    """Compute benchmark summary metrics over a collection of GradedItems."""
    n = len(items)
    if n == 0:
        return AggregateMetrics(
            total_items=0,
            accuracy=0.0,
            strict_accuracy=0.0,
            hallucination_rate=0.0,
            hallucination_rate_answered=None,
            distractor_adoption_rate=0.0,
            conflict_loser_adoption_rate=0.0,
            false_abstain_rate=None,
            format_error_rate=0.0,
            abstain_recall=None,
            abstain_precision=None,
            leakage_rate=None,
            unneeded_conflict_flag_rate=None,
            mean_citation_precision=None,
            mean_citation_recall=0.0,
            citation_exact_rate=0.0,
            fabricated_citation_rate=0.0,
        )

    acc = sum(1 for it in items if it.pass_answer) / n
    strict_acc = sum(1 for it in items if it.pass_strict) / n
    hallucinations = sum(1 for it in items if it.hallucinated)
    hall_rate = hallucinations / n

    answered_items = [it for it in items if it.normalized_answer is not None]
    hall_answered = (
        sum(1 for it in answered_items if it.hallucinated) / len(answered_items)
        if answered_items
        else None
    )

    distractor_adoptions = sum(1 for it in items if it.provenance == "distractor") / n
    conflict_loser_adoptions = sum(1 for it in items if it.provenance == "conflict_loser") / n

    answerable = [it for it in items if it.kind != "unanswerable"]
    false_abstain_rate = (
        sum(1 for it in answerable if it.failure_mode == "false_abstain") / len(answerable)
        if answerable
        else None
    )

    format_err_rate = sum(1 for it in items if it.format_error is not None) / n

    unanswerable = [it for it in items if it.kind == "unanswerable"]
    all_abstains = [
        it
        for it in items
        if it.failure_mode == "false_abstain" or (it.kind == "unanswerable" and it.pass_answer)
    ]
    abstain_recall = (
        sum(1 for it in unanswerable if it.pass_answer) / len(unanswerable)
        if unanswerable
        else None
    )
    abstain_precision = (
        sum(1 for it in unanswerable if it.pass_answer) / len(all_abstains)
        if all_abstains
        else None
    )

    # Leakage: unanswerable items where answer == gold
    leakage = (
        sum(1 for it in unanswerable if it.provenance == "gold") / len(unanswerable)
        if unanswerable
        else None
    )

    eligible_conflict = [it for it in items if it.kind != "unresolvable"]
    unneeded_conflict_rate = (
        sum(1 for it in eligible_conflict if it.false_conflict_flag) / len(eligible_conflict)
        if eligible_conflict
        else None
    )

    valid_precisions = [it.citation_precision for it in items if it.citation_precision is not None]
    mean_cite_prec = sum(valid_precisions) / len(valid_precisions) if valid_precisions else None
    mean_cite_rec = sum(it.citation_recall for it in items) / n
    cite_exact = sum(1 for it in items if it.citation_exact) / n
    fab_cite_rate = sum(1 for it in items if len(it.fabricated_citations) > 0) / n

    return AggregateMetrics(
        total_items=n,
        accuracy=acc,
        strict_accuracy=strict_acc,
        hallucination_rate=hall_rate,
        hallucination_rate_answered=hall_answered,
        distractor_adoption_rate=distractor_adoptions,
        conflict_loser_adoption_rate=conflict_loser_adoptions,
        false_abstain_rate=false_abstain_rate,
        format_error_rate=format_err_rate,
        abstain_recall=abstain_recall,
        abstain_precision=abstain_precision,
        leakage_rate=leakage,
        unneeded_conflict_flag_rate=unneeded_conflict_rate,
        mean_citation_precision=mean_cite_prec,
        mean_citation_recall=mean_cite_rec,
        citation_exact_rate=cite_exact,
        fabricated_citation_rate=fab_cite_rate,
    )
