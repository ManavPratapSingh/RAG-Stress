"""Citation grading (SPEC §8.4, FR-GRADE-4)."""

from pydantic import BaseModel


class CitationGrade(BaseModel):
    """Citation grading evaluation results."""

    cited_ids: list[str]
    support_ids: list[str]
    fabricated_citations: list[str]
    precision: float | None
    recall: float
    exact: bool


def grade_citations(
    cited_ids: list[str],
    support_ids: list[str],
    context_doc_ids: list[str],
) -> CitationGrade:
    """Evaluate citation precision, recall, exactness, and hallucinated/fabricated IDs."""
    cited_set = set(cited_ids)
    support_set = set(support_ids)
    context_set = set(context_doc_ids)

    fabricated = sorted(cited_set - context_set)

    # Precision: |cited ∩ support| / |cited| (None if cited is empty)
    if not cited_set:
        precision = None
    else:
        precision = len(cited_set & support_set) / len(cited_set)

    # Recall: |cited ∩ support| / |support| (1.0 if support is empty and cited is empty)
    if not support_set:
        recall = 1.0 if not cited_set else 0.0
    else:
        recall = len(cited_set & support_set) / len(support_set)

    exact = cited_set == support_set

    return CitationGrade(
        cited_ids=cited_ids,
        support_ids=support_ids,
        fabricated_citations=fabricated,
        precision=precision,
        recall=recall,
        exact=exact,
    )
