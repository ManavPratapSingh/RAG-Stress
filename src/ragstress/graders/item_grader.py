"""Item grading combining correctness, provenance, citations, and taxonomy (SPEC §8)."""

from pydantic import BaseModel

from ragstress.gen.schema import GenOutput, GenResult
from ragstress.graders.citations import grade_citations
from ragstress.graders.conflict import grade_conflict
from ragstress.graders.correctness import grade_correctness
from ragstress.graders.provenance import determine_provenance
from ragstress.graders.taxonomy import classify_failure


class GradedItem(BaseModel):
    """Complete grading outcome for a single evaluation item."""

    item_id: str
    kind: str
    value_type: str
    pass_answer: bool
    pass_strict: bool
    normalized_answer: str | None
    normalize_failed: bool
    provenance: str | None
    hallucinated: bool
    citation_precision: float | None
    citation_recall: float
    citation_exact: bool
    fabricated_citations: list[str]
    conflict_flagged: bool
    values_complete: bool
    false_conflict_flag: bool
    flagged_and_resolved: bool
    failure_mode: str | None
    format_error: str | None


def grade_item(
    item_id: str,
    kind: str,
    value_type: str,
    gen_result: GenResult,
    gold_value: str | None = None,
    gold_values: list[str] | None = None,
    twin_value: str | None = None,
    distractor_values: list[str] | None = None,
    gold_support_ids: list[str] | None = None,
    context_doc_ids: list[str] | None = None,
    retrieved_ids: list[str] | None = None,
    is_mode_b: bool = False,
) -> GradedItem:
    """Grade an item result comprehensively across all evaluation dimensions."""
    output = gen_result.output if gen_result.output is not None else GenOutput()
    format_error = gen_result.format_error

    support_ids = gold_support_ids or []
    ctx_ids = context_doc_ids or []
    ret_ids = retrieved_ids or ctx_ids

    # 1. Correctness
    corr = grade_correctness(
        kind=kind,
        output=output,
        gold_value=gold_value,
        gold_values=gold_values,
        value_type=value_type,
        gold_support_ids=support_ids,
        context_doc_ids=ctx_ids,
        format_error=format_error,
    )

    # 2. Provenance
    prov, hall = determine_provenance(
        answer=output.answer,
        value_type=value_type,
        gold_value=gold_value,
        twin_value=twin_value,
        distractor_values=distractor_values,
        gold_values=gold_values,
    )

    # 3. Citations
    cites = grade_citations(
        cited_ids=output.citations,
        support_ids=support_ids,
        context_doc_ids=ctx_ids,
    )

    # 4. Conflict
    conflict_res = grade_conflict(
        kind=kind,
        conflict_flag=output.conflict,
        conflicting_values=output.conflicting_values,
        gold_values=gold_values,
        value_type=value_type,
        pass_answer=corr.pass_answer,
    )

    # 5. Failure taxonomy
    failure_mode = None
    if not corr.pass_answer:
        failure_mode = classify_failure(
            format_error=format_error,
            is_mode_b=is_mode_b,
            kind=kind,
            gold_support_ids=support_ids,
            retrieved_ids=ret_ids,
            provenance=prov,
            abstain=output.abstain,
            conflict=output.conflict,
            values_complete=conflict_res.values_complete,
            answer_asserted=output.answer is not None,
        )

    return GradedItem(
        item_id=item_id,
        kind=kind,
        value_type=value_type,
        pass_answer=corr.pass_answer,
        pass_strict=corr.pass_strict,
        normalized_answer=corr.normalized_answer,
        normalize_failed=corr.normalize_failed,
        provenance=prov,
        hallucinated=hall,
        citation_precision=cites.precision,
        citation_recall=cites.recall,
        citation_exact=cites.exact,
        fabricated_citations=cites.fabricated_citations,
        conflict_flagged=conflict_res.conflict_flagged,
        values_complete=conflict_res.values_complete,
        false_conflict_flag=conflict_res.false_conflict_flag,
        flagged_and_resolved=conflict_res.flagged_and_resolved,
        failure_mode=failure_mode,
        format_error=format_error,
    )
