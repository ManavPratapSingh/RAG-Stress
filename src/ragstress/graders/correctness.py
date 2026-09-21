"""Answer and strict correctness grader across item kinds (SPEC §8.3, FR-GRADE-3)."""

from pydantic import BaseModel

from ragstress.gen.schema import GenOutput
from ragstress.graders.normalize import normalize_value


class CorrectnessGrade(BaseModel):
    """Correctness evaluation result."""

    pass_answer: bool
    pass_strict: bool
    normalized_answer: str | None
    normalize_failed: bool


def grade_correctness(
    kind: str,
    output: GenOutput,
    gold_value: str | None = None,
    gold_values: list[str] | None = None,
    value_type: str = "enum",
    gold_support_ids: list[str] | None = None,
    context_doc_ids: list[str] | None = None,
    format_error: str | None = None,
) -> CorrectnessGrade:
    """Evaluate pass_answer and pass_strict according to item kind."""
    if format_error is not None:
        return CorrectnessGrade(
            pass_answer=False,
            pass_strict=False,
            normalized_answer=None,
            normalize_failed=False,
        )

    support_ids = gold_support_ids or []
    context_ids = context_doc_ids or []
    cited_set = set(output.citations)
    support_set = set(support_ids)
    context_set = set(context_ids)

    no_fabricated = cited_set.issubset(context_set)

    # Normalize answer if provided
    norm_ans = None
    normalize_failed = False
    if output.answer is not None:
        norm_ans = normalize_value(output.answer, value_type)
        if norm_ans is None:
            normalize_failed = True

    # Normalize gold values
    norm_gold_value = None
    if gold_value is not None:
        ng = normalize_value(gold_value, value_type)
        norm_gold_value = ng if ng is not None else gold_value.strip().lower()

    norm_gold_values: set[str] = set()
    if gold_values:
        for g in gold_values:
            ng = normalize_value(g, value_type)
            norm_gold_values.add(ng if ng is not None else g.strip().lower())
    elif norm_gold_value:
        norm_gold_values.add(norm_gold_value)

    pass_answer = False
    pass_strict = False

    if kind in {"answerable_plain", "multihop", "plain"}:
        # normalized answer == gold, abstain = false
        if not output.abstain and norm_ans is not None and norm_ans == norm_gold_value:
            pass_answer = True
            if pass_answer and cited_set == support_set and no_fabricated and not output.conflict:
                pass_strict = True

    elif kind in {"dated", "authority"}:
        # answer == gold (newer/official)
        if not output.abstain and norm_ans is not None and norm_ans == norm_gold_value:
            pass_answer = True
            if pass_answer and support_set.issubset(cited_set) and no_fabricated:
                pass_strict = True

    elif kind == "unresolvable":
        # conflict = true AND normalized conflicting_values set == gold_values
        norm_reported: set[str] = set()
        for v in output.conflicting_values:
            nv = normalize_value(v, value_type)
            norm_reported.add(nv if nv is not None else v.strip().lower())

        if output.conflict and norm_reported == norm_gold_values:
            pass_answer = True
            if (
                pass_answer
                and output.answer is None
                and support_set.issubset(cited_set)
                and no_fabricated
            ):
                pass_strict = True

    elif kind == "unanswerable":
        # abstain = true
        if output.abstain:
            pass_answer = True
            if (
                pass_answer
                and output.answer is None
                and output.citations == []
                and not output.conflict
            ):
                pass_strict = True

    return CorrectnessGrade(
        pass_answer=pass_answer,
        pass_strict=pass_strict,
        normalized_answer=norm_ans,
        normalize_failed=normalize_failed,
    )
