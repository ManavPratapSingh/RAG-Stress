"""Conflict handling grading (SPEC §8.5, FR-GRADE-5)."""

from pydantic import BaseModel

from ragstress.graders.normalize import normalize_value


class ConflictGrade(BaseModel):
    """Conflict evaluation results."""

    conflict_flagged: bool
    values_complete: bool
    false_conflict_flag: bool
    flagged_and_resolved: bool


def grade_conflict(
    kind: str,
    conflict_flag: bool,
    conflicting_values: list[str],
    gold_values: list[str] | None,
    value_type: str,
    pass_answer: bool,
) -> ConflictGrade:
    """Evaluate conflict detection, completeness, and resolution flags."""
    # Complete values check for unresolvable
    values_complete = False
    if kind == "unresolvable" and gold_values:
        norm_gold = set()
        for g in gold_values:
            ng = normalize_value(g, value_type)
            norm_gold.add(ng if ng is not None else g.strip().lower())

        norm_reported = set()
        for r in conflicting_values:
            nr = normalize_value(r, value_type)
            norm_reported.add(nr if nr is not None else r.strip().lower())

        values_complete = norm_gold == norm_reported

    # False conflict flag: flagged on an item without a genuine unresolvable conflict
    false_conflict_flag = conflict_flag and (kind != "unresolvable")

    # Flagged and resolved: for dated/authority, flag true and answer correct
    flagged_and_resolved = False
    if kind in {"dated", "authority"}:
        flagged_and_resolved = conflict_flag and pass_answer

    return ConflictGrade(
        conflict_flagged=conflict_flag,
        values_complete=values_complete,
        false_conflict_flag=false_conflict_flag,
        flagged_and_resolved=flagged_and_resolved,
    )
