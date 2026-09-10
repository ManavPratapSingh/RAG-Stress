"""Failure taxonomy classification (SPEC §8.7, FR-GRADE-7)."""


def classify_failure(
    format_error: str | None,
    is_mode_b: bool,
    kind: str,
    gold_support_ids: list[str],
    retrieved_ids: list[str],
    provenance: str | None,
    abstain: bool,
    conflict: bool,
    values_complete: bool,
    answer_asserted: bool = False,
) -> str:
    """Classify failure mode on an item with pass_answer=False (first match wins)."""
    # 1. format_error
    if format_error is not None:
        return "format_error"

    # 2. retrieval_miss
    if is_mode_b and kind != "unanswerable":
        if not set(gold_support_ids).issubset(set(retrieved_ids)):
            return "retrieval_miss"

    # 3. stale_adopted
    if provenance == "conflict_loser":
        return "stale_adopted"

    # 4. distractor_adopted
    if provenance == "distractor":
        return "distractor_adopted"

    # 5. hallucinated
    if provenance == "none":
        return "hallucinated"

    # 6. false_abstain
    if kind != "unanswerable" and abstain:
        return "false_abstain"

    # 7. conflict_ignored
    if kind == "unresolvable" and not conflict and answer_asserted:
        return "conflict_ignored"

    # 8. conflict_incomplete
    if kind == "unresolvable" and conflict and not values_complete:
        return "conflict_incomplete"

    # 9. other_wrong
    return "other_wrong"
