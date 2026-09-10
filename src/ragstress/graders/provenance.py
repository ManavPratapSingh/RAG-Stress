"""Value provenance grader (SPEC §8.2, FR-GRADE-2)."""

from ragstress.graders.normalize import normalize_value


def determine_provenance(
    answer: str | None,
    value_type: str,
    gold_value: str | None = None,
    twin_value: str | None = None,
    distractor_values: list[str] | None = None,
    gold_values: list[str] | None = None,
) -> tuple[str | None, bool]:
    """Classify answer value provenance: gold, conflict_loser, distractor, none.

    Returns:
        (provenance, hallucinated): provenance is one of ("gold", "conflict_loser",
        "distractor", "none") or None if no answer was asserted. hallucinated is True
        iff provenance == "none".
    """
    if answer is None:
        return None, False

    norm_ans = normalize_value(answer, value_type)
    effective_ans = norm_ans if norm_ans is not None else answer.strip().lower()

    # 1. Gold check
    gold_set: set[str] = set()
    if gold_values:
        for g in gold_values:
            ng = normalize_value(g, value_type)
            gold_set.add(ng if ng is not None else g.strip().lower())
    elif gold_value:
        ng = normalize_value(gold_value, value_type)
        gold_set.add(ng if ng is not None else gold_value.strip().lower())

    if effective_ans in gold_set:
        return "gold", False

    # 2. Conflict loser check
    if twin_value is not None:
        nt = normalize_value(twin_value, value_type)
        effective_twin = nt if nt is not None else twin_value.strip().lower()
        if effective_ans == effective_twin:
            return "conflict_loser", False

    # 3. Distractor check
    if distractor_values:
        for dv in distractor_values:
            ndv = normalize_value(dv, value_type)
            effective_dv = ndv if ndv is not None else dv.strip().lower()
            if effective_ans == effective_dv:
                return "distractor", False

    # 4. None (appears in no passage in context)
    return "none", True
