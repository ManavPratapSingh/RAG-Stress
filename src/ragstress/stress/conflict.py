"""Conflict twins construction (SPEC §6.3, FR-STRESS-3)."""

import random
from datetime import timedelta
from typing import Literal

from ragstress.kb.facts import (
    ENTITY_KINDS,
    Fact,
    Passage,
    generate_attribute_value,
)
from ragstress.kb.render import render_passage


def build_conflict_twin(
    gold_fact: Fact,
    conflict_type: Literal["dated", "authority", "unresolvable"],
    rng: random.Random,
) -> Passage:
    """Build a conflict twin passage for the same (entity, attribute) with value' != value."""
    # Find attribute definition
    attr_defs = []
    for attrs in ENTITY_KINDS.values():
        for ad in attrs:
            if ad.attribute == gold_fact.attribute:
                attr_defs.append(ad)
    if not attr_defs:
        raise ValueError(f"Unknown attribute {gold_fact.attribute}")
    attr_def = attr_defs[0]

    # Generate twin value != gold_fact.value
    twin_val = gold_fact.value
    for _ in range(100):
        val = generate_attribute_value(attr_def, rng)
        if val != gold_fact.value:
            twin_val = val
            break

    # Determine date and status
    if conflict_type == "dated":
        # Gold is official, newer. Twin is official, older by >= 30 days
        twin_date = gold_fact.effective_date - timedelta(days=rng.randint(30, 180))
        twin_status = "official"
    elif conflict_type == "authority":
        # Gold is official. Twin is draft, newer by >= 30 days
        twin_date = gold_fact.effective_date + timedelta(days=rng.randint(30, 180))
        twin_status = "draft"
    elif conflict_type == "unresolvable":
        # Same date, same official status
        twin_date = gold_fact.effective_date
        twin_status = "official"
    else:
        raise ValueError(f"Invalid conflict type: {conflict_type}")

    twin_fact = Fact(
        fact_id=f"twin-{gold_fact.fact_id}",
        entity_id=gold_fact.entity_id,
        entity_name=gold_fact.entity_name,
        attribute=gold_fact.attribute,
        value=twin_val,
        value_type=gold_fact.value_type,
        unit=gold_fact.unit,
        effective_date=twin_date,
        status=twin_status,  # type: ignore[arg-type]
    )

    doc_id = f"twin-{gold_fact.fact_id}"
    title = f"{gold_fact.entity_name} Documentation (Update)"
    return render_passage(
        doc_id=doc_id,
        title=title,
        facts=[twin_fact],
        status=twin_status,
        effective_date=twin_date,
        rng=rng,
    )
