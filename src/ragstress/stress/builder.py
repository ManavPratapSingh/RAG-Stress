"""Context builder for Mode A stress experiments (SPEC §6.6-§6.8)."""

import random
from typing import Any

from pydantic import BaseModel, Field

from ragstress.kb.facts import Fact, Passage
from ragstress.kb.questions import Question
from ragstress.stress.conflict import build_conflict_twin
from ragstress.stress.distractors import generate_distractor_passages
from ragstress.stress.position import assign_opaque_doc_ids, order_passages
from ragstress.stress.spec import StressCondition


class Context(BaseModel):
    passages: list[Passage]
    labels: list[dict[str, Any]] = Field(default_factory=list)
    gold_support_ids: list[str] = Field(default_factory=list)
    relevant_ids: list[str] = Field(default_factory=list)
    gold_values: list[str] = Field(default_factory=list)


def build_context(
    question: Question,
    base_passages: list[Passage],
    all_facts: list[Fact],
    condition: StressCondition,
    seed: int,
    item_id: str,
) -> Context:
    """Build a deterministic context for an item under a StressCondition (SPEC §6)."""
    rng = random.Random(seed)

    # 1. Locate gold fact and gold passage
    gold_fact = next((f for f in all_facts if f.fact_id == question.gold_fact_id), None)
    if gold_fact is None:
        raise ValueError(f"Gold fact '{question.gold_fact_id}' not found")

    gold_passage = next(
        (p for p in base_passages if any(f.fact_id == gold_fact.fact_id for f in p.facts)),
        None,
    )
    if gold_passage is None:
        raise ValueError(f"Base passage for fact '{gold_fact.fact_id}' not found")

    # 2. Build conflict twin if applicable
    twin_passage: Passage | None = None
    twin_fact: Fact | None = None
    if condition.conflict != "none":
        twin_passage = build_conflict_twin(gold_fact, condition.conflict, rng)
        twin_fact = twin_passage.facts[0]

    # 3. Handle answerable vs unanswerable
    if not condition.answerable:
        # k_noise = 0 yields empty context
        if condition.k_noise == 0:
            return Context(
                passages=[],
                labels=[],
                gold_support_ids=[],
                relevant_ids=[],
                gold_values=[],
            )

        # Distractor generation without gold or twin
        distractors = generate_distractor_passages(
            question=question,
            gold_fact=gold_fact,
            base_passages=base_passages,
            all_facts=all_facts,
            k_noise=condition.k_noise,
            distractor_mix=condition.distractor_mix,
            rng=rng,
        )

        # Shuffle distractors
        rng.shuffle(distractors)
        ordered_entries = [(p, role, idx) for idx, (p, role) in enumerate(distractors)]
        final_passages = assign_opaque_doc_ids(ordered_entries, seed, item_id)

        labels = [
            {
                "doc_id": p.doc_id,
                "internal_id": ordered_entries[idx][0].doc_id,
                "role": p.role,
                "position": idx,
                "status": p.status,
                "effective_date": p.effective_date.isoformat(),
                "facts": [f.model_dump(mode="json") for f in p.facts],
            }
            for idx, p in enumerate(final_passages)
        ]

        return Context(
            passages=final_passages,
            labels=labels,
            gold_support_ids=[],
            relevant_ids=[],
            gold_values=[],
        )

    # 4. Answerable item: assemble passages
    others: list[tuple[Passage, str]] = []

    # Handle unresolvable conflict: both docs are considered gold support
    if condition.conflict == "unresolvable" and twin_passage is not None:
        # For unresolvable, "gold" is the one with smaller internal doc_id
        if gold_passage.doc_id < twin_passage.doc_id:
            primary_gold = gold_passage
            secondary = (twin_passage, "twin")
        else:
            primary_gold = twin_passage
            secondary = (gold_passage, "twin")
        others.append(secondary)
    elif twin_passage is not None:
        primary_gold = gold_passage
        others.append((twin_passage, "twin"))
    else:
        primary_gold = gold_passage

    # Generate distractors
    distractors = generate_distractor_passages(
        question=question,
        gold_fact=gold_fact,
        base_passages=base_passages,
        all_facts=all_facts,
        k_noise=condition.k_noise,
        distractor_mix=condition.distractor_mix,
        rng=rng,
        twin_fact=twin_fact,
    )
    others.extend(distractors)

    # 5. Order passages by position condition
    ordered_entries = order_passages(primary_gold, others, condition.position, rng)

    # 6. Assign opaque doc IDs
    final_passages = assign_opaque_doc_ids(ordered_entries, seed, item_id)

    # 7. Collect support and relevance labels
    labels = [
        {
            "doc_id": p.doc_id,
            "internal_id": ordered_entries[idx][0].doc_id,
            "role": p.role,
            "position": idx,
            "status": p.status,
            "effective_date": p.effective_date.isoformat(),
            "facts": [f.model_dump(mode="json") for f in p.facts],
        }
        for idx, p in enumerate(final_passages)
    ]

    gold_support_ids = []
    relevant_ids = []

    if condition.conflict == "unresolvable":
        gold_values = sorted([gold_fact.value, twin_fact.value]) if twin_fact else [gold_fact.value]
        gold_support_ids = [p.doc_id for p in final_passages if p.role in {"gold", "twin"}]
        relevant_ids = list(gold_support_ids)
    elif condition.conflict in {"dated", "authority"}:
        gold_values = [gold_fact.value]
        gold_support_ids = [p.doc_id for p in final_passages if p.role == "gold"]
        relevant_ids = [p.doc_id for p in final_passages if p.role in {"gold", "twin"}]
    else:
        gold_values = [gold_fact.value]
        gold_support_ids = [p.doc_id for p in final_passages if p.role == "gold"]
        relevant_ids = list(gold_support_ids)

    return Context(
        passages=final_passages,
        labels=labels,
        gold_support_ids=gold_support_ids,
        relevant_ids=relevant_ids,
        gold_values=gold_values,
    )
