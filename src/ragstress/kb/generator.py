"""Synthetic knowledge base generator (SPEC §4, FR-KB-*)."""

import random

from ragstress.kb.facts import (
    BaseKB,
    Fact,
    Passage,
    generate_facts_for_entity,
)
from ragstress.kb.names import generate_entity_name
from ragstress.kb.questions import Question, generate_question_for_fact
from ragstress.kb.render import render_passage


def generate_base_kb(
    seed: int,
    num_policies: int = 50,
    num_products: int = 50,
    num_facilities: int = 50,
) -> tuple[list[Passage], list[Fact], list[Question]]:
    """Generate a fully deterministic synthetic knowledge base.

    - 150 entities (50 Policy, 50 Product, 50 Facility)
    - 4 facts per entity (600 total facts)
    - 300 passages (2 facts per passage, 1-3 facts per passage invariant)
    - 600 questions (one per fact)
    - Byte-deterministic for any given seed
    """
    rng = random.Random(seed)
    existing_names: set[str] = set()

    entities: list[tuple[str, str, str]] = []  # (entity_id, entity_name, kind)

    # 1. Generate unique entities
    for i in range(1, num_policies + 1):
        eid = f"ent-pol-{i:03d}"
        ename = generate_entity_name("Policy", rng, existing_names)
        entities.append((eid, ename, "Policy"))

    for i in range(1, num_products + 1):
        eid = f"ent-prd-{i:03d}"
        ename = generate_entity_name("Product", rng, existing_names)
        entities.append((eid, ename, "Product"))

    for i in range(1, num_facilities + 1):
        eid = f"ent-fac-{i:03d}"
        ename = generate_entity_name("Facility", rng, existing_names)
        entities.append((eid, ename, "Facility"))

    # 2. Generate facts and passages for each entity
    all_passages: list[Passage] = []
    all_facts: list[Fact] = []
    doc_counter = 1

    for eid, ename, kind in entities:
        # Generate 4 facts for the entity
        entity_facts = generate_facts_for_entity(eid, ename, kind, rng, status="official")
        all_facts.extend(entity_facts)

        # Split 4 facts into 2 passages (2 facts each)
        part1_facts = entity_facts[:2]
        part2_facts = entity_facts[2:]

        # Passage 1
        doc_id_1 = f"doc-{doc_counter:04d}"
        doc_counter += 1
        p1_date = part1_facts[0].effective_date
        title1 = f"{ename} Documentation - Part 1"
        p1 = render_passage(
            doc_id=doc_id_1,
            title=title1,
            facts=part1_facts,
            status="official",
            effective_date=p1_date,
            rng=rng,
        )
        all_passages.append(p1)

        # Passage 2
        doc_id_2 = f"doc-{doc_counter:04d}"
        doc_counter += 1
        p2_date = part2_facts[0].effective_date
        title2 = f"{ename} Documentation - Part 2"
        p2 = render_passage(
            doc_id=doc_id_2,
            title=title2,
            facts=part2_facts,
            status="official",
            effective_date=p2_date,
            rng=rng,
        )
        all_passages.append(p2)

    # 3. Generate questions for all facts
    all_questions: list[Question] = []
    for idx, fact in enumerate(all_facts, start=1):
        qid = f"q-{idx:04d}"
        q = generate_question_for_fact(qid, fact, rng)
        all_questions.append(q)

    return all_passages, all_facts, all_questions


def generate_base_kb_container(seed: int = 42) -> BaseKB:
    """Generate and return BaseKB container object."""
    passages, facts, questions = generate_base_kb(seed=seed)
    return BaseKB(passages=passages, facts=facts, questions=questions)
