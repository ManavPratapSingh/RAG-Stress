import random

from ragstress.kb.facts import (
    generate_facts_for_entity,
)
from ragstress.kb.questions import (
    ATTRIBUTE_LABELS,
    QUESTION_TEMPLATES,
    Question,
    generate_question_for_fact,
)


def test_question_templates_count() -> None:
    """Requirement FR-KB-5: >= 3 paraphrases per attribute."""
    assert len(QUESTION_TEMPLATES) >= 3


def test_question_names_entity_and_label() -> None:
    """Requirement FR-KB-5: Question text MUST name exact entity and attribute label."""
    rng = random.Random(77)
    facts = generate_facts_for_entity("ent-fac-005", "Mirenta Data Center", "Facility", rng)

    for fact in facts:
        q = generate_question_for_fact("q-001", fact, rng)
        assert isinstance(q, Question)
        assert fact.entity_name in q.text, (
            f"Question '{q.text}' does not name entity '{fact.entity_name}'"
        )
        expected_label = ATTRIBUTE_LABELS[fact.attribute]
        assert expected_label in q.text, (
            f"Question '{q.text}' does not name label '{expected_label}'"
        )
        assert q.gold_value == fact.value
        assert q.gold_fact_id == fact.fact_id
