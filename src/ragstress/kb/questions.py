"""Question model and question generator (SPEC §4.5)."""

import random

from pydantic import BaseModel

from ragstress.kb.facts import Fact

QUESTION_TEMPLATES: list[str] = [
    "What is the {label} for {entity_name}?",
    "According to official documentation, what is the {label} specified for {entity_name}?",
    "Can you provide the {label} of {entity_name}?",
    "What is the stated {label} applicable to {entity_name}?",
]


class Question(BaseModel):
    question_id: str
    entity_id: str
    attribute: str
    text: str
    gold_value: str
    gold_fact_id: str


ATTRIBUTE_LABELS: dict[str, str] = {
    "notice_period": "notice period",
    "stipend_amount": "monthly stipend",
    "approval_threshold": "approval threshold",
    "review_cycle": "review cycle",
    "max_throughput": "maximum throughput",
    "warranty_period": "warranty period",
    "list_price": "list price",
    "max_operating_temp": "maximum operating temperature",
    "capacity": "capacity",
    "opening_date": "opening date",
    "floor_count": "number of floors",
    "security_level": "security level",
}


def generate_question_for_fact(
    question_id: str,
    fact: Fact,
    rng: random.Random,
) -> Question:
    """Generate a question for a fact, naming the exact entity name and attribute label."""
    label = ATTRIBUTE_LABELS.get(fact.attribute, fact.attribute.replace("_", " "))
    template = rng.choice(QUESTION_TEMPLATES)
    text = template.format(label=label, entity_name=fact.entity_name)

    return Question(
        question_id=question_id,
        entity_id=fact.entity_id,
        attribute=fact.attribute,
        text=text,
        gold_value=fact.value,
        gold_fact_id=fact.fact_id,
    )
