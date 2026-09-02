"""Fact models, attribute definitions, and base KB generation (SPEC §4.2, §4.3)."""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field


class Fact(BaseModel):
    fact_id: str
    entity_id: str
    entity_name: str
    attribute: str
    value: str  # canonical string form (see §8.1)
    value_type: Literal["int", "money", "date", "duration", "enum"]
    unit: str | None = None
    effective_date: date  # 2022-01-01 … 2025-05-31
    status: Literal["official", "draft"]


class Passage(BaseModel):
    doc_id: str  # internal id; replaced by opaque id in contexts (§6.5)
    title: str
    text: str
    status: Literal["official", "draft"]
    effective_date: date
    facts: list[Fact] = Field(default_factory=list)  # ground truth of what the text states (1–3)
    role: str | None = None  # set only inside a built context, see §6.6


@dataclass(frozen=True)
class AttributeDef:
    attribute: str
    value_type: Literal["int", "money", "date", "duration", "enum"]
    unit: str | None
    label: str


POLICY_ATTRIBUTES = [
    AttributeDef("notice_period", "duration", "days", "notice period"),
    AttributeDef("stipend_amount", "money", "USD", "monthly stipend"),
    AttributeDef("approval_threshold", "money", "USD", "approval threshold"),
    AttributeDef("review_cycle", "duration", "months", "review cycle"),
]

PRODUCT_ATTRIBUTES = [
    AttributeDef("max_throughput", "int", "requests/s", "maximum throughput"),
    AttributeDef("warranty_period", "duration", "months", "warranty period"),
    AttributeDef("list_price", "money", "USD", "list price"),
    AttributeDef("max_operating_temp", "int", "°C", "maximum operating temperature"),
]

FACILITY_ATTRIBUTES = [
    AttributeDef("capacity", "int", "people", "capacity"),
    AttributeDef("opening_date", "date", None, "opening date"),
    AttributeDef("floor_count", "int", "floors", "number of floors"),
    AttributeDef("security_level", "enum", None, "security level"),
]

ENTITY_KINDS = {
    "Policy": POLICY_ATTRIBUTES,
    "Product": PRODUCT_ATTRIBUTES,
    "Facility": FACILITY_ATTRIBUTES,
}

SECURITY_LEVELS = ["low", "medium", "high", "restricted"]


def _random_date(rng: random.Random, start_date: date, end_date: date) -> date:
    days_range = (end_date - start_date).days
    random_days = rng.randint(0, days_range)
    return start_date + timedelta(days=random_days)


def generate_attribute_value(attr_def: AttributeDef, rng: random.Random) -> str:
    """Generate a canonical string value adhering to §4.2 and §8.1."""
    attr = attr_def.attribute
    if attr == "notice_period":
        days = rng.randint(7, 90)
        return f"{days} days"
    elif attr == "stipend_amount":
        amt = rng.choice(range(50, 2001, 25))
        return f"{amt}.00"
    elif attr == "approval_threshold":
        amt = rng.choice(range(500, 20001, 250))
        return f"{amt}.00"
    elif attr == "review_cycle":
        months = rng.choice(range(3, 25))
        return f"{months} months"
    elif attr == "max_throughput":
        tp = rng.choice(range(100, 50001, 100))
        return str(tp)
    elif attr == "warranty_period":
        months = rng.choice(range(6, 61, 6))
        return f"{months} months"
    elif attr == "list_price":
        price = rng.choice(range(20, 9001, 20))
        return f"{price}.00"
    elif attr == "max_operating_temp":
        temp = rng.randint(30, 90)
        return str(temp)
    elif attr == "capacity":
        cap = rng.choice(range(20, 2001, 10))
        return str(cap)
    elif attr == "opening_date":
        d = _random_date(rng, date(2015, 1, 1), date(2025, 5, 31))
        return d.isoformat()
    elif attr == "floor_count":
        floors = rng.randint(1, 40)
        return str(floors)
    elif attr == "security_level":
        return rng.choice(SECURITY_LEVELS)
    else:
        raise ValueError(f"Unknown attribute: {attr}")


def generate_facts_for_entity(
    entity_id: str,
    entity_name: str,
    kind: str,
    rng: random.Random,
    status: Literal["official", "draft"] = "official",
    base_effective_date: date | None = None,
) -> list[Fact]:
    """Generate facts for all 4 attributes of an entity."""
    attributes = ENTITY_KINDS[kind]
    facts = []
    start_eff = date(2022, 1, 1)
    end_eff = date(2025, 5, 31)

    for attr_def in attributes:
        fact_id = f"fact-{entity_id}-{attr_def.attribute}"
        val = generate_attribute_value(attr_def, rng)
        eff_date = base_effective_date or _random_date(rng, start_eff, end_eff)
        fact = Fact(
            fact_id=fact_id,
            entity_id=entity_id,
            entity_name=entity_name,
            attribute=attr_def.attribute,
            value=val,
            value_type=attr_def.value_type,
            unit=attr_def.unit,
            effective_date=eff_date,
            status=status,
        )
        facts.append(fact)
    return facts
