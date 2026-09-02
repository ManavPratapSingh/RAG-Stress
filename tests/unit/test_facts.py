import random
from datetime import date

from ragstress.kb.facts import (
    ENTITY_KINDS,
    Fact,
    generate_attribute_value,
    generate_facts_for_entity,
)


def test_entity_kinds_and_attributes_spec() -> None:
    # 3 entity kinds, each with 4 typed attributes
    assert set(ENTITY_KINDS.keys()) == {"Policy", "Product", "Facility"}
    for kind, attrs in ENTITY_KINDS.items():
        assert len(attrs) == 4, f"{kind} should have exactly 4 attributes"


def test_attribute_value_ranges() -> None:
    rng = random.Random(999)
    for attrs in ENTITY_KINDS.values():
        for attr_def in attrs:
            for _ in range(50):
                val = generate_attribute_value(attr_def, rng)
                assert val is not None
                if attr_def.value_type == "int":
                    n = int(val)
                    if attr_def.attribute == "max_throughput":
                        assert 100 <= n <= 50000
                    elif attr_def.attribute == "max_operating_temp":
                        assert 30 <= n <= 90
                    elif attr_def.attribute == "capacity":
                        assert 20 <= n <= 2000
                    elif attr_def.attribute == "floor_count":
                        assert 1 <= n <= 40
                elif attr_def.value_type == "money":
                    amt = float(val)
                    if attr_def.attribute == "stipend_amount":
                        assert 50.0 <= amt <= 2000.0
                    elif attr_def.attribute == "approval_threshold":
                        assert 500.0 <= amt <= 20000.0
                    elif attr_def.attribute == "list_price":
                        assert 20.0 <= amt <= 9000.0
                elif attr_def.value_type == "duration":
                    assert val.endswith(" days") or val.endswith(" months")
                    num = int(val.split()[0])
                    if attr_def.attribute == "notice_period":
                        assert 7 <= num <= 90
                    elif attr_def.attribute == "review_cycle":
                        assert 3 <= num <= 24
                    elif attr_def.attribute == "warranty_period":
                        assert 6 <= num <= 60
                elif attr_def.value_type == "date":
                    dt = date.fromisoformat(val)
                    assert date(2015, 1, 1) <= dt <= date(2025, 5, 31)
                elif attr_def.value_type == "enum":
                    assert val in {"low", "medium", "high", "restricted"}


def test_fact_pydantic_validation() -> None:
    rng = random.Random(42)
    facts = generate_facts_for_entity("ent-pol-001", "Kelvora Travel Policy", "Policy", rng)
    assert len(facts) == 4
    for f in facts:
        assert isinstance(f, Fact)
        assert f.status == "official"
        assert date(2022, 1, 1) <= f.effective_date <= date(2025, 5, 31)

        # Ensure json round-trip works
        dumped = f.model_dump(mode="json")
        loaded = Fact.model_validate(dumped)
        assert loaded == f
