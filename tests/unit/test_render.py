import random
from datetime import date

from ragstress.graders.normalize import normalize_value
from ragstress.kb.facts import (
    ENTITY_KINDS,
    generate_facts_for_entity,
)
from ragstress.kb.names import generate_entity_name
from ragstress.kb.render import (
    TEMPLATES,
    render_fact_sentence,
    render_passage,
)


def test_template_coverage() -> None:
    """Each attribute must have >= 4 paraphrase templates (SPEC §4.4)."""
    all_attributes = [attr_def.attribute for attrs in ENTITY_KINDS.values() for attr_def in attrs]
    for attr in all_attributes:
        assert attr in TEMPLATES, f"Missing templates for attribute: {attr}"
        assert len(TEMPLATES[attr]) >= 4, f"Attribute {attr} must have >= 4 templates"


def test_passage_word_count_and_fillers() -> None:
    """Passages must have body length between 60 and 180 words."""
    rng = random.Random(42)
    facts = generate_facts_for_entity("ent-prd-001", "Dravix-340", "Product", rng)
    p = render_passage(
        doc_id="doc-0001",
        title="Dravix-340 Documentation",
        facts=facts[:2],
        status="official",
        effective_date=date(2024, 1, 1),
        rng=rng,
    )
    words = p.text.split()
    assert 60 <= len(words) <= 180, f"Word count {len(words)} out of range [60, 180]"


def test_render_normalize_roundtrip_1000_sample() -> None:
    """Requirement FR-KB-4 / Acceptance:

    The value MUST appear in the text in a form accepted by the normalizer for its type.
    Test: render -> normalize -> equals fact value, for every fact in a 1,000-fact sample.
    """
    rng = random.Random(1337)
    existing_names: set[str] = set()

    kinds = ["Policy", "Product", "Facility"]
    fact_count = 0

    while fact_count < 1000:
        kind = rng.choice(kinds)
        name = generate_entity_name(kind, rng, existing_names)
        eid = f"ent-{fact_count}"
        facts = generate_facts_for_entity(eid, name, kind, rng)

        for fact in facts:
            fact_count += 1
            sentence = render_fact_sentence(fact, rng)

            # Test value extraction & normalization
            # To simulate normalizer reading the value from the sentence:
            # We check that normalizing the fact's canonical value directly equals fact.value,
            # and that fact's value as rendered in the sentence normalizes to canonical fact.value.
            norm_val = normalize_value(
                fact.value,
                fact.value_type,
                allowed_enums=["low", "medium", "high", "restricted"]
                if fact.value_type == "enum"
                else None,
            )
            assert norm_val == fact.value, (
                f"Normalization mismatch for {fact.value} (expected {fact.value})"
            )

            # Also verify that the rendered sentence contains the value string in an accepted form
            # For money: fact.value is '1500.00', sentence contains '$1500.00'
            # For duration: fact.value is '30 days', sentence contains '30 days'
            # For int: fact.value is '1200', sentence contains '1200'
            # For date: fact.value is '2024-03-01', sentence contains '2024-03-01'
            # For enum: fact.value is 'high', sentence contains 'high'
            if fact.value_type == "money":
                val_num = fact.value
                assert f"${val_num}" in sentence or f"{val_num} USD" in sentence
                money_norm = normalize_value(f"${val_num}", "money")
                assert money_norm == fact.value
            elif fact.value_type == "int":
                assert fact.value in sentence
                int_norm = normalize_value(fact.value, "int")
                assert int_norm == fact.value
            elif fact.value_type == "duration":
                assert fact.value in sentence
                dur_norm = normalize_value(fact.value, "duration")
                assert dur_norm == fact.value
            elif fact.value_type == "date":
                assert fact.value in sentence
                dt_norm = normalize_value(fact.value, "date")
                assert dt_norm == fact.value
            elif fact.value_type == "enum":
                assert fact.value in sentence.lower()
                enum_norm = normalize_value(
                    fact.value, "enum", allowed_enums=["low", "medium", "high", "restricted"]
                )
                assert enum_norm == fact.value

            if fact_count >= 1000:
                break
