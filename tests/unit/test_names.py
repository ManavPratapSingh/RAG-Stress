import random
import re

from ragstress.kb.names import (
    BLOCKED_REAL_NAMES,
    generate_entity_name,
    generate_pseudo_word,
    is_pronounceable,
)


def test_is_pronounceable_rules() -> None:
    # Valid words
    assert is_pronounceable("Kelvora")
    assert is_pronounceable("Dravix")
    assert is_pronounceable("Mirenta")
    assert is_pronounceable("Vandor")

    # Whitelisted 3-consonant clusters
    assert is_pronounceable("Kestrel")  # str
    assert is_pronounceable("Spring")  # spr

    # Invalid: no vowels
    assert not is_pronounceable("bcdfgh")

    # Invalid: non-whitelisted 3+ consecutive consonants
    assert not is_pronounceable("Bdkrfa")
    assert not is_pronounceable("Xptzva")


def test_pseudo_word_properties() -> None:
    rng = random.Random(1234)
    words = [generate_pseudo_word(rng) for _ in range(200)]

    for word in words:
        assert 4 <= len(word) <= 12
        assert is_pronounceable(word)
        assert word.lower() not in BLOCKED_REAL_NAMES


def test_entity_name_kinds_and_uniqueness() -> None:
    rng = random.Random(42)
    names: set[str] = set()

    # Generate 50 Policy names
    policy_names = [generate_entity_name("Policy", rng, names) for _ in range(50)]
    assert len(policy_names) == 50
    for name in policy_names:
        assert name.endswith("Policy")

    # Generate 50 Product names
    product_names = [generate_entity_name("Product", rng, names) for _ in range(50)]
    assert len(product_names) == 50
    for name in product_names:
        assert re.search(r"-\d{3}$", name)

    # Generate 50 Facility names
    facility_names = [generate_entity_name("Facility", rng, names) for _ in range(50)]
    assert len(facility_names) == 50

    # Total 150 unique names
    assert len(names) == 150
