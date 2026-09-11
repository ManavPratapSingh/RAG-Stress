"""Tests for suite generation, condition mapping, and validation (SPEC §9, FR-SUITE-*)."""

from ragstress.kb.generator import generate_base_kb_container
from ragstress.suites.generator import generate_suite
from ragstress.suites.validator import validate_suite


def test_suite_generation_and_splits() -> None:
    kb = generate_base_kb_container(seed=42)

    # 1. Smoke suite generation
    smoke_items = generate_suite(name="smoke", kb=kb, seed=42)
    assert len(smoke_items) >= 50
    assert len(smoke_items) <= 100

    # Invariants
    item_ids = [it.item_id for it in smoke_items]
    assert len(item_ids) == len(set(item_ids)), "Item IDs must be strictly unique"

    groups = {it.group for it in smoke_items}
    assert {"controls", "distractor_sweep", "position_sweep", "conflict", "unanswerable"}.issubset(
        groups
    )

    # Splits (25% dev, 75% test)
    dev_count = sum(1 for it in smoke_items if it.split == "dev")
    test_count = sum(1 for it in smoke_items if it.split == "test")
    assert dev_count + test_count == len(smoke_items)
    dev_ratio = dev_count / len(smoke_items)
    assert 0.15 <= dev_ratio <= 0.35

    # Determinism: same seed generates identical items
    smoke_items_2 = generate_suite(name="smoke", kb=kb, seed=42)
    assert [it.model_dump() for it in smoke_items] == [it.model_dump() for it in smoke_items_2]


def test_validate_suite_passes_on_smoke_suite() -> None:
    kb = generate_base_kb_container(seed=42)
    smoke_items = generate_suite(name="smoke", kb=kb, seed=42)

    val_res = validate_suite(smoke_items, kb)
    assert val_res.valid is True
    assert val_res.errors == []
    assert val_res.oracle_strict_accuracy == 1.0
