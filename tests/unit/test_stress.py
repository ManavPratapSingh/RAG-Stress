import random
import re
from datetime import date

import pytest

from ragstress.kb.facts import Passage
from ragstress.kb.generator import generate_base_kb
from ragstress.stress.builder import build_context
from ragstress.stress.conflict import build_conflict_twin
from ragstress.stress.distractors import (
    create_near_miss_name,
    generate_distractor_passages,
)
from ragstress.stress.position import assign_opaque_doc_ids, order_passages
from ragstress.stress.spec import StressCondition


@pytest.fixture(scope="module")
def base_kb():
    passages, facts, questions = generate_base_kb(seed=42)
    return passages, facts, questions


def test_near_miss_name_edit_distance() -> None:
    rng = random.Random(42)
    original = "Kelvora Travel Policy"
    variant = create_near_miss_name(original, rng)
    assert variant != original

    # Check that variant still looks like a near miss (differs by 1-2 chars)
    # Simple Levenshtein distance check
    def lev(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return lev(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                ins = prev[j + 1] + 1
                dels = curr[j] + 1
                subs = prev[j] + (0 if c1 == c2 else 1)
                curr.append(min(ins, dels, subs))
            prev = curr
        return prev[-1]

    d = lev(original, variant)
    assert 1 <= d <= 2


def test_conflict_twins_dated_authority_unresolvable(base_kb) -> None:
    _, facts, _ = base_kb
    gold_fact = facts[0]
    rng = random.Random(42)

    # 1. Dated: twin is older by >= 30 days, both official
    twin_dated = build_conflict_twin(gold_fact, "dated", rng)
    assert twin_dated.status == "official"
    diff = (gold_fact.effective_date - twin_dated.effective_date).days
    assert diff >= 30
    assert twin_dated.facts[0].value != gold_fact.value

    # 2. Authority: twin is draft, effective date newer by >= 30 days
    twin_auth = build_conflict_twin(gold_fact, "authority", rng)
    assert twin_auth.status == "draft"
    diff_auth = (twin_auth.effective_date - gold_fact.effective_date).days
    assert diff_auth >= 30
    assert twin_auth.facts[0].value != gold_fact.value

    # 3. Unresolvable: same date, same official status, different value
    twin_unres = build_conflict_twin(gold_fact, "unresolvable", rng)
    assert twin_unres.status == "official"
    assert twin_unres.effective_date == gold_fact.effective_date
    assert twin_unres.facts[0].value != gold_fact.value


def test_distractor_value_collision_filter(base_kb) -> None:
    passages, facts, questions = base_kb
    q = questions[0]
    gold_fact = next(f for f in facts if f.fact_id == q.gold_fact_id)
    rng = random.Random(42)

    distractors = generate_distractor_passages(
        question=q,
        gold_fact=gold_fact,
        base_passages=passages,
        all_facts=facts,
        k_noise=10,
        distractor_mix={
            "same_attr_other_entity": 0.4,
            "same_entity_other_attr": 0.3,
            "near_miss_entity": 0.2,
            "random_irrelevant": 0.1,
        },
        rng=rng,
    )
    assert len(distractors) == 10
    # Verify no distractor fact shares the gold value for the same value type
    for d, _role in distractors:
        for f in d.facts:
            if f.value_type == gold_fact.value_type:
                assert f.value != gold_fact.value


def test_position_ordering() -> None:
    rng = random.Random(42)
    dummy_passages = [
        Passage(
            doc_id=f"doc-{i}",
            title=f"Doc {i}",
            text=f"Text {i}",
            status="official",
            effective_date=date(2023, 1, 1),
            facts=[],
        )
        for i in range(10)
    ]
    gold = dummy_passages[0]
    others = [(p, "distractor:random") for p in dummy_passages[1:]]

    # top -> index 0
    top_ordered = order_passages(gold=gold, others=others, position="top", rng=rng)
    assert top_ordered[0][0].doc_id == gold.doc_id

    # bottom -> index n-1
    bot_ordered = order_passages(gold=gold, others=others, position="bottom", rng=rng)
    assert bot_ordered[-1][0].doc_id == gold.doc_id

    # middle -> (10 - 1) // 2 = 4
    mid_ordered = order_passages(gold=gold, others=others, position="middle", rng=rng)
    assert mid_ordered[4][0].doc_id == gold.doc_id


def test_opaque_id_format_and_independence() -> None:
    dummy = Passage(
        doc_id="doc-0001",
        title="T",
        text="W",
        status="official",
        effective_date=date(2023, 1, 1),
        facts=[],
    )
    context_items = assign_opaque_doc_ids(
        ordered_entries=[(dummy, "gold", 0)],
        seed=42,
        item_id="item-test-01",
    )
    doc_id = context_items[0].doc_id
    assert re.match(r"^doc-[0-9a-z]{6}$", doc_id)


def test_build_context_determinism_and_invariants(base_kb) -> None:
    passages, facts, questions = base_kb
    q = questions[0]

    cond = StressCondition(
        k_noise=10,
        position="middle",
        conflict="dated",
        answerable=True,
    )

    ctx1 = build_context(q, passages, facts, cond, seed=1234, item_id="item-001")
    ctx2 = build_context(q, passages, facts, cond, seed=1234, item_id="item-001")

    # Byte-identical determinism
    assert ctx1.model_dump_json() == ctx2.model_dump_json()
    assert len(ctx1.passages) == 12  # gold + twin + 10 distractors

    # Labels and roles
    roles = [p.role for p in ctx1.passages]
    assert "gold" in roles
    assert "twin" in roles
    assert ctx1.gold_values == [q.gold_value]
    assert len(ctx1.gold_support_ids) == 1


def test_unanswerable_context_building(base_kb) -> None:
    passages, facts, questions = base_kb
    q = questions[0]

    # k_noise = 0 unanswerable -> empty context (closed-book)
    cond_cb = StressCondition(k_noise=0, answerable=False)
    ctx_cb = build_context(q, passages, facts, cond_cb, seed=42, item_id="item-cb")
    assert len(ctx_cb.passages) == 0

    # k_noise = 5 unanswerable -> no gold, >= 2 near_miss or same_attr
    cond_unans = StressCondition(k_noise=5, answerable=False)
    ctx_unans = build_context(q, passages, facts, cond_unans, seed=42, item_id="item-unans")
    assert len(ctx_unans.passages) == 5
    roles = [p.role for p in ctx_unans.passages]
    assert "gold" not in roles
    assert "twin" not in roles
    hard_count = sum(
        1
        for r in roles
        if r in {"distractor:near_miss_entity", "distractor:same_attr_other_entity"}
    )
    assert hard_count >= 2


def test_gold_vs_distractor_word_count_parity(base_kb) -> None:
    """Acceptance criterion SPEC §6.8: gold vs distractor word-count means within 10%."""
    passages, facts, questions = base_kb
    rng = random.Random(42)

    gold_counts = []
    distractor_counts = []

    for q in questions[:30]:
        gold_fact = next(f for f in facts if f.fact_id == q.gold_fact_id)
        gold_p = next(p for p in passages if any(f.fact_id == gold_fact.fact_id for f in p.facts))
        gold_counts.append(len(gold_p.text.split()))

        distractors = generate_distractor_passages(
            question=q,
            gold_fact=gold_fact,
            base_passages=passages,
            all_facts=facts,
            k_noise=10,
            distractor_mix={
                "same_attr_other_entity": 0.4,
                "same_entity_other_attr": 0.3,
                "near_miss_entity": 0.2,
                "random_irrelevant": 0.1,
            },
            rng=rng,
        )
        for d, _ in distractors:
            distractor_counts.append(len(d.text.split()))

    mean_gold = sum(gold_counts) / len(gold_counts)
    mean_dist = sum(distractor_counts) / len(distractor_counts)

    # Within 10%
    diff_pct = abs(mean_gold - mean_dist) / mean_gold
    assert diff_pct <= 0.10, (
        f"Mean gold ({mean_gold:.1f}) vs distractor ({mean_dist:.1f}) diff {diff_pct:.1%} > 10%"
    )
