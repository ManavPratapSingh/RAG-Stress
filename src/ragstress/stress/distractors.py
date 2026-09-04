"""Distractor generation and value-collision filtering (SPEC §6.2, FR-STRESS-2)."""

import random

from ragstress.kb.facts import (
    ENTITY_KINDS,
    Fact,
    Passage,
    generate_attribute_value,
)
from ragstress.kb.questions import Question
from ragstress.kb.render import render_passage


def create_near_miss_name(name: str, rng: random.Random) -> str:
    """Create a near-miss entity name differing by edit distance 1-2."""
    words = name.split()
    target_idx = 0
    target_word = words[0]

    # If first word has hyphen (e.g. Dravix-340), alter the pseudo-word part
    if "-" in target_word:
        base, suffix = target_word.split("-", 1)
        sub_pos = rng.randint(0, len(base) - 1)
        orig_char = base[sub_pos]
        vowels = "aeiou"
        replacement = rng.choice([v for v in vowels if v != orig_char.lower()] or ["x"])
        new_base = base[:sub_pos] + replacement + base[sub_pos + 1 :]
        words[0] = f"{new_base}-{suffix}"
        return " ".join(words)

    sub_pos = rng.randint(0, len(target_word) - 1)
    orig_char = target_word[sub_pos]
    replacement = rng.choice([c for c in "aeiou" if c != orig_char.lower()] or ["z"])
    new_word = target_word[:sub_pos] + replacement + target_word[sub_pos + 1 :]
    words[target_idx] = new_word
    return " ".join(words)


def _compute_mix_counts(k_noise: int, mix: dict[str, float]) -> dict[str, int]:
    """Calculate integer counts per type using largest-remainder rounding."""
    if k_noise == 0:
        return dict.fromkeys(mix, 0)
    floats = {k: k_noise * weight for k, weight in mix.items()}
    ints = {k: int(v) for k, v in floats.items()}
    remainder = k_noise - sum(ints.values())

    # Sort keys by descending fractional remainder
    sorted_remainders = sorted(mix.keys(), key=lambda k: floats[k] - ints[k], reverse=True)
    for i in range(remainder):
        ints[sorted_remainders[i]] += 1
    return ints


def generate_distractor_passages(
    question: Question,
    gold_fact: Fact,
    base_passages: list[Passage],
    all_facts: list[Fact],
    k_noise: int,
    distractor_mix: dict[str, float],
    rng: random.Random,
    twin_fact: Fact | None = None,
) -> list[tuple[Passage, str]]:
    """Generate k_noise distractor passages with value-collision filtering (SPEC §6.2).

    Returns list of (Passage, role_label).
    """
    if k_noise == 0:
        return []

    counts = _compute_mix_counts(k_noise, distractor_mix)
    distractors: list[tuple[Passage, str]] = []

    # Map entity_id to kind
    entity_kinds: dict[str, str] = {}
    for kind, attrs in ENTITY_KINDS.items():
        for ad in attrs:
            if ad.attribute == question.attribute:
                for f in all_facts:
                    if f.attribute == question.attribute:
                        entity_kinds[f.entity_id] = kind

    # Index facts
    facts_by_attr: dict[str, list[Fact]] = {}
    facts_by_entity: dict[str, list[Fact]] = {}
    for f in all_facts:
        facts_by_attr.setdefault(f.attribute, []).append(f)
        facts_by_entity.setdefault(f.entity_id, []).append(f)

    # Find attr_def
    target_attr_def = None
    for attrs in ENTITY_KINDS.values():
        for ad in attrs:
            if ad.attribute == question.attribute:
                target_attr_def = ad
                break

    forbidden_values = {gold_fact.value}
    if twin_fact is not None:
        forbidden_values.add(twin_fact.value)

    # 1. same_attr_other_entity
    same_attr_facts = [
        f
        for f in facts_by_attr.get(question.attribute, [])
        if f.entity_id != question.entity_id and f.value not in forbidden_values
    ]
    rng.shuffle(same_attr_facts)
    for f in same_attr_facts[: counts["same_attr_other_entity"]]:
        doc_id = f"dist-saoe-{f.fact_id}"
        p = render_passage(
            doc_id=doc_id,
            title=f"{f.entity_name} Documentation",
            facts=[f],
            status="official",
            effective_date=f.effective_date,
            rng=rng,
        )
        distractors.append((p, "distractor:same_attr_other_entity"))

    # 2. same_entity_other_attr
    same_entity_facts = [
        f
        for f in facts_by_entity.get(question.entity_id, [])
        if f.attribute != question.attribute
        and not (f.value_type == gold_fact.value_type and f.value in forbidden_values)
    ]
    rng.shuffle(same_entity_facts)
    for f in same_entity_facts[: counts["same_entity_other_attr"]]:
        doc_id = f"dist-seoa-{f.fact_id}"
        p = render_passage(
            doc_id=doc_id,
            title=f"{f.entity_name} Documentation",
            facts=[f],
            status="official",
            effective_date=f.effective_date,
            rng=rng,
        )
        distractors.append((p, "distractor:same_entity_other_attr"))

    # 3. near_miss_entity
    for _ in range(counts["near_miss_entity"]):
        near_name = create_near_miss_name(gold_fact.entity_name, rng)
        # Generate value != gold
        nm_val = gold_fact.value
        if target_attr_def is not None:
            for _ in range(50):
                v = generate_attribute_value(target_attr_def, rng)
                if v not in forbidden_values:
                    nm_val = v
                    break
        nm_fact = Fact(
            fact_id=f"nm-{rng.randint(1000, 9999)}",
            entity_id=f"nm-ent-{rng.randint(1000, 9999)}",
            entity_name=near_name,
            attribute=gold_fact.attribute,
            value=nm_val,
            value_type=gold_fact.value_type,
            unit=gold_fact.unit,
            effective_date=gold_fact.effective_date,
            status="official",
        )
        doc_id = f"dist-nm-{nm_fact.fact_id}"
        p = render_passage(
            doc_id=doc_id,
            title=f"{near_name} Documentation",
            facts=[nm_fact],
            status="official",
            effective_date=gold_fact.effective_date,
            rng=rng,
        )
        distractors.append((p, "distractor:near_miss_entity"))

    # 4. random_irrelevant
    irrelevant_passages = [
        p
        for p in base_passages
        if all(f.entity_id != question.entity_id for f in p.facts)
        and all(f.attribute != question.attribute for f in p.facts)
        and not any(
            f.value_type == gold_fact.value_type and f.value in forbidden_values for f in p.facts
        )
    ]
    rng.shuffle(irrelevant_passages)
    for p in irrelevant_passages[: counts["random_irrelevant"]]:
        dist_p = Passage(
            doc_id=f"dist-rand-{p.doc_id}",
            title=p.title,
            text=p.text,
            status=p.status,
            effective_date=p.effective_date,
            facts=p.facts,
        )
        distractors.append((dist_p, "distractor:random_irrelevant"))

    # If any count was short due to filtering, top up with safe irrelevant passages
    while len(distractors) < k_noise and irrelevant_passages:
        p = irrelevant_passages.pop()
        dist_p = Passage(
            doc_id=f"dist-rand-{p.doc_id}-{len(distractors)}",
            title=p.title,
            text=p.text,
            status=p.status,
            effective_date=p.effective_date,
            facts=p.facts,
        )
        distractors.append((dist_p, "distractor:random_irrelevant"))

    return distractors[:k_noise]
