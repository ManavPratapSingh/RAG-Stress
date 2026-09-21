"""Multi-hop question and relational bridge generator (PLAN.md Phase 9 Stretch)."""

import random

from ragstress.kb.facts import BaseKB, Fact, Passage
from ragstress.kb.questions import ATTRIBUTE_LABELS
from ragstress.multihop.schema import MultiHopQuestion

MULTIHOP_TEMPLATES: list[str] = [
    "What is the {label} of the policy governing {entity_name}?",
    "According to enterprise regulations, what is the {label} specified under the policy applicable to {entity_name}?",
    "For the regulatory policy that governs {entity_name}, what is the stated {label}?",
]


def generate_multihop_questions(
    kb: BaseKB,
    seed: int = 42,
    count: int = 20,
) -> list[MultiHopQuestion]:
    """Generate 2-hop compositional questions linking products/facilities to governing policies."""
    rng = random.Random(seed)

    # Separate passages by entity kind
    # Policy passages (Hop 2 targets)
    policy_passages: list[Passage] = [
        p
        for p in kb.passages
        if any("Policy" in f.entity_name for f in p.facts) and p.status == "official"
    ]
    # Product or Facility passages (Hop 1 bridges)
    client_passages: list[Passage] = [
        p
        for p in kb.passages
        if not any("Policy" in f.entity_name for f in p.facts) and p.status == "official"
    ]

    if not policy_passages or not client_passages:
        # Fallback if names do not contain Policy keyword
        policy_passages = kb.passages[len(kb.passages) // 2 :]
        client_passages = kb.passages[: len(kb.passages) // 2]

    multihop_questions: list[MultiHopQuestion] = []

    for i in range(count):
        p_hop1 = client_passages[i % len(client_passages)]
        p_hop2 = policy_passages[i % len(policy_passages)]

        hop1_fact = p_hop1.facts[0] if p_hop1.facts else None
        hop2_fact = p_hop2.facts[0] if p_hop2.facts else None

        if not hop1_fact or not hop2_fact:
            continue

        hop1_entity_id = hop1_fact.entity_id
        hop1_entity_name = hop1_fact.entity_name
        hop2_entity_id = hop2_fact.entity_id
        hop2_entity_name = hop2_fact.entity_name

        # Create bridge fact for Hop 1
        bridge_fact_id = f"fact-bridge-{i:04d}"
        bridge_fact = Fact(
            fact_id=bridge_fact_id,
            entity_id=hop1_entity_id,
            entity_name=hop1_entity_name,
            attribute="governing_policy",
            value=hop2_entity_name,
            value_type="enum",
            unit=None,
            effective_date=p_hop1.effective_date,
            status="official",
        )

        # Inject bridge sentence into p_hop1 text if not already present
        bridge_sentence = (
            f" Regulatory framework: {hop1_entity_name} is governed by {hop2_entity_name}."
        )
        if bridge_sentence not in p_hop1.text:
            p_hop1.text += bridge_sentence
            p_hop1.facts.append(bridge_fact)

        label = ATTRIBUTE_LABELS.get(hop2_fact.attribute, hop2_fact.attribute.replace("_", " "))
        template = rng.choice(MULTIHOP_TEMPLATES)
        text = template.format(label=label, entity_name=hop1_entity_name)

        q_id = f"q-mh-{i:04d}"
        multihop_questions.append(
            MultiHopQuestion(
                question_id=q_id,
                entity_id=hop1_entity_id,
                attribute=hop2_fact.attribute,
                text=text,
                gold_value=hop2_fact.value,
                gold_fact_id=hop2_fact.fact_id,
                entity_name=hop1_entity_name,
                attribute_label=label,
                hop1_entity_id=hop1_entity_id,
                hop1_entity_name=hop1_entity_name,
                hop2_entity_id=hop2_entity_id,
                hop2_entity_name=hop2_entity_name,
                bridge_relation="governing_policy",
                gold_support_fact_ids=[bridge_fact.fact_id, hop2_fact.fact_id],
                hop1_passage_id=p_hop1.doc_id,
                hop2_passage_id=p_hop2.doc_id,
            )
        )

    return multihop_questions
