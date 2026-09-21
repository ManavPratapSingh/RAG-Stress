"""Multi-hop question schemas and models (PLAN.md Phase 9 Stretch)."""

from ragstress.kb.questions import Question


class MultiHopQuestion(Question):
    """Question requiring 2-document retrieval and synthesis across linked entities."""

    kind: str = "multihop"
    hop1_entity_id: str
    hop1_entity_name: str
    hop2_entity_id: str
    hop2_entity_name: str
    bridge_relation: str = "governing_policy"
    gold_support_fact_ids: list[str] = []
    hop1_passage_id: str | None = None
    hop2_passage_id: str | None = None
