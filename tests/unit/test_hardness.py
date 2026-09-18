"""Unit tests for suite hardness, length parity, and ID leakage checks (SPEC §6.2, §6.8, §13)."""

from pathlib import Path

import pytest

from ragstress.kb.facts import BaseKB, Fact, Passage
from ragstress.kb.questions import Question
from ragstress.retrieval.embedders import HashEmbedder
from ragstress.suites.hardness import evaluate_suite_hardness
from ragstress.suites.schema import SuiteItem


@pytest.fixture(scope="module")
def base_kb() -> BaseKB:
    kb_dir = Path("data/kb")
    passages = [
        Passage.model_validate_json(line)
        for line in (kb_dir / "passages.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    facts = [
        Fact.model_validate_json(line)
        for line in (kb_dir / "facts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = [
        Question.model_validate_json(line)
        for line in (kb_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return BaseKB(passages=passages, facts=facts, questions=questions)


def test_suite_hardness_and_leakage(base_kb: BaseKB) -> None:
    smoke_file = Path("data/suites/smoke.jsonl")
    items = [
        SuiteItem.model_validate_json(line)
        for line in smoke_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    embedder = HashEmbedder()
    rep = evaluate_suite_hardness(items, base_kb, embedder=embedder)

    assert rep.passed is True
    assert rep.total_items_analyzed > 0
    assert rep.id_format_valid is True
    assert rep.id_leakage_detected is False
    assert rep.length_balanced is True
    assert rep.word_count_diff_ratio <= 0.10

    # Verify each distractor type is present and has positive hardness
    types = {s.distractor_type for s in rep.distractor_hardness}
    assert "same_attr_other_entity" in types
    assert "same_entity_other_attr" in types
    assert "near_miss_entity" in types
    assert "random_irrelevant" in types

    for s in rep.distractor_hardness:
        assert s.count > 0
        assert s.bm25_median > 0.0
        assert s.dense_median > 0.0
