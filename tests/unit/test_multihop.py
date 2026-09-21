"""Unit tests for Multi-Hop (2-document) reasoning stress factor (PLAN.md Phase 9 Stretch)."""

from pathlib import Path

import pytest

from ragstress.gen.prompts import SYSTEM_PROMPT, render_user_message
from ragstress.gen.schema import SUBMIT_ANSWER_SCHEMA
from ragstress.gen.scripted import ScriptedGenerator
from ragstress.graders.item_grader import grade_item
from ragstress.kb.facts import BaseKB
from ragstress.kb.generator import generate_base_kb
from ragstress.multihop.generator import generate_multihop_questions
from ragstress.multihop.schema import MultiHopQuestion
from ragstress.stress.builder import build_context
from ragstress.stress.spec import StressCondition


@pytest.fixture(scope="module")
def base_kb() -> BaseKB:
    passages, facts, questions = generate_base_kb(seed=42)
    return BaseKB(passages=passages, facts=facts, questions=questions)


def test_generate_multihop_questions(base_kb: BaseKB) -> None:
    multihop_qs = generate_multihop_questions(base_kb, seed=42, count=10)
    assert len(multihop_qs) == 10

    for q in multihop_qs:
        assert isinstance(q, MultiHopQuestion)
        assert q.kind == "multihop"
        assert len(q.gold_support_fact_ids) == 2
        assert q.hop1_entity_name is not None
        assert q.hop2_entity_name is not None
        assert q.gold_value is not None
        assert q.bridge_relation in ("governing_policy", "parent_facility")


def test_multihop_context_building(base_kb: BaseKB) -> None:
    multihop_qs = generate_multihop_questions(base_kb, seed=42, count=5)
    q = multihop_qs[0]

    cond = StressCondition(mode="A", k_noise=5, position="random")
    ctx = build_context(
        question=q,
        base_passages=base_kb.passages,
        all_facts=base_kb.facts,
        condition=cond,
        seed=42,
        item_id="mh-001",
    )

    # Invariant: both Hop 1 and Hop 2 documents must be in gold_support_ids
    assert len(ctx.gold_support_ids) == 2
    assert all(doc_id in [p.doc_id for p in ctx.passages] for doc_id in ctx.gold_support_ids)

    # Verify context passages count is 2 (hop1 + hop2) + 5 (distractors) = 7
    assert len(ctx.passages) == 7


def test_multihop_grading_both_hops_vs_single_hop(base_kb: BaseKB) -> None:
    multihop_qs = generate_multihop_questions(base_kb, seed=42, count=5)
    q = multihop_qs[0]

    cond = StressCondition(mode="A", k_noise=3)
    ctx = build_context(
        question=q,
        base_passages=base_kb.passages,
        all_facts=base_kb.facts,
        condition=cond,
        seed=42,
        item_id="mh-002",
    )

    doc_ids = [p.doc_id for p in ctx.passages]
    gold_support_ids = ctx.gold_support_ids
    assert len(gold_support_ids) == 2

    # 1. Oracle: cites both support passages and outputs gold answer
    oracle_gen = ScriptedGenerator(
        policy="oracle",
        gold_value=q.gold_value,
        context_labels=ctx.labels,
    )
    user_msg = render_user_message(ctx.passages, q.text)
    res_oracle = oracle_gen.generate(SYSTEM_PROMPT, user_msg, SUBMIT_ANSWER_SCHEMA)
    # Ensure oracle cites all gold_support_ids
    res_oracle.output.citations = list(gold_support_ids)

    g_oracle = grade_item(
        item_id="mh-002",
        kind="multihop",
        value_type="duration",
        gen_result=res_oracle,
        gold_value=q.gold_value,
        gold_values=[q.gold_value],
        twin_value=None,
        distractor_values=[],
        gold_support_ids=gold_support_ids,
        context_doc_ids=doc_ids,
    )
    assert g_oracle.pass_answer is True
    assert g_oracle.pass_strict is True
    assert g_oracle.failure_mode is None

    # 2. Incomplete Hop Citation: cites only 1 of the 2 required passages
    res_single = oracle_gen.generate(SYSTEM_PROMPT, user_msg, SUBMIT_ANSWER_SCHEMA)
    res_single.output.citations = [gold_support_ids[0]]  # only 1 hop cited

    g_single = grade_item(
        item_id="mh-002",
        kind="multihop",
        value_type="duration",
        gen_result=res_single,
        gold_value=q.gold_value,
        gold_values=[q.gold_value],
        twin_value=None,
        distractor_values=[],
        gold_support_ids=gold_support_ids,
        context_doc_ids=doc_ids,
    )
    # Answer is right, but citations fail recall
    assert g_single.pass_answer is True
    assert g_single.pass_strict is False
    assert g_single.citation_recall == 0.5
    assert g_single.failure_mode is None

    # 3. Wrong answer failure classification
    res_wrong = oracle_gen.generate(SYSTEM_PROMPT, user_msg, SUBMIT_ANSWER_SCHEMA)
    res_wrong.output.answer = "9999 days"
    res_wrong.output.citations = list(gold_support_ids)
    g_wrong = grade_item(
        item_id="mh-003",
        kind="multihop",
        value_type="duration",
        gen_result=res_wrong,
        gold_value=q.gold_value,
        gold_values=[q.gold_value],
        twin_value=None,
        distractor_values=[],
        gold_support_ids=gold_support_ids,
        context_doc_ids=doc_ids,
    )
    assert g_wrong.pass_answer is False
    assert g_wrong.pass_strict is False
    assert g_wrong.failure_mode == "hallucinated"

    # 4. Mode B partial retrieval miss: only 1 of the 2 support documents is retrieved
    g_retrieval_miss = grade_item(
        item_id="mh-004",
        kind="multihop",
        value_type="duration",
        gen_result=res_wrong,
        gold_value=q.gold_value,
        gold_values=[q.gold_value],
        twin_value=None,
        distractor_values=[],
        gold_support_ids=gold_support_ids,
        context_doc_ids=[gold_support_ids[0]],  # missing second hop doc in context
        is_mode_b=True,
        retrieved_ids=[gold_support_ids[0]],
    )
    assert g_retrieval_miss.pass_answer is False
    assert g_retrieval_miss.failure_mode == "retrieval_miss"


def test_multihop_suite_generation_and_validation(base_kb: BaseKB) -> None:
    from ragstress.suites.generator import generate_suite
    from ragstress.suites.validator import validate_suite

    items = generate_suite(name="multihop", kb=base_kb, seed=42)
    assert len(items) == 20
    assert all(it.group == "multihop" for it in items)

    # Validate invariants
    res = validate_suite(items, base_kb)
    assert res.valid is True
    assert len(res.errors) == 0
    assert res.oracle_strict_accuracy == 1.0


def test_multihop_suite_execution(base_kb: BaseKB, tmp_path: Path) -> None:
    from ragstress.gen.scripted import ScriptedGenerator
    from ragstress.runner.executor import run_suite
    from ragstress.suites.generator import generate_suite

    items = generate_suite(name="multihop", kb=base_kb, seed=42)[:4]
    gen = ScriptedGenerator(policy="oracle")
    out_dir = tmp_path / "multihop_run"

    run_suite(
        items=items,
        kb=base_kb,
        generator=gen,
        output_dir=out_dir,
        concurrency=1,
    )
    assert (out_dir / "results.jsonl").exists()
    lines = (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # 4 rows
