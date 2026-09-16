"""Unit tests for pluggable interventions (SPEC §12, FR-INT-*)."""

from datetime import date

from ragstress.gen.schema import GenOutput, GenResult
from ragstress.interventions.base import (
    BaselineIntervention,
    CheckPromptIntervention,
    LexicalRerankIntervention,
    ReorderEdgesIntervention,
    ScoreThresholdIntervention,
    get_intervention,
)
from ragstress.kb.facts import Fact, Passage
from ragstress.kb.questions import Question
from ragstress.stress.builder import Context


def make_passage(doc_id: str, text: str, pos: int = 0) -> Passage:
    return Passage(
        doc_id=doc_id,
        title=f"Doc {doc_id}",
        text=text,
        status="official",
        effective_date=date(2024, 1, 1),
        position=pos,
        facts=[
            Fact(
                fact_id=f"f-{doc_id}",
                entity_id="ent-1",
                entity_name="Dravix-340",
                attribute="warranty",
                value="24 months",
                value_type="duration",
                effective_date=date(2024, 1, 1),
                status="official",
            )
        ],
    )


def test_baseline_intervention_is_identity():
    """Baseline intervention leaves context and output untouched."""
    interv = BaselineIntervention()
    assert interv.name == "baseline"
    assert interv.extra_system() == ""

    p1 = make_passage("doc-1", "Some text")
    ctx = Context(passages=[p1])
    q = Question(
        question_id="q1",
        entity_id="ent-1",
        attribute="warranty",
        entity_name="Dravix-340",
        attribute_label="warranty",
        text="What is the warranty?",
        gold_fact_id="f-doc-1",
        gold_value="24 months",
    )

    t_ctx = interv.transform_context(ctx, q)
    assert t_ctx.passages == ctx.passages

    out = GenResult(output=GenOutput(answer="24 months", abstain=False))
    assert interv.postprocess(ctx, out) == out


def test_reorder_edges_places_highest_scores_at_exterior():
    """reorder_edges places best passages at start and end, lowest in the middle."""
    interv = ReorderEdgesIntervention()
    assert interv.name == "reorder_edges"

    # Passages with distinct lexical relevance to question
    p_high1 = make_passage("doc-1", "Dravix-340 warranty is exactly 24 months.")
    p_high2 = make_passage("doc-2", "Dravix-340 product warranty coverage documentation.")
    p_mid = make_passage("doc-3", "Dravix-340 specifications and physical dimensions.")
    p_low = make_passage("doc-4", "General corporate cafeteria guidelines.")

    q = Question(
        question_id="q1",
        entity_id="ent-1",
        attribute="warranty",
        entity_name="Dravix-340",
        attribute_label="warranty",
        text="What is the warranty for Dravix-340?",
        gold_fact_id="f-doc-1",
        gold_value="24 months",
    )

    ctx = Context(passages=[p_low, p_high1, p_mid, p_high2])
    reordered_ctx = interv.transform_context(ctx, q)
    reordered_ids = [p.doc_id for p in reordered_ctx.passages]

    # Highest relevance passages must sit at the edges (index 0 and index -1)
    assert reordered_ids[0] in ["doc-1", "doc-2"]
    assert reordered_ids[-1] in ["doc-1", "doc-2"]
    # Lowest relevance passage must be pushed into the interior
    assert reordered_ids[1] == "doc-4" or reordered_ids[2] == "doc-4"


def test_lexical_rerank_prioritizes_exact_matches():
    """lexical_rerank brings passages matching entity and attribute to front."""
    interv = LexicalRerankIntervention()
    assert interv.name == "lexical_rerank"

    p_unrelated = make_passage("doc-1", "Acme Corporation travel expense reimbursement policy.")
    p_entity_only = make_passage("doc-2", "Dravix-340 operating temperature and maintenance.")
    p_exact = make_passage("doc-3", "The warranty for Dravix-340 lasts 24 months.")

    q = Question(
        question_id="q1",
        entity_id="ent-1",
        attribute="warranty",
        entity_name="Dravix-340",
        attribute_label="warranty",
        text="What is the warranty for Dravix-340?",
        gold_fact_id="f-doc-3",
        gold_value="24 months",
    )

    ctx = Context(passages=[p_unrelated, p_entity_only, p_exact])
    reranked = interv.transform_context(ctx, q)

    assert reranked.passages[0].doc_id == "doc-3"
    assert reranked.passages[1].doc_id == "doc-2"
    assert reranked.passages[2].doc_id == "doc-1"


def test_check_prompt_appends_directive():
    """check_prompt appends explicit verification directives to system prompt."""
    interv = CheckPromptIntervention()
    assert interv.name == "check_prompt"
    extra = interv.extra_system()
    assert "exact entity" in extra.lower()
    assert "abstain" in extra.lower()


def test_score_threshold_abstains_below_threshold():
    """score_threshold forces abstention when top retrieval score is below threshold."""
    interv = ScoreThresholdIntervention(threshold=0.5)
    assert interv.name == "score_threshold"

    p1 = make_passage("doc-1", "Low score document")
    # Low score context (max score 0.3 < 0.5)
    ctx_low = Context(
        passages=[p1],
        labels=[{"doc_id": "doc-1", "score": 0.3}],
    )
    raw_out = GenResult(output=GenOutput(answer="fabricated answer", abstain=False))

    post_low = interv.postprocess(ctx_low, raw_out)
    assert post_low.output is not None
    assert post_low.output.abstain is True
    assert post_low.output.answer is None

    # High score context (score 0.8 >= 0.5)
    ctx_high = Context(
        passages=[p1],
        labels=[{"doc_id": "doc-1", "score": 0.8}],
    )
    post_high = interv.postprocess(ctx_high, raw_out)
    assert post_high.output is not None
    assert post_high.output.abstain is False
    assert post_high.output.answer == "fabricated answer"


def test_get_intervention_factory():
    """Verify intervention factory resolver."""
    assert isinstance(get_intervention("baseline"), BaselineIntervention)
    assert isinstance(get_intervention("reorder_edges"), ReorderEdgesIntervention)
    assert isinstance(get_intervention("lexical_rerank"), LexicalRerankIntervention)
    assert isinstance(get_intervention("check_prompt"), CheckPromptIntervention)
    assert isinstance(get_intervention("score_threshold"), ScoreThresholdIntervention)


def test_intervention_execution_and_comparison(tmp_path):
    """Verify runner executes interventions and compare_runs reports paired deltas (FR-INT-3)."""
    from ragstress.gen.scripted import ScriptedGenerator
    from ragstress.kb.facts import BaseKB
    from ragstress.kb.generator import generate_base_kb
    from ragstress.report.compare import compare_runs
    from ragstress.runner.executor import run_suite
    from ragstress.runner.schema import RunManifest
    from ragstress.suites.generator import generate_suite

    passages, facts, questions = generate_base_kb(seed=42)
    kb = BaseKB(passages=passages, facts=facts, questions=questions)
    items = generate_suite(name="smoke", kb=kb, seed=42)[:8]

    gen = ScriptedGenerator(policy="oracle")

    run_a = run_suite(
        items=items,
        kb=kb,
        generator=gen,
        output_dir=tmp_path / "run_baseline",
        concurrency=1,
        intervention="baseline",
    )
    run_b = run_suite(
        items=items,
        kb=kb,
        generator=gen,
        output_dir=tmp_path / "run_reorder",
        concurrency=1,
        intervention="reorder_edges",
    )

    man_a = RunManifest.model_validate_json((run_a / "manifest.json").read_text())
    man_b = RunManifest.model_validate_json((run_b / "manifest.json").read_text())
    assert man_a.intervention == "baseline"
    assert man_b.intervention == "reorder_edges"

    comparison_md = compare_runs(run_a, run_b)
    assert "Paired Significance Tests" in comparison_md
    assert "McNemar" in comparison_md
