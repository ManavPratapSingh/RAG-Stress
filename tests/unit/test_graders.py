"""Table-driven unit tests for deterministic graders and failure taxonomy (SPEC §8, FR-GRADE-*)."""

import pytest

from ragstress.gen.schema import GenOutput, GenResult
from ragstress.graders.citations import grade_citations
from ragstress.graders.conflict import grade_conflict
from ragstress.graders.correctness import grade_correctness
from ragstress.graders.item_grader import grade_item
from ragstress.graders.metrics import compute_aggregate_metrics
from ragstress.graders.provenance import determine_provenance
from ragstress.graders.taxonomy import classify_failure


def test_provenance_first_match_classification() -> None:
    # Set up context facts
    gold_val = "1200"
    twin_val = "1500"
    distractor_vals = ["2000", "3000"]

    # 1. Gold match
    prov, hall = determine_provenance("1,200", "int", gold_val, twin_val, distractor_vals)
    assert prov == "gold"
    assert hall is False

    # 2. Conflict loser match
    prov, hall = determine_provenance("1500", "int", gold_val, twin_val, distractor_vals)
    assert prov == "conflict_loser"
    assert hall is False

    # 3. Distractor match
    prov, hall = determine_provenance("2000 requests/s", "int", gold_val, twin_val, distractor_vals)
    assert prov == "distractor"
    assert hall is False

    # 4. None / Hallucinated
    prov, hall = determine_provenance("9999", "int", gold_val, twin_val, distractor_vals)
    assert prov == "none"
    assert hall is True


def test_correctness_by_item_kind() -> None:
    # 1. answerable_plain
    # pass_answer=True, pass_strict=True
    res_correct = GenResult(
        output=GenOutput(answer="24 months", citations=["doc-1"], abstain=False, conflict=False),
        model="test",
    )
    c1 = grade_correctness(
        kind="answerable_plain",
        output=res_correct.output,
        gold_value="24 months",
        value_type="duration",
        gold_support_ids=["doc-1"],
        context_doc_ids=["doc-1", "doc-2"],
    )
    assert c1.pass_answer is True
    assert c1.pass_strict is True

    # pass_answer=True, pass_strict=False (wrong citation)
    res_bad_cite = GenResult(
        output=GenOutput(answer="24 months", citations=["doc-2"], abstain=False, conflict=False),
        model="test",
    )
    c2 = grade_correctness(
        kind="answerable_plain",
        output=res_bad_cite.output,
        gold_value="24 months",
        value_type="duration",
        gold_support_ids=["doc-1"],
        context_doc_ids=["doc-1", "doc-2"],
    )
    assert c2.pass_answer is True
    assert c2.pass_strict is False

    # 2. dated
    c_dated = grade_correctness(
        kind="dated",
        output=GenOutput(answer="2024-05-01", citations=["doc-new"], conflict=True),
        gold_value="2024-05-01",
        value_type="date",
        gold_support_ids=["doc-new"],
        context_doc_ids=["doc-new", "doc-old"],
    )
    assert c_dated.pass_answer is True
    assert c_dated.pass_strict is True

    # 3. authority
    c_auth = grade_correctness(
        kind="authority",
        output=GenOutput(answer="$1,500.00", citations=["doc-official"], conflict=False),
        gold_value="1500.00",
        value_type="money",
        gold_support_ids=["doc-official"],
        context_doc_ids=["doc-official", "doc-draft"],
    )
    assert c_auth.pass_answer is True
    assert c_auth.pass_strict is True

    # 4. unresolvable
    c_unres_good = grade_correctness(
        kind="unresolvable",
        output=GenOutput(
            answer=None,
            conflict=True,
            conflicting_values=["1200", "1500"],
            citations=["doc-1", "doc-2"],
        ),
        gold_values=["1200", "1500"],
        value_type="int",
        gold_support_ids=["doc-1", "doc-2"],
        context_doc_ids=["doc-1", "doc-2"],
    )
    assert c_unres_good.pass_answer is True
    assert c_unres_good.pass_strict is True

    # 5. unanswerable
    c_unans_good = grade_correctness(
        kind="unanswerable",
        output=GenOutput(answer=None, abstain=True, citations=[]),
        gold_value=None,
        value_type="enum",
        gold_support_ids=[],
        context_doc_ids=["doc-d1"],
    )
    assert c_unans_good.pass_answer is True
    assert c_unans_good.pass_strict is True


def test_citations_precision_recall_and_fabricated() -> None:
    context_docs = ["doc-1", "doc-2", "doc-3"]
    support_docs = ["doc-1", "doc-2"]

    # Exactly correct
    c_exact = grade_citations(["doc-1", "doc-2"], support_docs, context_docs)
    assert c_exact.precision == 1.0
    assert c_exact.recall == 1.0
    assert c_exact.exact is True
    assert c_exact.fabricated_citations == []

    # Extra cited doc
    c_extra = grade_citations(["doc-1", "doc-2", "doc-3"], support_docs, context_docs)
    assert c_extra.precision == pytest.approx(2 / 3)
    assert c_extra.recall == 1.0
    assert c_extra.exact is False

    # Fabricated citation
    c_fab = grade_citations(["doc-1", "doc-fake"], support_docs, context_docs)
    assert c_fab.fabricated_citations == ["doc-fake"]
    assert c_fab.precision == 0.5
    assert c_fab.recall == 0.5
    assert c_fab.exact is False


def test_conflict_handling_metrics() -> None:
    # 1. Unresolvable with full values
    res1 = grade_conflict(
        kind="unresolvable",
        conflict_flag=True,
        conflicting_values=["1200", "1500"],
        gold_values=["1200", "1500"],
        value_type="int",
        pass_answer=True,
    )
    assert res1.conflict_flagged is True
    assert res1.values_complete is True
    assert res1.false_conflict_flag is False

    # 2. False conflict flag on plain question
    res2 = grade_conflict(
        kind="answerable_plain",
        conflict_flag=True,
        conflicting_values=[],
        gold_values=["1200"],
        value_type="int",
        pass_answer=True,
    )
    assert res2.conflict_flagged is True
    assert res2.false_conflict_flag is True

    # 3. Flagged and resolved on dated
    res3 = grade_conflict(
        kind="dated",
        conflict_flag=True,
        conflicting_values=[],
        gold_values=["2024-05-01"],
        value_type="date",
        pass_answer=True,
    )
    assert res3.flagged_and_resolved is True


def test_failure_taxonomy_ordered_first_match() -> None:
    # 1. format_error wins over all
    m1 = classify_failure(
        format_error="Syntax error in JSON",
        is_mode_b=False,
        kind="answerable_plain",
        gold_support_ids=["doc-1"],
        retrieved_ids=["doc-1"],
        provenance="distractor",
        abstain=False,
        conflict=False,
        values_complete=False,
    )
    assert m1 == "format_error"

    # 2. retrieval_miss in Mode B
    m2 = classify_failure(
        format_error=None,
        is_mode_b=True,
        kind="answerable_plain",
        gold_support_ids=["doc-1"],
        retrieved_ids=["doc-2"],
        provenance="none",
        abstain=False,
        conflict=False,
        values_complete=False,
    )
    assert m2 == "retrieval_miss"

    # 3. stale_adopted
    m3 = classify_failure(
        format_error=None,
        is_mode_b=False,
        kind="dated",
        gold_support_ids=["doc-1"],
        retrieved_ids=["doc-1", "doc-2"],
        provenance="conflict_loser",
        abstain=False,
        conflict=False,
        values_complete=False,
    )
    assert m3 == "stale_adopted"

    # 4. distractor_adopted
    m4 = classify_failure(
        format_error=None,
        is_mode_b=False,
        kind="answerable_plain",
        gold_support_ids=["doc-1"],
        retrieved_ids=["doc-1", "doc-2"],
        provenance="distractor",
        abstain=False,
        conflict=False,
        values_complete=False,
    )
    assert m4 == "distractor_adopted"

    # 5. hallucinated
    m5 = classify_failure(
        format_error=None,
        is_mode_b=False,
        kind="answerable_plain",
        gold_support_ids=["doc-1"],
        retrieved_ids=["doc-1"],
        provenance="none",
        abstain=False,
        conflict=False,
        values_complete=False,
    )
    assert m5 == "hallucinated"

    # 6. false_abstain
    m6 = classify_failure(
        format_error=None,
        is_mode_b=False,
        kind="answerable_plain",
        gold_support_ids=["doc-1"],
        retrieved_ids=["doc-1"],
        provenance=None,
        abstain=True,
        conflict=False,
        values_complete=False,
    )
    assert m6 == "false_abstain"


def test_mutation_test_flipped_gold_fails_correctness() -> None:
    # Mutation testing acceptance requirement: flipping gold value makes every correctness test fail
    res = GenResult(
        output=GenOutput(answer="1200", citations=["doc-1"], abstain=False, conflict=False),
        model="test",
    )
    c_pass = grade_correctness(
        kind="answerable_plain",
        output=res.output,
        gold_value="1200",
        value_type="int",
        gold_support_ids=["doc-1"],
        context_doc_ids=["doc-1"],
    )
    assert c_pass.pass_answer is True

    # Mutate gold value to 9999 -> must fail
    c_mutated = grade_correctness(
        kind="answerable_plain",
        output=res.output,
        gold_value="9999",
        value_type="int",
        gold_support_ids=["doc-1"],
        context_doc_ids=["doc-1"],
    )
    assert c_mutated.pass_answer is False


def test_scripted_generators_land_in_expected_taxonomy_bins() -> None:
    from ragstress.gen.prompts import SYSTEM_PROMPT
    from ragstress.gen.schema import SUBMIT_ANSWER_SCHEMA
    from ragstress.gen.scripted import ScriptedGenerator

    labels = [
        {"doc_id": "doc-gold", "role": "gold"},
        {"doc_id": "doc-dist", "role": "distractor", "facts": [{"value": "5000"}]},
        {"doc_id": "doc-twin", "role": "twin", "facts": [{"value": "1500"}]},
    ]
    distractor_vals = ["5000"]
    gold_val = "1200"
    twin_val = "1500"

    # 1. Oracle -> pass_answer=True
    gen_oracle = ScriptedGenerator(policy="oracle", gold_value=gold_val, context_labels=labels)
    res_oracle = gen_oracle.generate(SYSTEM_PROMPT, "user q", SUBMIT_ANSWER_SCHEMA)
    g_oracle = grade_item(
        item_id="item-1",
        kind="answerable_plain",
        value_type="int",
        gen_result=res_oracle,
        gold_value=gold_val,
        twin_value=twin_val,
        distractor_values=distractor_vals,
        gold_support_ids=["doc-gold"],
        context_doc_ids=["doc-gold", "doc-dist", "doc-twin"],
    )
    assert g_oracle.pass_answer is True
    assert g_oracle.failure_mode is None

    # 2. Abstain always on answerable plain -> false_abstain
    gen_abstain = ScriptedGenerator(policy="abstain_always")
    res_abstain = gen_abstain.generate(SYSTEM_PROMPT, "user q", SUBMIT_ANSWER_SCHEMA)
    g_abstain = grade_item(
        item_id="item-2",
        kind="answerable_plain",
        value_type="int",
        gen_result=res_abstain,
        gold_value=gold_val,
        twin_value=twin_val,
        distractor_values=distractor_vals,
        gold_support_ids=["doc-gold"],
        context_doc_ids=["doc-gold", "doc-dist", "doc-twin"],
    )
    assert g_abstain.pass_answer is False
    assert g_abstain.failure_mode == "false_abstain"

    # 3. Pick distractor -> distractor_adopted
    gen_dist = ScriptedGenerator(policy="pick_distractor", context_labels=labels)
    res_dist = gen_dist.generate(SYSTEM_PROMPT, "user q", SUBMIT_ANSWER_SCHEMA)
    g_dist = grade_item(
        item_id="item-3",
        kind="answerable_plain",
        value_type="int",
        gen_result=res_dist,
        gold_value=gold_val,
        twin_value=twin_val,
        distractor_values=distractor_vals,
        gold_support_ids=["doc-gold"],
        context_doc_ids=["doc-gold", "doc-dist", "doc-twin"],
    )
    assert g_dist.pass_answer is False
    assert g_dist.failure_mode == "distractor_adopted"

    # 4. Hallucinate -> hallucinated
    gen_hall = ScriptedGenerator(policy="hallucinate")
    res_hall = gen_hall.generate(SYSTEM_PROMPT, "user q", SUBMIT_ANSWER_SCHEMA)
    g_hall = grade_item(
        item_id="item-4",
        kind="answerable_plain",
        value_type="int",
        gen_result=res_hall,
        gold_value=gold_val,
        twin_value=twin_val,
        distractor_values=distractor_vals,
        gold_support_ids=["doc-gold"],
        context_doc_ids=["doc-gold", "doc-dist", "doc-twin"],
    )
    assert g_hall.pass_answer is False
    assert g_hall.failure_mode == "hallucinated"

    # 5. Ignore conflict -> conflict_ignored
    gen_ign = ScriptedGenerator(policy="ignore_conflict", context_labels=labels)
    res_ign = gen_ign.generate(SYSTEM_PROMPT, "user q", SUBMIT_ANSWER_SCHEMA)
    g_ign = grade_item(
        item_id="item-5",
        kind="unresolvable",
        value_type="int",
        gen_result=res_ign,
        gold_values=["1200", "1500"],
        twin_value=twin_val,
        distractor_values=distractor_vals,
        gold_support_ids=["doc-gold", "doc-twin"],
        context_doc_ids=["doc-gold", "doc-dist", "doc-twin"],
    )
    assert g_ign.pass_answer is False
    assert g_ign.failure_mode == "conflict_ignored"

    # Test aggregate metrics computation
    agg = compute_aggregate_metrics([g_oracle, g_abstain, g_dist, g_hall, g_ign])
    assert agg.total_items == 5
    assert agg.accuracy == 0.2  # 1 out of 5
