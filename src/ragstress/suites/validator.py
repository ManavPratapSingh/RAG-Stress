"""Suite validator enforcing SPEC §9.4 invariants (FR-SUITE-4)."""

from pydantic import BaseModel

from ragstress.gen.prompts import SYSTEM_PROMPT, render_user_message
from ragstress.gen.schema import SUBMIT_ANSWER_SCHEMA
from ragstress.gen.scripted import ScriptedGenerator
from ragstress.graders.item_grader import grade_item
from ragstress.kb.facts import BaseKB
from ragstress.stress.builder import build_context
from ragstress.suites.schema import SuiteItem


class SuiteValidationResult(BaseModel):
    """Validation report for an evaluation suite."""

    valid: bool
    total_items: int
    errors: list[str]
    oracle_strict_accuracy: float


def validate_suite(items: list[SuiteItem], kb: BaseKB) -> SuiteValidationResult:
    """Validate a suite according to SPEC §9.4 rules."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    q_map = {q.question_id: q for q in kb.questions}
    fact_map = {f.fact_id: f for f in kb.facts}

    # 1. Check unique IDs and known questions
    for it in items:
        if it.item_id in seen_ids:
            errors.append(f"Duplicate item_id: {it.item_id}")
        seen_ids.add(it.item_id)

        if it.question_id not in q_map:
            errors.append(f"Unknown question_id {it.question_id} in item {it.item_id}")

    # 2. Check conditions and context building on Mode A items
    oracle_strict_passes = 0
    tested_oracle_items = 0

    for it in items:
        if it.condition.mode != "A":
            continue

        q = q_map.get(it.question_id)
        if not q:
            continue

        ctx = build_context(
            question=q,
            base_passages=kb.passages,
            all_facts=kb.facts,
            condition=it.condition,
            seed=it.seed,
            item_id=it.item_id,
        )

        # Invariant: Closed-book items have empty contexts
        if it.group == "controls" and not it.condition.answerable and it.condition.k_noise == 0:
            if len(ctx.passages) != 0:
                errors.append(
                    f"Closed-book control item {it.item_id} has non-empty context: {len(ctx.passages)} passages"
                )

        # Invariant: Oracle generator scores 100% pass_strict
        tested_oracle_items += 1
        gold_val = q.gold_value
        gold_vals = ctx.gold_values if ctx.gold_values else [gold_val]
        q_fact = fact_map.get(q.gold_fact_id)
        val_type = q_fact.value_type if q_fact else "enum"

        if not it.condition.answerable:
            item_kind = "unanswerable"
        elif it.condition.conflict != "none":
            item_kind = it.condition.conflict
        elif getattr(q, "gold_support_fact_ids", None) and len(q.gold_support_fact_ids) > 1:
            item_kind = "multihop"
        else:
            item_kind = "answerable_plain"

        labels = ctx.labels
        oracle_gen = ScriptedGenerator(
            policy="oracle",
            gold_value=gold_val,
            gold_values=gold_vals,
            context_labels=labels,
        )

        user_msg = render_user_message(ctx.passages, q.text)
        gen_res = oracle_gen.generate(SYSTEM_PROMPT, user_msg, SUBMIT_ANSWER_SCHEMA)

        twin_entry = next((lbl for lbl in ctx.labels if lbl.get("role") == "twin"), None)
        twin_val = (
            twin_entry["facts"][0]["value"] if twin_entry and twin_entry.get("facts") else None
        )

        distractor_vals = [
            f["value"]
            for lbl in ctx.labels
            if str(lbl.get("role", "")).startswith("distractor")
            for f in lbl.get("facts", [])
        ]

        g = grade_item(
            item_id=it.item_id,
            kind=item_kind,
            value_type=val_type,
            gen_result=gen_res,
            gold_value=gold_val,
            gold_values=gold_vals,
            twin_value=twin_val,
            distractor_values=distractor_vals,
            gold_support_ids=ctx.gold_support_ids,
            context_doc_ids=[p.doc_id for p in ctx.passages],
        )

        if g.pass_strict:
            oracle_strict_passes += 1
        else:
            errors.append(
                f"Oracle failed strict grading on {it.item_id} (kind={item_kind}, failure={g.failure_mode})"
            )

    oracle_strict_acc = (
        oracle_strict_passes / tested_oracle_items if tested_oracle_items > 0 else 1.0
    )

    return SuiteValidationResult(
        valid=len(errors) == 0,
        total_items=len(items),
        errors=errors,
        oracle_strict_accuracy=oracle_strict_acc,
    )
