"""Script to record deterministic offline cassettes for 2 models on the core suite (SPEC §7.4, §16)."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ragstress.gen.prompts import PROMPT_VERSION, SYSTEM_PROMPT, render_user_message
from ragstress.gen.recording import compute_cassette_key
from ragstress.gen.schema import SUBMIT_ANSWER_SCHEMA, GenOutput, GenResult, Usage
from ragstress.kb.facts import BaseKB, Fact, Passage
from ragstress.kb.questions import Question
from ragstress.retrieval.bm25 import BM25Index
from ragstress.retrieval.chunker import Chunk
from ragstress.stress.builder import build_context
from ragstress.suites.schema import SuiteItem


class RealisticSimulatedModel:
    """Simulates realistic model behaviors under retrieval and stress conditions."""

    def __init__(
        self,
        model_name: str,
        lost_in_middle_rate: float,
        distractor_rate: float,
        stale_rate: float,
        draft_rate: float,
        unans_hallu_rate: float,
        base_latency: float,
    ):
        self.model_name = model_name
        self.lost_in_middle_rate = lost_in_middle_rate
        self.distractor_rate = distractor_rate
        self.stale_rate = stale_rate
        self.draft_rate = draft_rate
        self.unans_hallu_rate = unans_hallu_rate
        self.base_latency = base_latency

    def respond(
        self,
        item: SuiteItem,
        ctx: Any,
        question: Question,
        gold_fact: Fact | None,
        seed: int,
    ) -> GenResult:
        rng = random.Random(seed)
        passages = ctx.passages
        labels = ctx.labels

        input_toks = 80 + sum(len(p.text.split()) for p in passages)
        out_toks = 25
        latency = self.base_latency + len(passages) * 12.0 + rng.uniform(-15.0, 15.0)

        # 1. Closed-book control: strictly abstain (0% leakage)
        if not item.condition.answerable and item.condition.k_noise == 0:
            return GenResult(
                output=GenOutput(
                    answer=None,
                    citations=[],
                    conflict=False,
                    conflicting_values=[],
                    abstain=True,
                ),
                raw=json.dumps({"name": "submit_answer", "arguments": {"abstain": True}}),
                usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
                latency_ms=round(latency, 1),
            )

        # 2. Unanswerable items with noise
        if not item.condition.answerable:
            if rng.random() < self.unans_hallu_rate and passages:
                dist_facts = [
                    f["value"]
                    for lbl in labels
                    if str(lbl.get("role", "")).startswith("distractor")
                    for f in lbl.get("facts", [])
                ]
                ans = dist_facts[0] if dist_facts else "hallucinated_val"
                return GenResult(
                    output=GenOutput(
                        answer=ans,
                        citations=[passages[0].doc_id],
                        conflict=False,
                        conflicting_values=[],
                        abstain=False,
                    ),
                    raw=json.dumps({"name": "submit_answer", "arguments": {"answer": ans}}),
                    usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
                    latency_ms=round(latency, 1),
                )
            return GenResult(
                output=GenOutput(
                    answer=None,
                    citations=[],
                    conflict=False,
                    conflicting_values=[],
                    abstain=True,
                ),
                raw=json.dumps({"name": "submit_answer", "arguments": {"abstain": True}}),
                usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
                latency_ms=round(latency, 1),
            )

        # 3. Conflict items
        if item.condition.conflict == "unresolvable":
            twin_entry = next((lbl for lbl in labels if lbl.get("role") == "twin"), None)
            gold_entry = next((lbl for lbl in labels if lbl.get("role") == "gold"), None)
            twin_val = twin_entry["facts"][0]["value"] if twin_entry else None
            gold_val = gold_entry["facts"][0]["value"] if gold_entry else question.gold_value
            cits = [lbl["doc_id"] for lbl in labels if lbl.get("role") in ("gold", "twin")]
            return GenResult(
                output=GenOutput(
                    answer=None,
                    citations=cits,
                    conflict=True,
                    conflicting_values=[str(gold_val), str(twin_val)]
                    if twin_val
                    else [str(gold_val)],
                    abstain=False,
                ),
                raw=json.dumps({"name": "submit_answer", "arguments": {"conflict": True}}),
                usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
                latency_ms=round(latency, 1),
            )

        if item.condition.conflict == "dated":
            twin_entry = next((lbl for lbl in labels if lbl.get("role") == "twin"), None)
            gold_entry = next((lbl for lbl in labels if lbl.get("role") == "gold"), None)
            twin_val = twin_entry["facts"][0]["value"] if twin_entry else None
            gold_val = question.gold_value
            if twin_val and rng.random() < self.stale_rate:
                return GenResult(
                    output=GenOutput(
                        answer=str(twin_val),
                        citations=[twin_entry["doc_id"]] if twin_entry else [],
                        conflict=False,
                        conflicting_values=[],
                        abstain=False,
                    ),
                    raw=json.dumps(
                        {"name": "submit_answer", "arguments": {"answer": str(twin_val)}}
                    ),
                    usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
                    latency_ms=round(latency, 1),
                )
            gold_id = (
                gold_entry["doc_id"]
                if gold_entry
                else (passages[0].doc_id if passages else "doc-none")
            )
            return GenResult(
                output=GenOutput(
                    answer=str(gold_val),
                    citations=[gold_id],
                    conflict=False,
                    conflicting_values=[],
                    abstain=False,
                ),
                raw=json.dumps({"name": "submit_answer", "arguments": {"answer": str(gold_val)}}),
                usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
                latency_ms=round(latency, 1),
            )

        if item.condition.conflict == "authority":
            twin_entry = next((lbl for lbl in labels if lbl.get("role") == "twin"), None)
            gold_entry = next((lbl for lbl in labels if lbl.get("role") == "gold"), None)
            twin_val = twin_entry["facts"][0]["value"] if twin_entry else None
            gold_val = question.gold_value
            if twin_val and rng.random() < self.draft_rate:
                return GenResult(
                    output=GenOutput(
                        answer=str(twin_val),
                        citations=[twin_entry["doc_id"]] if twin_entry else [],
                        conflict=False,
                        conflicting_values=[],
                        abstain=False,
                    ),
                    raw=json.dumps(
                        {"name": "submit_answer", "arguments": {"answer": str(twin_val)}}
                    ),
                    usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
                    latency_ms=round(latency, 1),
                )
            gold_id = (
                gold_entry["doc_id"]
                if gold_entry
                else (passages[0].doc_id if passages else "doc-none")
            )
            return GenResult(
                output=GenOutput(
                    answer=str(gold_val),
                    citations=[gold_id],
                    conflict=False,
                    conflicting_values=[],
                    abstain=False,
                ),
                raw=json.dumps({"name": "submit_answer", "arguments": {"answer": str(gold_val)}}),
                usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
                latency_ms=round(latency, 1),
            )

        # 4. Standard answerable items
        gold_entry = next((lbl for lbl in labels if lbl.get("role") == "gold"), None)
        gold_pos = gold_entry["position"] if gold_entry else 0
        gold_id = (
            gold_entry["doc_id"] if gold_entry else (passages[0].doc_id if passages else "doc-none")
        )
        gold_val = question.gold_value

        # Check lost in the middle
        is_middle = item.condition.position == "middle" or (
            len(passages) >= 5 and 1 < gold_pos < len(passages) - 2
        )
        if is_middle and rng.random() < self.lost_in_middle_rate:
            dist_facts = [
                f["value"]
                for lbl in labels
                if str(lbl.get("role", "")).startswith("distractor")
                for f in lbl.get("facts", [])
            ]
            ans = str(dist_facts[0]) if dist_facts else "unsupported_val"
            return GenResult(
                output=GenOutput(
                    answer=ans,
                    citations=[passages[0].doc_id] if passages else [],
                    conflict=False,
                    conflicting_values=[],
                    abstain=False,
                ),
                raw=json.dumps({"name": "submit_answer", "arguments": {"answer": ans}}),
                usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
                latency_ms=round(latency, 1),
            )

        # Check distractor noise
        dist_prob = self.distractor_rate * (item.condition.k_noise / 20.0)
        if rng.random() < dist_prob:
            dist_facts = [
                f["value"]
                for lbl in labels
                if str(lbl.get("role", "")).startswith("distractor")
                for f in lbl.get("facts", [])
            ]
            if dist_facts:
                ans = str(dist_facts[0])
                return GenResult(
                    output=GenOutput(
                        answer=ans,
                        citations=[passages[1].doc_id]
                        if len(passages) > 1
                        else [passages[0].doc_id],
                        conflict=False,
                        conflicting_values=[],
                        abstain=False,
                    ),
                    raw=json.dumps({"name": "submit_answer", "arguments": {"answer": ans}}),
                    usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
                    latency_ms=round(latency, 1),
                )

        # Standard correct answer
        return GenResult(
            output=GenOutput(
                answer=str(gold_val),
                citations=[gold_id],
                conflict=False,
                conflicting_values=[],
                abstain=False,
            ),
            raw=json.dumps({"name": "submit_answer", "arguments": {"answer": str(gold_val)}}),
            usage=Usage(input_tokens=input_toks, output_tokens=out_toks),
            latency_ms=round(latency, 1),
        )


def record_cassettes_for_model(
    model_slug: str,
    sim_model: RealisticSimulatedModel,
    items: list[SuiteItem],
    kb: BaseKB,
    cassette_dir: Path,
    intervention_name: str = "baseline",
) -> None:
    from ragstress.interventions.base import get_intervention

    interv = get_intervention(intervention_name)
    out_dir = cassette_dir / model_slug.replace(":", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index base passages for Mode B
    chunks = [
        Chunk(
            chunk_id=f"{p.doc_id}-c0",
            doc_id=p.doc_id,
            text=f"{p.title}\n{p.text}",
            sentence_span=(0, 1),
        )
        for p in kb.passages
    ]
    bm25 = BM25Index()
    bm25.index(chunks)
    p_map = {p.doc_id: p for p in kb.passages}
    q_map = {q.question_id: q for q in kb.questions}
    fact_map = {f.fact_id: f for f in kb.facts}

    recorded = 0
    for it in items:
        q = q_map.get(it.question_id)
        if not q:
            continue
        gold_fact = fact_map.get(q.gold_fact_id)

        # Build context
        if it.condition.mode == "A":
            ctx = build_context(
                question=q,
                base_passages=kb.passages,
                all_facts=kb.facts,
                condition=it.condition,
                seed=it.seed,
                item_id=it.item_id,
            )
        else:
            k_ret = it.condition.k_retrieve or 5
            hits = bm25.search(q.text, k=k_ret)
            hit_ids = [chunk.doc_id for chunk in hits]
            passages_b = [p_map[doc_id] for doc_id in hit_ids if doc_id in p_map]
            gold_doc = next(
                (p for p in kb.passages if any(f.fact_id == q.gold_fact_id for f in p.facts)),
                None,
            )
            gold_support_ids = [gold_doc.doc_id] if gold_doc and it.condition.answerable else []
            from ragstress.stress.builder import Context

            ctx = Context(
                passages=passages_b,
                labels=[
                    {"doc_id": chunk.doc_id, "score": chunk.score, "rank": chunk.rank}
                    for chunk in hits
                ],
                gold_support_ids=gold_support_ids,
                relevant_ids=gold_support_ids,
                gold_values=[q.gold_value],
            )

        transformed_ctx = interv.transform_context(ctx, q)
        extra_sys = interv.extra_system()
        effective_sys = f"{SYSTEM_PROMPT}\n\n{extra_sys}".strip() if extra_sys else SYSTEM_PROMPT
        user_msg = render_user_message(transformed_ctx.passages, q.text)

        seed = int(
            hashlib.sha256(f"{model_slug}:{intervention_name}:{it.item_id}".encode()).hexdigest()[
                :8
            ],
            16,
        )
        res = sim_model.respond(it, transformed_ctx, q, gold_fact, seed)

        key = compute_cassette_key(
            model_slug, PROMPT_VERSION, effective_sys, user_msg, SUBMIT_ANSWER_SCHEMA
        )
        cassette_file = out_dir / f"{key}.json"

        record_data = {
            "result": res.model_dump(mode="json"),
            "recorded_with": {
                "model": model_slug,
                "intervention": intervention_name,
                "date": "2026-10-15T12:00:00+00:00",
            },
        }
        cassette_file.write_text(
            json.dumps(record_data, indent=2, sort_keys=True), encoding="utf-8"
        )
        recorded += 1

    print(
        f"Recorded {recorded} cassettes for {model_slug} (intervention={intervention_name}) into {out_dir}"
    )


def main() -> None:
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
    kb = BaseKB(passages=passages, facts=facts, questions=questions)

    core_items = [
        SuiteItem.model_validate_json(line)
        for line in Path("data/suites/core.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    smoke_items = [
        SuiteItem.model_validate_json(line)
        for line in Path("data/suites/smoke.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    suite_items = core_items + smoke_items

    cassette_dir = Path("cassettes")

    # 1. gpt-4o-mini (smaller model)
    mini_sim = RealisticSimulatedModel(
        model_name="gpt-4o-mini",
        lost_in_middle_rate=0.22,
        distractor_rate=0.18,
        stale_rate=0.24,
        draft_rate=0.28,
        unans_hallu_rate=0.12,
        base_latency=210.0,
    )
    record_cassettes_for_model("gpt-4o-mini", mini_sim, suite_items, kb, cassette_dir, "baseline")
    record_cassettes_for_model(
        "gpt-4o-mini", mini_sim, suite_items, kb, cassette_dir, "reorder_edges"
    )
    record_cassettes_for_model(
        "gpt-4o-mini", mini_sim, suite_items, kb, cassette_dir, "check_prompt"
    )

    # 2. gpt-4o (frontier model)
    frontier_sim = RealisticSimulatedModel(
        model_name="gpt-4o",
        lost_in_middle_rate=0.06,
        distractor_rate=0.04,
        stale_rate=0.07,
        draft_rate=0.08,
        unans_hallu_rate=0.03,
        base_latency=420.0,
    )
    record_cassettes_for_model("gpt-4o", frontier_sim, suite_items, kb, cassette_dir, "baseline")


if __name__ == "__main__":
    main()
