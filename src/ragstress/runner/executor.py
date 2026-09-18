"""Benchmark runner executing suites concurrently (SPEC §10, FR-RUN-*)."""

import concurrent.futures
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragstress.gen.prompts import PROMPT_VERSION, SYSTEM_PROMPT, render_user_message
from ragstress.gen.schema import SUBMIT_ANSWER_SCHEMA, GenResult
from ragstress.gen.scripted import ScriptedGenerator
from ragstress.graders.item_grader import grade_item
from ragstress.interventions.base import get_intervention
from ragstress.kb.facts import BaseKB
from ragstress.retrieval.bm25 import BM25Index
from ragstress.retrieval.chunker import Chunk
from ragstress.retrieval.metrics import compute_retrieval_metrics
from ragstress.runner.schema import ItemTrace, ResultRow, RunManifest
from ragstress.stress.builder import Context, build_context
from ragstress.suites.schema import SuiteItem


def _get_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _evaluate_single_item(
    item: SuiteItem,
    trial: int,
    kb: BaseKB,
    generator: Any,
    q_map: dict[str, Any],
    fact_map: dict[str, Any],
    run_id: str,
    bm25: BM25Index,
    intervention: Any,
) -> tuple[ItemTrace, ResultRow]:
    q = q_map[item.question_id]
    condition = item.condition

    retrieval_metrics_dict: dict[str, Any] = {}
    retrieved_traces: list[dict[str, Any]] | None = None
    distractor_vals: list[str] = []

    if condition.mode == "A":
        ctx = build_context(
            question=q,
            base_passages=kb.passages,
            all_facts=kb.facts,
            condition=condition,
            seed=item.seed + trial,
            item_id=item.item_id,
        )
        gold_support_ids = ctx.gold_support_ids
        item_kind = (
            "unanswerable"
            if not condition.answerable
            else (condition.conflict if condition.conflict != "none" else "answerable_plain")
        )
        twin_entry = next((lbl for lbl in ctx.labels if lbl.get("role") == "twin"), None)
        twin_val = (
            twin_entry["facts"][0]["value"] if twin_entry and twin_entry.get("facts") else None
        )
        gold_vals = ctx.gold_values if ctx.gold_values else [q.gold_value]
        distractor_vals = [
            f["value"]
            for lbl in ctx.labels
            if str(lbl.get("role", "")).startswith("distractor")
            for f in lbl.get("facts", [])
        ]
    else:
        # Mode B retrieval
        k_ret = condition.k_retrieve or 5
        hits = bm25.search(q.text, k=k_ret)
        hit_ids = [chunk.doc_id for chunk in hits]
        retrieved_traces = [
            {"doc_id": chunk.doc_id, "rank": chunk.rank, "score": chunk.score} for chunk in hits
        ]

        # Target gold docs
        gold_doc = next(
            (p for p in kb.passages if any(f.fact_id == q.gold_fact_id for f in p.facts)),
            None,
        )
        gold_support_ids = [gold_doc.doc_id] if gold_doc and condition.answerable else []
        item_kind = "answerable_plain" if condition.answerable else "unanswerable"

        # Map retrieved passages
        p_map = {p.doc_id: p for p in kb.passages}
        passages_b = [p_map[doc_id] for doc_id in hit_ids if doc_id in p_map]
        twin_val = None
        gold_vals = [q.gold_value]

        for p in passages_b:
            for f in p.facts:
                if f.value != q.gold_value:
                    distractor_vals.append(f.value)

        m = compute_retrieval_metrics(hit_ids, set(gold_support_ids), k=k_ret)
        retrieval_metrics_dict = {
            "retrieval_hit": m.hit,
            "retrieval_recall": m.recall,
            "retrieval_precision": m.precision,
            "retrieval_mrr": m.mrr,
            "retrieval_ndcg": m.ndcg,
        }
        ctx = Context(
            passages=passages_b,
            labels=[
                {"doc_id": chunk.doc_id, "score": chunk.score, "rank": chunk.rank} for chunk in hits
            ],
            gold_support_ids=gold_support_ids,
            relevant_ids=gold_support_ids,
            gold_values=gold_vals,
        )

    # Apply intervention context transform
    transformed_ctx = intervention.transform_context(ctx, q)
    passages = list(transformed_ctx.passages)
    context_labels = (
        transformed_ctx.labels if transformed_ctx.labels else [p.model_dump() for p in passages]
    )

    gold_val = q.gold_value
    q_fact = fact_map.get(q.gold_fact_id)
    val_type = q_fact.value_type if q_fact else "enum"

    # Prompt rendering
    user_msg = render_user_message(passages, q.text)
    extra_sys = intervention.extra_system()
    effective_sys = f"{SYSTEM_PROMPT}\n\n{extra_sys}".strip() if extra_sys else SYSTEM_PROMPT

    # Generator setup (clone scripted if needed)
    if isinstance(generator, ScriptedGenerator):
        gen_under_test = ScriptedGenerator(
            policy=generator.policy,
            gold_value=gold_val,
            gold_values=gold_vals,
            context_labels=context_labels,
        )
    else:
        gen_under_test = generator

    raw_gen_res: GenResult = gen_under_test.generate(effective_sys, user_msg, SUBMIT_ANSWER_SCHEMA)
    gen_res = intervention.postprocess(transformed_ctx, raw_gen_res)

    # Grade
    graded = grade_item(
        item_id=item.item_id,
        kind=item_kind,
        value_type=val_type,
        gen_result=gen_res,
        gold_value=gold_val,
        gold_values=gold_vals,
        twin_value=twin_val,
        distractor_values=distractor_vals,
        gold_support_ids=gold_support_ids,
        context_doc_ids=[p.doc_id for p in passages],
        retrieved_ids=[p.doc_id for p in passages] if condition.mode == "B" else None,
        is_mode_b=(condition.mode == "B"),
    )

    trace = ItemTrace(
        run_id=run_id,
        item_id=item.item_id,
        trial=trial,
        generator=getattr(generator, "name", "unknown"),
        prompt_version=PROMPT_VERSION,
        intervention=intervention.name,
        condition=condition.model_dump(mode="json"),
        context=[
            p.model_dump(mode="json") if hasattr(p, "model_dump") else p.__dict__ for p in passages
        ],
        retrieved=retrieved_traces,
        raw_output=gen_res.raw,
        parsed=gen_res.output.model_dump(mode="json") if gen_res.output else None,
        format_error=gen_res.format_error,
        grades=graded.model_dump(mode="json"),
        usage=gen_res.usage.model_dump(mode="json"),
        latency_ms=gen_res.latency_ms,
    )

    row = ResultRow(
        run_id=run_id,
        item_id=item.item_id,
        group=item.group,
        split=item.split,
        kind=item_kind,
        mode=condition.mode,
        k_noise=condition.k_noise,
        position=condition.position,
        conflict=condition.conflict,
        k_retrieve=condition.k_retrieve,
        retriever="bm25" if condition.mode == "B" else None,
        generator=getattr(generator, "name", "unknown"),
        intervention=intervention.name,
        pass_answer=graded.pass_answer,
        pass_strict=graded.pass_strict,
        provenance=graded.provenance,
        hallucinated=graded.hallucinated,
        failure_mode=graded.failure_mode,
        abstain=graded.pass_answer
        if item_kind == "unanswerable"
        else (gen_res.output.abstain if gen_res.output else False),
        conflict_flagged=graded.conflict_flagged,
        false_conflict_flag=graded.false_conflict_flag,
        values_complete=graded.values_complete,
        citation_precision=graded.citation_precision,
        citation_recall=graded.citation_recall,
        citation_exact=graded.citation_exact,
        fabricated_citations=graded.fabricated_citations,
        retrieval_hit=retrieval_metrics_dict.get("retrieval_hit"),
        retrieval_recall=retrieval_metrics_dict.get("retrieval_recall"),
        retrieval_precision=retrieval_metrics_dict.get("retrieval_precision"),
        retrieval_mrr=retrieval_metrics_dict.get("retrieval_mrr"),
        retrieval_ndcg=retrieval_metrics_dict.get("retrieval_ndcg"),
        input_tokens=gen_res.usage.input_tokens,
        output_tokens=gen_res.usage.output_tokens,
        cost_usd=gen_res.usage.cost_usd,
        latency_ms=gen_res.latency_ms,
    )

    return trace, row


def run_suite(
    items: list[SuiteItem],
    kb: BaseKB,
    generator: Any,
    output_dir: Path | str,
    concurrency: int = 4,
    trials: int = 1,
    intervention: str | Any = "baseline",
    run_id: str | None = None,
) -> Path:
    """Execute a benchmark suite across items and trials, saving traces and results."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    traces_dir = out_path / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    interv_obj = get_intervention(intervention) if isinstance(intervention, str) else intervention

    started_at = datetime.now(UTC).isoformat()
    active_run_id = run_id or (f"run_{items[0].seed}" if items else "run_default")

    q_map = {q.question_id: q for q in kb.questions}
    fact_map = {f.fact_id: f for f in kb.facts}

    # Pre-build Mode B BM25 index
    bm25 = BM25Index()
    chunks = [
        Chunk(
            chunk_id=f"{p.doc_id}-c0",
            doc_id=p.doc_id,
            text=f"{p.title}\n{p.text}",
            sentence_span=(0, 1),
        )
        for p in kb.passages
    ]
    bm25.index(chunks)

    tasks: list[tuple[SuiteItem, int]] = []
    for t in range(trials):
        for it in items:
            tasks.append((it, t))

    all_traces: list[ItemTrace] = []
    all_rows: list[ResultRow] = []

    if concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_task = {
                executor.submit(
                    _evaluate_single_item,
                    it,
                    trial,
                    kb,
                    generator,
                    q_map,
                    fact_map,
                    active_run_id,
                    bm25,
                    interv_obj,
                ): (it.item_id, trial)
                for it, trial in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                trace, row = future.result()
                all_traces.append(trace)
                all_rows.append(row)
    else:
        for it, trial in tasks:
            trace, row = _evaluate_single_item(
                it,
                trial,
                kb,
                generator,
                q_map,
                fact_map,
                active_run_id,
                bm25,
                interv_obj,
            )
            all_traces.append(trace)
            all_rows.append(row)

    # Sort outputs strictly by (item_id, trial) for byte-identical determinism
    all_traces.sort(key=lambda t: (t.item_id, t.trial))
    all_rows.sort(key=lambda r: (r.item_id, r.run_id))

    # Write traces
    for tr in all_traces:
        trace_file = traces_dir / f"{tr.item_id}__{tr.trial}.json"
        trace_file.write_text(
            json.dumps(tr.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # Write results.jsonl
    results_file = out_path / "results.jsonl"
    with results_file.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(
                json.dumps(r.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
            )

    finished_at = datetime.now(UTC).isoformat()

    # Compute suite and KB hashes
    suite_json = json.dumps([it.model_dump(mode="json") for it in items], sort_keys=True)
    suite_hash = hashlib.sha256(suite_json.encode()).hexdigest()
    kb_hash = hashlib.sha256(str(len(kb.passages)).encode()).hexdigest()

    manifest = RunManifest(
        run_id=active_run_id,
        generator=getattr(generator, "name", "unknown"),
        prompt_version=PROMPT_VERSION,
        git_sha=_get_git_sha(),
        kb_hash=kb_hash,
        suite_name=items[0].group if items else "empty",
        suite_hash=suite_hash,
        total_items=len(all_rows),
        concurrency=concurrency,
        trials=trials,
        intervention=interv_obj.name,
        started_at=started_at,
        finished_at=finished_at,
    )
    manifest_file = out_path / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return out_path
