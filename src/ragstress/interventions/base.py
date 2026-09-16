"""Intervention protocols and pluggable mitigation implementations (SPEC §12, FR-INT-*)."""

import re
from typing import Protocol, runtime_checkable

from ragstress.gen.schema import GenOutput, GenResult
from ragstress.kb.facts import Passage
from ragstress.kb.questions import Question
from ragstress.stress.builder import Context


@runtime_checkable
class Intervention(Protocol):
    """Protocol for RAG interventions and mitigations (SPEC §12.1)."""

    name: str

    def transform_context(self, ctx: Context, q: Question) -> Context:
        """Transform context passages before prompt rendering."""
        ...

    def extra_system(self) -> str:
        """Extra prompt instructions appended AFTER base prompt; never edits R1-R6."""
        ...

    def postprocess(self, ctx: Context, out: GenResult) -> GenResult:
        """Postprocess generator output before grading."""
        ...


class BaselineIntervention:
    """Identity intervention (SPEC §12.1)."""

    name: str = "baseline"

    def transform_context(self, ctx: Context, q: Question) -> Context:
        return ctx

    def extra_system(self) -> str:
        return ""

    def postprocess(self, ctx: Context, out: GenResult) -> GenResult:
        return out


class ReorderEdgesIntervention:
    """Reorders passages so highest-scoring sit at start and end, lowest in middle (SPEC §12.2)."""

    name: str = "reorder_edges"

    def transform_context(self, ctx: Context, q: Question) -> Context:
        if not ctx.passages:
            return ctx

        # Determine passage scores
        score_map: dict[str, float] = {}
        if ctx.labels:
            for lbl in ctx.labels:
                doc_id = lbl.get("doc_id")
                if doc_id and "score" in lbl:
                    score_map[doc_id] = float(lbl["score"])

        # If no retriever scores present, score via question term overlap
        q_tokens = set(re.findall(r"\w+", q.text.lower()))
        for p in ctx.passages:
            if p.doc_id not in score_map:
                p_tokens = re.findall(r"\w+", p.text.lower())
                overlap = sum(1 for tok in p_tokens if tok in q_tokens)
                score_map[p.doc_id] = float(overlap)

        # Sort descending by score, breaking ties by doc_id
        sorted_passages = sorted(
            ctx.passages,
            key=lambda p: (score_map.get(p.doc_id, 0.0), p.doc_id),
            reverse=True,
        )

        n = len(sorted_passages)
        reordered: list[Passage] = [sorted_passages[0]] * n
        left, right = 0, n - 1
        for i, p in enumerate(sorted_passages):
            if i % 2 == 0:
                reordered[left] = p
                left += 1
            else:
                reordered[right] = p
                right -= 1

        new_passages = reordered
        return Context(
            passages=new_passages,
            labels=ctx.labels,
            gold_support_ids=ctx.gold_support_ids,
            relevant_ids=ctx.relevant_ids,
            gold_values=ctx.gold_values,
        )

    def extra_system(self) -> str:
        return ""

    def postprocess(self, ctx: Context, out: GenResult) -> GenResult:
        return out


class LexicalRerankIntervention:
    """Reranks passages by deterministic match of entity name and attribute label (SPEC §12.2)."""

    name: str = "lexical_rerank"

    def transform_context(self, ctx: Context, q: Question) -> Context:
        if not ctx.passages:
            return ctx

        e_target = (
            getattr(q, "entity_name", None)
            or (q.text.split("for")[-1].strip("? .") if "for" in q.text else "")
        ).lower()
        a_target = (getattr(q, "attribute_label", None) or q.attribute.replace("_", " ")).lower()

        # Sort descending by (entity_match, attribute_match) with stable initial index tie-breaking
        sorted_items = sorted(
            enumerate(ctx.passages),
            key=lambda item: (
                1 if e_target and e_target in item[1].text.lower() else 0,
                1 if a_target and a_target in item[1].text.lower() else 0,
                -item[0],
            ),
            reverse=True,
        )

        new_passages = [p for _, p in sorted_items]
        return Context(
            passages=new_passages,
            labels=ctx.labels,
            gold_support_ids=ctx.gold_support_ids,
            relevant_ids=ctx.relevant_ids,
            gold_values=ctx.gold_values,
        )

    def extra_system(self) -> str:
        return ""

    def postprocess(self, ctx: Context, out: GenResult) -> GenResult:
        return out


class CheckPromptIntervention:
    """Appends explicit checking instructions to system prompt (SPEC §12.2)."""

    name: str = "check_prompt"

    def transform_context(self, ctx: Context, q: Question) -> Context:
        return ctx

    def extra_system(self) -> str:
        return (
            "INSPECTION DIRECTIVE:\n"
            "Before answering, carefully verify whether any passage matches the exact entity name "
            "and attribute in the question. If no passage provides direct, authoritative evidence "
            "for that exact entity, set abstain to true."
        )

    def postprocess(self, ctx: Context, out: GenResult) -> GenResult:
        return out


class ScoreThresholdIntervention:
    """Mode B mitigation forcing abstention when top retrieval score is below threshold (SPEC §12.2)."""

    name: str = "score_threshold"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def transform_context(self, ctx: Context, q: Question) -> Context:
        return ctx

    def extra_system(self) -> str:
        return ""

    def postprocess(self, ctx: Context, out: GenResult) -> GenResult:
        if ctx.labels:
            max_score = max((float(lbl.get("score", 0.0)) for lbl in ctx.labels), default=0.0)
            if max_score < self.threshold:
                return GenResult(
                    output=GenOutput(
                        answer=None,
                        abstain=True,
                        conflict=False,
                        conflicting_values=[],
                        citations=[],
                    ),
                    raw=out.raw,
                    format_error=None,
                    usage=out.usage,
                    latency_ms=out.latency_ms,
                )
        return out


def get_intervention(name: str) -> Intervention:
    """Factory resolver for interventions."""
    clean_name = name.strip().lower()
    if clean_name == "baseline":
        return BaselineIntervention()
    if clean_name == "reorder_edges":
        return ReorderEdgesIntervention()
    if clean_name == "lexical_rerank":
        return LexicalRerankIntervention()
    if clean_name == "check_prompt":
        return CheckPromptIntervention()
    if clean_name == "score_threshold":
        return ScoreThresholdIntervention()
    raise ValueError(f"Unknown intervention: '{name}'")
