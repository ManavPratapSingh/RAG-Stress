"""Runner schemas for manifests, item traces, and results (SPEC §10, FR-RUN-*)."""

from typing import Any

from pydantic import BaseModel, Field


class RunManifest(BaseModel):
    """Manifest recording run environment, versions, and metadata (SPEC §10.6)."""

    run_id: str
    generator: str
    prompt_version: str
    git_sha: str
    kb_hash: str
    suite_name: str
    suite_hash: str
    total_items: int
    concurrency: int
    trials: int
    started_at: str
    finished_at: str


class ItemTrace(BaseModel):
    """Full per-item execution trace (SPEC §10.4)."""

    run_id: str
    item_id: str
    trial: int = 0
    generator: str
    prompt_version: str
    intervention: str = "baseline"
    condition: dict[str, Any]
    context: list[dict[str, Any]]
    retrieved: list[dict[str, Any]] | None = None
    raw_output: str = ""
    parsed: dict[str, Any] | None = None
    format_error: str | None = None
    grades: dict[str, Any]
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class ResultRow(BaseModel):
    """Flattened evaluation result row for tabular analysis (SPEC §10.5)."""

    run_id: str
    item_id: str
    group: str
    split: str
    kind: str
    mode: str
    k_noise: int
    position: str
    conflict: str
    k_retrieve: int | None
    retriever: str | None
    generator: str
    intervention: str
    pass_answer: bool
    pass_strict: bool
    provenance: str | None
    hallucinated: bool
    failure_mode: str | None
    abstain: bool
    conflict_flagged: bool
    false_conflict_flag: bool
    values_complete: bool
    citation_precision: float | None
    citation_recall: float
    citation_exact: bool
    fabricated_citations: list[str]
    retrieval_hit: float | None = None
    retrieval_recall: float | None = None
    retrieval_precision: float | None = None
    retrieval_mrr: float | None = None
    retrieval_ndcg: float | None = None
    distractor_rate: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
