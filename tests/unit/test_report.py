"""Unit tests for reporting and comparative analysis (SPEC §11, FR-REP-2, FR-REP-3)."""

import json
from pathlib import Path

import pytest

from ragstress.report.compare import compare_runs
from ragstress.report.generator import generate_report
from ragstress.report.pricing import PricingEngine
from ragstress.runner.schema import ResultRow, RunManifest


@pytest.fixture
def sample_run_dir(tmp_path: Path) -> Path:
    """Create a mock run directory with manifest, traces, and results."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir(parents=True)

    manifest = RunManifest(
        run_id="run_001",
        generator="openai:gpt-4o-mini",
        prompt_version="1",
        git_sha="abcd123",
        kb_hash="kb123456",
        suite_name="smoke",
        suite_hash="suite123456",
        total_items=6,
        concurrency=1,
        trials=1,
        started_at="2026-10-15T12:00:00Z",
        finished_at="2026-10-15T12:00:05Z",
    )
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f)

    rows = [
        # 1. Oracle control (pass)
        ResultRow(
            run_id="run_001",
            item_id="ctrl-oracle-0001",
            group="control",
            split="test",
            kind="plain",
            mode="A",
            k_noise=0,
            position="top",
            conflict="none",
            k_retrieve=None,
            retriever=None,
            generator="openai:gpt-4o-mini",
            intervention="baseline",
            pass_answer=True,
            pass_strict=True,
            provenance="gold",
            hallucinated=False,
            failure_mode=None,
            abstain=False,
            conflict_flagged=False,
            false_conflict_flag=False,
            values_complete=True,
            citation_precision=1.0,
            citation_recall=1.0,
            citation_exact=True,
            fabricated_citations=[],
            input_tokens=100,
            output_tokens=20,
            latency_ms=250.0,
        ),
        # 2. Closed-book control (abstain, no leakage)
        ResultRow(
            run_id="run_001",
            item_id="ctrl-closed-0001",
            group="control",
            split="test",
            kind="unanswerable",
            mode="A",
            k_noise=0,
            position="top",
            conflict="none",
            k_retrieve=None,
            retriever=None,
            generator="openai:gpt-4o-mini",
            intervention="baseline",
            pass_answer=True,
            pass_strict=True,
            provenance="none",
            hallucinated=False,
            failure_mode=None,
            abstain=True,
            conflict_flagged=False,
            false_conflict_flag=False,
            values_complete=True,
            citation_precision=None,
            citation_recall=1.0,
            citation_exact=True,
            fabricated_citations=[],
            input_tokens=50,
            output_tokens=15,
            latency_ms=180.0,
        ),
        # 3. Distractor sweep k=5 (pass)
        ResultRow(
            run_id="run_001",
            item_id="dist-sweep-0001-k5",
            group="distractor_sweep",
            split="test",
            kind="plain",
            mode="A",
            k_noise=5,
            position="top",
            conflict="none",
            k_retrieve=None,
            retriever=None,
            generator="openai:gpt-4o-mini",
            intervention="baseline",
            pass_answer=True,
            pass_strict=True,
            provenance="gold",
            hallucinated=False,
            failure_mode=None,
            abstain=False,
            conflict_flagged=False,
            false_conflict_flag=False,
            values_complete=True,
            citation_precision=1.0,
            citation_recall=1.0,
            citation_exact=True,
            fabricated_citations=[],
            input_tokens=400,
            output_tokens=25,
            latency_ms=310.0,
        ),
        # 4. Position sweep middle k=10 (fail hallucinated)
        ResultRow(
            run_id="run_001",
            item_id="pos-sweep-0001-middle-k10",
            group="position_sweep",
            split="test",
            kind="plain",
            mode="A",
            k_noise=10,
            position="middle",
            conflict="none",
            k_retrieve=None,
            retriever=None,
            generator="openai:gpt-4o-mini",
            intervention="baseline",
            pass_answer=False,
            pass_strict=False,
            provenance="none",
            hallucinated=True,
            failure_mode="hallucinated",
            abstain=False,
            conflict_flagged=False,
            false_conflict_flag=False,
            values_complete=False,
            citation_precision=0.0,
            citation_recall=0.0,
            citation_exact=False,
            fabricated_citations=["doc-fabricated"],
            input_tokens=800,
            output_tokens=30,
            latency_ms=450.0,
        ),
        # 5. Conflict dated (stale adopted)
        ResultRow(
            run_id="run_001",
            item_id="conf-sweep-0001-dated",
            group="conflict",
            split="test",
            kind="dated",
            mode="A",
            k_noise=10,
            position="middle",
            conflict="dated",
            k_retrieve=None,
            retriever=None,
            generator="openai:gpt-4o-mini",
            intervention="baseline",
            pass_answer=False,
            pass_strict=False,
            provenance="conflict_loser",
            hallucinated=False,
            failure_mode="stale_adopted",
            abstain=False,
            conflict_flagged=False,
            false_conflict_flag=False,
            values_complete=False,
            citation_precision=1.0,
            citation_recall=0.5,
            citation_exact=False,
            fabricated_citations=[],
            input_tokens=750,
            output_tokens=22,
            latency_ms=390.0,
        ),
        # 6. Unanswerable k=10 (abstain recall)
        ResultRow(
            run_id="run_001",
            item_id="unans-sweep-0001-k10",
            group="unanswerable",
            split="test",
            kind="unanswerable",
            mode="A",
            k_noise=10,
            position="middle",
            conflict="none",
            k_retrieve=None,
            retriever=None,
            generator="openai:gpt-4o-mini",
            intervention="baseline",
            pass_answer=True,
            pass_strict=True,
            provenance="none",
            hallucinated=False,
            failure_mode=None,
            abstain=True,
            conflict_flagged=False,
            false_conflict_flag=False,
            values_complete=True,
            citation_precision=None,
            citation_recall=1.0,
            citation_exact=True,
            fabricated_citations=[],
            input_tokens=600,
            output_tokens=18,
            latency_ms=290.0,
        ),
    ]

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")

    return run_dir


def test_pricing_engine():
    """Verify PricingEngine cost calculation and fallback handling."""
    engine = PricingEngine()
    cost = engine.compute_cost("openai:gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.15 + 0.60)

    unknown_cost = engine.compute_cost("unknown-model", 100, 100)
    assert unknown_cost is None


def test_generate_report_all_sections(sample_run_dir: Path):
    """Verify generate_report renders all 11 required sections."""
    report_md = generate_report(sample_run_dir)

    # 1. Header
    assert "# RAG-Stress Benchmark Report" in report_md
    assert "openai:gpt-4o-mini" in report_md
    assert "smoke" in report_md

    # 2. Controls
    assert "## 2. Controls" in report_md
    assert "Oracle Accuracy" in report_md
    assert "Closed-Book Abstain Rate" in report_md

    # 3. Distractor Curve
    assert "## 3. Distractor Noise Curve" in report_md

    # 4. Position Sweep
    assert "## 4. Position & Lost-in-the-Middle" in report_md

    # 5. Conflict Dynamics
    assert "## 5. Knowledge Conflict Dynamics" in report_md

    # 6. Abstention
    assert "## 6. Abstention Performance" in report_md

    # 7. Citations
    assert "## 7. Citation Attribution" in report_md

    # 8. End-to-End
    assert "## 8. End-to-End Retrieval" in report_md

    # 9. Failure Taxonomy
    assert "## 9. Failure Taxonomy" in report_md

    # 10. Cost & Latency
    assert "## 10. Latency and Cost" in report_md

    # 11. Reproduction Command
    assert "## 11. Reproduction Command" in report_md
    assert "ragstress run" in report_md


def test_compare_runs(tmp_path: Path, sample_run_dir: Path):
    """Verify compare_runs computes McNemar and paired deltas on identical item sets."""
    # Create run B with one flipped item (hallucinated -> passed)
    run_b_dir = tmp_path / "run_002"
    run_b_dir.mkdir(parents=True)

    manifest_b = RunManifest(
        run_id="run_002",
        generator="openai:gpt-4o-mini",
        prompt_version="1",
        git_sha="abcd123",
        kb_hash="kb123456",
        suite_name="smoke",
        suite_hash="suite123456",
        total_items=6,
        concurrency=1,
        trials=1,
        started_at="2026-10-15T12:05:00Z",
        finished_at="2026-10-15T12:05:05Z",
    )
    with open(run_b_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_b.model_dump(), f)

    # Read run A rows and flip row 4
    with open(sample_run_dir / "results.jsonl", encoding="utf-8") as f:
        rows = [ResultRow.model_validate_json(line) for line in f]

    rows[3].pass_answer = True
    rows[3].pass_strict = True
    rows[3].hallucinated = False
    rows[3].failure_mode = None

    with open(run_b_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")

    comparison = compare_runs(sample_run_dir, run_b_dir)
    assert "# Run Comparison:" in comparison
    assert "run_001" in comparison
    assert "run_002" in comparison
    assert "McNemar" in comparison
    assert "Flipped Items" in comparison
    assert "pos-sweep-0001-middle-k10" in comparison


def test_compare_runs_mismatch_raises(tmp_path: Path, sample_run_dir: Path):
    """Refuse comparison on mismatched item sets."""
    run_c_dir = tmp_path / "run_003"
    run_c_dir.mkdir(parents=True)
    manifest = RunManifest(
        run_id="run_003",
        generator="test",
        prompt_version="1",
        git_sha="sha",
        kb_hash="kb",
        suite_name="smoke",
        suite_hash="suite",
        total_items=1,
        concurrency=1,
        trials=1,
        started_at="now",
        finished_at="now",
    )
    with open(run_c_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f)
    # Write a valid row with an item_id not in sample_run_dir
    different_row = ResultRow(
        run_id="run_003",
        item_id="completely-different-id",
        group="control",
        split="test",
        kind="plain",
        mode="A",
        k_noise=0,
        position="top",
        conflict="none",
        k_retrieve=None,
        retriever=None,
        generator="test",
        intervention="baseline",
        pass_answer=True,
        pass_strict=True,
        provenance="gold",
        hallucinated=False,
        failure_mode=None,
        abstain=False,
        conflict_flagged=False,
        false_conflict_flag=False,
        values_complete=True,
        citation_precision=1.0,
        citation_recall=1.0,
        citation_exact=True,
        fabricated_citations=[],
    )
    with open(run_c_dir / "results.jsonl", "w", encoding="utf-8") as f:
        f.write(different_row.model_dump_json() + "\n")

    with pytest.raises(ValueError, match="Item set mismatch"):
        compare_runs(sample_run_dir, run_c_dir)
