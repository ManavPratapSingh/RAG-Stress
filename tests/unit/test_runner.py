"""Tests for benchmark execution runner and determinism (SPEC §10, FR-RUN-*)."""

import hashlib
import json
import tempfile
from pathlib import Path

from ragstress.gen.scripted import ScriptedGenerator
from ragstress.kb.generator import generate_base_kb_container
from ragstress.runner.executor import run_suite
from ragstress.suites.generator import generate_suite


def test_runner_mode_a_determinism_and_replay() -> None:
    kb = generate_base_kb_container(seed=42)
    smoke_items = generate_suite(name="smoke", kb=kb, seed=42)[:10]

    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        out1 = Path(tmp1)
        out2 = Path(tmp2)

        # Run 1 with oracle generator
        gen1 = ScriptedGenerator(policy="oracle")
        run_suite(items=smoke_items, kb=kb, generator=gen1, output_dir=out1, concurrency=2)

        # Run 2 with oracle generator
        gen2 = ScriptedGenerator(policy="oracle")
        run_suite(items=smoke_items, kb=kb, generator=gen2, output_dir=out2, concurrency=2)

        results1 = (out1 / "results.jsonl").read_text(encoding="utf-8")
        results2 = (out2 / "results.jsonl").read_text(encoding="utf-8")

        h1 = hashlib.sha256(results1.encode()).hexdigest()
        h2 = hashlib.sha256(results2.encode()).hexdigest()

        assert h1 == h2, "Replay runs must produce byte-identical results.jsonl"

        # Verify trace files exist and are valid JSON
        trace_files = list((out1 / "traces").glob("*.json"))
        assert len(trace_files) == 10
        trace_sample = json.loads(trace_files[0].read_text(encoding="utf-8"))
        assert "run_id" in trace_sample
        assert "item_id" in trace_sample
        assert "grades" in trace_sample

        # Verify manifest
        manifest_path = out1 / "manifest.json"
        assert manifest_path.exists()
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["total_items"] == 10
        assert "git_sha" in manifest_data


def test_runner_mode_b_retrieval_and_metrics() -> None:
    kb = generate_base_kb_container(seed=42)
    # Get an e2e Mode B item
    items = generate_suite(name="smoke", kb=kb, seed=42)
    e2e_items = [it for it in items if it.condition.mode == "B"]
    if not e2e_items:
        # Fallback: force one item to Mode B
        item = items[0].model_copy(deep=True)
        item.condition.mode = "B"
        item.condition.k_retrieve = 5
        e2e_items = [item]
    else:
        e2e_items = e2e_items[:3]

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        gen = ScriptedGenerator(policy="oracle")
        run_suite(items=e2e_items, kb=kb, generator=gen, output_dir=out_dir, concurrency=1)

        results = [
            json.loads(line) for line in (out_dir / "results.jsonl").read_text().splitlines()
        ]
        assert len(results) == len(e2e_items)
        assert results[0]["mode"] == "B"
        assert "retrieval_hit" in results[0]
