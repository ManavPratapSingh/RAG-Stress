"""Command Line Interface for RAG-Stress (SPEC §13)."""

import hashlib
import json
from pathlib import Path
from typing import Any

import typer

from ragstress.kb.generator import generate_base_kb

app = typer.Typer(
    name="ragstress",
    help="RAG Hallucination & Retrieval Stress-Test Harness",
    no_args_is_help=True,
)


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@app.callback()
def main() -> None:
    """RAG-Stress: RAG Hallucination & Retrieval Stress-Test Harness."""
    pass


@app.command("gen-kb")
def gen_kb_command(
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for generation"),
    out: Path = typer.Option(Path("data/kb"), "--out", "-o", help="Output directory"),
) -> None:
    """Generate synthetic knowledge base, facts, passages, and questions (SPEC §4)."""
    out.mkdir(parents=True, exist_ok=True)

    passages, facts, questions = generate_base_kb(seed=seed)

    # 1. passages.jsonl
    passages_lines = [json.dumps(p.model_dump(mode="json"), sort_keys=True) for p in passages]
    passages_content = "\n".join(passages_lines) + "\n"
    passages_file = out / "passages.jsonl"
    passages_file.write_text(passages_content, encoding="utf-8")
    passages_sha256 = _compute_sha256(passages_content.encode("utf-8"))

    # 2. facts.jsonl
    facts_lines = [json.dumps(f.model_dump(mode="json"), sort_keys=True) for f in facts]
    facts_content = "\n".join(facts_lines) + "\n"
    facts_file = out / "facts.jsonl"
    facts_file.write_text(facts_content, encoding="utf-8")
    facts_sha256 = _compute_sha256(facts_content.encode("utf-8"))

    # 3. questions.jsonl
    questions_lines = [json.dumps(q.model_dump(mode="json"), sort_keys=True) for q in questions]
    questions_content = "\n".join(questions_lines) + "\n"
    questions_file = out / "questions.jsonl"
    questions_file.write_text(questions_content, encoding="utf-8")
    questions_sha256 = _compute_sha256(questions_content.encode("utf-8"))

    # 4. manifest.json
    manifest = {
        "seed": seed,
        "passage_count": len(passages),
        "fact_count": len(facts),
        "question_count": len(questions),
        "files": {
            "facts.jsonl": facts_sha256,
            "passages.jsonl": passages_sha256,
            "questions.jsonl": questions_sha256,
        },
    }
    manifest_content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out / "manifest.json").write_text(manifest_content, encoding="utf-8")

    typer.echo(
        f"Generated base KB: {len(passages)} passages, {len(facts)} facts, "
        f"{len(questions)} questions into {out} (seed={seed})"
    )


@app.command("generate-suite")
def generate_suite_command(
    name: str = typer.Option("smoke", "--name", "-n", help="Suite name (smoke or core)"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for generation"),
    kb_dir: Path = typer.Option(Path("data/kb"), "--kb-dir", help="Path to base KB directory"),
    out: Path = typer.Option(Path("data/suites"), "--out", "-o", help="Output directory"),
) -> None:
    """Generate evaluation suite items (SPEC §9.2)."""
    from ragstress.kb.facts import BaseKB, Fact, Passage
    from ragstress.kb.questions import Question
    from ragstress.suites.generator import generate_suite

    out.mkdir(parents=True, exist_ok=True)

    # Load or generate KB
    if (kb_dir / "passages.jsonl").exists():
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
    else:
        passages, facts, questions = generate_base_kb(seed=seed)
        kb = BaseKB(passages=passages, facts=facts, questions=questions)

    items = generate_suite(name=name, kb=kb, seed=seed)
    suite_file = out / f"{name}.jsonl"
    lines = [json.dumps(it.model_dump(mode="json"), sort_keys=True) for it in items]
    suite_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    typer.echo(f"Generated suite '{name}' with {len(items)} items to {suite_file}")


@app.command("validate-suite")
def validate_suite_command(
    suite_path: Path = typer.Argument(..., help="Path to suite .jsonl file"),
    kb_dir: Path = typer.Option(Path("data/kb"), "--kb-dir", help="Path to base KB directory"),
    seed: int = typer.Option(42, "--seed", "-s", help="Seed if generating base KB"),
) -> None:
    """Validate suite invariants (SPEC §9.4)."""
    from ragstress.kb.facts import BaseKB, Fact, Passage
    from ragstress.kb.questions import Question
    from ragstress.suites.schema import SuiteItem
    from ragstress.suites.validator import validate_suite

    if not suite_path.exists():
        typer.secho(f"Error: suite file {suite_path} does not exist", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    items = [
        SuiteItem.model_validate_json(line)
        for line in suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Load or generate KB
    if (kb_dir / "passages.jsonl").exists():
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
    else:
        passages, facts, questions = generate_base_kb(seed=seed)
        kb = BaseKB(passages=passages, facts=facts, questions=questions)

    res = validate_suite(items, kb)
    if res.valid:
        typer.secho(
            f"Suite validation PASSED for {res.total_items} items (oracle strict accuracy: {res.oracle_strict_accuracy:.1%})",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(f"Suite validation FAILED with {len(res.errors)} errors:", fg=typer.colors.RED)
        for err in res.errors[:10]:
            typer.echo(f"  - {err}")
        if len(res.errors) > 10:
            typer.echo(f"  ... and {len(res.errors) - 10} more errors")
        raise typer.Exit(code=1)


@app.command("run")
def run_command_cli(
    suite: Path = typer.Option(
        Path("data/suites/smoke.jsonl"), "--suite", help="Path to suite file"
    ),
    generator: str = typer.Option(
        "scripted:oracle", "--generator", "-g", help="Generator identifier"
    ),
    output_dir: Path = typer.Option(Path("runs/latest"), "--out", "-o", help="Output directory"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Worker concurrency"),
    trials: int = typer.Option(1, "--trials", "-t", help="Number of trials per item"),
    intervention: str = typer.Option(
        "baseline",
        "--intervention",
        "-i",
        help="Intervention to evaluate (baseline, reorder_edges, lexical_rerank, check_prompt, score_threshold)",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Seed if generating base KB"),
    kb_dir: Path = typer.Option(Path("data/kb"), "--kb-dir", help="Path to base KB directory"),
) -> None:
    """Execute benchmark runner on a suite (SPEC §10)."""
    from ragstress.gen.recording import RecordingGenerator
    from ragstress.gen.replay import ReplayGenerator
    from ragstress.gen.scripted import ScriptedGenerator
    from ragstress.kb.facts import BaseKB, Fact, Passage
    from ragstress.kb.questions import Question
    from ragstress.runner.executor import run_suite
    from ragstress.suites.generator import generate_suite
    from ragstress.suites.schema import SuiteItem

    # Load or generate KB
    if (kb_dir / "passages.jsonl").exists():
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
    else:
        passages, facts, questions = generate_base_kb(seed=seed)
        kb = BaseKB(passages=passages, facts=facts, questions=questions)

    # Load items
    if suite.exists():
        items = [
            SuiteItem.model_validate_json(line)
            for line in suite.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        # Fallback to generating smoke suite
        items = generate_suite(name="smoke", kb=kb, seed=seed)

    # Instantiate generator
    gen_instance: Any
    if generator.startswith("scripted:"):
        policy = generator.split(":", 1)[1]
        gen_instance = ScriptedGenerator(policy=policy)
    elif generator.startswith("replay:"):
        slug = generator.split(":", 1)[1]
        gen_instance = ReplayGenerator(model_slug=slug)
    elif generator.startswith("openai:"):
        from ragstress.gen.openai_gen import OpenAIGenerator

        model_name = generator.split(":", 1)[1]
        live_gen = OpenAIGenerator(model=model_name)
        gen_instance = RecordingGenerator(inner_generator=live_gen, model_slug=model_name)
    else:
        gen_instance = ScriptedGenerator(policy="oracle")

    run_path = run_suite(
        items=items,
        kb=kb,
        generator=gen_instance,
        output_dir=output_dir,
        concurrency=concurrency,
        trials=trials,
        intervention=intervention,
    )
    typer.secho(
        f"Benchmark run complete. Results and traces written to: {run_path}", fg=typer.colors.GREEN
    )


@app.command("report")
def report_command(
    run_dir: Path = typer.Argument(
        ..., help="Path to run directory containing results.jsonl and manifest.json"
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Optional path to output markdown file (default: <run_dir>/report.md)",
    ),
) -> None:
    """Generate markdown evaluation report from a benchmark run (SPEC §11.2)."""
    from ragstress.report.generator import generate_report

    if not run_dir.exists():
        typer.secho(f"Error: run directory {run_dir} does not exist", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        report_md = generate_report(run_dir)
        target = out or (run_dir / "report.md")
        target.write_text(report_md, encoding="utf-8")
        typer.secho(f"Report generated successfully: {target}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error generating report: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


@app.command("compare")
def compare_command(
    run_a: Path = typer.Argument(..., help="Path to first run directory (Run A)"),
    run_b: Path = typer.Argument(..., help="Path to second run directory (Run B)"),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Optional path to output comparison markdown file"
    ),
) -> None:
    """Compare two benchmark runs using paired statistics (SPEC §11.3)."""
    from ragstress.report.compare import compare_runs

    for p in [run_a, run_b]:
        if not p.exists():
            typer.secho(f"Error: run directory {p} does not exist", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    try:
        comp_md = compare_runs(run_a, run_b)
        if out:
            out.write_text(comp_md, encoding="utf-8")
            typer.secho(f"Comparison written to: {out}", fg=typer.colors.GREEN)
        else:
            typer.echo(comp_md)
    except ValueError as e:
        typer.secho(f"Comparison refused: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.secho(f"Error comparing runs: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


def gen_kb_standalone() -> None:
    """Entrypoint for standalone gen-kb CLI command."""
    gen_app = typer.Typer(add_completion=False)
    gen_app.command()(gen_kb_command)
    gen_app()


if __name__ == "__main__":
    app()
