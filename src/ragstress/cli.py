"""Command Line Interface for RAG-Stress (SPEC §13)."""

import hashlib
import json
from pathlib import Path

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


def gen_kb_standalone() -> None:
    """Entrypoint for standalone gen-kb CLI command."""
    gen_app = typer.Typer(add_completion=False)
    gen_app.command()(gen_kb_command)
    gen_app()


if __name__ == "__main__":
    app()
