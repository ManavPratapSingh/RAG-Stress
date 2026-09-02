import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from ragstress.cli import app

runner = CliRunner()


def hash_directory_files(dir_path: Path) -> dict[str, str]:
    """Calculate SHA-256 for all regular files in a directory."""
    hashes = {}
    for file_path in sorted(dir_path.glob("*")):
        if file_path.is_file():
            content = file_path.read_bytes()
            hashes[file_path.name] = hashlib.sha256(content).hexdigest()
    return hashes


def test_gen_kb_determinism_identical_hashes(tmp_path: Path) -> None:
    """Requirement FR-KB-6 / Acceptance:

    gen-kb --seed 42 twice -> byte-identical files with identical SHA-256 hashes.
    """
    out1 = tmp_path / "kb1"
    out2 = tmp_path / "kb2"

    result1 = runner.invoke(app, ["gen-kb", "--seed", "42", "--out", str(out1)])
    assert result1.exit_code == 0, f"Run 1 failed: {result1.output}"

    result2 = runner.invoke(app, ["gen-kb", "--seed", "42", "--out", str(out2)])
    assert result2.exit_code == 0, f"Run 2 failed: {result2.output}"

    hashes1 = hash_directory_files(out1)
    hashes2 = hash_directory_files(out2)

    assert hashes1, "No files generated in out1"
    assert hashes1 == hashes2, (
        f"Hashes mismatch between runs with same seed: {hashes1} vs {hashes2}"
    )

    # Also verify byte-for-byte identity of all files
    for fname in hashes1:
        f1_bytes = (out1 / fname).read_bytes()
        f2_bytes = (out2 / fname).read_bytes()
        assert f1_bytes == f2_bytes, f"Byte mismatch in file {fname}"


def test_gen_kb_different_seeds_produce_different_output(tmp_path: Path) -> None:
    """Requirement FR-KB-6: Different seeds MUST produce different output."""
    out1 = tmp_path / "kb_seed42"
    out2 = tmp_path / "kb_seed99"

    res1 = runner.invoke(app, ["gen-kb", "--seed", "42", "--out", str(out1)])
    assert res1.exit_code == 0

    res2 = runner.invoke(app, ["gen-kb", "--seed", "99", "--out", str(out2)])
    assert res2.exit_code == 0

    h1 = hash_directory_files(out1)
    h2 = hash_directory_files(out2)

    assert h1["passages.jsonl"] != h2["passages.jsonl"]
    assert h1["facts.jsonl"] != h2["facts.jsonl"]
    assert h1["questions.jsonl"] != h2["questions.jsonl"]


def test_gen_kb_invariants(tmp_path: Path) -> None:
    """Verify SPEC §4.3-§4.5 structural invariants:

    - 300 base passages
    - Each states 1-3 facts about ONE entity
    - Every (entity, attribute) appears in exactly one base passage, status=official
    - Entity names are unique
    """
    out = tmp_path / "kb"
    result = runner.invoke(app, ["gen-kb", "--seed", "42", "--out", str(out)])
    assert result.exit_code == 0

    passages_file = out / "passages.jsonl"
    facts_file = out / "facts.jsonl"
    questions_file = out / "questions.jsonl"

    assert passages_file.exists()
    assert facts_file.exists()
    assert questions_file.exists()

    passages = [json.loads(line) for line in passages_file.read_text().strip().split("\n")]
    facts = [json.loads(line) for line in facts_file.read_text().strip().split("\n")]
    questions = [json.loads(line) for line in questions_file.read_text().strip().split("\n")]

    # 300 passages in base KB
    assert len(passages) == 300

    # Entity names unique across KB
    entity_names = set()
    for f in facts:
        entity_names.add(f["entity_name"])
    assert len(entity_names) == 150  # 50 Policy + 50 Product + 50 Facility

    # Passage constraints
    seen_entity_attrs = set()
    for p in passages:
        p_facts = p["facts"]
        assert 1 <= len(p_facts) <= 3, (
            f"Passage {p['doc_id']} has {len(p_facts)} facts (expected 1-3)"
        )
        assert p["status"] == "official"

        # All facts in a passage belong to the same entity
        p_entities = {f["entity_id"] for f in p_facts}
        assert len(p_entities) == 1, (
            f"Passage {p['doc_id']} contains facts for multiple entities: {p_entities}"
        )

        for f in p_facts:
            key = (f["entity_id"], f["attribute"])
            assert key not in seen_entity_attrs, f"Duplicate fact in base KB for {key}"
            seen_entity_attrs.add(key)
            assert f["status"] == "official"

    # Exactly one base passage per (entity, attribute)
    assert len(seen_entity_attrs) == len(facts)
    assert len(facts) == 600

    # Questions correspond to facts
    assert len(questions) == len(facts)
    for q in questions:
        assert q["gold_value"] is not None
        assert q["gold_fact_id"] is not None
