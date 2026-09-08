from datetime import date
from pathlib import Path

import pytest

from ragstress.gen.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    render_user_message,
)
from ragstress.gen.recording import (
    CassetteMiss,
    RecordingGenerator,
    compute_cassette_key,
)
from ragstress.gen.replay import ReplayGenerator
from ragstress.gen.schema import (
    SUBMIT_ANSWER_SCHEMA,
    validate_tool_arguments,
)
from ragstress.gen.scripted import ScriptedGenerator
from ragstress.kb.facts import Passage
from ragstress.kb.generator import generate_base_kb
from ragstress.stress.builder import build_context
from ragstress.stress.spec import StressCondition


def test_submit_answer_schema_and_validation() -> None:
    # Valid call
    valid_args = {
        "answer": "1200",
        "abstain": False,
        "conflict": False,
        "conflicting_values": [],
        "citations": ["doc-abc123"],
    }
    out, err = validate_tool_arguments(valid_args)
    assert err is None
    assert out is not None
    assert out.answer == "1200"
    assert out.abstain is False

    # Schema violation: abstain=True but answer is provided
    invalid_args = {
        "answer": "1200",
        "abstain": True,
        "conflict": False,
        "conflicting_values": [],
        "citations": [],
    }
    out_err, err_msg = validate_tool_arguments(invalid_args)
    assert out_err is None
    assert err_msg is not None
    assert "abstain" in err_msg.lower()


def test_prompt_rendering() -> None:
    p = Passage(
        doc_id="doc-k3f9ab",
        title="Test Title",
        text="Test Text",
        status="official",
        effective_date=date(2024, 3, 1),
        facts=[],
    )
    user_msg = render_user_message([p], "What is the notice period?")
    assert "[doc_id=doc-k3f9ab | status=official | effective=2024-03-01]" in user_msg
    assert "Test Title" in user_msg
    assert "Test Text" in user_msg
    assert "Question: What is the notice period?" in user_msg


def test_scripted_generator_policies() -> None:
    # Set up a real context
    passages, facts, questions = generate_base_kb(seed=42)
    q = questions[0]
    cond = StressCondition(k_noise=3, conflict="dated")
    ctx = build_context(q, passages, facts, cond, seed=42, item_id="it-01")
    user_msg = render_user_message(ctx.passages, q.text)

    # 1. Oracle policy
    gen_oracle = ScriptedGenerator(
        policy="oracle", context_labels=ctx.labels, gold_value=q.gold_value
    )
    res_oracle = gen_oracle.generate(SYSTEM_PROMPT, user_msg, SUBMIT_ANSWER_SCHEMA)
    assert res_oracle.format_error is None
    assert res_oracle.output.answer == q.gold_value
    assert res_oracle.output.abstain is False

    # 2. Abstain always policy
    gen_abstain = ScriptedGenerator(policy="abstain_always")
    res_abstain = gen_abstain.generate(SYSTEM_PROMPT, user_msg, SUBMIT_ANSWER_SCHEMA)
    assert res_abstain.output.abstain is True
    assert res_abstain.output.answer is None

    # 3. Hallucinate policy
    gen_hallu = ScriptedGenerator(policy="hallucinate")
    res_hallu = gen_hallu.generate(SYSTEM_PROMPT, user_msg, SUBMIT_ANSWER_SCHEMA)
    assert res_hallu.output.answer == "hallucinated_unsupported_value_99999"

    # 4. Fabricate citation policy
    gen_fab = ScriptedGenerator(policy="fabricate_citation", gold_value=q.gold_value)
    res_fab = gen_fab.generate(SYSTEM_PROMPT, user_msg, SUBMIT_ANSWER_SCHEMA)
    assert "doc-fabric8" in res_fab.output.citations


def test_cassette_record_and_replay(tmp_path: Path) -> None:
    cassette_dir = tmp_path / "cassettes"
    model = "scripted:oracle"
    system = "System Instructions"
    user = "User Question"
    schema = SUBMIT_ANSWER_SCHEMA

    # Mock inner generator
    inner = ScriptedGenerator(policy="abstain_always")

    # 1. Record
    rec_gen = RecordingGenerator(inner, cassette_dir=cassette_dir, model_slug=model)
    res1 = rec_gen.generate(system, user, schema)
    assert res1.output.abstain is True

    # Verify cassette exists
    key = compute_cassette_key(model, PROMPT_VERSION, system, user, schema)
    cassette_file = cassette_dir / model.replace(":", "_") / f"{key}.json"
    assert cassette_file.exists()

    # 2. Replay
    replay_gen = ReplayGenerator(cassette_dir=cassette_dir, model_slug=model)
    res2 = replay_gen.generate(system, user, schema)
    assert res2.output.abstain is True
    assert res2.output.answer == res1.output.answer

    # 3. Miss raises CassetteMiss
    with pytest.raises(CassetteMiss):
        replay_gen.generate(system, "Unseen Question", schema)
