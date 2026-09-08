"""Generator output models and submit_answer schema (SPEC §7.1, FR-GEN-1)."""

from typing import Any

from pydantic import BaseModel, Field, ValidationError


class GenOutput(BaseModel):
    answer: str | None = None
    abstain: bool = False
    conflict: bool = False
    conflicting_values: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class GenResult(BaseModel):
    output: GenOutput | None = None  # None on format error
    raw: str = ""
    format_error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0


SUBMIT_ANSWER_SCHEMA: dict[str, Any] = {
    "name": "submit_answer",
    "description": "Submit structured answer to the question using only provided passages.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": ["string", "null"],
                "description": "Direct answer string or null if abstaining.",
            },
            "abstain": {
                "type": "boolean",
                "description": "True if no passage states the answer for the exact entity and attribute asked.",
            },
            "conflict": {
                "type": "boolean",
                "description": "True if passages genuinely disagree and R3-R4 cannot decide.",
            },
            "conflicting_values": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of conflicting values if conflict is true.",
            },
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of doc_id strings for all supporting passages.",
            },
        },
        "required": ["answer", "abstain", "conflict", "conflicting_values", "citations"],
    },
}


def validate_tool_arguments(args: dict[str, Any]) -> tuple[GenOutput | None, str | None]:
    """Validate tool call arguments against schema rules (SPEC §7.1).

    Enforces logical invariants:
    - abstain=True with non-null answer is a schema violation.
    - conflict=True with < 2 conflicting values is a schema violation.
    """
    try:
        out = GenOutput.model_validate(args)
    except ValidationError as e:
        return None, f"Schema validation error: {e}"

    if out.abstain and out.answer is not None:
        return None, "Invalid tool call: abstain=True requires answer=None"

    if out.conflict and len(out.conflicting_values) < 2:
        return None, "Invalid tool call: conflict=True requires at least 2 conflicting_values"

    return out, None
