"""Generator stack package."""

from ragstress.gen.openai_gen import OpenAIGenerator
from ragstress.gen.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    render_user_message,
)
from ragstress.gen.recording import (
    CassetteMiss,
    RecordingGenerator,
    canonical_json,
    compute_cassette_key,
)
from ragstress.gen.replay import ReplayGenerator
from ragstress.gen.schema import (
    SUBMIT_ANSWER_SCHEMA,
    GenOutput,
    GenResult,
    Usage,
    validate_tool_arguments,
)
from ragstress.gen.scripted import ScriptedGenerator

__all__ = [
    "PROMPT_VERSION",
    "SUBMIT_ANSWER_SCHEMA",
    "SYSTEM_PROMPT",
    "CassetteMiss",
    "GenOutput",
    "GenResult",
    "OpenAIGenerator",
    "RecordingGenerator",
    "ReplayGenerator",
    "ScriptedGenerator",
    "Usage",
    "canonical_json",
    "compute_cassette_key",
    "render_user_message",
    "validate_tool_arguments",
]
