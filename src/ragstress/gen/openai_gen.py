"""OpenAI generator adapter with forced tool call (SPEC §7.3)."""

import json
import time
from typing import Any

from ragstress.gen.schema import GenResult, Usage, validate_tool_arguments


class OpenAIGenerator:
    """OpenAI generator requiring forced submit_answer tool call."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ):
        try:
            import openai  # type: ignore[import-not-found,import-untyped]
        except ImportError as exc:
            raise ImportError("openai package is required to use OpenAIGenerator") from exc

        self.name = f"openai:{model}"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> GenResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tools = [{"type": "function", "function": schema}]
        tool_choice = {"type": "function", "function": {"name": schema["name"]}}

        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                tool_choice=tool_choice,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            choice = response.choices[0]
            tool_calls = choice.message.tool_calls

            if not tool_calls:
                return GenResult(
                    output=None,
                    raw=choice.message.content or "",
                    format_error="Model did not call forced tool submit_answer",
                    usage=Usage(
                        input_tokens=response.usage.prompt_tokens if response.usage else 0,
                        output_tokens=response.usage.completion_tokens if response.usage else 0,
                    ),
                    latency_ms=elapsed_ms,
                )

            tc = tool_calls[0]
            raw_args = tc.function.arguments
            parsed_args = json.loads(raw_args)
            output, format_error = validate_tool_arguments(parsed_args)

            usage = Usage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            return GenResult(
                output=output,
                raw=raw_args,
                format_error=format_error,
                usage=usage,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GenResult(
                output=None,
                raw="",
                format_error=f"Provider error: {e}",
                usage=Usage(),
                latency_ms=elapsed_ms,
            )
