"""Model pricing resolution and cost computation (SPEC §11.1, §14)."""

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


class PricingEngine:
    """Computes generation cost from config/models.yaml."""

    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            # Default search relative to repo root
            config_path = Path(__file__).resolve().parents[3] / "config" / "models.yaml"

        self.version = "unknown"
        self.models: dict[str, dict[str, float]] = {}

        if config_path.is_file():
            with open(config_path, encoding="utf-8") as f:
                data: dict[str, Any] = yaml.safe_load(f) or {}
                self.version = str(data.get("version", "unknown"))
                self.models = data.get("models", {})

    def resolve_model_key(self, generator_str: str) -> str | None:
        """Resolve generator identifier to pricing key."""
        gen = generator_str.strip()
        if gen in self.models:
            return gen
        # Strip provider prefix: "openai:gpt-4o-mini" -> "gpt-4o-mini"
        if ":" in gen:
            prefix, rest = gen.split(":", 1)
            if rest in self.models:
                return rest
            if prefix in self.models:
                return prefix
        return None

    def compute_cost(self, generator: str, input_tokens: int, output_tokens: int) -> float | None:
        """Compute cost in USD. Returns None if model is unknown."""
        key = self.resolve_model_key(generator)
        if key is None:
            return None
        rates = self.models[key]
        in_rate = rates.get("input_per_mtok", 0.0)
        out_rate = rates.get("output_per_mtok", 0.0)
        cost = (input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate
        return float(cost)
