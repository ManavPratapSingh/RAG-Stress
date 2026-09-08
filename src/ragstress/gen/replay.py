"""Offline cassette replay generator (SPEC §7.4, FR-GEN-4)."""

import json
from pathlib import Path
from typing import Any

from ragstress.gen.prompts import PROMPT_VERSION
from ragstress.gen.recording import CassetteMiss, compute_cassette_key
from ragstress.gen.schema import GenResult


class ReplayGenerator:
    """Zero-network generator that serves pre-recorded responses from disk cassettes."""

    def __init__(
        self,
        cassette_dir: Path | str = "cassettes",
        model_slug: str = "default",
    ):
        self.model_slug = model_slug
        self.cassette_dir = Path(cassette_dir) / model_slug.replace(":", "_")
        self.name = f"replay:{model_slug}"

    def generate(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> GenResult:
        key = compute_cassette_key(self.model_slug, PROMPT_VERSION, system, user, schema, params)
        cassette_file = self.cassette_dir / f"{key}.json"

        if not cassette_file.exists():
            raise CassetteMiss(
                f"Cassette miss for model '{self.model_slug}', key '{key}'. "
                f"File not found: {cassette_file}. Replay mode forbids network calls."
            )

        data = json.loads(cassette_file.read_text(encoding="utf-8"))
        res_dict = data.get("result", {})
        return GenResult.model_validate(res_dict)
