"""Cassette recording utilities and RecordingGenerator (SPEC §7.4, FR-GEN-4)."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragstress.gen.prompts import PROMPT_VERSION
from ragstress.gen.schema import GenResult


class CassetteMiss(Exception):
    """Raised when replaying from a cassette that does not exist offline."""


def canonical_json(data: Any) -> str:
    """Format JSON with sorted keys and no whitespace separators."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def compute_cassette_key(
    model: str,
    prompt_version: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> str:
    """Compute sha256 hash of canonical JSON query dictionary (SPEC §7.4)."""
    payload = {
        "model": model,
        "prompt_version": prompt_version,
        "system": system,
        "user": user,
        "schema": schema,
        "params": params or {},
    }
    canon = canonical_json(payload)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class RecordingGenerator:
    """Wraps an active generator and records outputs to disk cassettes."""

    def __init__(
        self,
        inner_generator: Any,
        cassette_dir: Path | str = "cassettes",
        model_slug: str | None = None,
    ):
        self.inner = inner_generator
        slug = model_slug or getattr(inner_generator, "name", "unknown")
        self.model_slug: str = str(slug)
        self.cassette_dir = Path(cassette_dir) / self.model_slug.replace(":", "_")
        self.cassette_dir.mkdir(parents=True, exist_ok=True)
        self.name = f"recording:{self.model_slug}"

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> GenResult:
        result: GenResult = self.inner.generate(system, user, schema)

        key = compute_cassette_key(self.model_slug, PROMPT_VERSION, system, user, schema)
        cassette_file = self.cassette_dir / f"{key}.json"

        record_data = {
            "result": result.model_dump(mode="json"),
            "recorded_with": {
                "model": self.model_slug,
                "date": datetime.now(UTC).isoformat(),
            },
        }
        cassette_file.write_text(
            json.dumps(record_data, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result
