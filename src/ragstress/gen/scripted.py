"""Deterministic scripted test double generators (SPEC §7.3, FR-GEN-3)."""

from typing import Any

from ragstress.gen.schema import GenOutput, GenResult, Usage


class ScriptedGenerator:
    """Deterministic generator with scripted test policies."""

    def __init__(
        self,
        policy: str = "oracle",
        context_labels: list[dict[str, Any]] | None = None,
        gold_value: str | None = None,
        gold_values: list[str] | None = None,
    ):
        self.name = f"scripted:{policy}"
        self.policy = policy
        self.context_labels = context_labels or []
        self.gold_value = gold_value
        self.gold_values = gold_values or ([gold_value] if gold_value else [])

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> GenResult:
        labels = self.context_labels

        if self.policy == "oracle":
            # If no gold passage in context -> abstain
            gold_entries = [e for e in labels if e.get("role") == "gold"]
            twin_entries = [e for e in labels if e.get("role") == "twin"]

            if not gold_entries and not twin_entries:
                output = GenOutput(abstain=True)
            elif len(self.gold_values) > 1:
                # Unresolvable conflict
                citations = [e["doc_id"] for e in labels if e.get("role") in {"gold", "twin"}]
                output = GenOutput(
                    answer=None,
                    abstain=False,
                    conflict=True,
                    conflicting_values=self.gold_values,
                    citations=citations,
                )
            else:
                citations = [
                    e["doc_id"] for e in labels if e.get("role") in {"gold", "gold_bridge"}
                ]
                output = GenOutput(
                    answer=self.gold_value,
                    abstain=False,
                    conflict=False,
                    conflicting_values=[],
                    citations=citations,
                )

        elif self.policy == "abstain_always":
            output = GenOutput(abstain=True)

        elif self.policy == "first_passage":
            if labels:
                first = labels[0]
                facts = first.get("facts", [])
                val = facts[0]["value"] if facts else "unknown"
                output = GenOutput(answer=val, citations=[first["doc_id"]])
            else:
                output = GenOutput(abstain=True)

        elif self.policy == "hallucinate":
            output = GenOutput(
                answer="hallucinated_unsupported_value_99999",
                abstain=False,
                citations=[],
            )

        elif self.policy == "ignore_conflict":
            # Answer with twin value if present
            twin_entries = [e for e in labels if e.get("role") == "twin"]
            if twin_entries:
                twin_doc = twin_entries[0]
                facts = twin_doc.get("facts", [])
                val = facts[0]["value"] if facts else "twin_val"
                output = GenOutput(
                    answer=val,
                    conflict=False,
                    citations=[twin_doc["doc_id"]],
                )
            else:
                output = GenOutput(answer=self.gold_value)

        elif self.policy == "pick_distractor":
            distractors = [e for e in labels if str(e.get("role", "")).startswith("distractor")]
            if distractors:
                dist = distractors[0]
                facts = dist.get("facts", [])
                val = facts[0]["value"] if facts else "distractor_val"
                output = GenOutput(answer=val, citations=[dist["doc_id"]])
            else:
                output = GenOutput(answer="distractor_val")

        elif self.policy == "fabricate_citation":
            output = GenOutput(
                answer=self.gold_value,
                citations=["doc-fabric8"],
            )

        else:
            raise ValueError(f"Unknown scripted policy: {self.policy}")

        return GenResult(
            output=output,
            raw=output.model_dump_json(),
            format_error=None,
            usage=Usage(input_tokens=100, output_tokens=25, cost_usd=0.0),
            latency_ms=10.0,
        )
