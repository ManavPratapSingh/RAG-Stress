"""System prompts, prompt versioning, and user message formatting (SPEC §7.2, FR-GEN-2)."""

from ragstress.kb.facts import Passage

PROMPT_VERSION = "1"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided passages. Each passage has a header with "
    "its doc_id, status (official or draft) and effective date. Rules:\n"
    "R1. Use only information in the passages; do not use outside knowledge.\n"
    "R2. If no passage states the answer for the exact entity and attribute asked, abstain.\n"
    "R3. If passages disagree, prefer an official passage over a draft.\n"
    "R4. Among official passages that disagree, prefer the one with the latest effective date.\n"
    "R5. If passages disagree and R3–R4 cannot decide (same status and same effective date), do not choose: "
    "set conflict=true and list all conflicting values.\n"
    "R6. Cite the doc_id of every passage that supports your answer.\n"
    "Respond only by calling submit_answer."
)


def render_user_message(passages: list[Passage], question_text: str) -> str:
    """Render the user prompt per SPEC §7.2 format."""
    sections = ["Passages:\n"]

    for p in passages:
        header = (
            f"[doc_id={p.doc_id} | status={p.status} | effective={p.effective_date.isoformat()}]"
        )
        sections.append(f"{header}\n{p.title}\n{p.text}\n")

    sections.append(f"Question: {question_text}")
    return "\n".join(sections)
