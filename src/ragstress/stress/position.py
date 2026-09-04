"""Position ordering and opaque doc ID assignment (SPEC §6.4, §6.5)."""

import base64
import hashlib
import random
from typing import Literal

from ragstress.kb.facts import Passage


def order_passages(
    gold: Passage,
    others: list[tuple[Passage, str]],
    position: Literal["top", "middle", "bottom", "random"],
    rng: random.Random,
) -> list[tuple[Passage, str, int]]:
    """Order gold and others according to position specification (SPEC §6.4)."""
    n = 1 + len(others)
    if position == "top":
        p = 0
    elif position == "middle":
        p = (n - 1) // 2
    elif position == "bottom":
        p = n - 1
    elif position == "random":
        p = rng.randint(0, n - 1)
    else:
        raise ValueError(f"Unknown position: {position}")

    shuffled_others = list(others)
    rng.shuffle(shuffled_others)

    ordered: list[tuple[Passage, str, int]] = []
    other_idx = 0
    for idx in range(n):
        if idx == p:
            ordered.append((gold, "gold", idx))
        else:
            passage, role = shuffled_others[other_idx]
            ordered.append((passage, role, idx))
            other_idx += 1

    return ordered


def make_opaque_doc_id(seed: int, item_id: str, internal_id: str) -> str:
    """Derive context doc_id of form doc- + 6 base32 characters (SPEC §6.5)."""
    key = f"{seed}:{item_id}:{internal_id}".encode()
    digest = hashlib.sha256(key).digest()
    b32 = base64.b32encode(digest).decode("ascii").lower()
    return f"doc-{b32[:6]}"


def assign_opaque_doc_ids(
    ordered_entries: list[tuple[Passage, str, int]],
    seed: int,
    item_id: str,
) -> list[Passage]:
    """Assign opaque doc_ids and set role on passages."""
    results: list[Passage] = []
    for passage, role, _ in ordered_entries:
        opaque_id = make_opaque_doc_id(seed, item_id, passage.doc_id)
        # Clone passage with opaque doc_id and role
        new_p = Passage(
            doc_id=opaque_id,
            title=passage.title,
            text=passage.text,
            status=passage.status,
            effective_date=passage.effective_date,
            facts=passage.facts,
            role=role,
        )
        results.append(new_p)
    return results
