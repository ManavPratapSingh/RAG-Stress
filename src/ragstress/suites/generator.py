"""Suite generator for smoke and core benchmarks (SPEC §9.2, FR-SUITE-2)."""

import hashlib
import random
from typing import Literal

from ragstress.kb.facts import BaseKB
from ragstress.stress.spec import StressCondition
from ragstress.suites.schema import SuiteItem


def _assign_split(question_id: str) -> Literal["dev", "test"]:
    h = int(hashlib.sha256(f"split:{question_id}".encode()).hexdigest(), 16)
    return "dev" if (h % 100) < 25 else "test"


def generate_suite(
    name: str = "smoke",
    kb: BaseKB | None = None,
    seed: int = 42,
) -> list[SuiteItem]:
    """Generate a standard benchmark suite (smoke or core) from a knowledge base."""
    if kb is None:
        from ragstress.kb.generator import generate_base_kb_container

        kb = generate_base_kb_container(seed=seed)

    rng = random.Random(seed)
    questions = sorted(kb.questions, key=lambda q: q.question_id)
    n_q = len(questions)

    is_smoke = name == "smoke"
    items: list[SuiteItem] = []

    if name == "multihop":
        from ragstress.multihop.generator import generate_multihop_questions

        mh_qs = generate_multihop_questions(kb, seed=seed, count=20)
        existing_ids = {q.question_id for q in kb.questions}
        for mq in mh_qs:
            if mq.question_id not in existing_ids:
                kb.questions.append(mq)

        for i, mq in enumerate(mh_qs):
            split = _assign_split(mq.question_id)
            cond_mode: Literal["A", "B"] = "B" if (i % 4 == 3) else "A"
            k_n = 0 if (i % 4 == 0) else (5 if (i % 4 == 1) else 10)
            items.append(
                SuiteItem(
                    item_id=f"multihop-{i:04d}-{cond_mode.lower()}-k{k_n if cond_mode == 'A' else 5}",
                    split=split,
                    group="multihop",
                    question_id=mq.question_id,
                    seed=rng.randint(0, 2**31 - 1),
                    condition=StressCondition(
                        k_noise=k_n if cond_mode == "A" else 0,
                        position="random",
                        conflict="none",
                        answerable=True,
                        mode=cond_mode,
                        k_retrieve=5,
                    ),
                )
            )
        items.sort(key=lambda it: it.item_id)
        return items

    # 1. Controls: Oracle (k=0, answerable) & Closed-book (k=0, unanswerable)
    n_ctrl = 4 if is_smoke else 60
    ctrl_indices = rng.sample(range(n_q), min(n_ctrl, n_q))
    for i, idx in enumerate(ctrl_indices):
        q = questions[idx]
        split = _assign_split(q.question_id)

        # Oracle control
        items.append(
            SuiteItem(
                item_id=f"ctrl-oracle-{i:04d}",
                split=split,
                group="controls",
                question_id=q.question_id,
                seed=rng.randint(0, 2**31 - 1),
                condition=StressCondition(
                    k_noise=0,
                    position="top",
                    conflict="none",
                    answerable=True,
                    mode="A",
                ),
            )
        )
        # Closed-book control
        items.append(
            SuiteItem(
                item_id=f"ctrl-closed-{i:04d}",
                split=split,
                group="controls",
                question_id=q.question_id,
                seed=rng.randint(0, 2**31 - 1),
                condition=StressCondition(
                    k_noise=0,
                    position="random",
                    conflict="none",
                    answerable=False,
                    mode="A",
                ),
            )
        )

    # 2. Distractor sweep: k_noise in {0, 5, 10, 25, 50}
    n_dist = 4 if is_smoke else 60
    dist_indices = rng.sample(range(n_q), min(n_dist, n_q))
    k_dist_levels = [0, 5, 10, 25, 50]
    for i, idx in enumerate(dist_indices):
        q = questions[idx]
        split = _assign_split(q.question_id)
        for k in k_dist_levels:
            items.append(
                SuiteItem(
                    item_id=f"dist-sweep-{i:04d}-k{k}",
                    split=split,
                    group="distractor_sweep",
                    question_id=q.question_id,
                    seed=rng.randint(0, 2**31 - 1),
                    condition=StressCondition(
                        k_noise=k,
                        position="random",
                        conflict="none",
                        answerable=True,
                        mode="A",
                    ),
                )
            )

    # 3. Position sweep: positions in {top, middle, bottom} x k_noise in {10, 30}
    n_pos = 2 if is_smoke else 60
    pos_indices = rng.sample(range(n_q), min(n_pos, n_q))
    positions: list[Literal["top", "middle", "bottom"]] = ["top", "middle", "bottom"]
    k_pos_levels = [10, 30]
    for i, idx in enumerate(pos_indices):
        q = questions[idx]
        split = _assign_split(q.question_id)
        for k in k_pos_levels:
            for p in positions:
                items.append(
                    SuiteItem(
                        item_id=f"pos-sweep-{i:04d}-{p}-k{k}",
                        split=split,
                        group="position_sweep",
                        question_id=q.question_id,
                        seed=rng.randint(0, 2**31 - 1),
                        condition=StressCondition(
                            k_noise=k,
                            position=p,
                            conflict="none",
                            answerable=True,
                            mode="A",
                        ),
                    )
                )

    # 4. Conflict: dated, authority, unresolvable (k_noise=10, position=random)
    n_conf = 2 if is_smoke else 40
    conf_indices = rng.sample(range(n_q), min(n_conf, n_q))
    conflict_kinds: list[Literal["dated", "authority", "unresolvable"]] = [
        "dated",
        "authority",
        "unresolvable",
    ]
    for i, idx in enumerate(conf_indices):
        q = questions[idx]
        split = _assign_split(q.question_id)
        for ck in conflict_kinds:
            items.append(
                SuiteItem(
                    item_id=f"conflict-{i:04d}-{ck}",
                    split=split,
                    group="conflict",
                    question_id=q.question_id,
                    seed=rng.randint(0, 2**31 - 1),
                    condition=StressCondition(
                        k_noise=10,
                        position="random",
                        conflict=ck,
                        answerable=True,
                        mode="A",
                    ),
                )
            )

    # 5. Unanswerable: k_noise in {5, 10, 25}
    n_unans = 2 if is_smoke else 40
    unans_indices = rng.sample(range(n_q), min(n_unans, n_q))
    k_unans_levels = [5, 10, 25]
    for i, idx in enumerate(unans_indices):
        q = questions[idx]
        split = _assign_split(q.question_id)
        for k in k_unans_levels:
            items.append(
                SuiteItem(
                    item_id=f"unans-{i:04d}-k{k}",
                    split=split,
                    group="unanswerable",
                    question_id=q.question_id,
                    seed=rng.randint(0, 2**31 - 1),
                    condition=StressCondition(
                        k_noise=k,
                        position="random",
                        conflict="none",
                        answerable=False,
                        mode="A",
                    ),
                )
            )

    # 6. E2E Mode B
    n_e2e = 6 if is_smoke else 100
    e2e_indices = rng.sample(range(n_q), min(n_e2e, n_q))
    for i, idx in enumerate(e2e_indices):
        q = questions[idx]
        split = _assign_split(q.question_id)
        # Mix of plain, conflict, unanswerable per SPEC §9.2
        if i % 3 == 0:
            c_kind: Literal["none", "dated", "authority", "unresolvable"] = "none"
            ans = True
        elif i % 3 == 1:
            c_kind = "dated"
            ans = True
        else:
            c_kind = "none"
            ans = False

        items.append(
            SuiteItem(
                item_id=f"e2e-{i:04d}-k5",
                split=split,
                group="e2e",
                question_id=q.question_id,
                seed=rng.randint(0, 2**31 - 1),
                condition=StressCondition(
                    k_noise=0,
                    position="random",
                    conflict=c_kind,
                    answerable=ans,
                    mode="B",
                    k_retrieve=5,
                ),
            )
        )

    # Ensure deterministic output ordering by item_id
    items.sort(key=lambda it: it.item_id)
    return items
