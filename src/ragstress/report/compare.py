"""Comparative paired analysis between two benchmark runs (SPEC §11.3, FR-REP-3)."""

from pathlib import Path

from ragstress.runner.schema import ResultRow, RunManifest
from ragstress.stats.bootstrap import paired_bootstrap_mean_difference
from ragstress.stats.mcnemar import mcnemar_exact_test


def compare_runs(run_a_dir: Path, run_b_dir: Path) -> str:
    """Compare two benchmark runs on identical item sets using paired statistics."""
    manifest_a_file = run_a_dir / "manifest.json"
    results_a_file = run_a_dir / "results.jsonl"
    manifest_b_file = run_b_dir / "manifest.json"
    results_b_file = run_b_dir / "results.jsonl"

    for p in [manifest_a_file, results_a_file, manifest_b_file, results_b_file]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing run file: {p}")

    with open(manifest_a_file, encoding="utf-8") as f:
        manifest_a = RunManifest.model_validate_json(f.read())
    with open(manifest_b_file, encoding="utf-8") as f:
        manifest_b = RunManifest.model_validate_json(f.read())

    rows_a_map: dict[str, ResultRow] = {}
    with open(results_a_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = ResultRow.model_validate_json(line)
                rows_a_map[row.item_id] = row

    rows_b_map: dict[str, ResultRow] = {}
    with open(results_b_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = ResultRow.model_validate_json(line)
                rows_b_map[row.item_id] = row

    # Invariant: Refuse on item-set mismatch
    if sorted(rows_a_map.keys()) != sorted(rows_b_map.keys()):
        raise ValueError(
            f"Item set mismatch: Run A has {len(rows_a_map)} items, Run B has {len(rows_b_map)} items."
        )

    common_ids = sorted(rows_a_map.keys())
    aligned_a = [rows_a_map[iid] for iid in common_ids]
    aligned_b = [rows_b_map[iid] for iid in common_ids]

    # Metrics for paired analysis
    # 1. Answer Accuracy
    a_pass_ans = [float(r.pass_answer) for r in aligned_a]
    b_pass_ans = [float(r.pass_answer) for r in aligned_b]
    diff_ans, ci_low_ans, ci_high_ans = paired_bootstrap_mean_difference(b_pass_ans, a_pass_ans)
    b_ans, c_ans, p_ans = mcnemar_exact_test(
        [r.pass_answer for r in aligned_b], [r.pass_answer for r in aligned_a]
    )

    # 2. Strict Accuracy
    a_pass_strict = [float(r.pass_strict) for r in aligned_a]
    b_pass_strict = [float(r.pass_strict) for r in aligned_b]
    diff_st, ci_low_st, ci_high_st = paired_bootstrap_mean_difference(b_pass_strict, a_pass_strict)
    b_st, c_st, p_st = mcnemar_exact_test(
        [r.pass_strict for r in aligned_b], [r.pass_strict for r in aligned_a]
    )

    # 3. Hallucination Rate
    a_hall = [float(r.hallucinated) for r in aligned_a]
    b_hall = [float(r.hallucinated) for r in aligned_b]
    diff_hall, ci_low_hall, ci_high_hall = paired_bootstrap_mean_difference(b_hall, a_hall)
    b_h, c_h, p_hall = mcnemar_exact_test(
        [r.hallucinated for r in aligned_b], [r.hallucinated for r in aligned_a]
    )

    # 4. False Abstain Rate (SPEC §12.3)
    a_false_abs = [float(r.abstain and r.kind != "unanswerable") for r in aligned_a]
    b_false_abs = [float(r.abstain and r.kind != "unanswerable") for r in aligned_b]
    diff_fa, ci_low_fa, ci_high_fa = paired_bootstrap_mean_difference(b_false_abs, a_false_abs)
    b_fa, c_fa, p_fa = mcnemar_exact_test(
        [bool(x) for x in b_false_abs], [bool(x) for x in a_false_abs]
    )

    lines = [
        f"# Run Comparison: `{manifest_a.run_id}` vs `{manifest_b.run_id}`",
        "",
        "## Configuration",
        f"- **Run A:** `{manifest_a.generator}` (run_id: `{manifest_a.run_id}`)",
        f"- **Run B:** `{manifest_b.generator}` (run_id: `{manifest_b.run_id}`)",
        f"- **Total Shared Items:** {len(common_ids)}",
        "",
        "## Paired Significance Tests (Run B − Run A)",
        "| Metric | Mean Delta | 95% Bootstrap CI | McNemar ($b/c$) | McNemar $p$-value |",
        "|---|---|---|---|---|",
        f"| Accuracy (`pass_answer`) | {diff_ans * 100:+.1f}% | [{ci_low_ans * 100:+.1f}%, {ci_high_ans * 100:+.1f}%] | {b_ans} / {c_ans} | {p_ans:.4f} |",
        f"| Strict Accuracy (`pass_strict`) | {diff_st * 100:+.1f}% | [{ci_low_st * 100:+.1f}%, {ci_high_st * 100:+.1f}%] | {b_st} / {c_st} | {p_st:.4f} |",
        f"| Hallucination Rate | {diff_hall * 100:+.1f}% | [{ci_low_hall * 100:+.1f}%, {ci_high_hall * 100:+.1f}%] | {b_h} / {c_h} | {p_hall:.4f} |",
        f"| False Abstain Rate | {diff_fa * 100:+.1f}% | [{ci_low_fa * 100:+.1f}%, {ci_high_fa * 100:+.1f}%] | {b_fa} / {c_fa} | {p_fa:.4f} |",
        "",
    ]

    # Flipped items
    flipped = []
    for ra, rb in zip(aligned_a, aligned_b, strict=True):
        if ra.pass_answer != rb.pass_answer or ra.failure_mode != rb.failure_mode:
            flipped.append((ra, rb))

    lines.extend(
        [
            f"## Flipped Items ({len(flipped)} changes)",
            "| Item ID | Run A Pass | Run B Pass | Run A Taxonomy | Run B Taxonomy |",
            "|---|---|---|---|---|",
        ]
    )
    if flipped:
        for ra, rb in flipped:
            a_p = "PASS" if ra.pass_answer else "FAIL"
            b_p = "PASS" if rb.pass_answer else "FAIL"
            a_tax = f"`{ra.failure_mode or 'pass'}`"
            b_tax = f"`{rb.failure_mode or 'pass'}`"
            lines.append(f"| `{ra.item_id}` | {a_p} | {b_p} | {a_tax} | {b_tax} |")
    else:
        lines.append("| None | - | - | - | - |")

    lines.append("")
    return "\n".join(lines)
