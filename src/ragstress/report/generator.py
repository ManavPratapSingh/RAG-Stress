"""Report generator rendering Markdown summaries (SPEC §11.2, FR-REP-2)."""

from pathlib import Path

import numpy as np

from ragstress.report.pricing import PricingEngine
from ragstress.runner.schema import ResultRow, RunManifest
from ragstress.stats.position import compute_lost_in_the_middle
from ragstress.stats.wilson import wilson_score_interval


def _fmt_prop(k: int, n: int) -> str:
    """Format proportion with 95% Wilson CI."""
    if n == 0:
        return "N/A"
    p = float(k) / float(n)
    low, high = wilson_score_interval(k, n)
    return f"{p * 100:.1f}% [{low * 100:.1f}%, {high * 100:.1f}%]"


def generate_report(run_dir: Path) -> str:
    """Generate Markdown evaluation report from a run directory."""
    manifest_file = run_dir / "manifest.json"
    results_file = run_dir / "results.jsonl"

    if not manifest_file.is_file() or not results_file.is_file():
        raise FileNotFoundError(f"Missing manifest.json or results.jsonl in {run_dir}")

    with open(manifest_file, encoding="utf-8") as f:
        manifest = RunManifest.model_validate_json(f.read())

    rows: list[ResultRow] = []
    with open(results_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(ResultRow.model_validate_json(line))

    pricing = PricingEngine()
    total_in_toks = sum(r.input_tokens for r in rows)
    total_out_toks = sum(r.output_tokens for r in rows)
    total_cost = pricing.compute_cost(manifest.generator, total_in_toks, total_out_toks)
    correct_count = sum(1 for r in rows if r.pass_answer)
    cost_per_correct = (
        (total_cost / correct_count) if total_cost is not None and correct_count > 0 else None
    )

    # 1. Header
    gen_errors = sum(1 for r in rows if r.failure_mode == "generator_error")
    closed_rows = [r for r in rows if r.group == "control" and r.kind == "unanswerable"]
    # SPEC §8.6: leakage_rate = unanswerable items whose answer == gold value (answered correctly without evidence)
    leakage_count = sum(1 for r in closed_rows if not r.abstain and r.provenance == "gold")
    leakage_rate = (leakage_count / len(closed_rows)) if closed_rows else 0.0

    lines = [
        f"# RAG-Stress Benchmark Report: `{manifest.run_id}`",
        "",
        "## 1. Executive Summary & Metadata",
        f"- **Generator:** `{manifest.generator}`",
        f"- **Prompt Version:** `{manifest.prompt_version}`",
        f"- **Git SHA:** `{manifest.git_sha}`",
        f"- **KB Hash:** `{manifest.kb_hash}`",
        f"- **Suite:** `{manifest.suite_name}` (hash: `{manifest.suite_hash}`)",
        f"- **Total Items:** {len(rows)} (across {manifest.trials} trial(s))",
        f"- **Generator Errors:** {gen_errors}",
        "",
    ]

    if closed_rows and leakage_rate > 0.05:
        lines.extend(
            [
                f"> [!WARNING] **LEAKAGE_WARNING:** Closed-book leakage rate is {leakage_rate * 100:.1f}%, which exceeds the 5.0% threshold! Evidence of pretraining contamination or guessable values.",
                "",
            ]
        )

    # 2. Controls
    oracle_rows = [r for r in rows if r.group == "control" and r.kind != "unanswerable"]
    oracle_acc = (
        _fmt_prop(sum(1 for r in oracle_rows if r.pass_strict), len(oracle_rows))
        if oracle_rows
        else "N/A"
    )
    closed_abstain = (
        _fmt_prop(sum(1 for r in closed_rows if r.abstain), len(closed_rows))
        if closed_rows
        else "N/A"
    )
    closed_leakage = _fmt_prop(leakage_count, len(closed_rows)) if closed_rows else "N/A"

    lines.extend(
        [
            "## 2. Controls",
            "| Control Metric | Result (95% CI) | Status |",
            "|---|---|---|",
            f"| Oracle Accuracy (`pass_strict`) | {oracle_acc} | Baseline calibration |",
            f"| Closed-Book Abstain Rate | {closed_abstain} | Noise-free abstention |",
            f"| Parametric Leakage Rate | {closed_leakage} | Target < 5.0% |",
            "",
        ]
    )

    # 3. Distractor Curve
    dist_rows = [r for r in rows if r.group == "distractor_sweep"]
    lines.extend(
        [
            "## 3. Distractor Noise Curve",
            "| Noise ($k$) | Items | Accuracy (`pass_answer`) | Hallucination Rate | Distractor Adoption |",
            "|---|---|---|---|---|",
        ]
    )
    if dist_rows:
        k_levels = sorted({r.k_noise for r in dist_rows})
        for k in k_levels:
            k_items = [r for r in dist_rows if r.k_noise == k]
            n_k = len(k_items)
            acc = _fmt_prop(sum(1 for r in k_items if r.pass_answer), n_k)
            hall = _fmt_prop(sum(1 for r in k_items if r.hallucinated), n_k)
            dist_adopt = _fmt_prop(sum(1 for r in k_items if r.provenance == "distractor"), n_k)
            lines.append(f"| {k} | {n_k} | {acc} | {hall} | {dist_adopt} |")
    else:
        lines.append("| N/A | 0 | N/A | N/A | N/A |")
    lines.append("")

    # 4. Position & Lost-in-the-Middle
    pos_rows = [r for r in rows if r.group == "position_sweep"]
    lines.extend(
        [
            "## 4. Position & Lost-in-the-Middle",
            "| Noise ($k$) | Top Acc | Middle Acc | Bottom Acc | Lost-in-the-Middle Score (95% CI) |",
            "|---|---|---|---|---|",
        ]
    )
    if pos_rows:
        pos_scores = compute_lost_in_the_middle(pos_rows)
        for k, sc in sorted(pos_scores.items()):
            t_acc = f"{sc['acc_top'] * 100:.1f}%"
            m_acc = f"{sc['acc_middle'] * 100:.1f}%"
            b_acc = f"{sc['acc_bottom'] * 100:.1f}%"
            score_ci = (
                f"{sc['score'] * 100:.1f} [{sc['ci_lower'] * 100:.1f}, {sc['ci_upper'] * 100:.1f}]"
            )
            lines.append(f"| {k} | {t_acc} | {m_acc} | {b_acc} | {score_ci} |")
    else:
        lines.append("| N/A | N/A | N/A | N/A | N/A |")
    lines.append("")

    # 5. Knowledge Conflict Dynamics
    conf_rows = [r for r in rows if r.group == "conflict"]
    lines.extend(
        [
            "## 5. Knowledge Conflict Dynamics",
            "| Conflict Kind | Items | Accuracy | Stale Adopted | Conflict Ignored | Flagged & Resolved | Values Complete |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    if conf_rows:
        for ck in ["dated", "authority", "unresolvable"]:
            ck_items = [r for r in conf_rows if r.conflict == ck]
            n_ck = len(ck_items)
            if n_ck > 0:
                acc = f"{sum(1 for r in ck_items if r.pass_answer) / n_ck * 100:.1f}%"
                stale = f"{sum(1 for r in ck_items if r.provenance == 'conflict_loser') / n_ck * 100:.1f}%"
                ignored = f"{sum(1 for r in ck_items if r.failure_mode == 'conflict_ignored') / n_ck * 100:.1f}%"
                flag_res = f"{sum(1 for r in ck_items if r.conflict_flagged and r.pass_answer) / n_ck * 100:.1f}%"
                vals_comp = f"{sum(1 for r in ck_items if r.values_complete) / n_ck * 100:.1f}%"
                lines.append(
                    f"| {ck} | {n_ck} | {acc} | {stale} | {ignored} | {flag_res} | {vals_comp} |"
                )
    else:
        lines.append("| N/A | 0 | N/A | N/A | N/A | N/A | N/A |")

    # False conflict flag rate
    plain_rows = [r for r in rows if r.group != "conflict" and r.conflict == "none"]
    false_flags = sum(1 for r in plain_rows if r.false_conflict_flag)
    false_flag_rate = _fmt_prop(false_flags, len(plain_rows))
    lines.extend(
        [
            "",
            f"- **False Conflict Flag Rate on Plain Items:** {false_flag_rate}",
            "",
        ]
    )

    # 6. Abstention Performance
    unans_rows = [r for r in rows if r.kind == "unanswerable"]
    ans_rows = [r for r in rows if r.kind != "unanswerable"]
    all_abstains = sum(1 for r in rows if r.abstain)
    true_abstains = sum(1 for r in unans_rows if r.abstain)
    false_abstains = sum(1 for r in ans_rows if r.abstain)

    abs_rec = _fmt_prop(true_abstains, len(unans_rows))
    abs_prec = _fmt_prop(true_abstains, all_abstains) if all_abstains > 0 else "N/A"
    false_abs_rate = _fmt_prop(false_abstains, len(ans_rows))

    lines.extend(
        [
            "## 6. Abstention Performance",
            "| Metric | Result (95% CI) | Description |",
            "|---|---|---|",
            f"| Abstain Recall | {abs_rec} | True abstains / All unanswerable items |",
            f"| Abstain Precision | {abs_prec} | True abstains / All abstentions |",
            f"| False Abstain Rate | {false_abs_rate} | Unwarranted abstains on answerable items |",
            "",
        ]
    )

    # 7. Citation Attribution
    valid_prec = [r.citation_precision for r in rows if r.citation_precision is not None]
    mean_prec = f"{float(np.mean(valid_prec)) * 100:.1f}%" if valid_prec else "N/A"
    mean_rec = f"{float(np.mean([r.citation_recall for r in rows])) * 100:.1f}%"
    exact_match = _fmt_prop(sum(1 for r in rows if r.citation_exact), len(rows))
    fab_count = sum(1 for r in rows if len(r.fabricated_citations) > 0)
    fab_rate = _fmt_prop(fab_count, len(rows))

    lines.extend(
        [
            "## 7. Citation Attribution",
            "| Metric | Result | Description |",
            "|---|---|---|",
            f"| Citation Precision | {mean_prec} | Cited context passages containing supporting facts |",
            f"| Citation Recall | {mean_rec} | Fraction of necessary gold passages cited |",
            f"| Exact Citation Rate | {exact_match} | Exact set match between cited and gold passages |",
            f"| Fabricated Citation Rate | {fab_rate} | Citations of IDs not present in context |",
            "",
        ]
    )

    # 8. End-to-End Retrieval
    e2e_rows = [r for r in rows if r.mode == "B"]
    lines.extend(
        [
            "## 8. End-to-End Retrieval",
        ]
    )
    if e2e_rows:
        lines.extend(
            [
                "| Retriever | $K$ | Hit Rate | Recall | Precision | MRR | nDCG |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        combos = sorted({(r.retriever or "unknown", r.k_retrieve or 0) for r in e2e_rows})
        for ret, k in combos:
            c_rows = [
                r
                for r in e2e_rows
                if (r.retriever or "unknown") == ret and (r.k_retrieve or 0) == k
            ]
            hit = f"{float(np.mean([r.retrieval_hit or 0.0 for r in c_rows])) * 100:.1f}%"
            rec = f"{float(np.mean([r.retrieval_recall or 0.0 for r in c_rows])) * 100:.1f}%"
            prec = f"{float(np.mean([r.retrieval_precision or 0.0 for r in c_rows])) * 100:.1f}%"
            mrr = f"{float(np.mean([r.retrieval_mrr or 0.0 for r in c_rows])):.3f}"
            ndcg = f"{float(np.mean([r.retrieval_ndcg or 0.0 for r in c_rows])):.3f}"
            lines.append(f"| {ret} | {k} | {hit} | {rec} | {prec} | {mrr} | {ndcg} |")

        e2e_fails = [r for r in e2e_rows if not r.pass_answer]
        miss_count = sum(1 for r in e2e_fails if r.failure_mode == "retrieval_miss")
        miss_share = f"{miss_count / len(e2e_fails) * 100:.1f}%" if e2e_fails else "0.0%"

        hit_rows = [r for r in e2e_rows if (r.retrieval_hit or 0.0) >= 1.0]
        acc_hit = _fmt_prop(sum(1 for r in hit_rows if r.pass_answer), len(hit_rows))
        miss_rows = [r for r in e2e_rows if (r.retrieval_hit or 0.0) < 1.0]
        acc_miss = _fmt_prop(sum(1 for r in miss_rows if r.pass_answer), len(miss_rows))

        lines.extend(
            [
                "",
                f"- **Share of Failures Attributed to Retrieval Miss:** {miss_share}",
                f"- **Accuracy Conditional on Retrieval Hit:** {acc_hit}",
                f"- **Accuracy Conditional on Retrieval Miss:** {acc_miss}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Evaluation performed in Mode A (pre-synthesized context matrices). End-to-end retriever measurements not applicable.",
                "",
            ]
        )

    # 9. Failure Taxonomy
    tax_bins = [
        "format_error",
        "retrieval_miss",
        "stale_adopted",
        "distractor_adopted",
        "hallucinated",
        "conflict_ignored",
        "false_abstain",
    ]
    lines.extend(
        [
            "## 9. Failure Taxonomy",
            "| Failure Category | Count | Percentage |",
            "|---|---|---|",
        ]
    )
    for b in tax_bins:
        cnt = sum(1 for r in rows if r.failure_mode == b)
        pct = (cnt / len(rows) * 100) if rows else 0.0
        lines.append(f"| `{b}` | {cnt} | {pct:.1f}% |")

    pass_cnt = sum(1 for r in rows if r.failure_mode is None or r.pass_strict)
    pass_pct = (pass_cnt / len(rows) * 100) if rows else 0.0
    lines.extend(
        [
            f"| `pass` | {pass_cnt} | {pass_pct:.1f}% |",
            "",
        ]
    )

    # 10. Cost & Latency
    lats = [r.latency_ms for r in rows]
    lat_mean = f"{float(np.mean(lats)):.1f} ms" if lats else "N/A"
    lat_p50 = f"{float(np.percentile(lats, 50)):.1f} ms" if lats else "N/A"
    lat_p95 = f"{float(np.percentile(lats, 95)):.1f} ms" if lats else "N/A"
    lat_p99 = f"{float(np.percentile(lats, 99)):.1f} ms" if lats else "N/A"
    cost_str = f"${total_cost:.4f}" if total_cost is not None else "N/A (unknown model pricing)"
    c_per_corr = f"${cost_per_correct:.4f}" if cost_per_correct is not None else "N/A"

    lines.extend(
        [
            "## 10. Latency and Cost",
            "| Metric | Value |",
            "|---|---|",
            f"| Mean Latency | {lat_mean} |",
            f"| Latency p50 / p95 / p99 | {lat_p50} / {lat_p95} / {lat_p99} |",
            f"| Total Input Tokens | {total_in_toks:,} |",
            f"| Total Output Tokens | {total_out_toks:,} |",
            f"| Pricing Table Version | {pricing.version} |",
            f"| Total Cost (USD) | {cost_str} |",
            f"| Cost Per Correct Item | {c_per_corr} |",
            "",
        ]
    )

    # 11. Reproduction Command
    lines.extend(
        [
            "## 11. Reproduction Command",
            "```bash",
            f"ragstress run --generator {manifest.generator} --suite data/suites/{manifest.suite_name}.jsonl --trials {manifest.trials} --concurrency {manifest.concurrency}",
            "```",
            "",
        ]
    )

    return "\n".join(lines)
