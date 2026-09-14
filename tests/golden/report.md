# RAG-Stress Benchmark Report: `golden_fixture_run`

## 1. Executive Summary & Metadata
- **Generator:** `openai:gpt-4o-mini`
- **Prompt Version:** `1.0`
- **Git SHA:** `golden0001`
- **KB Hash:** `golden_kb_hash_12345`
- **Suite:** `smoke` (hash: `golden_suite_hash_67890`)
- **Total Items:** 4 (across 1 trial(s))
- **Generator Errors:** 0

## 2. Controls
| Control Metric | Result (95% CI) | Status |
|---|---|---|
| Oracle Accuracy (`pass_strict`) | 100.0% [20.7%, 100.0%] | Baseline calibration |
| Closed-Book Abstain Rate | 100.0% [20.7%, 100.0%] | Noise-free abstention |
| Parametric Leakage Rate | 0.0% [0.0%, 79.3%] | Target < 5.0% |

## 3. Distractor Noise Curve
| Noise ($k$) | Items | Accuracy (`pass_answer`) | Hallucination Rate | Distractor Adoption |
|---|---|---|---|---|
| 5 | 1 | 100.0% [20.7%, 100.0%] | 0.0% [0.0%, 79.3%] | 0.0% [0.0%, 79.3%] |

## 4. Position & Lost-in-the-Middle
| Noise ($k$) | Top Acc | Middle Acc | Bottom Acc | Lost-in-the-Middle Score (95% CI) |
|---|---|---|---|---|
| N/A | N/A | N/A | N/A | N/A |

## 5. Knowledge Conflict Dynamics
| Conflict Kind | Items | Accuracy | Stale Adopted | Conflict Ignored | Flagged & Resolved | Values Complete |
|---|---|---|---|---|---|---|
| dated | 1 | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |

- **False Conflict Flag Rate on Plain Items:** 0.0% [0.0%, 56.1%]

## 6. Abstention Performance
| Metric | Result (95% CI) | Description |
|---|---|---|
| Abstain Recall | 100.0% [20.7%, 100.0%] | True abstains / All unanswerable items |
| Abstain Precision | 100.0% [20.7%, 100.0%] | True abstains / All abstentions |
| False Abstain Rate | 0.0% [0.0%, 56.1%] | Unwarranted abstains on answerable items |

## 7. Citation Attribution
| Metric | Result | Description |
|---|---|---|
| Citation Precision | 100.0% | Cited context passages containing supporting facts |
| Citation Recall | 87.5% | Fraction of necessary gold passages cited |
| Exact Citation Rate | 75.0% [30.1%, 95.4%] | Exact set match between cited and gold passages |
| Fabricated Citation Rate | 0.0% [0.0%, 49.0%] | Citations of IDs not present in context |

## 8. End-to-End Retrieval
Evaluation performed in Mode A (pre-synthesized context matrices). End-to-end retriever measurements not applicable.

## 9. Failure Taxonomy
| Failure Category | Count | Percentage |
|---|---|---|
| `format_error` | 0 | 0.0% |
| `retrieval_miss` | 0 | 0.0% |
| `stale_adopted` | 1 | 25.0% |
| `distractor_adopted` | 0 | 0.0% |
| `hallucinated` | 0 | 0.0% |
| `conflict_ignored` | 0 | 0.0% |
| `false_abstain` | 0 | 0.0% |
| `pass` | 3 | 75.0% |

## 10. Latency and Cost
| Metric | Value |
|---|---|
| Mean Latency | 240.0 ms |
| Latency p50 / p95 / p99 | 245.0 ms / 314.0 ms / 318.8 ms |
| Total Input Tokens | 1,050 |
| Total Output Tokens | 98 |
| Pricing Table Version | 2024-10-01 |
| Total Cost (USD) | $0.0002 |
| Cost Per Correct Item | $0.0001 |

## 11. Reproduction Command
```bash
ragstress run --generator openai:gpt-4o-mini --suite data/suites/smoke.jsonl --trials 1 --concurrency 1
```
