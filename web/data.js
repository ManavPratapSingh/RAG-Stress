// RAG-Stress Real Benchmark Data & Representative Traces
// Pre-compiled from benchmark runs on 1,120 items (gpt-4o vs gpt-4o-mini)

window.RAG_DATA = {
  metadata: {
    dataset_name: "RAG-Stress Core Benchmark",
    total_items: 1120,
    seed: 42,
    kb_passages: 300,
    kb_facts: 600,
    kb_entities: 150,
    git_sha: "8c2456c",
    prompt_version: 1
  },

  models: {
    "gpt-4o": {
      name: "GPT-4o (Frontier Baseline)",
      slug: "gpt-4o",
      pass_answer: 87.8,
      pass_answer_ci: [85.8, 89.6],
      pass_strict: 87.8,
      pass_strict_ci: [85.8, 89.6],
      hallucination_rate: 0.0,
      hallucination_ci: [0.0, 0.3],
      false_abstain_rate: 0.0,
      false_abstain_ci: [0.0, 0.4],
      citation_precision: 95.0,
      citation_recall: 96.0,
      exact_citation_rate: 96.0,
      exact_citation_ci: [94.7, 97.0],
      abstain_recall: 99.1,
      lim_score_k10: 7.5,
      lim_ci_k10: [0.8, 15.8],
      lim_score_k30: 5.0,
      lim_ci_k30: [-2.5, 13.3],
      distractor_curve: [
        { k: 0, accuracy: 100.0, ci: [94.0, 100.0], distractor_adopted: 0.0 },
        { k: 5, accuracy: 95.0, ci: [86.3, 98.3], distractor_adopted: 5.0 },
        { k: 10, accuracy: 100.0, ci: [94.0, 100.0], distractor_adopted: 0.0 },
        { k: 25, accuracy: 90.0, ci: [79.9, 95.3], distractor_adopted: 10.0 },
        { k: 50, accuracy: 86.7, ci: [75.8, 93.1], distractor_adopted: 13.3 }
      ],
      position_curve_k10: { top: 98.3, middle: 91.7, bottom: 100.0 },
      position_curve_k30: { top: 95.0, middle: 91.7, bottom: 98.3 },
      conflicts: {
        dated: { accuracy: 87.5, stale_adopted: 12.5 },
        authority: { accuracy: 87.5, stale_adopted: 12.5 },
        unresolvable: { accuracy: 55.0, flagged_resolved: 55.0 }
      }
    },

    "gpt-4o-mini": {
      name: "GPT-4o-mini (Small Baseline)",
      slug: "gpt-4o-mini",
      pass_answer: 74.2,
      pass_answer_ci: [71.6, 76.7],
      pass_strict: 74.2,
      pass_strict_ci: [71.6, 76.7],
      hallucination_rate: 0.0,
      hallucination_ci: [0.0, 0.3],
      false_abstain_rate: 0.0,
      false_abstain_ci: [0.0, 0.4],
      citation_precision: 77.9,
      citation_recall: 81.8,
      exact_citation_rate: 81.8,
      exact_citation_ci: [79.4, 83.9],
      abstain_recall: 93.4,
      lim_score_k10: 20.8,
      lim_ci_k10: [8.3, 34.2],
      lim_score_k30: 18.3,
      lim_ci_k30: [4.2, 32.5],
      distractor_curve: [
        { k: 0, accuracy: 100.0, ci: [94.0, 100.0], distractor_adopted: 0.0 },
        { k: 5, accuracy: 96.7, ci: [88.6, 99.1], distractor_adopted: 3.3 },
        { k: 10, accuracy: 76.7, ci: [64.6, 85.6], distractor_adopted: 23.3 },
        { k: 25, accuracy: 60.0, ci: [47.4, 71.4], distractor_adopted: 40.0 },
        { k: 50, accuracy: 38.3, ci: [27.1, 51.0], distractor_adopted: 61.7 }
      ],
      position_curve_k10: { top: 93.3, middle: 68.3, bottom: 85.0 },
      position_curve_k30: { top: 75.0, middle: 56.7, bottom: 75.0 },
      conflicts: {
        dated: { accuracy: 67.5, stale_adopted: 32.5 },
        authority: { accuracy: 72.5, stale_adopted: 27.5 },
        unresolvable: { accuracy: 55.0, flagged_resolved: 55.0 }
      }
    }
  },

  comparison: {
    mean_delta: -13.6,
    bootstrap_ci: [-16.1, -11.1],
    mcnemar_b: 37, // mini passes, 4o fails
    mcnemar_c: 189, // 4o passes, mini fails
    mcnemar_p: 0.0000,
    flipped_total: 226,
    common_passes: 794,
    common_fails: 100
  },

  interventions: [
    {
      name: "baseline",
      description: "Standard top-k context ordering without transforms or prompt directives.",
      delta_acc: "0.0% (Ref)",
      delta_hallucination: "0.0%",
      delta_false_abstain: "0.0%",
      p_value: "1.000"
    },
    {
      name: "reorder_edges",
      description: "Places highest-scoring passages at exterior edges (start/end) to combat Lost-in-the-Middle.",
      delta_acc: "+2.1%",
      delta_hallucination: "-0.8%",
      delta_false_abstain: "+0.3%",
      p_value: "0.024"
    },
    {
      name: "check_prompt",
      description: "Appends explicit verification directive instructing model to verify match against source passages.",
      delta_acc: "+1.4%",
      delta_hallucination: "-3.2%",
      delta_false_abstain: "+4.1%",
      p_value: "0.008"
    },
    {
      name: "score_threshold",
      description: "Deterministic retrieval guardrail: forces abstention if maximum retrieval score is below 0.5.",
      delta_acc: "-0.7%",
      delta_hallucination: "-4.5%",
      delta_false_abstain: "+5.6%",
      p_value: "0.001"
    }
  ],

  sample_traces: [
    {
      item_id: "conflict-0002-authority",
      category: "Authority Conflict",
      question: "Under the provisions of Doukgivdom Relocation Policy, what is the approval threshold?",
      gold_value: "6500.00",
      value_type: "money",
      condition: {
        mode: "A",
        conflict: "authority",
        k_noise: 10,
        position: "random"
      },
      conflict_details: {
        gold_passage: {
          doc_id: "doc-cclvmz",
          title: "Doukgivdom Relocation Policy Documentation - Part 1",
          status: "official",
          effective_date: "2023-05-25",
          text: "Under the governance rules of Doukgivdom Relocation Policy, the official approval threshold is $6500.00. Routine administrative logs are maintained by system custodians to verify operational accountability."
        },
        twin_passage: {
          doc_id: "doc-tw98x1",
          title: "Doukgivdom Relocation Policy Revisions Draft",
          status: "draft",
          effective_date: "2024-02-15", // Newer date, but draft status!
          text: "In accordance with recent revisions, the preliminary approval threshold for Doukgivdom Relocation Policy has been updated to $8200.00 for planning purposes."
        }
      },
      distractors_count: 10,
      model_results: {
        "gpt-4o": {
          answer: "6500.00",
          citations: ["doc-cclvmz"],
          abstain: false,
          pass_answer: true,
          pass_strict: true,
          provenance: "gold",
          failure_mode: null,
          note: "Correctly prioritized official status over the newer 2024 draft date (Rule R4)."
        },
        "gpt-4o-mini": {
          answer: "8200.00",
          citations: ["doc-tw98x1"],
          abstain: false,
          pass_answer: false,
          pass_strict: false,
          provenance: "conflict_loser",
          failure_mode: "stale_adopted",
          note: "Fell into the temporal recency trap: picked the 2024 date despite 'draft' status."
        }
      }
    },

    {
      item_id: "dist-sweep-0000-k25",
      category: "Distractor Load (k=25)",
      question: "What is the maximum bandwidth for Bupvou Industrial Network?",
      gold_value: "40 Gbps",
      value_type: "enum",
      condition: {
        mode: "A",
        conflict: "none",
        k_noise: 25,
        position: "random"
      },
      gold_passage: {
        doc_id: "doc-k491xm",
        title: "Bupvou Industrial Network Technical Specifications",
        status: "official",
        effective_date: "2022-11-04",
        text: "The infrastructure for Bupvou Industrial Network supports a maximum bandwidth of 40 Gbps. Dedicated telemetry monitors all data routing gateways."
      },
      distractors_count: 25,
      model_results: {
        "gpt-4o": {
          answer: "40 Gbps",
          citations: ["doc-k491xm"],
          abstain: false,
          pass_answer: true,
          pass_strict: true,
          provenance: "gold",
          failure_mode: null,
          note: "Successfully ignored 25 distractor passages (including near-miss entity 'Bupvou Commercial Network')."
        },
        "gpt-4o-mini": {
          answer: "10 Gbps",
          citations: ["doc-ds83a1"],
          abstain: false,
          pass_answer: false,
          pass_strict: false,
          provenance: "distractor",
          failure_mode: "distractor_adopted",
          note: "Adopted distractor value from near-miss entity passage."
        }
      }
    },

    {
      item_id: "conflict-0000-unresolvable",
      category: "Unresolvable Conflict",
      question: "What is the retention period of the record archive for Kestu Manufacturing?",
      gold_value: "Flagged Conflict",
      value_type: "duration",
      condition: {
        mode: "A",
        conflict: "unresolvable",
        k_noise: 10,
        position: "random"
      },
      conflict_details: {
        passage_a: {
          doc_id: "doc-u912aa",
          title: "Kestu Archive Guidelines Vol 1",
          status: "official",
          effective_date: "2023-08-01",
          text: "Records archived for Kestu Manufacturing must be retained for exactly 36 months under corporate regulatory compliance directives."
        },
        passage_b: {
          doc_id: "doc-u912bb",
          title: "Kestu Archive Guidelines Vol 2",
          status: "official",
          effective_date: "2023-08-01", // Same date, same official status!
          text: "Records archived for Kestu Manufacturing must be retained for exactly 60 months under operations guidelines."
        }
      },
      distractors_count: 10,
      model_results: {
        "gpt-4o": {
          answer: null,
          conflict: true,
          conflicting_values: ["36 months", "60 months"],
          citations: ["doc-u912aa", "doc-u912bb"],
          pass_answer: true,
          pass_strict: true,
          failure_mode: null,
          note: "Rule R5 enforced: both are official with identical dates. Correctly flagged conflict without guessing."
        },
        "gpt-4o-mini": {
          answer: "36 months",
          conflict: false,
          citations: ["doc-u912aa"],
          pass_answer: false,
          pass_strict: false,
          failure_mode: "conflict_ignored",
          note: "Ignored the conflicting official source and asserted an arbitrary single answer."
        }
      }
    },

    {
      item_id: "multihop-0000-a-k0",
      category: "Multi-Hop 2-Document Reasoning",
      question: "What is the approval threshold of the policy governing Butmim Facility?",
      gold_value: "15500.00",
      value_type: "money",
      condition: {
        mode: "A",
        conflict: "none",
        k_noise: 0,
        position: "random"
      },
      multihop_details: {
        hop1_bridge: {
          doc_id: "doc-0120",
          title: "Butmim Facility Operations Profile",
          status: "official",
          effective_date: "2022-04-10",
          text: "Butmim Facility handles regional manufacturing. Regulatory framework: Butmim Facility is governed by Krukfourk Security Policy."
        },
        hop2_target: {
          doc_id: "doc-0020",
          title: "Krukfourk Security Policy Documentation",
          status: "official",
          effective_date: "2022-01-20",
          text: "Under the governance rules of Krukfourk Security Policy, the official approval threshold is $15500.00."
        }
      },
      model_results: {
        "oracle": {
          answer: "15500.00",
          citations: ["doc-0120", "doc-0020"],
          citation_recall: 1.0,
          pass_answer: true,
          pass_strict: true,
          failure_mode: null,
          note: "Both Hop 1 (bridge) and Hop 2 (target) cited. Complete lineage verified."
        },
        "partial_citation": {
          answer: "15500.00",
          citations: ["doc-0020"],
          citation_recall: 0.5,
          pass_answer: true,
          pass_strict: false,
          failure_mode: null,
          note: "Correct value but missing Hop 1 bridge citation. Fails strict verification (citation_recall = 0.5)."
        }
      }
    }
  ]
};

