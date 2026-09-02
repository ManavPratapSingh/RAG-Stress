# Project Summary: RAG-Stress

RAG-Stress is a reproducible benchmark designed to stress-test RAG pipelines under controlled adversarial noise (distractors, contradictions, position bias, missing evidence) using deterministic ground truth without an LLM-as-judge.

### Current Status
- **Phase 1 Complete:** Synthetic knowledge base generator, fact modeling, passage rendering, question generation, and value normalization implemented and verified.
- **Acceptance:** `gen-kb --seed 42` produces bit-exact identical files across independent runs.
- **Test Suite:** 52 unit tests passing, covering determinism, pronounceability, value ranges, passage lengths, template paraphrases, and 1,000-fact render-normalize roundtrips.
- **Lint & Types:** Ruff checks and format checks pass; Mypy reports 0 issues across all source files.
