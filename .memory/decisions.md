# Decision Log

### D-001 Entity and Passage Partitioning
- **Context:** SPEC §4.3 mandates a Base KB of 300 passages, each stating 1–3 facts about one entity, with every `(entity, attribute)` appearing in exactly one base passage (`status = "official"`).
- **Decision:** Generate 150 entities (50 Policy, 50 Product, 50 Facility), each having 4 typed attributes (600 facts total). Split each entity's 4 facts evenly into 2 passages (2 facts each), producing exactly 300 passages.
- **Alternatives Considered:** Unequal partitions (e.g. 3 facts + 1 fact, or variable facts across entities).
- **Trade-offs:** Even 2-fact distribution ensures balanced passage length and uniform distractor potential across the corpus.
- **Status:** Active.

### D-002 Typer CLI Multi-Command Structure
- **Context:** Typer defaults to single-command mode when only one command is defined unless explicitly configured, which broke `ragstress gen-kb`.
- **Decision:** Add `@app.callback()` to ensure `ragstress` acts as a multi-command CLI ready for future subcommands (`gen-suite`, `run`, etc.), while providing a standalone entrypoint `gen_kb_standalone` for `gen-kb`.
- **Alternatives Considered:** Separate CLI scripts or click.
- **Trade-offs:** Minimal boilerplate in `cli.py` while ensuring both `ragstress gen-kb` and standalone `gen-kb` work identically.
- **Status:** Active.

### D-003 Early Implementation of Normalizer
- **Context:** SPEC §4.4 states: "Test: render -> normalize -> equals fact value, for every fact in a 1,000-fact sample." Normalizer is formally specified in SPEC §8.1.
- **Decision:** Implement `graders/normalize.py` during Phase 1 to satisfy and verify the render-normalize contract immediately.
- **Alternatives Considered:** Mock normalizer in tests.
- **Trade-offs:** Slightly more code upfront in Phase 1, but guarantees that Phase 1 outputs strictly conform to SPEC §8.1 before downstream phases rely on them.
- **Status:** Active.

