# AGENTS.md

Instructions for AI Coding Agents working on RAG-Stress:

1. **Read PLAN.md and SPEC.md**: Always align with specifications in `DOCS/PLAN.md` and `DOCS/SPEC.md`. If PLAN and SPEC disagree, SPEC wins.
2. **Work one phase at a time**: Do not jump ahead to future phases. Focus strictly on the requirements for the active phase.
3. **Write tests first**: Adhere to test-driven development. Verify determinism, boundary conditions, and acceptance criteria before declaring a phase complete.
4. **Use the project-explainer skill after each phase**: Maintain `docs/explainer/` with living documentation, decision logs, interview questions, and real traces from verified runs.
5. **Maintain persistent project memory**: Keep `.memory/` updated after every phase with architectural insights, module schemas, patterns, and decisions.
6. **Run `make test` and `make lint` before stopping**: All tests must pass offline without network access. Coverage on core modules must meet SPEC NFR-6 (≥85%).
7. **Don't build outside the current phase**: Keep solutions minimal, robust, and cleanly scoped.

