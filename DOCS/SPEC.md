# SPEC.md — RAG-Stress

**RAG Hallucination & Retrieval Stress-Test Harness**
Spec version 1.0 · Companion to `PLAN.md` (PLAN = how/when; SPEC = what must be true). If they disagree, **SPEC wins**.

**MUST / MUST NOT / SHOULD / MAY** are used as in RFC 2119. Requirements carry IDs (`FR-KB-2`, `NFR-1`, …) so tests, commits and explainer chapters can cite them.

---

## 1. Purpose and scope

### 1.1 Purpose
Measure how a RAG pipeline (or just its generator) behaves under **controlled** adversarial conditions (distractors, contradictions, position shifts, missing evidence) using **deterministic ground truth**, and attribute each failure to retrieval or generation.

### 1.2 In scope
Synthetic knowledge-base generation; question and suite generation; retrieval stack (chunking, BM25, dense, hybrid, vector stores, metrics); stress engine; generator interface with structured output and record/replay; deterministic graders and failure taxonomy; interventions; statistics and reports; CLI, Docker, CI.

### 1.3 Non-goals
- No LLM-as-judge in the core path (an optional, calibrated judge stretch is allowed but MUST NOT affect core metrics).
- No real-world corpora, no web crawling, no fine-tuning, no UI, no hosted services.
- Not a general RAG leaderboard; a synthetic, controlled instrument.

### 1.4 Definitions
| Term | Meaning |
|---|---|
| **Fact** | Structured record `(entity, attribute, value, effective_date, status)`. |
| **Passage** | A rendered text document stating 1–3 facts about one entity, plus metadata. |
| **Gold passage** | The passage that states the fact a question asks about. |
| **Twin** | A generated passage stating a *different* value for the same `(entity, attribute)` as the gold passage. |
| **Distractor** | A passage that does not state the asked `(entity, attribute)` but resembles it. |
| **Item** | `(question, StressCondition, seed)`: one graded test case. |
| **Context** | Ordered list of passages shown to the generator. |
| **Mode A** | *Generator-only*: the stress engine builds the context directly. |
| **Mode B** | *End-to-end*: a retriever fetches the context from an index containing noise. |
| **Oracle item** | Gold passage only, no noise. **Closed-book item**: unanswerable with `k_noise = 0` (empty context). |

---

## 2. Global requirements

### 2.1 Functional summary
| ID | Requirement |
|---|---|
| FR-1 | The KB, questions, suites, and contexts MUST be fully determined by seeds. |
| FR-2 | Ground truth MUST come from structured fact records, never from parsing model output or an LLM. |
| FR-3 | Every item MUST carry role labels for every passage in its context. |
| FR-4 | Failures MUST be attributed to exactly one taxonomy mode. |
| FR-5 | Mode B MUST separate retrieval misses from generation failures. |
| FR-6 | Every suite MUST include oracle and closed-book controls. |
| FR-7 | A full suite MUST be re-runnable offline from cassettes with byte-identical output. |
| FR-8 | Reports MUST include confidence intervals, per-factor curves, and paired comparisons when comparing runs. |

### 2.2 Non-functional
| ID | Requirement |
|---|---|
| NFR-1 | **Determinism:** two `bench-replay` runs MUST produce `results.jsonl` with identical SHA-256. |
| NFR-2 | **Speed:** replay of the `core` suite (≈1,100 items) SHOULD finish in < 90 s with the hash embedder and flat store. |
| NFR-3 | **Isolation:** no shared mutable state between items; concurrency MUST NOT change results. |
| NFR-4 | **Offline tests:** unit tests and replay MUST work with networking disabled (enforced by test). |
| NFR-5 | **Secrets:** API keys from environment only; never in traces, cassettes, logs, reports. |
| NFR-6 | **Quality:** ruff, type checker, pytest pass in CI; coverage ≥ 85% on `kb/`, `stress/`, `retrieval/`, `graders/`, `report/stats.py`. |
| NFR-7 | **Portability:** Python ≥ 3.11; Linux/macOS; Docker image runs non-root. |
| NFR-8 | **Dependencies:** core = `pydantic>=2`, `numpy`, `pyyaml`, `typer`, `scipy`; extras = `sentence-transformers`, `qdrant-client`, provider SDKs, `matplotlib`; dev = `pytest`, `ruff`, `mypy`/`pyright`, `pytest-socket`. |

---

## 3. Repository layout (normative)
See `PLAN.md` §1. Module names there are binding.

---

## 4. Knowledge base and questions (`kb/`)

### 4.1 Names (FR-KB-1)
- Entity names MUST be built from seeded pseudo-word generation (syllable lists) plus a fixed type noun, e.g. `Kelvora Travel Policy`, `Dravix-340`, `Mirenta Data Center`.
- Names MUST be unique within a KB; MUST NOT be drawn from real company/brand lists; MUST pass a pronounceability filter (no 3+ consecutive consonants beyond a small whitelist).

### 4.2 Entity kinds and attributes (FR-KB-2)
| Kind | Attribute | Value type | Range / domain | Unit/label |
|---|---|---|---|---|
| Policy | `notice_period` | duration | 7–90 days | "notice period" |
| Policy | `stipend_amount` | money | 50–2,000 USD | "monthly stipend" |
| Policy | `approval_threshold` | money | 500–20,000 USD | "approval threshold" |
| Policy | `review_cycle` | duration | 3–24 months | "review cycle" |
| Product | `max_throughput` | int | 100–50,000 | "maximum throughput (requests/s)" |
| Product | `warranty_period` | duration | 6–60 months | "warranty period" |
| Product | `list_price` | money | 20–9,000 USD | "list price" |
| Product | `max_operating_temp` | int | 30–90 °C | "maximum operating temperature" |
| Facility | `capacity` | int | 20–2,000 | "capacity (people)" |
| Facility | `opening_date` | date | 2015-01-01…2025-05-31 | "opening date" |
| Facility | `floor_count` | int | 1–40 | "number of floors" |
| Facility | `security_level` | enum | low, medium, high, restricted | "security level" |

### 4.3 Fact and passage models (FR-KB-3)
```python
class Fact(BaseModel):
    fact_id: str
    entity_id: str
    entity_name: str
    attribute: str
    value: str  # canonical string form (see §8.1)
    value_type: Literal["int", "money", "date", "duration", "enum"]
    effective_date: date  # 2022-01-01 … 2025-05-31
    status: Literal["official", "draft"]


class Passage(BaseModel):
    doc_id: str  # internal id; replaced by opaque id in contexts (§6.5)
    title: str
    text: str
    status: Literal["official", "draft"]
    effective_date: date
    facts: list[Fact]  # ground truth of what the text states (1–3)
    role: str | None = None  # set only inside a built context, see §6.6
```
- Base KB: 300 passages, each stating 1–3 facts about **one** entity; every `(entity, attribute)` appears in **exactly one** base passage, `status = official`.
- Values drawn from seeded RNG within §4.2 ranges; for any two entities of the same kind and attribute, values SHOULD differ (collisions allowed only if unavoidable; the stress engine enforces its own filter, §6.2).

### 4.4 Rendering (FR-KB-4)
- ≥ 4 paraphrase templates per attribute; template choice from seeded RNG.
- Body length 60–180 words, padded with sentences from a fixed neutral filler list.
- **One renderer for all passages** (gold, twins, distractors). No role-specific text, no role-specific formatting.
- The value MUST appear in the text in a form accepted by the normalizer for its type (§8.1). Test: render → normalize → equals fact value, for every fact in a 1,000-fact sample.

### 4.5 Questions (FR-KB-5)
```python
class Question(BaseModel):
    question_id: str
    entity_id: str
    attribute: str
    text: str
    gold_value: str
    gold_fact_id: str
```
≥ 3 paraphrases per attribute (e.g., "What is the {label} for {entity_name}?"). Question text MUST name the exact entity and attribute label.

### 4.6 Determinism tests (FR-KB-6)
`gen-kb --seed S` twice → byte-identical output; different seeds → different output; all invariants in §4.3–§4.5 hold (property tests over several seeds).

---

## 5. Retrieval stack (`retrieval/`)

### 5.1 Chunker (FR-RET-1)
`chunk(passage, max_words) -> list[Chunk]` splitting on sentence boundaries, never mid-sentence; `Chunk{chunk_id, doc_id, text, sentence_span}`. Default `max_words = 200` (so most passages are a single chunk). The *relevant chunk* for a fact is the chunk containing the sentence that states it.

### 5.2 BM25 (FR-RET-2)
- Own implementation. Tokenizer: lowercase, `\w+`. Parameters `k1 = 1.5`, `b = 0.75`. IDF: `ln(1 + (N − n + 0.5) / (n + 0.5))`.
- MUST include a unit test against a hand-computed 3-document example.
- Ties broken by ascending `chunk_id`.

### 5.3 Embedders (FR-RET-3)
```python
class Embedder(Protocol):
    name: str; dim: int
    def embed(self, texts: list[str]) -> np.ndarray  # float32, L2-normalized, shape (n, dim)
```
- `HashEmbedder` (default in tests/CI): feature hashing of word unigrams, word bigrams and character 3-grams into `dim = 512` buckets with signed hashing (stable hash, e.g., blake2b), then L2-normalize. Fully deterministic, no downloads.
- `SentenceTransformerEmbedder` (extra): pinned model name **and revision**, CPU, float32. Embeddings cached on disk keyed by `sha256(model|revision|text)`.
- Indexed text = `title + "\n" + body` (metadata header is **not** embedded).

### 5.4 Vector stores (FR-RET-4)
```python
class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray) -> None
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]  # score desc, tie → id asc
```
- `FlatStore`: exact cosine via matrix product (default).
- `QdrantStore` (extra): `qdrant-client` in-memory mode, cosine distance.
- Parity test: on the `core` E2E queries, `QdrantStore` recall@10 against `FlatStore` top-10 MUST be ≥ 0.95; the measured value is reported.

### 5.5 Retrievers (FR-RET-5)
`bm25`, `dense`, `hybrid` (RRF over `bm25` and `dense` with `k_rrf = 60`; ties by id). `retrieve(query, K) -> list[RetrievedChunk{doc_id, chunk_id, score, rank}]`.

### 5.6 Retrieval metrics (FR-RET-6)
Per query with relevant set `R` (non-empty) and ranking `r_1..r_K`:
- `Hit@K = 1[R ∩ top-K ≠ ∅]`
- `Recall@K = |R ∩ top-K| / |R|`
- `Precision@K = |R ∩ top-K| / K`
- `MRR = 1 / rank of the first relevant` (0 if none in top-K)
- `nDCG@K` with binary relevance and `log2(rank+1)` discount
- `distractor_rate@K = |injected-noise ∩ top-K| / K`
Queries with `R = ∅` (unanswerable) are excluded from these metrics and reported separately by `distractor_rate@K`. Unit-tested on hand-built rankings.

---

## 6. Stress engine (`stress/`)

### 6.1 Condition model (FR-STRESS-1)
```python
class StressCondition(BaseModel):
    k_noise: int = 0  # number of distractors
    position: Literal["top", "middle", "bottom", "random"] = "random"
    conflict: Literal["none", "dated", "authority", "unresolvable"] = "none"
    answerable: bool = True
    distractor_mix: dict[str, float] = {  # must sum to 1.0
        "same_attr_other_entity": 0.4,
        "same_entity_other_attr": 0.3,
        "near_miss_entity": 0.2,
        "random_irrelevant": 0.1,
    }
    mode: Literal["A", "B"] = "A"
    k_retrieve: int | None = None  # Mode B only
```
Item kind (derived): `answerable_plain` (conflict none, answerable) · `dated` · `authority` · `unresolvable` · `unanswerable` (answerable=false). `answerable=false` with `conflict != none` is invalid.

### 6.2 Distractors (FR-STRESS-2)
| Type | Construction |
|---|---|
| `same_attr_other_entity` | Same attribute as the question, a different entity of the same kind. |
| `same_entity_other_attr` | Same entity, a different attribute. |
| `near_miss_entity` | Same attribute; entity name differs from the target by edit distance 1–2; different value. |
| `random_irrelevant` | Random base passage with no entity/attribute overlap with the question. |
- **Value-collision filter:** any distractor fact whose `value_type` equals the gold value's type MUST have a value ≠ gold value (and ≠ twin value). Enforced and tested.
- Distractors are rendered by the same renderer (§4.4) and MUST NOT be the gold/twin passages.
- Mix counts: `round(k_noise × mix)` per type with largest-remainder rounding; sampling without replacement from a seeded RNG.
- **Hardness metric:** for each distractor, `hardness = score(distractor, question) / score(gold, question)` under BM25 and under the dense retriever. The suite validator MUST report medians per type, and the median for `same_attr_other_entity`, `same_entity_other_attr`, `near_miss_entity` SHOULD be ≥ 0.6 under BM25 (if not, templates are too dissimilar: fix the generator).

### 6.3 Conflict twins (FR-STRESS-3)
Twin = rendered passage for the same `(entity, attribute)` with `value' ≠ value` (same value type, within range).
| `conflict` | Gold doc | Twin doc | Expected behavior |
|---|---|---|---|
| `dated` | official, effective `E_g` | official, effective `E_t` where `E_g − E_t ≥ 30 days` (gold is newer) | Answer = gold value |
| `authority` | official, effective `E_g` | **draft**, effective `E_t > E_g` by ≥ 30 days | Answer = gold (official) value, even though the draft is newer |
| `unresolvable` | official, `E` | official, same `E` | Flag `conflict=true` and list both values; no single answer |
For `dated`/`authority`, resolution follows task rules R3–R4 (§7.2). For `unresolvable` there is no gold single value: `gold_values = {value, value'}`.

### 6.4 Position (FR-STRESS-4)
Let `n` = number of passages in the context. Gold index `p`: `top → 0`, `middle → (n−1)//2`, `bottom → n−1`, `random → seeded uniform`. All other passages (distractors, twin) are shuffled by the seeded RNG into the remaining slots. For `unresolvable`, "gold" is the passage with the lexicographically smaller internal `doc_id`.

### 6.5 Opaque IDs (FR-STRESS-5)
After ordering, each passage receives a context `doc_id` of the form `doc-` + 6 base32 characters derived from `sha256(f"{seed}:{item_id}:{internal_id}")`. IDs MUST be independent of role and position. Test: over 1,000 items, a classifier using only the ID string (e.g., sorted rank, hash prefix) predicts role no better than chance (χ² test, α = 0.01), and ID sort order shows no correlation with gold position.

### 6.6 Labels (FR-STRESS-6)
Each built context stores, per passage: `{doc_id, internal_id, role, position, status, effective_date, facts}` where `role ∈ {gold, twin, distractor:<type>}`. Also stored: `gold_support_ids`, `relevant_ids` (§9.3), `gold_values`.

### 6.7 Answerability (FR-STRESS-7)
When `answerable = false`, the gold (and any twin) passage is excluded from the context, and **at least 2** `near_miss_entity` or `same_attr_other_entity` distractors are included when `k_noise ≥ 5`. `k_noise = 0` yields the empty-context closed-book item.

### 6.8 Builder tests (FR-STRESS-8)
Same `(item, seed)` → byte-identical context; word-count means of gold vs distractors within 10%; no distractor contains the gold value; labels consistent with content; hardness report generated.

---

## 7. Generator under test (`gen/`)

### 7.1 Interface (FR-GEN-1)
```python
class GenOutput(BaseModel):
    answer: str | None; abstain: bool; conflict: bool
    conflicting_values: list[str]; citations: list[str]
class GenResult(BaseModel):
    output: GenOutput | None      # None on format error
    raw: str; format_error: str | None
    usage: Usage; latency_ms: float
class Generator(Protocol):
    name: str
    def generate(self, system: str, user: str, schema: dict) -> GenResult
```
Structured output MUST be obtained via a **forced tool call** `submit_answer` (or the provider's equivalent). Schema violations (e.g., `abstain=true` with a non-null `answer`) are `format_error`.

### 7.2 Prompts (FR-GEN-2)
`SYSTEM_PROMPT` (constant; `PROMPT_VERSION = "1"`):

> You answer questions using ONLY the provided passages. Each passage has a header with its doc_id, status (official or draft) and effective date. Rules:
> R1. Use only information in the passages; do not use outside knowledge.
> R2. If no passage states the answer for the exact entity and attribute asked, abstain.
> R3. If passages disagree, prefer an official passage over a draft.
> R4. Among official passages that disagree, prefer the one with the latest effective date.
> R5. If passages disagree and R3–R4 cannot decide (same status and same effective date), do not choose: set conflict=true and list all conflicting values.
> R6. Cite the doc_id of every passage that supports your answer.
> Respond only by calling submit_answer.

User message:
```
Passages:

[doc_id=doc-k3f9ab | status=official | effective=2024-03-01]
<title>
<text>

[doc_id=...]
...

Question: <question text>
```
R1–R6 are the **task definition** and are always present. Interventions (§12) MUST NOT remove or weaken them.

### 7.3 Provided generators (FR-GEN-3)
| Generator | Behavior |
|---|---|
| `OpenAIGenerator`, `AnthropicGenerator` | Real models; `temperature = 0`; `max_tokens = 400`. At least one is required. |
| `ScriptedGenerator(policy)` | Deterministic test doubles: `oracle` (reads labels, always correct), `abstain_always`, `first_passage`, `hallucinate` (answers with a value present in no passage), `ignore_conflict` (picks the twin's value), `pick_distractor`, `fabricate_citation`. |
| `RecordingGenerator` / `ReplayGenerator` | Cassettes per §7.4. Replay never touches the network. |

### 7.4 Cassettes (FR-GEN-4)
Key = `sha256(canonical_json({model, prompt_version, system, user, schema, params}))`. File `cassettes/<model_slug>/<key>.json` stores the `GenResult` plus `{recorded_with:{model,date}}`. Miss → `CassetteMiss` (no network fallthrough unless `--record`). Canonical JSON = sorted keys, no whitespace, UTF-8. In replay, `latency_ms` comes from the cassette.

---

## 8. Graders (`graders/`)
All graders are pure functions of `(item, context labels, GenResult)`; each is table-driven unit-tested.

### 8.1 Normalization (FR-GRADE-1)
| Type | Accepted (examples) | Rejected (examples) | Canonical |
|---|---|---|---|
| int | `1,200`, `1200`, `1200 requests/s` | `1.2k`, `about 1200` | `"1200"` |
| money | `$1,500`, `1500 USD`, `1,500.00` | `1.5k`, `~$1,500` | `"1500.00"` |
| date | `2024-03-01`, `March 1, 2024`, `1 Mar 2024` | `03/01/2024` (ambiguous day/month) | `"2024-03-01"` |
| duration | `30 days`, `1 month`, `6 months` | `half a year`, `0.5 years` | `"<n> days"` or `"<n> months"` (no cross-unit conversion; the unit must match) |
| enum | case-insensitive exact match | synonyms | lowercase |

Unparseable answers normalize to `None` and count as wrong (not as format errors), with `normalize_failed = true` recorded. Rendered passages MUST state values only in accepted forms, so a correct answer copied from a passage always normalizes.

### 8.2 Value provenance (FR-GRADE-2)
For an asserted answer `v` (not abstain, not None) classify, **first match wins**:
1. `gold` (equals gold value; for `unresolvable`: equals either of `gold_values`)
2. `conflict_loser` (equals the twin's value, when a twin exists and is not gold)
3. `distractor` (equals the value of any fact in any distractor passage in the context)
4. `none` (value appears in no passage in the context)

`hallucinated = (provenance == none)`.
Notes: grounding is computed from the **structured `facts`** of the passages in the context, not by text search. The value-collision filter (§6.2) prevents accidental matches.

### 8.3 Correctness by item kind (FR-GRADE-3)
| Kind | `pass_answer` | `pass_strict` (adds) |
|---|---|---|
| `answerable_plain` | normalized answer == gold, `abstain = false` | citations exactly the gold support set, no fabricated IDs, `conflict = false` |
| `dated` | answer == gold (newer) | citations include the gold doc, no fabricated IDs; `conflict` flag may be either |
| `authority` | answer == gold (official) | as `dated` |
| `unresolvable` | `conflict = true` AND normalized `conflicting_values` set == `gold_values` | `answer is None`, citations ⊇ both docs, no fabricated IDs |
| `unanswerable` | `abstain = true` | `answer is None`, `citations == []`, `conflict = false` |
`format_error` → both pass levels false.

### 8.4 Citations (FR-GRADE-4)
- `fabricated_citations` = cited IDs not in the context.
- Citation precision = |cited ∩ support| / |cited| (undefined if cited is empty → recorded as null); recall = |cited ∩ support| / |support|; `exact` = set equality. `support` = `gold_support_ids` (gold doc; both docs for `unresolvable`; empty for `unanswerable`).

### 8.5 Conflict handling (FR-GRADE-5)
Recorded for every item: `conflict_flagged`, `values_complete` (listed values == `gold_values`, `unresolvable` only), `false_conflict_flag` (flag on a kind without a genuine unresolvable conflict). For `dated`/`authority`, also `flagged_and_resolved` (flag true and answer correct).

### 8.6 Metrics (FR-GRADE-6)
Over any slice of items:
- `accuracy` = mean `pass_answer`; `strict_accuracy` = mean `pass_strict`.
- `hallucination_rate` = hallucinated items / all items; `hallucination_rate_answered` = hallucinated items / items where the model asserted an answer.
- `distractor_adoption_rate`, `conflict_loser_adoption_rate` (= stale/draft adoption), `false_abstain_rate` (answerable items abstained), `format_error_rate`.
- `abstain_recall` = abstained / unanswerable items; `abstain_precision` = correct abstains / all abstains.
- `leakage_rate` = unanswerable items whose answer == gold value (answered correctly with no evidence in context). Closed-book slice is the headline leakage control; `leakage_rate > 0.05` raises a `LEAKAGE_WARNING` in the report.
- `unneeded_conflict_flag_rate` = `false_conflict_flag` / items eligible.
- Citation metrics: mean precision/recall, `exact_rate`, `fabricated_citation_rate`.

### 8.7 Failure taxonomy (FR-GRADE-7)
Assigned to every item with `pass_answer = false`; **first match wins**:
| # | Mode | Rule |
|---|---|---|
| 1 | `format_error` | `format_error` set |
| 2 | `retrieval_miss` | Mode B, answerable kind, and `gold_support_ids ⊄ retrieved` (for `unresolvable`: either support doc missing) |
| 3 | `stale_adopted` | provenance == `conflict_loser` |
| 4 | `distractor_adopted` | provenance == `distractor` |
| 5 | `hallucinated` | provenance == `none` |
| 6 | `false_abstain` | answerable kind and `abstain = true` |
| 7 | `conflict_ignored` | `unresolvable`, `conflict = false`, answer asserted |
| 8 | `conflict_incomplete` | `unresolvable`, `conflict = true`, `values_complete = false` |
| 9 | `other_wrong` | default (including `normalize_failed`) |
Note: an *unanswerable* item answered with a distractor value is `distractor_adopted`; with an unsupported value, `hallucinated`.

---

## 9. Suites and Mode B (`suites/`)

### 9.1 Item and suite format (FR-SUITE-1)
`data/suites/<name>.jsonl`, one item per line:
```json
{"item_id":"dist-sweep-0042-k25","split":"test","group":"distractor_sweep",
 "question_id":"q-0173","seed":918273,
 "condition":{"k_noise":25,"position":"random","conflict":"none","answerable":true,
              "distractor_mix":{...},"mode":"A","k_retrieve":null}}
```
Items have a `split ∈ {dev, test}`; prompt/intervention tuning is allowed on `dev` only; reported numbers use `test`.

### 9.2 Suite definitions (FR-SUITE-2)
Questions per cell come from a seeded stratified sample over entity kinds and attributes; the **same question set** is reused across levels of a factor (so paired comparisons are valid).

| Group | Design | Items (core) |
|---|---|---|
| `controls` | oracle (k=0, answerable) · closed-book (k=0, unanswerable) | 60 × 2 = 120 |
| `distractor_sweep` | 60 q × `k_noise ∈ {0,5,10,25,50}`, position random | 300 |
| `position_sweep` | 60 q × `position ∈ {top,middle,bottom}` × `k_noise ∈ {10,30}` | 360 |
| `conflict` | 40 q × `{dated, authority, unresolvable}`, `k_noise = 10`, position random | 120 |
| `unanswerable` | 40 q × `k_noise ∈ {5,10,25}` | 120 |
| `e2e` | 100 questions: 50 plain, 30 conflict (10 each kind), 20 unanswerable; retrievers × `K ∈ {3,5,10}` run as separate configs | 100 per config |
Total `core` Mode-A items ≈ 1,020 + 100 per E2E config. `smoke` = 6–8 questions per cell (~60 items). Each item's `split` = `dev` for 25% of questions (by question_id hash), else `test`.
`position_sweep` includes `k_noise = 10` and `30` so position effects can be seen in short and long contexts.

### 9.3 Mode B (FR-SUITE-3)
- Index = base passages (chunked) + a **noise pool** (default 1,500 distractor passages generated per §6.2 using the E2E questions as targets; unlabeled to the retriever) + **twins** for conflict-type E2E questions. For `unanswerable` E2E items, the gold passage (and twin) are **excluded from the index for that run** (use a separate index variant `index_no_gold`, built once and shared by all unanswerable items).
- `relevant_ids` = `gold_support_ids ∪ {twin}` for conflict kinds; `{gold}` for plain; ∅ for unanswerable.
- The retrieved top-K, in rank order (best first), becomes the context (after any intervention). Doc IDs are opaque as in §6.5.
- Retrieval metrics (§5.6) are computed per item; `retrieval_miss` per §8.7.

### 9.4 Validation (FR-SUITE-4) — `validate-suite` MUST enforce
1. Unique `item_id`; known `question_id`.
2. Conditions valid per §6.1.
3. Every Mode-A item builds a context passing §6.8 checks.
4. Oracle generator scores 100% `pass_strict` on every solvable item and every unanswerable item whose context is built (it reads labels).
5. `abstain_always` fails every solvable item; `hallucinate` yields `hallucinated` on every item it answers; `ignore_conflict` fails every `unresolvable` item.
6. Closed-book items have empty contexts.
7. Hardness report (§6.2) and ID-leakage test (§6.5) pass.

---

## 10. Runner (`runner/`)

### 10.1 Algorithm (FR-RUN-1)
```
for item in suite (sorted by item_id):
    q = question(item.question_id)
    ctx = build_context_modeA(item) if mode=="A" else retrieve_context(item, retriever, K)
    ctx = intervention.transform_context(ctx, q)        # §12
    system, user = render_prompt(ctx, q, intervention)  # §7.2
    gen = generator.generate(system, user, SUBMIT_ANSWER_SCHEMA)
    grades = grade(item, ctx.labels, gen)               # §8
    write trace + result row
```
Trials: `--trials` defaults to 1 (temperature 0). With `n > 1`, reports use per-item means.

### 10.2 Concurrency (FR-RUN-2)
`--concurrency N` (default 4). Output files are sorted by `(item_id, trial)` regardless of completion order. No module-level mutable state.

### 10.3 Errors (FR-RUN-3)
Provider errors retry with exponential backoff (max 5; seeded jitter in tests). A permanent error records `generator_error` and excludes the item from denominators (reported prominently, never silently).

### 10.4 Trace (`traces/<item_id>__<trial>.json`) (FR-RUN-4)
```json
{"run_id":"…","item_id":"…","trial":0,"generator":"openai:gpt-4o-mini","prompt_version":"1",
 "intervention":"baseline","condition":{…},
 "context":[{"doc_id":"doc-k3f9ab","role":"distractor:near_miss_entity","position":0,
             "status":"official","effective_date":"2024-03-01","facts":[…]}, …],
 "retrieved":[{"doc_id":"…","rank":1,"score":7.31}],         // Mode B only
 "raw_output":"…","parsed":{…},"format_error":null,
 "grades":{…§8…},"usage":{…},"latency_ms":812.4}
```
No timestamps in traces or results (wall-clock lives only in `manifest.json`).

### 10.5 Result row (`results.jsonl`) (FR-RUN-5)
```
run_id, item_id, group, split, kind, mode, k_noise, position, conflict, k_retrieve, retriever,
generator, intervention, pass_answer, pass_strict, provenance, hallucinated, failure_mode,
abstain, conflict_flagged, false_conflict_flag, values_complete,
citation_precision, citation_recall, citation_exact, fabricated_citations,
retrieval_hit, retrieval_recall, retrieval_precision, retrieval_mrr, retrieval_ndcg, distractor_rate,
input_tokens, output_tokens, cost_usd, latency_ms
```

### 10.6 Manifest (`manifest.json`) (FR-RUN-6)
Generator, params, versions, git sha, KB hash, suite hash, embedder + revision, store, retriever, intervention, CLI args, wall-clock start/end.

---

## 11. Statistics and reports (`report/`)

### 11.1 Statistics (FR-REP-1) — unit-tested against known values
- **Proportion CI:** 95% Wilson interval. State in the report that items sharing a question are not independent.
- **Paired difference of means (same items, two runs/conditions):** paired bootstrap, 10,000 resamples of items, seeded; report mean difference and 95% percentile CI.
- **Paired binary comparison:** exact McNemar (two-sided binomial on discordant pairs `b`, `c`); report `b`, `c`, p-value.
- **Lost-in-the-middle score** = `mean(acc_top, acc_bottom) − acc_middle` on `position_sweep` (per `k_noise`), with a paired-bootstrap CI over questions.
- **Latency/cost:** mean, p50, p95, p99 (nearest-rank); `cost_usd` from `config/models.yaml`; cost per correct item.
- Multiple comparisons: the report lists the number of tests run; Bonferroni-adjusted p-values are shown alongside raw ones for the intervention table.

### 11.2 Report (`ragstress report <run_dir>` → `report.md`) (FR-REP-2)
Sections, in order:
1. Header: generator, params, versions, KB/suite hashes, item counts, `generator_error` count, `LEAKAGE_WARNING` if any.
2. **Controls:** oracle accuracy, closed-book abstain rate, leakage rate.
3. **Distractor curve:** `accuracy`, `hallucination_rate`, `distractor_adoption_rate` vs `k_noise` (table; PNG optional) with Wilson CIs.
4. **Position:** accuracy by position × `k_noise`; lost-in-the-middle score with CI.
5. **Conflict:** per kind: accuracy, stale-adoption, conflict-ignored, flagged-and-resolved, values-complete; false-conflict-flag rate on plain items.
6. **Abstention:** recall, precision, false-abstain rate.
7. **Citations:** precision, recall, exact, fabricated rate.
8. **End-to-end:** retrieval metrics per retriever and `K`; share of failures that are `retrieval_miss`; accuracy conditional on retrieval hit vs miss.
9. **Failure taxonomy** (counts and %).
10. Cost and latency.
11. Reproduction command.

### 11.3 Compare (`ragstress compare <runA> <runB>`) (FR-REP-3)
Requires identical item sets. Reports paired deltas (bootstrap CI) and McNemar results for `pass_answer` and `hallucinated`; lists flipped items (regressions/improvements) and their taxonomy changes. Refuses (exit ≠ 0) on item-set mismatch.

### 11.4 Golden test (FR-REP-4)
A fixture run in `tests/golden/` renders to a committed `report.md`; any diff fails the test.

---

## 12. Interventions (`interventions/`)

### 12.1 Interface (FR-INT-1)
```python
class Intervention(Protocol):
    name: str
    def transform_context(self, ctx: Context, q: Question) -> Context
    def extra_system(self) -> str          # appended AFTER the base prompt; never edits R1–R6
    def postprocess(self, ctx: Context, out: GenResult) -> GenResult
```
`baseline` is the identity.

### 12.2 Provided interventions (FR-INT-2)
| Name | Effect |
|---|---|
| `reorder_edges` | Reorder so the highest-scoring passages sit at the start and end, lowest in the middle (deterministic; score = retriever score in Mode B, BM25 vs question in Mode A). |
| `lexical_rerank` | Rerank by deterministic lexical match of question entity name and attribute label (exact-name matches first). Optional `cross_encoder_rerank` (extra, pinned model). |
| `check_prompt` | Appends instructions to first list passages matching the *exact* entity and attribute, then answer. |
| `score_threshold` | Mode B: abstain (post-process to `abstain=true`) when the top retrieval score is below a threshold chosen on `dev` only. |

### 12.3 Evaluation protocol (FR-INT-3)
Each intervention is run on the **same items** as the baseline; thresholds/prompts are tuned on `dev`, reported on `test`. The intervention table in the report lists paired deltas for `accuracy`, `hallucination_rate`, `false_abstain_rate` and `strict_accuracy`, so trade-offs are visible.

---

## 13. CLI (FR-CLI-1)
```
ragstress gen-kb --seed S --out data/kb
ragstress gen-suite --kb data/kb --suite {smoke,core} --seed S --out data/suites/<name>.jsonl
ragstress validate-suite data/suites/<name>.jsonl
ragstress hardness data/suites/<name>.jsonl             # distractor hardness + style/ID leakage checks
ragstress retrieval-eval --suite … --retriever {bm25,dense,hybrid} --embedder {hash,minilm}
                         --store {flat,qdrant} --k 3,5,10
ragstress run --generator openai:gpt-4o-mini --suite data/suites/core.jsonl
              [--intervention NAME] [--retriever … --k-retrieve K]
              [--trials N] [--concurrency N] [--out runs/<id>] [--record | --replay]
              [--group GLOB] [--split test]
ragstress report runs/<id>
ragstress compare runs/A runs/B
```
Generator specs: `openai:<model>`, `anthropic:<model>`, `scripted:<policy>`. `--replay` is default in CI and forbids network.
Makefile: `setup`, `test`, `lint`, `validate`, `bench-replay`, `bench`, `report`, `docker`.

## 14. Configuration
`config/models.yaml`: per-model `input_per_mtok`, `output_per_mtok` (data, not code; the report prints the price-table version used; unknown model → cost `null` + warning). `config/defaults.yaml`: chunk size, retriever params, suite sizes, bootstrap settings.

---

## 15. Testing requirements
| Layer | Must cover |
|---|---|
| Unit | name/fact/render determinism and invariants; normalizers (incl. rejected formats); BM25 vs hand-computed; hash embedder determinism; stores' tie-break; retrieval metrics; each distractor type and collision filter; twins per conflict kind; position placement; opaque-ID independence; each grader table-driven; taxonomy order; stats vs known values (Wilson, McNemar, bootstrap with fixed seed) |
| Integration | scripted generators land in their intended taxonomy bins; oracle = 100% on validate-suite; closed-book contexts empty; concurrency 1 vs 8 → identical `results.jsonl`; replay-twice hash equality; Qdrant/Flat parity |
| Golden | report snapshot |
| Meta | socket blocking for all non-live tests; mutation test (corrupting gold values must fail correctness tests) |
Every `FR-*` ID SHOULD be referenced by at least one test name or docstring.

---

## 16. Acceptance criteria (Definition of Done)
- [ ] FR-1…FR-8 and NFR-1…NFR-8 satisfied.
- [ ] `core` suite generated and `validate-suite` passes; hardness and ID-leakage checks pass.
- [ ] Oracle generator scores 100% `pass_strict` on validated items; `hallucinate`, `ignore_conflict`, `abstain_always` fail where §9.4 says.
- [ ] `make bench-replay` twice → identical `results.jsonl` SHA-256.
- [ ] Two real generators recorded on `core` with cassettes committed; closed-book leakage reported (and < 5%, else the generator or names are fixed and the suite regenerated).
- [ ] Report includes controls, distractor curve, position table with lost-in-the-middle score + CI, conflict table, abstention, citations, E2E retrieval + miss attribution, taxonomy, cost/latency.
- [ ] At least two interventions evaluated with paired statistics; the README states the measured effects honestly (including nulls and trade-offs).
- [ ] CI green (lint, types, tests, coverage, replay bench); README documents architecture, how to add a stress factor, how to record cassettes, limitations (§17).

---

## 17. Known limitations (state them in the README)
- Synthetic, templated text is easier to parse than real documents; absolute numbers will not transfer to production corpora.
- Single-fact lookups only (multi-hop is a stretch); no tables, long documents, or OCR noise.
- Distractor "hardness" is measured with lexical/dense scores, not human judgment.
- Position effects depend on context length, prompt format, and model; one prompt format is tested.
- Items sharing a question are correlated; intervals are approximate.
- Deterministic grading cannot assess free-form explanation quality or partially correct answers.
- Cassette replay validates the harness and grading, not new model behavior.
- Closed-book near-zero leakage shows facts are fictional, not that the model has no priors about *numeric ranges* (it may guess plausible values; chance collisions are possible).

## 18. Assumptions and open decisions
| # | Assumption | Revisit if |
|---|---|---|
| A1 | Fictional enterprise-policy domain, three entity kinds. | You want a domain closer to your target employer. |
| A2 | `FlatStore` + `HashEmbedder` default; real vector DB as an adapter. | You want the dense results to reflect a real embedding model by default. |
| A3 | Structured output through a forced tool call. | The provider's tool calling changes behavior vs free text; add a free-text variant. |
| A4 | One question per item, single-fact answers. | You add multi-hop. |
| A5 | `temperature = 0`, one trial. | You want variance estimates; raise `--trials` and report per-item means. |
| A6 | Draft-vs-official and date rules are stated in the prompt. | You want to test whether models *infer* such rules without being told (a separate, valid experiment). |
