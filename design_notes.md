# Design Notes — Policy Q&A Bot

## 1. Chunking & Ingestion Strategy

### Parent-Document Retrieval

The system decouples search precision from generation context using a two-tier chunking approach:

- **Child chunks (~800 chars / ~200 tokens):** Focused segments indexed natively in FAISS. Their concentrated embeddings enable precise cosine similarity matching for specific facts (dollar limits, time periods, exclusion terms).
- **Parent chunks (~3000 chars / ~750 tokens):** Larger contextual blocks sent to the LLM, sized to fall within the PRD's 400–800 token requirement. When a child chunk matches a query, the corresponding parent provides enough surrounding context for coherent answer synthesis.

### Two-Pass Ingestion 

To guarantee absolutely zero content gaps, ingestion uses a robust two-pass strategy:
1. **Pass 1 (Structured Item-by-Item):** Parses all text-bearing labels, tracking visual boundaries and headings recursively. This correctly ingests complex structures natively (tables, recursive lists, footnotes, and even OCR text from picture elements).
2. **Pass 2 (Page-Level Reconciliation):** Performs a sweep evaluating text captured per page against total page text. If the structured extraction yields an unusually low percentage of total text (due to exotic layouts), supplementary chunks are generated combining all available text on that page with the most recently known section header.

---

## 2. Agentic RAG Pipeline (3-Chain Architecture)

The engine uses a multi-chain agentic approach where each LLM or specialized model call serves a distinct role:

```
User Question
     │
     ▼
[Chain 1: Query Reformulator]     ← LLM generates domain-specific search queries
     │
     ▼
[Hybrid Retrieval: FAISS + BM25]  ← Multi-query retrieval merges & deduplicates
     │
     ▼
[Chain 2: Cross-Encoder Reranker] ← ms-marco-MiniLM scores all candidates
     │                              (Hard backstop gate at -3.0 applies)
     ▼
[Chain 3: Conditioned Generator]  ← CoT derives answer & simultaneously classifies
     │
     ▼
Final Answer
```

### Chain 1 — LLM Query Reformulation

The `QueryReformulator` DSPy Signature generates 2–3 alternative search queries using insurance jargon, legal terms, and synonyms. This generalizes to any user question without domain-specific lookups.

Example: "Is wear-and-tear covered?" → the LLM might generate:
- "wear and tear gradual deterioration exclusion"
- "depreciation loss policy coverage"
- "general exclusions deterioration damage"

### Dual-BM25 & Hybrid Retrieval

Retrieval merges results from all queries (original + reformulated), deduplicated by content. 
Keyword matching uses a *Dual-BM25* setup: both individual child chunks and their full parent chunks are indexed textually. This design resolves matches where a dense keyword appears sparsely in the broader parent context but not in the precise targeted child text snippet. Search hits from vector search, child-BM25, and parent-BM25 are unified using Reciprocal Rank Fusion (RRF) with heavy BM25 multipliers.

### Chain 2 — Relevance Gate (Cross-Encoder)

All deduplicated candidate parent chunks are reranked using the MS MARCO cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
A configurable score threshold acts as a hard filter: chunks mapping to score `< -3.0` indicate an objective disconnect from the query's topic context. This prevents hallucination by severing completely irrelevant chunks (e.g. bilingual boilerplate) from reaching the context window.

### Chain 3 — Answer Generator & Simultaneous Classifier

The single `PolicySignature` passes context block strings to the 4B model and requests a unified CoT approach to simultaneously write the answer and strictly classify it.

The classification rules dictate exactly one output state:
- **DEFINITIVE:** The context directly addresses the question's topic. Coverage is confirmed, or explicitly excluded.
- **CONDITIONAL:** Coverage relies on a contingency, specific timeline, or endorsement condition. 
- **REFUSAL:** The system is unable to locate relevant subject matter, falling back to a structured refusal string.

---

## 3. Negative-Question Handling Logic

Insurance policies are structured around exclusions. The system handles scenarios gracefully:

| Scenario | System Behavior (Retrieval/Gate → Generation) |
|---|---|
| **Explicit exclusion found** | Cross-encoder scores high (exclusion is relevant to the question) → LLM outputs DEFINITIVE and writes definitive "No" answer citing the exclusion |
| **No relevant information whatsoever** | Cross-encoder scores very low (< -3.0) → Hard backstop rejects context → Generator outputs REFUSAL status with exact refusal string |
| **Near-Miss / Ambiguous (related but insufficient)** | Cross-encoder scores pass backstop, but context lacks hard facts → LLM evaluates and falls back to REFUSAL, returning refusal string alongside an explanation of the related clause |

---

## 4. Evaluation Architecture

The evaluation suite uses a 2-layer hybrid approach to eliminate false pass/fail from LLM judge non-determinism:

```
Answer → [Rule-Based Pre-Check] → [LLM Judge]
              │                         │
        Deterministic             Fallback for
        PASS/FAIL rules           nuanced cases
```

1. **Rule-based pre-check:** Deterministic checks that short-circuit to a final verdict:
   - Out-of-Scope queries strictly require `policy_found == False` (system-wide refusal execution).
   - Near-Miss queries strictly require `policy_found == True` (system successfully retrieved near-matches instead of crashing/totally refusing immediately).
   - Mandatory check that the `Sources:` string exists exactly.
2. **LLM judge fallback:** `ChainOfThought(AnswerCorrectness)` handles nuanced factual evaluation for the final `PASS`/`FAIL` classification for in-domain definitive queries.

### Limitation: Same-Model Judge Bias

The LLM judge currently uses the same `gemma3:4b` backend as the answer generator. This creates a potential bias — the judge may broadly share the same reasoning patterns. The rule-based layer aggressively mitigates this error vector by handling the clear-cut system/structural edge-cases without LLM-judging involvement.
