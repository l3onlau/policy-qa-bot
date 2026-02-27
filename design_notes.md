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

## 2. Prompt Design & In-Context Learning

The core generation prompt (`PolicySignature`) is managed dynamically via DSPy. It enforces strict grounded generation by demanding that the LLM answers questions using exactly the provided context and populates a distinct `sources` field with the pre-formatted citations.

### DSPy Few-Shot Compilation
Instead of manual prompt engineering, the system optimizes the prompt via `dspy.BootstrapFewShot`. The prompt is compiled offline using a purely synthetic, domain-agnostic dataset (e.g., "Space Fleet Operational Directives"). By explicitly training on unrelated material, the LLM learns the *structural pattern* of factual extraction and structured citation formatting without overfitting to actual insurance terminology, drastically improving zero-shot performance on complex insurance clauses.

### Post-Retrieval Context Distillation
To maximize prompt efficiency and prevent "lost-in-the-middle" hallucination on large parent chunks, the retrieved context is distilled immediately prior to generation. Large text blocks are split into overlapping sentences. The cross-encoder rerank pipeline identifies only the 15 most relevant semantic sentences. Only these highly concentrated, factual micro-blocks are provided in the `context` field of the prompt, saving massive token compute and focusing the LLM's attention precisely.

---

## 3. Negative-Question Handling Logic

Insurance policies are structured around exclusions. The system handles scenarios gracefully:

| Scenario | System Behavior |
|---|---|
| **Out-of-Scope (Conversational)** | An `_is_valid_query` regex intercepts and refuses queries like "hello" instantly, bypassing LLM compute entirely. |
| **Explicit exclusion found** | Cross-encoder scores high (exclusion is relevant to the question) → LLM outputs definitively and writes a "No" answer citing the exclusion. |
| **No relevant information whatsoever** | Cross-encoder scores very low (< -5.0) → Hard backstop rejects context entirely → System short-circuits to the PRD refusal string: *"I cannot find a definitive answer in the provided policy wording."* |
| **Ambiguous / Hallucinated Connections** | A Natural Language Inference (NLI) model explicitly evaluates the LLM's generated facts against the raw chunk. If entailment is low or contradicted, generation is blocked and defaults to the refusal string. |
