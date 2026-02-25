# 📄 Policy Q&A Bot

A Retrieval-Augmented Generation (RAG) assistant for insurance policy documents. It answers user queries strictly from provided policy wordings and provides clause-level citations for every claim.

## Features

- **Two-Pass Ingestion Strategy:** Uses [Docling](https://github.com/DS4SD/docling) for high-fidelity structured extraction (tables, lists, OCR), followed by a page-level reconciliation pass to guarantee zero missing content due to layout edge cases.
- **Parent-Document Retrieval:** Indexes small child chunks for precise search but returns larger parent chunks for richer LLM context.
- **Dual-BM25 Hybrid Retrieval:** Combines FAISS vector search with separate BM25 keyword matching for both child and parent chunks via Reciprocal Rank Fusion (RRF), ensuring broad contextual terms and specific keywords are reliably surfaced.
- **Agentic 3-Chain RAG Pipeline:** Uses [DSPy](https://github.com/stanfordnlp/dspy) for query reformulation, reranking relevance gating, and a unified generation-classification step that natively enforces grounded answers and structural constraints.
- **2-Layer Hybrid Evaluation:** Rule-based pre-checks target definitive edge cases (Near-Miss, Out-of-Scope), followed by an LLM-judge fallback. This guarantees rigorous adherence to product requirements while eliminating standard LLM judge non-determinism.

---

## Architecture

```
User Question
    │
    ▼
[Chain 1: LLM Query Reformulator] ──► 2-3 domain-adapted search queries
    │
    ▼
[FAISS + Dual BM25 Multi-Query Retrieval] ──► Merged + deduplicated candidates
    │
    ▼
[Chain 2: Cross-Encoder Reranker + Gate] ──► Filtered top-k chunks with a hard relevance backstop
    │
    ▼
[Chain 3: Grounded Answer Generation] ──► DSPy CoT derives the answer and simultaneous classification
```

### Models

| Component | Model | Runtime |
|---|---|---|
| LLM | `gemma3:4b-it-q4_K_M` | Ollama |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace (Sentence Transformers) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace |

---

## System Requirements

This project is optimized for performance on consumer-grade hardware.

- **GPU:** NVIDIA GTX 1660 Ti or better (6GB+ VRAM recommended)
- **CUDA:** 11.7+ (Required for GPU-accelerated embeddings and reranking)
- **CPU:** 4+ cores
- **RAM:** 8GB+ (16GB recommended)

> [!NOTE]
> While a GPU is recommended for the reranker to ensure low latency, all components can run on CPU if a GPU is unavailable. Ollama models will automatically fall back to CPU.

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally

### Installation

```bash
git clone https://github.com/l3onlau/policy-qa-bot.git
cd policy-qa-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Pull Models

```bash
ollama pull gemma3:4b-it-q4_K_M
```

### Data Preparation

Place PDF policy documents in the `./data/` directory.

---

## Usage

### Interactive CLI

```bash
python main.py
```

On first run, the system uses a two-pass parser on all PDFs, builds the FAISS index, and persists it to `./faiss_db/`. Subsequent runs load the index instantly.

### Evaluation Suite

```bash
python test_bot.py
```

Runs 10 test cases from `tests.json` and writes results to `eval_report.json`. The evaluator uses a 2-layer approach:

1. **Rule-based pre-check** — deterministic checks for correct refusal/definitive answer classification specifically regarding Near-Miss and Out-of-Scope queries, and verification of presence of the `Sources:` block.
2. **LLM judge** — `ChainOfThought` fallback for nuanced factual accuracy evaluation.

---

## Configuration

All models and paths are configured via `.env`:

```env
DSPY_LM_MODEL=ollama_chat/gemma3:4b-it-q4_K_M
DSPY_API_BASE=http://localhost:11434
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
FAISS_INDEX_PATH=./faiss_db/policy_index.faiss
DATA_DIR=./data
TOP_K_RERANK=10
RELEVANCE_THRESHOLD=-5.0
```

To swap models, update the `.env` values — no code changes required.

---

## Project Structure

```
├── main.py              # Interactive CLI entry point
├── test_bot.py          # 2-layer hybrid evaluation suite
├── tests.json           # 10 test cases (5 in-domain, 3 near-miss, 2 out-of-scope)
├── eval_report.json     # Generated evaluation results
├── prd.md               # Product requirements document
├── design_notes.md      # Architecture and design decisions
├── src/
│   ├── __init__.py      # Package marker
│   ├── engine.py        # 3-chain agentic RAG pipeline (reformulate → retrieve & gate → generate)
│   ├── ingestion.py     # 2-pass PDF parsing and parent-child chunking
│   ├── vectorstore.py   # FAISS + Dual BM25 hybrid store with RRF
│   └── utils.py         # Shared utilities (citations, env vars)
├── data/                # Source PDF policy documents
└── faiss_db/            # Persisted FAISS index and metadata
```

---

### System Limitations & Hardware Constraints

This architecture is strictly optimized for local consumer hardware (e.g., 6GB VRAM GPUs), introducing necessary performance trade-offs.

* **Evaluation Limitations (4B Models):** To fit within memory limits, the system relies on a heavily quantized small-parameter model (`gemma3:4b-it-q4_K_M`). This model is underpowered for complex reasoning and suffers from massive "same-model bias" when acting as an LLM judge for its own outputs. As a result, the `test_bot.py` evaluation suite typically scores around 5 ± 2, with an expected variance of ± 2 false positives/negatives.
* **Reranker as a "Hard Gate":** Small models struggle to execute clean refusals and often try to force out-of-scope concepts into their answers. The Cross-Encoder reranker functions as a mandatory mathematical gate to counteract this. By calculating exact relevance probabilities and aggressively dropping chunks, it physically starves the LLM of irrelevant context to force deterministic "Out-of-Scope" refusals.

> [!WARNING]
> **Automatic Memory Management:** To ensure stable 100% GPU inference and prevent Out-of-Memory (OOM) crashes on limited hardware, the codebase automatically manages memory by clearing Ollama's VRAM and flushing the CUDA cache between heavy workloads (e.g., transitioning between document ingestion and evaluation).
