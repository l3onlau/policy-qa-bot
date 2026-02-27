# 📄 Policy Q&A Bot

A Retrieval-Augmented Generation (RAG) assistant for insurance policy documents. It answers user queries strictly from provided policy wordings and provides clause-level citations for every claim.

## Features

- **Robust Ingestion:** Uses Docling for high-fidelity extraction of complex structural PDFs natively.
- **Concurrent Retrieval:** Async fetching via FAISS vector search and Dual-BM25 keyword matching (parent & child chunks explicitly merged via RRF).
- **Post-Retrieval Distillation:** Distills large 800-token chunks into smaller factual sentences for precise generation.
- **DSPy Few-Shot Generation:** Enforces rigid factual citations using a prompt structurally compiled on synthetic offline data.

---

### Models

| Component | Model | Runtime |
|---|---|---|
| LLM | `qwen3:4b-instruct-2507-q4_K_M` | Ollama |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` | HuggingFace (8-bit Quantized) |
| Reranker | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` | HuggingFace (8-bit Quantized) |
| NLI Judge | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | HuggingFace (8-bit Quantized) |

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
ollama pull qwen3:4b-instruct-2507-q4_K_M
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

All models and paths are configured via `config.py`:

```config
DSPY_LM_MODEL=ollama_chat/qwen3:4b-instruct-2507-q4_K_M
DSPY_API_BASE=http://localhost:11434
EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
RERANKER_MODEL=tomaarsen/Qwen3-Reranker-0.6B-seq-cls
NLI_MODEL=MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
FAISS_INDEX_PATH=./faiss_db/policy_index.faiss
DATA_DIR=./data
TOP_K_RERANK=10
RELEVANCE_THRESHOLD=-5.0

# AI Engineering Flags
USE_SEMANTIC_CACHE=True
USE_INTENT_ROUTING=True
USE_QUERY_REFORMULATION=True
USE_CHUNK_DISTILLATION=True
USE_NLI_ENTAILMENT_CHECK=True
MAX_RETRIES=1
```

To swap models, update the `config.py` values — no code changes required.

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
│   ├── engine.py        # 4-chain agentic RAG pipeline (reformulate → retrieve & gate → generate → verify)
│   ├── ingestion.py     # 2-pass PDF parsing and parent-child chunking
│   ├── vectorstore.py   # FAISS + Dual BM25 hybrid store with RRF
│   └── utils.py         # Shared utilities (citations, env vars)
├── data/                # Source PDF policy documents
└── faiss_db/            # Persisted FAISS index and metadata
```

---

### System Limitations & Hardware Constraints

This architecture is strictly optimized for local consumer hardware (e.g., 6GB VRAM GPUs), introducing necessary performance trade-offs.

* **Evaluation Limitations (4B Models):** To fit within memory limits, the system relies on a heavily quantized small-parameter model (`qwen3:4b-instruct-2507-q4_K_M`). This model is underpowered for complex reasoning and suffers from massive "same-model bias" when acting as an LLM judge for its own outputs. As a result, the `test_bot.py` evaluation suite typically scores around 5 ± 2, with an expected variance of ± 2 false positives/negatives.
* **Reranker as a "Hard Gate":** Small models struggle to execute clean refusals and often try to force out-of-scope concepts into their answers. The Cross-Encoder reranker functions as a mandatory mathematical gate to counteract this. By calculating exact relevance probabilities and aggressively dropping chunks, it physically starves the LLM of irrelevant context to force deterministic "Out-of-Scope" refusals.

> [!WARNING]
> **Automatic Memory Management:** To ensure stable execution even on constrained hardware (e.g. 5GB VRAM limits), the system strictly utilizes 8-bit dynamic quantization + CPU offloading for the majority of the Transformer pipelines where supported. Furthermore, heavy processing contexts (4096 tokens) are batched down to micro-batches (batch_size=1) to prevent quadratic attention OOM spikes natively during inference.
