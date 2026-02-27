from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM (via Ollama)
    dspy_lm_model: str = "ollama_chat/qwen3:4b-instruct-2507-q4_K_M"
    dspy_api_base: str = "http://localhost:11434"

    # HuggingFace Models (local)
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

    # Paths
    faiss_index_path: str = "./faiss_db/policy_index.faiss"
    data_dir: str = "./data"

    # Retrieval Tuning
    top_k_rerank: int = 10
    relevance_threshold: float = 0.0

    # AI Engineering Flags - Toggle pipeline components
    use_semantic_cache: bool = True
    use_intent_routing: bool = True
    use_query_reformulation: bool = True
    use_chunk_distillation: bool = True
    use_nli_entailment_check: bool = True
    max_retries: int = 1

    # Enable optimized few-shot prompt from optimize_rag.py
    # WARNING: Turning this on drastically increases Ollama's context window.
    # If running locally on a 5GB VRAM limit alongside the Embedder, Reranker,
    # and NLI Judge, this is prone to cause CUDA Out Of Memory (OOM) crashes.
    use_compiled_prompt: bool = False

    # Debug Model
    debug_mode: bool = False
    debug_log_file: str = "rag_debug.log"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
