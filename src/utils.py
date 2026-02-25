import os
import requests
import gc
import torch
from typing import Dict, Any


def format_citation(metadata: Dict[str, Any]) -> str:
    """
    Format document metadata into a standardized citation string.

    Example output: "Policy_A.pdf §3.2 (Exclusions), p.12"
    """
    doc = metadata.get("doc_name", "Unknown Document")
    section = metadata.get("section", "N/A")
    clause = metadata.get("clause_number", "")
    page = metadata.get("page", "??")

    clause_str = f" §{clause}" if clause else ""
    return f"{doc}{clause_str} ({section}), p.{page}"


def get_env_var(var_name: str, default: str = "") -> str:
    """Retrieve an environment variable with an optional default."""
    return os.getenv(var_name, default)


def free_ollama_memory():
    """Request the Ollama service to unload models and free GPU VRAM."""
    api_base = get_env_var("DSPY_API_BASE", "http://localhost:11434")
    print("🧹 Requesting Ollama to free up memory...")
    try:
        response = requests.get(f"{api_base}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                model_name = model.get("name")
                requests.post(
                    f"{api_base}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=5,
                )
            print(f"   ✅ Unloaded {len(models)} models.")
    except Exception as e:
        print(f"   ⚠️ Could not free Ollama memory: {e}")


def ensure_ingested(vectorstore: Any):
    """Check vectorstore and trigger document ingestion if empty."""
    if vectorstore.count() == 0:
        print("📥 Vectorstore is empty. Starting document ingestion...")

        free_ollama_memory()

        from .ingestion import PolicyIngestor

        ingestor = PolicyIngestor()
        data_dir = get_env_var("DATA_DIR", "./data")

        docs = ingestor.process_pdfs(data_dir)

        if docs:
            vectorstore.add_documents(docs)
            print(f"   ✅ Indexed {len(docs)} document chunks.")

        print("🧹 Reclaiming system resources...")
        del ingestor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"✅ Vectorstore loaded with {vectorstore.count()} chunks.")
