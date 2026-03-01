from typing import Dict, Any
from config import settings


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


def ensure_ingested(vectorstore: Any):
    """Check vectorstore and trigger document ingestion if empty."""
    if vectorstore.count() == 0:
        print("📥 Vectorstore is empty. Starting document ingestion...")

        from .ingestion import PolicyIngestor

        ingestor = PolicyIngestor()
        data_dir = settings.data_dir

        docs = ingestor.process_pdfs(data_dir)

        if docs:
            vectorstore.add_documents(docs)
            print(f"   ✅ Indexed {len(docs)} document chunks.")

    else:
        print(f"✅ Vectorstore loaded with {vectorstore.count()} chunks.")
