import os
import pickle
import re
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

from config import settings


def tokenize_text(text: str) -> List[str]:
    """
    Regex-based tokenizer that strips punctuation and isolates alphanumeric words.
    Enables matching exact numbers and identifiers within complex strings.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class PolicyVectorStore:
    """
    Persistent hybrid vector store using FAISS (semantic) and BM25 (keyword).

    Uses a parent-document retrieval strategy: child chunks are indexed for
    precise search, but parent chunks (larger context) are returned for LLM
    consumption. Supports optional metadata pre-filtering.

    BM25 indexes are simultaneously maintained for both child and parent chunks.
    This ensures that keywords appearing sparsely across the broader parent context
    (e.g., sidebar contact numbers related to body text) can still be reliably matched.
    """

    def __init__(self):
        self.model_name = settings.embed_model
        self.index_path = settings.faiss_index_path
        self.meta_path = self.index_path.replace(".faiss", "_meta.pkl")

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        self.embedder = SentenceTransformer(
            self.model_name,
            model_kwargs={
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                ),
            },
        )
        # Upgrade reasonable context limit
        if hasattr(self.embedder, "max_seq_length"):
            self.embedder.max_seq_length = 4096

        self.documents: List[str] = []
        self.parent_documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.index: Optional[faiss.Index] = None
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_parent: Optional[BM25Okapi] = None

        self.load_index()

    def count(self) -> int:
        """Returns the number of indexed child chunks."""
        return len(self.documents)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Embed and index new documents.

        Args:
            documents: List of dicts with 'content', 'parent_content', and 'metadata'.
        """
        if not documents:
            return

        texts = [doc["content"] for doc in documents]
        parent_texts = [doc.get("parent_content", doc["content"]) for doc in documents]
        metas = [doc["metadata"] for doc in documents]

        print(f"🧠 Generating embeddings for {len(texts)} child chunks...")

        # Lower outer batch size for better logging progress tracking
        batch_size = 10
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            print(f"   ⏳ Embedding batch {i // batch_size + 1} of {total_batches}...")
            # Enforce tiny iteration batch_size (2-4 max) to prevent quadratic
            # attention memory spikes from 4096 max_length on only 5GB of VRAM
            batch_emb = self.embedder.encode(
                batch_texts, batch_size=2, show_progress_bar=False
            )
            all_embeddings.extend(batch_emb)

        embeddings = np.array(all_embeddings).astype("float32")
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)
        self.documents.extend(texts)
        self.parent_documents.extend(parent_texts)
        self.metadatas.extend(metas)

        # Rebuild BM25 keyword indexes (child + parent) using the regex tokenizer
        tokenized_corpus = [tokenize_text(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        tokenized_parents = [tokenize_text(doc) for doc in self.parent_documents]
        self.bm25_parent = BM25Okapi(tokenized_parents)

        self.save_index()

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filters: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Hybrid search combining FAISS vector search and heavily-weighted BM25 keyword matching,
        fused via Reciprocal Rank Fusion (RRF). Returns parent documents.
        """
        if self.index is None or self.count() == 0:
            return {"documents": [], "metadatas": [], "distances": []}

        # Apply metadata pre-filtering if specified
        if filters:
            allowed_indices = set()
            for idx, meta in enumerate(self.metadatas):
                match = all(
                    meta.get(key, "").lower() == value.lower()
                    for key, value in filters.items()
                )
                if match:
                    allowed_indices.add(idx)
            if not allowed_indices:
                return {"documents": [], "metadatas": [], "distances": []}
        else:
            allowed_indices = None

        # 1. Vector search (cosine similarity via inner product on normalized vectors)
        query_embedding = self.embedder.encode([query_text]).astype("float32")
        faiss.normalize_L2(query_embedding)

        search_k = n_results * 8 if filters else n_results * 4
        faiss_distances, faiss_indices = self.index.search(query_embedding, search_k)
        faiss_results = [i for i in faiss_indices[0] if i != -1]

        if allowed_indices is not None:
            faiss_results = [i for i in faiss_results if i in allowed_indices]

        # 2. Keyword search (exact term matching on both child AND parent chunks)
        tokenized_query = tokenize_text(query_text)
        bm25_pool_size = n_results * 5

        bm25_results = []
        if self.bm25:
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_results_raw = np.argsort(bm25_scores)[::-1][:bm25_pool_size].tolist()
            bm25_results = [i for i in bm25_results_raw if bm25_scores[i] > 0]
            if allowed_indices is not None:
                bm25_results = [i for i in bm25_results if i in allowed_indices]

        # Parent-doc BM25: catches keywords that only appear in the broader
        # parent context (e.g. sidebar phone numbers merged into body text).
        bm25_parent_results = []
        if self.bm25_parent:
            parent_scores = self.bm25_parent.get_scores(tokenized_query)
            parent_raw = np.argsort(parent_scores)[::-1][:bm25_pool_size].tolist()
            bm25_parent_results = [i for i in parent_raw if parent_scores[i] > 0]
            if allowed_indices is not None:
                bm25_parent_results = [
                    i for i in bm25_parent_results if i in allowed_indices
                ]

        # 3. Reciprocal Rank Fusion (RRF) with heavy BM25 bias
        fused_scores: Dict[int, float] = {}
        k = 60

        for rank, idx in enumerate(faiss_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        for rank, idx in enumerate(bm25_results):
            # 3.0 multiplier aggressively pushes exact keyword matches to the top
            fused_scores[idx] = fused_scores.get(idx, 0) + 3.0 / (k + rank + 1)

        for rank, idx in enumerate(bm25_parent_results):
            # 2.0 multiplier for parent-doc keyword matches (slightly lower
            # weight because parent text is broader and less precise)
            fused_scores[idx] = fused_scores.get(idx, 0) + 2.0 / (k + rank + 1)

        # Sort by fused score, then deduplicate BEFORE truncating to n_results
        # (previously dedup happened after, which could lose good candidates)
        reranked_indices = sorted(
            fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True
        )

        # Deduplicate by parent_id THEN limit to n_results
        seen_parents = set()
        deduped_indices = []
        for idx in reranked_indices:
            parent_id = self.metadatas[idx].get("parent_id", str(idx))
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                deduped_indices.append(idx)
            if len(deduped_indices) >= n_results:
                break

        return {
            "documents": [
                (
                    self.parent_documents[i]
                    if i < len(self.parent_documents)
                    else self.documents[i]
                )
                for i in deduped_indices
            ],
            "metadatas": [self.metadatas[i] for i in deduped_indices],
            "distances": [fused_scores[i] for i in deduped_indices],
        }

    def save_index(self):
        """Persist the FAISS index and metadata to disk."""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(
                    {
                        "docs": self.documents,
                        "parent_docs": self.parent_documents,
                        "metas": self.metadatas,
                    },
                    f,
                )

    def load_index(self):
        """Load existing FAISS index and metadata from disk."""
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["docs"]
                    self.parent_documents = data.get("parent_docs", data["docs"])
                    self.metadatas = data["metas"]

                if self.documents:
                    # Rebuild BM25 indexes using the regex tokenizer
                    tokenized_corpus = [tokenize_text(doc) for doc in self.documents]
                    self.bm25 = BM25Okapi(tokenized_corpus)

                    tokenized_parents = [
                        tokenize_text(doc) for doc in self.parent_documents
                    ]
                    self.bm25_parent = BM25Okapi(tokenized_parents)
            except Exception as e:
                print(f"⚠️ Failed to load existing index: {e}. Starting fresh.")
