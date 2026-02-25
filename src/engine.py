import dspy
from sentence_transformers import CrossEncoder
from .utils import get_env_var
from typing import Any

REFUSAL_STRING = "I cannot find a definitive answer in the provided policy wording."

ANSWER_TYPE_DEFINITIVE = "DEFINITIVE"
ANSWER_TYPE_REFUSAL = "REFUSAL"
ANSWER_TYPE_CONDITIONAL = "CONDITIONAL"

# ─────────────────────────────────────────────────────────────────────────────
# Chain 1: Query Reformulator
# Generates multiple search queries to handle vocabulary variance.
# ─────────────────────────────────────────────────────────────────────────────


class QueryReformulator(dspy.Signature):
    """
    You are an insurance policy search expert. Given a user question, generate
    alternative search queries that would help find the relevant policy clauses.

    RULES:
    1. Produce exactly 3 short search queries separated by newlines.
    2. Each query should use different vocabulary — insurance jargon, legal terms,
       and plain-language synonyms.
    3. Think about WHERE in a policy document the answer would appear:
       exclusions, definitions, general conditions, coverage sections, etc.
    4. Keep each query under 15 words.
    """

    question = dspy.InputField(desc="The user's insurance question.")
    search_queries = dspy.OutputField(
        desc="Exactly 3 alternative search queries, one per line."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chain 3: Policy Answer Generator & Classifier
# Core generation logic for policy-grounded answers. Generates the answer
# while simultaneously classifying it based on context relevance.
# ─────────────────────────────────────────────────────────────────────────────


class PolicySignature(dspy.Signature):
    """
    You are a strict insurance compliance assistant. Answer questions using ONLY the provided context.

    CLASSIFICATION RULES — follow in order:
    1. DEFINITIVE ANSWER (answer_type = "DEFINITIVE"):
       - The context directly addresses the question's topic — either confirming coverage or stating an explicit exclusion.
       - If the context explicitly says something is excluded or not covered, state confidently that it is NOT covered. An explicit exclusion IS a definitive answer.
       - Always extract specific numbers, dollar amounts, time periods, and limits when present.
    2. CONDITIONAL ANSWER (answer_type = "CONDITIONAL"):
       - Coverage depends on specific conditions, options, or endorsements mentioned in the context.
       - State what conditions apply.
    3. REFUSAL (answer_type = "REFUSAL"):
       - Use ONLY when the question's core topic does NOT appear in the context AT ALL.
       - The context has zero direct mentions of the specific concept being asked about.
       - The question is entirely outside the scope of insurance policy wording.

    CRITICAL DISTINCTION:
    - If the context mentions the EXACT topic and says it's excluded → DEFINITIVE
    - If the context only has tangentially related topics but never directly addresses the question → REFUSAL
    - If coverage depends on policyholder choices or conditions → CONDITIONAL

    CITATION RULE: Always populate the `sources` field with the pre-formatted citations provided in the context.
    """

    context = dspy.InputField(
        desc="Relevant policy document chunks with pre-formatted citations."
    )
    question = dspy.InputField(desc="The user's specific insurance inquiry.")
    answer_type = dspy.OutputField(
        desc="EXACTLY one of: 'DEFINITIVE', 'CONDITIONAL', or 'REFUSAL'."
    )
    reasoning_and_answer = dspy.OutputField(
        desc="The grounded answer with specific facts, or explanation of why no answer exists."
    )
    sources = dspy.OutputField(
        desc="The pre-formatted citations copied from the context."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PolicyRAG: Agentic Multi-Chain Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class PolicyRAG(dspy.Module):
    """
    3-chain agentic RAG pipeline:
      Chain 1: LLM Query Reformulation (vocabulary expansion)
      Chain 2: Relevance Gating (cross-encoder score threshold)
      Chain 3: Grounded Generation (LLM derives answer and classifies it simultaneously)
    """

    def __init__(self, vectorstore: Any):
        super().__init__()
        self.vectorstore = vectorstore
        self.reformulator = dspy.Predict(QueryReformulator)

        self.generator = dspy.ChainOfThought(PolicySignature)

        self.reranker = CrossEncoder(
            get_env_var("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )

        self.top_k = int(get_env_var("TOP_K_RERANK", "10"))
        # We use a loose threshold to allow potential context, but use a strict backstop later
        self.relevance_threshold = float(get_env_var("RELEVANCE_THRESHOLD", "-5.0"))

    def forward(self, question: str) -> dspy.Prediction:
        # ── Chain 1: LLM Query Reformulation ──────────────────────────────
        # Generate domain-adapted search queries to improve retrieval recall
        try:
            reformulated = self.reformulator(question=question)
            alt_queries = [
                q.strip()
                for q in reformulated.search_queries.strip().split("\n")
                if q.strip()
            ][:3]
        except Exception:
            alt_queries = []

        # Multi-query retrieval: merge results from original + reformulated queries
        all_queries = [question] + alt_queries
        seen_texts = set()
        merged_docs = []
        merged_metas = []

        for q in all_queries:
            results = self.vectorstore.query(q, n_results=20)
            for doc, meta in zip(
                results.get("documents", []), results.get("metadatas", [])
            ):
                doc_hash = hash(doc[:200])  # Deduplicate by content prefix
                if doc_hash not in seen_texts:
                    seen_texts.add(doc_hash)
                    merged_docs.append(doc)
                    merged_metas.append(meta)

        if not merged_docs:
            return dspy.Prediction(
                answer=f"{REFUSAL_STRING} No relevant documents were found.",
                policy_found=False,
                answer_type=ANSWER_TYPE_REFUSAL,
                retrieved_chunks=[],
            )

        # ── Chain 2: Cross-Encoder Reranking + Relevance Gate ─────────────
        pairs = [[question, doc] for doc in merged_docs]
        scores = self.reranker.predict(pairs)

        combined = sorted(
            zip(merged_docs, merged_metas, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        # ── Chain 3: Generation & Classification ──────────────────────────
        context_blocks = []
        for doc_text, meta, _score in combined[: self.top_k]:
            clause = meta.get("clause_number", "N/A")
            section = meta.get("section", "N/A")
            page = meta.get("page", "N/A")
            doc_name = meta.get("doc_name", "N/A")

            exact_citation = f"{doc_name} §{clause} ({section}), p.{page}"
            block = f"DOCUMENT TEXT:\n{doc_text}\nCitation To Use: {exact_citation}\n"
            context_blocks.append(block)

        full_context = "\n---\n".join(context_blocks)

        pred = self.generator(context=full_context, question=question)

        raw_answer_type = str(pred.answer_type).strip().upper()

        # Hard backstop: if the CrossEncoder score is extremely low (< -3.0),
        # force a REFUSAL regardless of what the LLM hallucinated, because the context
        # objectively does not address the question.
        max_score = combined[0][2] if combined else -999.0
        if max_score < -3.0:
            answer_type = ANSWER_TYPE_REFUSAL
        else:
            if raw_answer_type in (
                ANSWER_TYPE_DEFINITIVE,
                ANSWER_TYPE_CONDITIONAL,
                ANSWER_TYPE_REFUSAL,
            ):
                answer_type = raw_answer_type
            else:
                answer_type = ANSWER_TYPE_REFUSAL

        # ── Final Assembly ────────────────────────────────────────────────
        is_refusal = answer_type == ANSWER_TYPE_REFUSAL
        final_answer = pred.reasoning_and_answer

        # Enforce refusal string format if it is a refusal
        if is_refusal and REFUSAL_STRING.lower() not in final_answer.lower():
            final_answer = f"{REFUSAL_STRING}\n\n{final_answer}"

        if pred.sources:
            final_answer = f"{final_answer}\n\nSources: {pred.sources}"

        return dspy.Prediction(
            answer=final_answer,
            policy_found=not is_refusal,
            answer_type=answer_type,
            retrieved_chunks=[
                {"text": r[0], "metadata": r[1], "score": float(r[2])}
                for r in combined[: self.top_k]
            ],
            max_relevance_score=float(max_score),
        )


def get_rag_system(vectorstore: Any) -> PolicyRAG:
    """Factory function to create a configured PolicyRAG instance."""
    return PolicyRAG(vectorstore)


def setup_dspy():
    """Configure DSPy with the LLM backend specified in environment variables."""
    lm = dspy.LM(
        model=get_env_var("DSPY_LM_MODEL"),
        api_base=get_env_var("DSPY_API_BASE"),
        temperature=0.0,  # Deterministic for grounded policy answers
        top_p=0.95,  # Gemma default; relaxed since temp=0 dominates
        top_k=40,  # Gemma default range; prevents repetition traps
        max_tokens=512,  # Safe for 6GB VRAM, enough for policy answers
        frequency_penalty=0.1,  # Reduces repetition in small models
        # presence_penalty=0.0, # Unsupported by litellm's ollama_chat
    )
    dspy.configure(lm=lm)
