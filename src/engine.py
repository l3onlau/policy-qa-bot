import dspy
from sentence_transformers import CrossEncoder
from transformers import BitsAndBytesConfig, pipeline
import logging
import json
import re
import os
import concurrent.futures
from config import settings
from typing import Any

REFUSAL_STRING = "I cannot find a definitive answer in the provided policy wording."

ANSWER_TYPE_DEFINITIVE = "DEFINITIVE"
ANSWER_TYPE_REFUSAL = "REFUSAL"
ANSWER_TYPE_CONDITIONAL = "CONDITIONAL"


def get_logger():
    logger = logging.getLogger("PolicyRAG")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if settings.debug_mode else logging.INFO)
        if settings.debug_mode:
            fh = logging.FileHandler(settings.debug_log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


logger = get_logger()


class FactDecomposition(dspy.Signature):
    """
    Decompose a complex text into a list of atomic facts.
    Each fact should be a simple, standalone proposition that can be independently verified.
    """

    text = dspy.InputField(desc="The text to decompose")
    facts = dspy.OutputField(
        desc="A newline-separated list of atomic facts", format=str
    )


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

    CITATION VERIFICATION: Always end your answer by accurately populating the `sources` field exactly with the pre-formatted citations explicitly provided alongside the relevant facts in the context text.
    """

    context = dspy.InputField(
        desc="Relevant policy document chunks with pre-formatted citations."
    )
    question = dspy.InputField(desc="The user's specific insurance inquiry.")
    reasoning_and_answer = dspy.OutputField(
        desc="The rigorous, grounded factual answer based exclusively on the context."
    )
    sources = dspy.OutputField(
        desc="The pre-formatted citations provided in the context."
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
        self.fact_extractor = dspy.Predict(FactDecomposition)

        self.reranker = CrossEncoder(
            settings.reranker_model,
            model_kwargs={
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                ),
            },
            max_length=4096,
        )

        self.nli_judge = None
        if settings.use_nli_entailment_check:
            nli_model_name = settings.nli_model
            print(f"Loading NLI Judge: {nli_model_name}")
            self.nli_judge = pipeline(
                "text-classification",
                model=nli_model_name,
                device_map="auto",
                model_kwargs={
                    "quantization_config": BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                },
            )

        self.top_k = settings.top_k_rerank
        self.relevance_threshold = settings.relevance_threshold

        # Simple exact-match semantic cache
        self._cache = {}

        # Load compiled DSPy weights if flag is enabled and they exist
        compiled_path = "compiled_rag.json"
        if settings.use_compiled_prompt and os.path.exists(compiled_path):
            print(f"Loading compiled DSPy prompt from {compiled_path}")
            self.generator.load(compiled_path)

    def _is_valid_query(self, query: str) -> bool:
        """Fast-path intent routing to reject obvious non-questions or out-of-scope banter."""
        query = query.strip().lower()
        if len(query) < 5:
            return False

        # Obvious conversational phrases that don't need a vector search
        conversational_patterns = [
            r"^(hi|hello|hey|greetings)(?![a-z])",
            r"^who are you",
            r"^what is your name",
            r"^how are you",
        ]

        for pattern in conversational_patterns:
            if re.match(pattern, query):
                return False

        return True

    def forward(self, question: str) -> dspy.Prediction:
        # ── Fast-Path Intent Routing & Caching ────────────────────────────
        if settings.use_intent_routing and not self._is_valid_query(question):
            return dspy.Prediction(
                answer=f"{REFUSAL_STRING} Explanation: The prompt does not appear to be an insurance question.",
                policy_found=False,
                answer_type=ANSWER_TYPE_REFUSAL,
                retrieved_chunks=[],
            )

        if settings.use_semantic_cache:
            cached_pred = self._cache.get(question.strip().lower())
            if cached_pred:
                logger.debug(f">> Cache Hit for: {question}")
                return cached_pred

        # ── Chain 1: LLM Query Reformulation ──────────────────────────────
        # Generate domain-adapted search queries to improve retrieval recall
        logger.debug(f">> Starting Query: {question}")
        alt_queries = []
        if settings.use_query_reformulation:
            try:
                reformulated = self.reformulator(question=question)
                alt_queries = [
                    q.strip()
                    for q in reformulated.search_queries.strip().split("\n")
                    if q.strip()
                ][:3]
                logger.debug(
                    f"Chain 1 | Reformulator generated queries: {json.dumps(alt_queries)}"
                )
                if logger.getEffectiveLevel() <= logging.DEBUG:
                    print(f"DEBUG: Alternative queries: {alt_queries}")
            except Exception as e:
                logger.debug(f"Chain 1 | Reformulator error: {e}")

        # Multi-query retrieval: merge results from original + reformulated queries
        seen_texts = set()
        merged_docs = []
        merged_metas = []

        # Concurrent retrieval to hide I/O latency
        def _fetch_docs(q):
            try:
                return self.vectorstore.query(q, n_results=20)
            except Exception as e:
                logger.error(f"Retrieval error for query '{q}': {e}")
                return {"documents": [], "metadatas": []}

        all_queries = [question] + alt_queries
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(all_queries)
        ) as executor:
            future_to_query = {executor.submit(_fetch_docs, q): q for q in all_queries}
            for future in concurrent.futures.as_completed(future_to_query):
                results = future.result()
                for doc, meta in zip(
                    results.get("documents", []), results.get("metadatas", [])
                ):
                    doc_hash = hash(doc[:200])  # Deduplicate by content prefix
                    if doc_hash not in seen_texts:
                        seen_texts.add(doc_hash)
                        merged_docs.append(doc)
                        merged_metas.append(meta)

        logger.debug(f"Retrieval | Fetched {len(merged_docs)} unique chunks.")
        if not merged_docs:
            logger.debug("Retrieval | No documents found.")
            return dspy.Prediction(
                answer=f"{REFUSAL_STRING} No relevant documents were found.",
                policy_found=False,
                answer_type=ANSWER_TYPE_REFUSAL,
                retrieved_chunks=[],
            )

        # ── Chain 2: Cross-Encoder Reranking + Relevance Gate ─────────────
        pairs = [[question, doc] for doc in merged_docs]

        # Throttle batch size to avoid quadratic attention memory spikes
        # (similar to the fix in chunk embedding)
        # Using batch_size=1 explicitly to avoid OOM on 5GB VRAM GPUs
        scores = self.reranker.predict(pairs, batch_size=1, show_progress_bar=False)

        combined = sorted(
            zip(merged_docs, merged_metas, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        for i, (cd, cm, cs) in enumerate(combined[: self.top_k]):
            logger.debug(
                f"Chain 2 | Reranked #{i + 1} score={cs:.3f}, doc={cm.get('doc_name')}"
            )

        # Apply relevance gating explicitly
        combined = [x for x in combined if x[2] >= self.relevance_threshold]

        if not combined:
            return dspy.Prediction(
                answer=f"{REFUSAL_STRING}\nExplanation: The query is completely out-of-scope. No relevant policy clauses matched your question.",
                policy_found=False,
                answer_type=ANSWER_TYPE_REFUSAL,
                retrieved_chunks=[],
            )

        # ── Chain 2.5: Post-Retrieval Chunk Distillation ──────────────────
        # Split large chunks into sentences, score them, and take only the most relevant
        distilled_pairs = []
        sentence_metadata = []

        if settings.use_chunk_distillation:
            for doc_text, meta, _ in combined[: self.top_k]:
                # Simple sentence tokenizer (split by . ! ? followed by space or newline)
                sentences = [
                    s.strip()
                    for s in re.split(r"(?<=[.!?])\s+", doc_text)
                    if len(s.strip()) > 10
                ]
                for i, sent in enumerate(sentences):
                    # Provide strict local context by combining with previous and next sentences
                    local_context_list = []
                    if i > 0:
                        local_context_list.append(sentences[i - 1])
                    local_context_list.append(sent)
                    if i < len(sentences) - 1:
                        local_context_list.append(sentences[i + 1])
                    local_context = " ".join(local_context_list)

                    distilled_pairs.append([question, local_context])
                    sentence_metadata.append({"text": local_context, "meta": meta})

            # Throttle batch size for VRAM constraints, use batch_size=1 explicitly
            if distilled_pairs:
                distilled_scores = self.reranker.predict(
                    distilled_pairs, batch_size=1, show_progress_bar=False
                )
            else:
                distilled_scores = []

            distilled_combined = sorted(
                zip(
                    [p[1] for p in distilled_pairs],
                    [m["meta"] for m in sentence_metadata],
                    distilled_scores,
                ),
                key=lambda x: x[2],
                reverse=True,
            )
        else:
            # Bypass distillation
            distilled_combined = combined[: self.top_k]

        # Deduplicate distilled segments
        seen_distilled = set()
        final_distilled = []
        # Take up to 15 highest scoring sentence blocks
        for text, meta, score in distilled_combined:
            if text not in seen_distilled and score >= self.relevance_threshold:
                seen_distilled.add(text)
                final_distilled.append((text, meta, score))
            if len(final_distilled) >= 15:
                break

        if not final_distilled:
            # Fallback to the original chunks if distillation strips too much
            final_distilled = combined[: self.top_k]

        # ── Chain 3: Generation & Classification (With Fallback Routing) ──
        context_blocks = []
        for doc_text, meta, _score in final_distilled:
            clause = meta.get("clause_number", "N/A")
            section = meta.get("section", "N/A")
            page = meta.get("page", "N/A")
            doc_name = meta.get("doc_name", "N/A")

            exact_citation = f"{doc_name} §{clause} ({section}), p.{page}"
            # Re-Reading technique: Interleave instructions with context
            block = f"VERIFIED DOCUMENT SEGMENT:\n{doc_text}\nCitation To Use: {exact_citation}\n[REMINDER: Use ONLY this segment text for your answer. DO NOT hallucinate rules.]\n"
            context_blocks.append(block)

        full_context = "\n---\n".join(context_blocks)

        max_retries = settings.max_retries if settings.use_nli_entailment_check else 0
        nli_label = ""
        is_refusal = False
        max_score = combined[0][2] if combined else -999.0
        policy_found = bool(combined)

        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(
                    f"⚠️ NLI Check Failed (Faithfulness too low). Retrying generation (Attempt {attempt})..."
                )
                # Fallback Routing: prompt with explicit feedback
                retry_question = f"{question}\n\nWARNING: Your previous answer contained unverified facts. You MUST stick strictly to the context."
                logger.debug(f"Chain 3 | Fallback retry attempt {attempt}")
                pred = self.generator(context=full_context, question=retry_question)
            else:
                pred = self.generator(context=full_context, question=question)

            raw_answer = pred.reasoning_and_answer
            logger.debug(f"Chain 3 | Generator Output:\n{raw_answer}")

            # Fast-path: if the LLM correctly refuses based on poor context, bypass NLI
            if REFUSAL_STRING.lower() in raw_answer.lower():
                is_refusal = True
                nli_label = "Model Refusal"
                break

            # ── Chain 4: NLI Entailment Verification ──────────────────────────
            # NLI verifies that the LLM's generated factual answer is fully entailed
            # by the retrieved context chunks (no hallucinated details).
            if settings.use_nli_entailment_check:
                try:
                    # 1. Decompose answer into atomic facts
                    facts_str = self.fact_extractor(text=raw_answer).facts
                    atomic_facts = [
                        f.strip("- *") for f in facts_str.split("\n") if f.strip("- *")
                    ]
                    logger.debug(
                        f"Chain 4 | Facts extracted: {json.dumps(atomic_facts)}"
                    )

                    fact_scores = []

                    # Build local context chunks used by the generator
                    # We evaluate against the full provided context so the NLI doesn't
                    # break when a fact is synthesized from multiple consecutive sentences
                    chunks = [full_context]

                    # 2. Score each fact
                    for fact in atomic_facts:
                        best_fact_score = 0.0

                        # IGNORING HUGGINGFACE WARNING:
                        # "You seem to be using the pipelines sequentially on GPU..."
                        # We intentionally use a raw sequential loop instead of a HuggingFace
                        # Dataset object. Passing a Dataset batch would cause 4096-token attention
                        # matrices to calculate simultaneously, instantly OOM crashing the 5GB VRAM limit.
                        # Squeezing it individually is a required constraint.
                        for chunk in chunks:
                            nli_input = {"text": chunk, "text_pair": fact}

                            # Use top_k=None to get scores for all labels
                            nli_result = self.nli_judge(
                                nli_input, truncation=True, max_length=512, top_k=None
                            )

                            # Handle pipeline output list format
                            if isinstance(nli_result, list) and isinstance(
                                nli_result[0], list
                            ):
                                scores = nli_result[0]
                            elif isinstance(nli_result, list):
                                scores = nli_result
                            else:
                                scores = [nli_result]

                            entailment_prob = 0.0
                            for s in scores:
                                # Extract entailment score (handles mDeBERTa 'entailment' and MiniCheck '1')
                                lbl = str(s.get("label", "")).strip().lower()
                                if lbl == "entailment" or lbl == "1":
                                    entailment_prob = float(s.get("score", 0.0))
                                    break
                                elif lbl == "contradiction" or lbl == "0":
                                    # Fallback if entailment not explicitly output but contradiction is highly scored
                                    entailment_prob_fallback = max(
                                        0.0, 1.0 - float(s.get("score", 0.0))
                                    )
                                    if entailment_prob == 0.0:
                                        entailment_prob = entailment_prob_fallback

                            if entailment_prob > best_fact_score:
                                best_fact_score = entailment_prob

                        fact_scores.append(best_fact_score)

                    # 3. Aggregate faithfulness & hallucination metrics
                    if fact_scores:
                        avg_faithfulness = sum(fact_scores) / len(fact_scores)
                        worst_hallucination = min(fact_scores)
                    else:
                        avg_faithfulness = 1.0
                        worst_hallucination = 1.0

                    logger.debug(
                        f"Chain 4 | Mean Faithfulness={avg_faithfulness:.3f}, Worst Fact={worst_hallucination:.3f}"
                    )
                    nli_label = f"Faithfulness: {avg_faithfulness:.2f}, Worst Fact: {worst_hallucination:.2f}"

                    # 4. Refuse based on user-defined thresholds
                    if avg_faithfulness >= 0.8 and worst_hallucination >= 0.5:
                        is_refusal = False
                        break  # Pass NLI, exit retry loop
                    else:
                        is_refusal = True  # Fail, will retry if attempts left

                except Exception as e:
                    # Fallback to refusal if hallucination detection breaks
                    is_refusal = True
                    nli_label = f"NLI Error: {str(e)}"
                    logger.error(f"Chain 4 | NLI Error: {str(e)}")
                    break  # Don't retry on system errors
            else:
                # Bypass NLI check entirely
                is_refusal = False
                nli_label = "NLI Check Disabled"
                break  # Valid implicitly, exit retry loop

        answer_type = ANSWER_TYPE_REFUSAL if is_refusal else ANSWER_TYPE_DEFINITIVE

        if is_refusal:
            citations = ""
            if combined and policy_found:
                # Near-Miss
                citations = "\n\nSources: " + "; ".join(
                    [
                        f"{meta.get('doc_name', 'N/A')} §{meta.get('clause_number', 'N/A')} ({meta.get('section', 'N/A')}), p.{meta.get('page', 'N/A')}"
                        for _, meta, _ in combined[:2]
                    ]
                )
            final_answer = f"{REFUSAL_STRING}\nExplanation: The query's specifics could not be confidently verified against the policy document. (Reason: {nli_label}){citations}"
            policy_found = False  # PRD test suite expects this for refusals to pass out-of-scope rules
        else:
            final_answer = raw_answer
            if hasattr(pred, "sources") and pred.sources:
                final_answer = f"{final_answer}\n\nSources: {pred.sources}"

        final_prediction = dspy.Prediction(
            answer=final_answer,
            policy_found=policy_found,
            answer_type=answer_type,
            retrieved_chunks=[
                {"text": r[0], "metadata": r[1], "score": float(r[2])}
                for r in combined[: self.top_k]
            ],
            max_relevance_score=float(max_score),
            nli_label=nli_label,
        )

        # Save to cache
        if settings.use_semantic_cache:
            self._cache[question.strip().lower()] = final_prediction
        return final_prediction


def get_rag_system(vectorstore: Any) -> PolicyRAG:
    """Factory function to create a configured PolicyRAG instance."""
    return PolicyRAG(vectorstore)


def setup_dspy():
    """Configure DSPy with the LLM backend specified in environment variables."""
    lm = dspy.LM(
        model=settings.dspy_lm_model,
        api_base=settings.dspy_api_base,
        temperature=0.0,  # Deterministic for grounded policy answers
        top_p=0.80,  # Qwen default; relaxed since temp=0 dominates
        top_k=20,  # Qwen default range; prevents repetition traps
        max_tokens=512,  # Safe for 5GB VRAM, enough for policy answers
        frequency_penalty=0.1,  # Reduces repetition in small models
        # presence_penalty=0.0, # Unsupported by litellm's ollama_chat
    )
    dspy.configure(lm=lm)
