import dspy
from sentence_transformers import CrossEncoder
from transformers import BitsAndBytesConfig
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


class FaithfulnessJudge(dspy.Signature):
    """
    You are an insurance compliance auditor. Verify if the 'claim' is grounded.

    SPECIAL HANDLING:
    1. REFUSAL VALIDATION: If the claim starts with 'I cannot find a definitive answer...', it is FAITHFUL if the context truly does not contain a specific answer or is ambiguous.
    2. CITATION CHECK: Ensure the assistant's 'Sources' actually appear in the context headers/metadata.
    3. NO HALLUCINATION: If the claim adds terms not in the context, it is UNFAITHFUL (is_faithful=False).
    """

    context: str = dspy.InputField(
        desc="Policy wording segments including metadata (doc_name, section, page)."
    )
    claim: str = dspy.InputField(desc="The assistant's response and citations.")

    is_faithful: bool = dspy.OutputField(
        desc="True if the claim is a direct factual derivation or a valid PRD-mandated refusal. False if it fabricates terms or cites non-existent sections."
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

    question: str = dspy.InputField(desc="The user's insurance question.")
    search_queries: str = dspy.OutputField(
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

    CRITICAL INSTRUCTIONS:
    1. OUT-OF-SCOPE: If the question is unrelated to insurance (e.g., recipes, games), set can_answer=False and sources="None".
    2. EXCLUSION CHECK: You MUST check for 'Exclusions', 'Limitations', or 'General Conditions' that may override a coverage clause.
    3. CITATIONS: Use the pre-formatted citations provided in the context blocks.
    """

    context: str = dspy.InputField(
        desc="Relevant policy document chunks with pre-formatted citations."
    )
    question: str = dspy.InputField(desc="The user's specific insurance inquiry.")

    can_answer: bool = dspy.OutputField(
        desc="True only if context provides a definitive answer. False if out-of-scope, ambiguous, or if specific conditions/exclusions are mentioned but the full impact is unclear."
    )

    reasoning_and_answer: str = dspy.OutputField(
        desc="""If can_answer is True, provide a rigorous answer. 
        If can_answer is False, you MUST start with: 'I cannot find a definitive answer in the provided policy wording.' 
        Then, briefly explain the ambiguity or out-of-scope nature and list the closest related clauses if applicable."""
    )

    sources: str = dspy.OutputField(
        desc="The pre-formatted citations (e.g., 'Doc.pdf §Section, p.X') from the context, or 'None' if out-of-scope."
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
            settings.reranker_model,
            model_kwargs={
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                ),
            },
            max_length=1024,
        )

        self.faithfulness_judge = dspy.ChainOfThought(FaithfulnessJudge)

        self.top_k_rerank = settings.top_k_rerank
        self.relevance_threshold = 0.3

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
                answer=f"{REFUSAL_STRING}\nExplanation: The prompt does not appear to be an insurance question.\n\nSources: None",
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
            except Exception as e:
                logger.debug(f"Chain 1 | Reformulator error: {e}")

        # Multi-query retrieval: merge results from original + reformulated queries
        seen_texts = set()
        merged_docs = []
        merged_metas = []

        # Concurrent retrieval to hide I/O latency
        def _fetch_docs(q):
            try:
                return self.vectorstore.query(q, top_k_retrieve=settings.top_k_retrieve)
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
                answer=f"{REFUSAL_STRING}\nExplanation: No relevant documents were found in the vector space.\n\nSources: None",
                policy_found=False,
                answer_type=ANSWER_TYPE_REFUSAL,
                retrieved_chunks=[],
            )

        # ── Chain 2: Cross-Encoder Reranking + Relevance Gate ─────────────
        pairs = [[question, doc] for doc in merged_docs]

        # Throttle batch size to avoid quadratic attention memory spikes
        # (similar to the fix in chunk embedding)
        # Using batch_size=1 explicitly to avoid OOM on 6GB VRAM GPUs
        scores = self.reranker.predict(pairs, batch_size=1, show_progress_bar=False)

        combined = sorted(
            zip(merged_docs, merged_metas, scores),
            key=lambda x: x[2],
            reverse=True,
        )

        for i, (cd, cm, cs) in enumerate(combined[: self.top_k_rerank]):
            logger.debug(
                f"Chain 2 | Reranked #{i + 1} score={cs:.3f}, doc={cm.get('doc_name')}"
            )

        # Instead of applying relevance gating explicitly here on the aggregate chunk score,
        # we filter out low-relevance sentences during chunk distillation.
        # This prevents us from dropping chunks that contain a highly relevant sentence
        # but have a low overall aggregate score.

        # ── Chain 2.5: Post-Retrieval Chunk Distillation ──────────────────
        # Split large chunks into sentences, score them, and take only the most relevant
        distilled_pairs = []
        sentence_metadata = []

        if settings.use_chunk_distillation:
            for doc_text, meta, _ in combined[: self.top_k_rerank]:
                if meta.get("is_table"):
                    # Bypass sentence splitting for tables, keep intact
                    distilled_pairs.append([question, doc_text])
                    sentence_metadata.append({"text": doc_text, "meta": meta})
                    continue

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
            distilled_combined = combined[: self.top_k_rerank]

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
            if not settings.use_chunk_distillation:
                final_distilled = [
                    x
                    for x in combined[: self.top_k_rerank]
                    if x[2] >= self.relevance_threshold
                ]

            if not final_distilled:
                return dspy.Prediction(
                    answer=f"{REFUSAL_STRING}\nExplanation: The query's specifics could not be confidently verified against the policy document. (Reason: No relevant clauses matched your question.)\n\nSources: None",
                    policy_found=False,
                    answer_type=ANSWER_TYPE_REFUSAL,
                    retrieved_chunks=[],
                )

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

        max_retries = (
            settings.max_retries if settings.use_faithfulness_judge_check else 0
        )
        refusal_label = ""
        is_refusal = False
        max_score = combined[0][2] if combined else -999.0
        policy_found = bool(combined)

        logger.debug(
            f"Chain 3 | Generator Input:\ncontext: {full_context}\nquestion: {question}"
        )
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(
                    f"⚠️ LLM Check Failed (Faithfulness fail). Retrying generation (Attempt {attempt})..."
                )
                # Fallback Routing: prompt with explicit feedback
                retry_question = f"{question}\n\nWARNING: Your previous answer contained unverified facts. You MUST stick strictly to the context."
                logger.debug(f"Chain 3 | Fallback retry attempt {attempt}")
                pred = self.generator(context=full_context, question=retry_question)
            else:
                pred = self.generator(context=full_context, question=question)

            raw_answer = pred.reasoning_and_answer
            raw_sources = getattr(pred, "sources", "None")

            # CRITICAL FIX: Concatenate answer and sources for the judge
            full_claim_for_judge = f"{raw_answer}\n\nSources: {raw_sources}"

            logger.debug(f"Chain 3 | Generator Output:\n{full_claim_for_judge}")

            can_answer = getattr(pred, "can_answer", True)
            if isinstance(can_answer, str):
                can_answer = can_answer.strip().lower() == "true"

            if not can_answer or REFUSAL_STRING.lower() in raw_answer.lower():
                is_refusal = True
                refusal_label = "Model Refusal"
                break

            # ── Chain 4: LLM as a judge Verification ──────────────────────────
            if settings.use_faithfulness_judge_check:
                try:
                    # Pass the concatenated claim
                    pred_judge = self.faithfulness_judge(
                        context=full_context, claim=full_claim_for_judge
                    )

                    # Simple boolean check from the 4B model
                    is_faithful_verdict = getattr(pred_judge, "is_faithful", False)

                    # Ensure it's a bool (in case dspy/model returns a truthy string)
                    if isinstance(is_faithful_verdict, str):
                        is_faithful_verdict = (
                            is_faithful_verdict.strip().lower() == "true"
                        )

                    logger.debug(
                        f"Chain 4 | Judge Verdict (is_faithful): {is_faithful_verdict}"
                    )

                    if is_faithful_verdict:
                        # Success (Faithful)
                        is_refusal = False
                        break  # Pass Faithfulness, exit retry loop
                    else:
                        # Failure
                        logger.debug(
                            "Chain 4 | Rejected by Judge: claim not strictly supported."
                        )
                        is_refusal = True  # Fail, will retry if attempts left
                        refusal_label = "Judge Failure"

                except Exception as e:
                    logger.error(f"Chain 4 | Error inside FaithfulnessJudge: {e}")
                    break  # Don't retry on system errors
            else:
                # Bypass Faithfulness Judge check entirely
                is_refusal = False
                refusal_label = "Faithfulness Judge Check Disabled"
                break  # Valid implicitly, exit retry loop

        answer_type = ANSWER_TYPE_REFUSAL if is_refusal else ANSWER_TYPE_DEFINITIVE

        if is_refusal:
            cleaned_answer = (
                re.compile(re.escape(REFUSAL_STRING), re.IGNORECASE)
                .sub("", raw_answer)
                .strip()
            )

            if "model refusal" in refusal_label.lower():
                # The LLM set can_answer=False and explained it.
                if not cleaned_answer.startswith("Explanation:"):
                    cleaned_answer = f"Explanation: {cleaned_answer}"
                final_answer = f"{REFUSAL_STRING}\n{cleaned_answer}"
            else:
                # Judge Failure or other refusal: preserve the generated explanation and closest clauses
                final_answer = f"{REFUSAL_STRING}\nExplanation: The query's specifics could not be confidently verified against the policy document. (Reason: {refusal_label})\nContext Analysis: {cleaned_answer}"

            # CRITICAL FIX: Extract and append sources universally for all refusals
            sources_val = getattr(pred, "sources", "").strip()
            if (
                sources_val
                and sources_val.lower() != "none"
                and "Sources:" not in final_answer
            ):
                final_answer = f"{final_answer}\n\nSources: {sources_val}"

            if "Sources:" not in final_answer:
                final_answer = f"{final_answer}\n\nSources: None"

            policy_found = False
        else:
            final_answer = raw_answer
            sources_val = getattr(pred, "sources", "").strip()
            if (
                sources_val
                and sources_val.lower() != "none"
                and "Sources:" not in final_answer
            ):
                final_answer = f"{final_answer}\n\nSources: {sources_val}"
            elif "Sources:" not in final_answer:
                final_answer = f"{final_answer}\n\nSources: None"

        final_prediction = dspy.Prediction(
            answer=final_answer,
            policy_found=policy_found,
            answer_type=answer_type,
            retrieved_chunks=[
                {"text": r[0], "metadata": r[1], "score": float(r[2])}
                for r in combined[: self.top_k_rerank]
            ],
            max_relevance_score=float(max_score),
            refusal_label=refusal_label,
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
        max_tokens=1024,  # Safe for 6GB VRAM, enough for policy answers
        frequency_penalty=0.1,  # Reduces repetition in small models
        # presence_penalty=0.0, # Unsupported by litellm's ollama_chat
    )
    dspy.configure(lm=lm)
