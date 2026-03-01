import json
from typing import Literal
import dspy
from src.engine import PolicyRAG, setup_dspy
from src.vectorstore import PolicyVectorStore
from src.utils import ensure_ingested
from dotenv import load_dotenv


class AnswerCorrectness(dspy.Signature):
    """
    You are a strict, objective grader evaluating an AI's 'generated_answer' against an 'expected_answer'.

    CRITICAL RULES:
    1. CONTRADICTION = FAIL: If generated_answer says 'unknown' when expected_answer has a fact (or vice versa).
    2. YES/NO MISMATCH = FAIL: Any flip in coverage status is an automatic FAIL.
    3. CITATION HYGIENE = FAIL: If the generated_answer includes citations for a question it admitted it couldn't answer (Out-of-Scope), or if citations are missing when they should be present.
    4. VERBATIM REFUSAL: The generated_answer MUST include the specific PRD refusal string if it cannot answer.
    """

    question: str = dspy.InputField()
    expected_answer: str = dspy.InputField(desc="The ground truth logic and citations.")
    generated_answer: str = dspy.InputField(desc="The AI output to evaluate.")

    verdict: Literal["PASS", "FAIL"] = dspy.OutputField(
        desc="EXACTLY 'PASS' or 'FAIL'."
    )
    reason: str = dspy.OutputField(
        desc="Concise explanation focusing on contradictions or citation errors."
    )


def rule_based_precheck(test: dict, pred: dspy.Prediction) -> tuple[str, str] | None:
    answer = pred.answer
    is_refusal = (
        "I cannot find a definitive answer in the provided policy wording." in answer
    )

    # Rule 1: Out-of-scope questions must be refused
    if test.get("type", "unknown") == "out-of-scope":
        if is_refusal:
            return "PASS", "Properly refused out-of-scope question according to PRD."
        else:
            return "FAIL", "Failed to clearly refuse out-of-scope question."

    # Rule 2: In-domain or near-miss questions that are answered MUST contain citations
    if not is_refusal:
        if "Sources:" not in answer:
            return "FAIL", "Missing mandatory 'Sources:' citation required by PRD."

    # For anything else, fall back to LLM judge
    return None


def run_evals():
    """Run the evaluation suite using a 2-layer hybrid evaluator (Rules + LLM)."""
    load_dotenv()
    setup_dspy()

    print("🚀 Initializing Vectorstore and ensuring data is ingested...")
    vectorstore = PolicyVectorStore()
    ensure_ingested(vectorstore)
    rag = PolicyRAG(vectorstore)

    llm_judge = dspy.ChainOfThought(AnswerCorrectness)

    with open("tests.json", "r") as f:
        tests = json.load(f)["test_set"]

    results = []
    print(f"\n🧪 Running {len(tests)} Tests...\n")

    for test in tests:
        print(f"Testing ID {test['id']}: {test['question'][:50]}...")
        pred = rag(question=test["question"])

        # Layer 1: Rule-based pre-check (deterministic PRD rules)
        rule_result = rule_based_precheck(test, pred)

        if rule_result:
            verdict, reason = rule_result
            eval_method = "rule_based"
            evaluator_reasoning = (
                reason if verdict == "PASS" else "Failed deterministic PRD rule check."
            )
        else:
            # Layer 2: LLM judge for factual accuracy check
            try:
                eval_result = llm_judge(
                    question=test["question"],
                    expected_answer=test["expected_answer"],
                    generated_answer=pred.answer,
                )
                verdict = "PASS" if "PASS" in eval_result.verdict.upper() else "FAIL"
                reason = eval_result.reason
                eval_method = "llm_judge"
                evaluator_reasoning = eval_result.reasoning
            except Exception as e:
                verdict = "FAIL"
                reason = f"LLM Judge encountered an error: {e}"
                eval_method = "error"
                evaluator_reasoning = "Error during generation."

        results.append(
            {
                "id": test["id"],
                "question": test["question"],
                "answer": pred.answer,
                "verdict": verdict,
                "reason": reason,
                "eval_method": eval_method,
                "evaluator_reasoning": evaluator_reasoning,
                # "retrieved_chunks": pred.retrieved_chunks # Optional: for debugging
            }
        )

    with open("eval_report.json", "w") as f:
        json.dump(results, f, indent=2)

    passed = len([r for r in results if r.get("verdict") == "PASS"])
    total = len(tests)
    methods = {}
    for r in results:
        m = r.get("eval_method", "unknown")
        methods[m] = methods.get(m, 0) + 1

    print(f"\n📊 EVALUATION COMPLETE: {passed}/{total} PASSED")
    print(f"   Eval methods used: {methods}")


if __name__ == "__main__":
    run_evals()
