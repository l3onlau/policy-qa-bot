import json
import dspy
from src.engine import PolicyRAG, setup_dspy
from src.vectorstore import PolicyVectorStore
from src.utils import ensure_ingested
from dotenv import load_dotenv


class AnswerCorrectness(dspy.Signature):
    """
    You are a strict, objective grader evaluating an AI's answer against a required behavior.

    CRITICAL GRADING RULES:
    1. FACT CHECK: Read the 'expected_behavior'. If it requires a specific number, name, or fact, verify that exact detail exists in the 'generated_answer'. If it is missing, you MUST output FAIL.
    2. CONTRADICTION: If the 'generated_answer' says the opposite of the 'expected_behavior' (e.g., saying something is covered when it should be excluded), output FAIL.
    3. PASS CONDITION: Only output PASS if the generated answer clearly and correctly fulfills the exact instructions in the 'expected_behavior'.
    4. CONSISTENCY: Your 'verdict' MUST be consistent with your 'reason'. If your reason explains the answer is correct, you MUST output PASS. Never contradict yourself.
    5. LENIENT ON FORMAT: Focus on whether the SUBSTANCE of the answer is correct, not how it is phrased.
    """

    question = dspy.InputField()
    expected_behavior = dspy.InputField(
        desc="The exact logical behavior or facts the answer MUST contain."
    )
    generated_answer = dspy.InputField()
    verdict = dspy.OutputField(
        desc="Output EXACTLY 'PASS' or 'FAIL'. Must be consistent with your reason."
    )
    reason = dspy.OutputField(
        desc="A 1-sentence explanation of why it passed or failed."
    )


def rule_based_precheck(test: dict, pred: dspy.Prediction) -> tuple[str, str] | None:
    """
    Deterministic pre-check for hard-rule violations mandated by the PRD.
    Distinguishes between Out-of-Scope refusals and Near-Miss refusals.
    """
    test_type = test.get("type", "").lower()
    answer_lower = pred.answer.lower()

    is_out_of_scope = "out-of-scope" in test_type
    is_near_miss = "near-miss" in test_type

    # Rule 1a: Out-of-Scope queries should NOT find relevant policy documents
    if is_out_of_scope and pred.policy_found:
        return (
            "FAIL",
            "Out-of-Scope: Expected a complete refusal (policy_found=False) but model indicated policy was found.",
        )

    # Rule 1b: Near-Miss queries MUST find a policy to reference, even if refusing
    if is_near_miss and not pred.policy_found:
        return (
            "FAIL",
            "Near-Miss: Expected policy_found=True to reference related clauses, but none were found.",
        )

    # Rule 2: Mandatory Citation check (PRD Section 3.1)
    if "sources:" not in answer_lower:
        return ("FAIL", "Missing mandatory 'Sources:' citation block.")

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
                    expected_behavior=test["expected_behavior"],
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
