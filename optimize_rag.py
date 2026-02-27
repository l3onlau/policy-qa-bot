import dspy
from dotenv import load_dotenv

# Import the specific module architecture we want to optimize
from src.engine import PolicySignature, setup_dspy


def build_unrelated_dataset():
    """
    Builds a synthetic dataset completely unrelated to insurance policies.
    The goal is purely to teach the LLM the *pattern* of verifiable citation,
    strict adherence to the provided context, and refusal when facts are missing.
    Domain: Intergalactic Space Fleet Regulations
    """

    examples = [
        # Pass example (Definitive Answer)
        dspy.Example(
            context="VERIFIED DOCUMENT SEGMENT:\nAll Class-4 starships must disengage hyperdrive within 50,000 km of a major space station to prevent gravimetric shearing. Violations incur a 500-credit fine.\nCitation To Use: Fleet Ops Manual §4.1 (Navigation), p.12\n[REMINDER: Use ONLY this segment text for your answer. DO NOT hallucinate rules.]\n",
            question="What is the fine for engaging hyperdrive near a space station?",
            reasoning_and_answer="Based on the context, a Class-4 starship violating the rule to disengage hyperdrive within 50,000 km of a major space station incurs a 500-credit fine.",
            sources="Fleet Ops Manual §4.1 (Navigation), p.12",
        ).with_inputs("context", "question"),
        # Pass example (Definitive Answer - Different wording)
        dspy.Example(
            context="VERIFIED DOCUMENT SEGMENT:\nCrew members of alien descent (Non-Terran) are granted an additional 4 cycles of solar rest per standard orbit, provided they submit form X-99 to the bio-medical bay.\nCitation To Use: HR Directives §8.2 (Leave), p.44\n[REMINDER: Use ONLY this segment text for your answer. DO NOT hallucinate rules.]\n",
            question="Do aliens get extra vacation time?",
            reasoning_and_answer="Yes, according to the directives, crew members of alien descent (Non-Terran) receive an additional 4 cycles of solar rest per standard orbit, but they must submit form X-99 to the bio-medical bay to claim it.",
            sources="HR Directives §8.2 (Leave), p.44",
        ).with_inputs("context", "question"),
        # Pass example (Definitive Answer - Multiple Constraints)
        dspy.Example(
            context="VERIFIED DOCUMENT SEGMENT:\nThe primary plasma weapon may only be fired if: 1) Shields are below 20%, 2) The captain issues a Code Red, and 3) The target is a confirmed hostile dreadnought.\nCitation To Use: Combat Protocol §9.9 (Weapons), p.88\n[REMINDER: Use ONLY this segment text for your answer. DO NOT hallucinate rules.]\n",
            question="Can I fire the plasma weapon at a small fighter?",
            reasoning_and_answer="No. The rules explicitly state the primary plasma weapon may only be fired if the target is a confirmed hostile dreadnought, along with two other conditions (shields below 20% and a Code Red issued by the captain).",
            sources="Combat Protocol §9.9 (Weapons), p.88",
        ).with_inputs("context", "question"),
    ]
    return examples


class ExactCitationMetric:
    """
    Metric to ensure the LLM outputs the answer and precisely copies the provided citation.
    """

    def __call__(self, example, pred, trace=None):
        # 1. Check if the required source is in the output sources
        expected_source = example.sources.lower().strip()
        pred_source = pred.sources.lower().strip() if hasattr(pred, "sources") else ""
        has_citation = expected_source in pred_source

        # 2. Check if the answer isn't empty and has reasonable length
        pred_ans = pred.reasoning_and_answer
        has_answer = len(pred_ans) > 20

        # Return boolean score
        return bool(has_citation and has_answer)


def run_optimization():
    print("🚀 Initializing DSPy and loading Space Fleet dataset...")
    load_dotenv()
    setup_dspy()

    trainset = build_unrelated_dataset()
    metric = ExactCitationMetric()

    print("🧠 Starting BootstrapFewShot optimization...")
    # BootstrapFewShot compiles the prompt by running the examples through the signature
    # and keeping the traces that pass the metric.
    from dspy.teleprompt import BootstrapFewShot

    teleprompter = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=3,  # Keep it small to save context window
        max_labeled_demos=3,
    )

    # We compile ONLY the PolicySignature (the final generator layer)
    # We use a dummy module just to wrap the signature for compilation
    class DummyGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_answer = dspy.ChainOfThought(PolicySignature)

        def forward(self, context, question):
            return self.generate_answer(context=context, question=question)

    dummy_program = DummyGenerator()

    print("⏳ Compiling prompt. This will invoke the LLM several times...")
    compiled_generator = teleprompter.compile(dummy_program, trainset=trainset)

    save_path = "compiled_rag.json"
    print(f"💾 Saving compiled prompt weights to {save_path}...")
    compiled_generator.generate_answer.save(save_path)

    print(
        "✅ Optimization complete. The PolicyRAG engine will now load these weights automatically."
    )


if __name__ == "__main__":
    run_optimization()
