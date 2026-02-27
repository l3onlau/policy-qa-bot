"""
Policy Q&A Bot — Entry Point

Interactive CLI for querying insurance policy documents using a
retrieval-augmented generation (RAG) pipeline.
"""

from src.engine import get_rag_system, setup_dspy
from src.vectorstore import PolicyVectorStore
from src.utils import ensure_ingested


def main():
    """Launch the interactive policy Q&A assistant."""

    print("🚀 Initializing Policy Q&A Bot...")
    vectorstore = PolicyVectorStore()
    setup_dspy()
    ensure_ingested(vectorstore)

    rag_system = get_rag_system(vectorstore)

    print("\n--- Policy Assistant Online (Type 'exit' to quit) ---")

    try:
        while True:
            query = input("\n👤 Question: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            print("🤖 Analyzing policy documentation...")
            response = rag_system(question=query)
            print(f"\n📝 Answer:\n{response.answer}")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Shutting down.")


if __name__ == "__main__":
    main()
