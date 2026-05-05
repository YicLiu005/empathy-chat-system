import json

from sentiment_analyzer import analyze_sentiment


def print_result(result):
    print("\nAnalysis Result")
    print("=" * 60)

    print(f"Sentiment: {result.get('sentiment', 'unknown')}")
    print(f"Confidence: {result.get('confidence', 0.0)}")
    print(f"Sarcasm Detected: {result.get('sarcasm_detected', False)}")

    print("\nReason:")
    print(result.get("reason", ""))

    evidence = result.get("evidence", [])
    if evidence:
        print("\nEvidence:")
        for i, item in enumerate(evidence, start=1):
            print(f"{i}. {item}")

    retrieved_context = result.get("retrieved_context", [])
    if retrieved_context:
        print("\nRetrieved Context:")
        for i, ctx in enumerate(retrieved_context, start=1):
            print("-" * 60)
            print(f"Rank: {i}")
            print(f"Source: {ctx.get('source_file', 'unknown')}")
            print(f"Chunk ID: {ctx.get('chunk_id', 'unknown')}")
            print(f"Score: {ctx.get('score', 0.0)}")
            print(ctx.get("text", "")[:500])

    print("=" * 60)


def main():
    print("Sentiment RAG System")
    print("Type 'exit' to quit.")
    print()

    while True:
        user_text = input("Enter a sentence/comment: ").strip()

        if user_text.lower() == "exit":
            print("Goodbye!")
            break

        if not user_text:
            print("Input cannot be empty.\n")
            continue

        try:
            result = analyze_sentiment(
                user_text=user_text,
                use_rag=True,
                top_k=5,
            )

            print_result(result)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()