from typing import List, Any


def format_contexts(contexts: List[Any]) -> str:
    """
    Format retrieved RAG contexts into prompt text.

    Each context is expected to have:
    - chunk_id
    - source_file
    - text
    - score

    This function also supports plain strings for compatibility.
    """

    if not contexts:
        return "No relevant knowledge was retrieved."

    formatted_blocks = []

    for i, ctx in enumerate(contexts, start=1):
        if isinstance(ctx, str):
            formatted_blocks.append(
                f"[Context {i}]\n"
                f"{ctx}"
            )
        else:
            chunk_id = getattr(ctx, "chunk_id", f"context_{i}")
            source_file = getattr(ctx, "source_file", "unknown")
            score = getattr(ctx, "score", 0.0)
            text = getattr(ctx, "text", str(ctx))

            formatted_blocks.append(
                f"[Context {i}]\n"
                f"Source: {source_file}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Retrieval Score: {score:.4f}\n"
                f"Content:\n{text}"
            )

    return "\n\n".join(formatted_blocks)


def build_sentiment_prompt(
    user_text: str,
    contexts: List[Any] | None = None,
    use_rag: bool = True,
) -> str:
    """
    Build the final prompt for sentiment analysis.
    """

    user_text = user_text.strip()

    if not user_text:
        raise ValueError("user_text cannot be empty.")

    contexts = contexts or []

    if use_rag:
        context_text = format_contexts(contexts)
    else:
        context_text = "RAG is disabled. Analyze only based on the user text."

    prompt = f"""
You are a sentiment analysis assistant.

Your task is to analyze the true sentiment of the user text.

You must classify the sentiment into exactly one of the following labels:

- positive
- negative
- neutral
- mixed

Definitions:
- positive: The user expresses satisfaction, approval, happiness, or praise.
- negative: The user expresses dissatisfaction, criticism, anger, frustration, disappointment, or complaint.
- neutral: The user mainly states facts without clear emotional attitude.
- mixed: The user expresses both positive and negative opinions.

Important rules:
1. Do not judge sentiment only by individual positive or negative words.
2. Pay special attention to sarcasm, irony, exaggeration, and contrast.
3. If a sentence contains positive words but describes a bad situation, it may be negative sarcasm.
4. If the retrieved knowledge is relevant, use it as supporting evidence.
5. If the retrieved knowledge is not relevant, do not force it into the reasoning.
6. The final answer must be valid JSON only.
7. Do not output markdown.
8. Do not add explanations outside the JSON object.
9. The confidence score is only a rough model estimate.
10. Do not always output the same confidence value.
11. The backend code may recalibrate confidence after the model output.

Confidence calibration rules:
- Use 0.90 to 1.00 only when the sentiment is very explicit and directly supported by strong evidence in the user text.
- Use 0.80 to 0.89 when the sentiment is clear but requires some interpretation, such as sarcasm, irony, or contrast.
- Use 0.65 to 0.79 when the sentiment is likely but not fully explicit.
- Use 0.50 to 0.64 when the sentiment is ambiguous, weak, or highly context-dependent.
- Use below 0.50 only when the sentiment is very uncertain.
- For neutral statements, avoid giving extremely high confidence unless the text is clearly factual and emotionless.
- For sarcasm, reduce confidence slightly if the judgment depends heavily on interpretation.

User Text:
{user_text}

Retrieved Knowledge:
{context_text}

Please return the result in this exact JSON format:

{{
  "sentiment": "positive | negative | neutral | mixed",
  "confidence": 0.0,
  "reason": "One short sentence explaining the sentiment.",
  "sarcasm_detected": true,
  "evidence": [
    "Short evidence 1",
    "Short evidence 2"
  ]
}}
""".strip()

    return prompt


def build_no_rag_sentiment_prompt(user_text: str) -> str:
    """
    Build a sentiment analysis prompt without RAG.
    """

    return build_sentiment_prompt(
        user_text=user_text,
        contexts=[],
        use_rag=False,
    )


if __name__ == "__main__":
    sample_text = "Great, another update that breaks everything."

    sample_contexts = [
        "Text: Great, another update that breaks everything.\n"
        "Label: sarcastic\n"
        "Meaning: The user is unhappy because the update caused problems.\n"
        "Sentiment Hint: negative"
    ]

    prompt = build_sentiment_prompt(
        user_text=sample_text,
        contexts=sample_contexts,
        use_rag=True,
    )

    print(prompt)