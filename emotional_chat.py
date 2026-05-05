from typing import Dict, List, Any

from sentiment_analyzer import analyze_sentiment, call_gemini_text


def build_response_strategy(sentiment: str) -> str:
    sentiment = (sentiment or "unknown").lower().strip()

    if sentiment == "positive":
        return "Maintain a positive tone, acknowledge the user's good experience, and encourage them to share more."

    if sentiment == "negative":
        return "First acknowledge and validate the user's frustration, then offer calm, positive, and practical support."

    if sentiment == "mixed":
        return "Acknowledge both the positive and negative parts of the user's message, and help separate the issues clearly."

    if sentiment == "neutral":
        return "Keep the response objective and friendly, and ask for more context if needed."

    return "Stay supportive and open-ended, understand the user's message first, and then respond based on the context."


def build_emotional_chat_prompt(
    user_text: str,
    sentiment_result: Dict[str, Any],
    history: List[Dict[str, str]],
) -> str:
    sentiment = sentiment_result.get("sentiment", "unknown")
    confidence = sentiment_result.get("confidence", 0.0)
    reason = sentiment_result.get("reason", "")
    sarcasm_detected = sentiment_result.get("sarcasm_detected", False)
    evidence = sentiment_result.get("evidence", [])
    retrieved_context = sentiment_result.get("retrieved_context", [])

    recent_history = history[-8:]

    history_text = ""
    for msg in recent_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        history_text += f"{role}: {content}\n"

    if evidence:
        evidence_text = ""
        for i, item in enumerate(evidence, start=1):
            evidence_text += f"{i}. {item}\n"
    else:
        evidence_text = "No explicit evidence provided.\n"

    if retrieved_context:
        retrieved_text = ""
        for i, ctx in enumerate(retrieved_context[:3], start=1):
            source_file = ctx.get("source_file", "unknown")
            chunk_id = ctx.get("chunk_id", "unknown")
            score = ctx.get("score", 0.0)
            text = ctx.get("text", "")

            retrieved_text += (
                f"[Retrieved Knowledge {i}]\n"
                f"Source: {source_file}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Score: {score}\n"
                f"Content: {text[:500]}\n\n"
            )
    else:
        retrieved_text = "No retrieved knowledge was found.\n"

    prompt = f"""
You are an emotionally supportive AI chat assistant.

The user is having a multi-turn conversation with you.

Your task is to generate a natural assistant reply based on:
1. the user's latest message,
2. the conversation history,
3. the detected current sentiment,
4. the retrieved knowledge used to support the sentiment analysis.

Important rules:
1. Reply naturally and supportively.
2. Do not mention API, JSON, model, prompt, backend, or implementation details.
3. Keep the main reply concise and helpful.
4. After the main reply, add one English parenthetical note.
5. The English parenthetical note must include:
   - the user's likely current emotion,
   - why the AI thinks the user may feel that way,
   - what retrieved knowledge or similar examples support this judgment,
   - what response strategy the AI should use next.
6. The final output must follow this structure:

Main reply to the user.

(Emotion analysis: The user may currently feel ...; Reason: ...; Knowledge-base support: ...; Response strategy: ...)

Conversation History:
{history_text}

Latest User Message:
{user_text}

Detected Current Sentiment:
- sentiment: {sentiment}
- confidence: {confidence}
- sarcasm_detected: {sarcasm_detected}
- reason: {reason}

Evidence from sentiment analyzer:
{evidence_text}

Retrieved Knowledge:
{retrieved_text}

Now write the assistant reply.
""".strip()

    return prompt


def generate_emotional_chat_reply(
    user_text: str,
    sentiment_result: Dict[str, Any],
    history: List[Dict[str, str]],
) -> str:
    prompt = build_emotional_chat_prompt(
        user_text=user_text,
        sentiment_result=sentiment_result,
        history=history,
    )

    reply = call_gemini_text(prompt).strip()

    sentiment = sentiment_result.get("sentiment", "unknown")
    reason = sentiment_result.get("reason", "")
    evidence = sentiment_result.get("evidence", [])
    retrieved_context = sentiment_result.get("retrieved_context", [])

    if "(Emotion analysis:" not in reply:
        evidence_text = "; ".join(evidence[:2]) if evidence else "The user's message contains clear emotional cues."

        if retrieved_context:
            top_ctx = retrieved_context[0]
            source_file = top_ctx.get("source_file", "unknown")
            chunk_id = top_ctx.get("chunk_id", "unknown")
            rag_basis = f"The knowledge base retrieved a similar example from {source_file}, chunk {chunk_id}."
        else:
            rag_basis = "No highly relevant knowledge-base example was retrieved for this turn."

        strategy = build_response_strategy(sentiment)

        reply = (
            f"{reply}\n\n"
            f"(Emotion analysis: The user may currently be experiencing a {sentiment} emotional state; "
            f"Reason: {reason or evidence_text}; "
            f"Knowledge-base support: {rag_basis}; "
            f"Response strategy: {strategy})"
        )

    return reply


def process_emotional_chat_turn(
    user_text: str,
    history: List[Dict[str, str]],
    top_k: int = 5,
) -> Dict[str, Any]:
    user_text = user_text.strip()

    if not user_text:
        raise ValueError("user_text cannot be empty.")

    sentiment_result = analyze_sentiment(
        user_text=user_text,
        use_rag=True,
        top_k=top_k,
    )

    assistant_reply = generate_emotional_chat_reply(
        user_text=user_text,
        sentiment_result=sentiment_result,
        history=history,
    )

    return {
        "reply": assistant_reply,
        "sentiment_result": sentiment_result,
    }


if __name__ == "__main__":
    chat_history = []

    print("Emotional Chat Test")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ").strip()

        if user_input.lower() == "exit":
            break

        if not user_input:
            print("Input cannot be empty.\n")
            continue

        chat_history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            result = process_emotional_chat_turn(
                user_text=user_input,
                history=chat_history,
                top_k=5,
            )

            assistant_reply = result["reply"]

            chat_history.append(
                {
                    "role": "assistant",
                    "content": assistant_reply,
                }
            )

            print(f"\nAssistant: {assistant_reply}\n")

        except Exception as e:
            print(f"Error: {e}")