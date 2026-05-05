import json
import re
from typing import Any, Dict, List

from retriever import SimpleMarkdownRetriever, RetrievedChunk
from sentiment_analyzer import call_gemini_text


# =========================
# Global Retriever Cache
# =========================

_need_retriever_instance: SimpleMarkdownRetriever | None = None


def get_need_retriever() -> SimpleMarkdownRetriever:
    """
    Create and cache a retriever for user need and empathy strategy knowledge.
    """

    global _need_retriever_instance

    if _need_retriever_instance is None:
        _need_retriever_instance = SimpleMarkdownRetriever(
            knowledge_base_dir="knowledge_base",
            chunk_size=350,
            chunk_overlap=60,
        )

    return _need_retriever_instance


# =========================
# Context Formatting
# =========================

def format_need_contexts(contexts: List[RetrievedChunk]) -> str:
    """
    Format retrieved user-need / empathy-strategy contexts into prompt text.
    """

    if not contexts:
        return (
            "No external need or empathy strategy knowledge was retrieved. "
            "Analyze only based on the conversation history, user text, and emotion analysis result."
        )

    blocks = []

    for i, ctx in enumerate(contexts, start=1):
        blocks.append(
            f"[Context {i}]\n"
            f"Source: {ctx.source_file}\n"
            f"Chunk ID: {ctx.chunk_id}\n"
            f"Retrieval Score: {ctx.score:.4f}\n"
            f"Content:\n{ctx.text}"
        )

    return "\n\n".join(blocks)


def format_chat_history(
    chat_history: List[Dict[str, str]] | None,
    max_turns: int = 6,
) -> str:
    """
    Format recent chat history for need detection.

    Only recent turns are used to avoid making the prompt too long.
    """

    if not chat_history:
        return "No previous conversation history."

    recent_history = chat_history[-max_turns:]

    lines = []

    for msg in recent_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip()

        if content:
            lines.append(f"{role}: {content}")

    if not lines:
        return "No previous conversation history."

    return "\n".join(lines)


# =========================
# Prompt
# =========================

def build_need_detection_prompt(
    user_text: str,
    emotion_result: Dict[str, Any],
    contexts: List[RetrievedChunk],
    chat_history: List[Dict[str, str]] | None = None,
    use_rag: bool = True,
) -> str:
    """
    Build prompt for user need detection.
    """

    user_text = user_text.strip()

    if not user_text:
        raise ValueError("user_text cannot be empty.")

    context_text = format_need_contexts(contexts)
    history_text = format_chat_history(chat_history)

    sentiment = emotion_result.get("sentiment", "unknown")
    confidence = emotion_result.get("confidence", 0.0)
    reason = emotion_result.get("reason", "")
    sarcasm_detected = emotion_result.get("sarcasm_detected", False)
    evidence = emotion_result.get("evidence", [])

    rag_status = (
        "RAG is enabled. Use the retrieved knowledge when it is relevant."
        if use_rag
        else "RAG is disabled. Do not assume any external knowledge was retrieved."
    )

    prompt = f"""
You are a user-need detection assistant for multi-turn empathetic dialogue.

Your task is to identify what kind of support the user currently needs.

The system has already analyzed the user's current emotion.
Now you must infer the user's support need by considering:
1. the latest user message,
2. the recent conversation history,
3. the current emotion analysis result,
4. retrieved knowledge if RAG is enabled.

Choose exactly one user_need from the following labels:

- emotional_validation
- reassurance
- practical_guidance
- emotional_regulation
- action_planning
- clarification

Definitions:
- emotional_validation: The user mainly needs their feelings to be recognized, accepted, and understood.
- reassurance: The user feels anxious, insecure, worried, or uncertain and needs calming encouragement.
- practical_guidance: The user wants concrete advice, a solution, or help solving a problem.
- emotional_regulation: The user is overwhelmed, angry, panicked, or emotionally flooded and needs help calming down.
- action_planning: The user is ready to act and needs structured next steps.
- clarification: The user's need is unclear, and the assistant should ask a gentle follow-up question.

Important rules:
1. Do not choose a need based only on sentiment label.
2. Focus on what kind of support the user is implicitly or explicitly asking for.
3. Use recent conversation history to resolve short or ambiguous messages.
4. If the user expresses pain but does not ask for advice, emotional_validation is usually better than practical_guidance.
5. If the user asks "what should I do" or "how can I fix this", practical_guidance or action_planning may be better.
6. If the user sounds overwhelmed or unable to calm down, choose emotional_regulation.
7. If the user is worried about future outcomes, choose reassurance.
8. If the latest message is short but continues a previous planning discussion, choose action_planning rather than clarification.
9. If the latest message is short but continues a previous advice-seeking discussion, choose practical_guidance rather than clarification.
10. Choose clarification only when the user's need is unclear even after considering conversation history.
11. Use retrieved knowledge only if RAG is enabled and the knowledge is relevant.
12. Return valid JSON only.
13. Do not output markdown.
14. Keep the reason under 30 words.
15. Keep the recommended_strategy under 30 words.

RAG Setting:
{rag_status}

Recent Conversation History:
{history_text}

Latest User Text:
{user_text}

Emotion Analysis Result:
- sentiment: {sentiment}
- confidence: {confidence}
- sarcasm_detected: {sarcasm_detected}
- reason: {reason}
- evidence: {evidence}

Retrieved Knowledge:
{context_text}

Return the result in this exact JSON format:

{{
  "user_need": "emotional_validation | reassurance | practical_guidance | emotional_regulation | action_planning | clarification",
  "confidence": 0.0,
  "reason": "One short sentence explaining the selected need.",
  "recommended_strategy": "One short sentence describing how the assistant should respond."
}}
""".strip()

    return prompt


# =========================
# JSON Parsing
# =========================

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from model output.
    This version is tolerant when the model output is incomplete.
    """

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    user_need_match = re.search(r'"user_need"\s*:\s*"([^"]+)"', text)
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', text, flags=re.DOTALL)
    strategy_match = re.search(r'"recommended_strategy"\s*:\s*"([^"]*)', text, flags=re.DOTALL)

    user_need = user_need_match.group(1) if user_need_match else "clarification"

    try:
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
    except ValueError:
        confidence = 0.5

    reason = reason_match.group(1).strip() if reason_match else (
        "The model output was incomplete, so the system used a fallback need detection result."
    )

    recommended_strategy = strategy_match.group(1).strip() if strategy_match else (
        "Ask a gentle follow-up question to better understand the user's situation."
    )

    return {
        "user_need": user_need,
        "confidence": confidence,
        "reason": reason,
        "recommended_strategy": recommended_strategy,
    }


# =========================
# Result Normalization
# =========================

def normalize_need_result(
    parsed_result: Dict[str, Any],
    retrieved_contexts: List[RetrievedChunk],
    raw_model_output: str,
    use_rag: bool,
) -> Dict[str, Any]:
    """
    Normalize need detection result.
    """

    user_need = str(parsed_result.get("user_need", "clarification")).lower().strip()

    valid_needs = {
        "emotional_validation",
        "reassurance",
        "practical_guidance",
        "emotional_regulation",
        "action_planning",
        "clarification",
    }

    if user_need not in valid_needs:
        user_need = "clarification"

    confidence = parsed_result.get("confidence", 0.0)

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    reason = str(parsed_result.get("reason", "")).strip()
    recommended_strategy = str(parsed_result.get("recommended_strategy", "")).strip()

    retrieved_context_list = []

    for ctx in retrieved_contexts:
        retrieved_context_list.append(
            {
                "chunk_id": ctx.chunk_id,
                "source_file": ctx.source_file,
                "score": round(ctx.score, 4),
                "text": ctx.text,
            }
        )

    return {
        "user_need": user_need,
        "confidence": confidence,
        "reason": reason,
        "recommended_strategy": recommended_strategy,
        "retrieved_context": retrieved_context_list,
        "raw_model_output": raw_model_output,
        "use_rag": use_rag,
    }


# =========================
# Main Public Function
# =========================

def detect_user_need(
    user_text: str,
    emotion_result: Dict[str, Any],
    chat_history: List[Dict[str, str]] | None = None,
    top_k: int = 4,
    use_rag: bool = True,
) -> Dict[str, Any]:
    """
    Detect the user's support need based on:
    - latest user text
    - recent conversation history
    - current emotion analysis result
    - optionally retrieved knowledge from user_needs.md and empathy_strategies.md
    """

    user_text = user_text.strip()

    if not user_text:
        raise ValueError("user_text cannot be empty.")

    retrieved_contexts: List[RetrievedChunk] = []

    if use_rag:
        retriever = get_need_retriever()

        retrieved_contexts = retriever.retrieve(
            query=user_text,
            top_k=top_k,
            source_filter=[
                "user_needs.md",
                "empathy_strategies.md",
            ],
        )

    prompt = build_need_detection_prompt(
        user_text=user_text,
        emotion_result=emotion_result,
        contexts=retrieved_contexts,
        chat_history=chat_history,
        use_rag=use_rag,
    )

    raw_output = call_gemini_text(prompt)

    parsed_result = extract_json_from_text(raw_output)

    final_result = normalize_need_result(
        parsed_result=parsed_result,
        retrieved_contexts=retrieved_contexts,
        raw_model_output=raw_output,
        use_rag=use_rag,
    )

    return final_result


# =========================
# Local Test
# =========================

if __name__ == "__main__":
    from sentiment_analyzer import analyze_sentiment

    print("User Need Detector Test")
    print("Type 'exit' to quit.\n")

    chat_history_for_test: List[Dict[str, str]] = []

    while True:
        text = input("User: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            print("Input cannot be empty.\n")
            continue

        try:
            chat_history_for_test.append(
                {
                    "role": "user",
                    "content": text,
                }
            )

            emotion = analyze_sentiment(
                user_text=text,
                use_rag=True,
                top_k=5,
            )

            need_with_rag = detect_user_need(
                user_text=text,
                emotion_result=emotion,
                chat_history=chat_history_for_test,
                top_k=4,
                use_rag=True,
            )

            need_without_rag = detect_user_need(
                user_text=text,
                emotion_result=emotion,
                chat_history=chat_history_for_test,
                top_k=4,
                use_rag=False,
            )

            print("\nEmotion Result:")
            print(json.dumps(emotion, indent=2, ensure_ascii=False))

            print("\nNeed Result With RAG:")
            print(json.dumps(need_with_rag, indent=2, ensure_ascii=False))

            print("\nNeed Result Without RAG:")
            print(json.dumps(need_without_rag, indent=2, ensure_ascii=False))

            print("-" * 60)

        except Exception as e:
            print(f"Error: {e}")