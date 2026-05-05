from typing import Dict, Any, List

from sentiment_analyzer import analyze_sentiment, call_gemini_plain_text
from need_detector import detect_user_need
from emotion_tracker import build_emotion_state, analyze_emotion_trajectory


# =========================
# Format Chat History
# =========================

def format_chat_history(chat_history: List[Dict[str, str]], max_turns: int = 8) -> str:
    """
    Format recent chat history for prompt input.
    """

    if not chat_history:
        return "No previous conversation history."

    recent_history = chat_history[-max_turns:]

    lines = []

    for msg in recent_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


# =========================
# Build Empathetic Response Prompt
# =========================

def build_empathetic_response_prompt(
    user_text: str,
    chat_history: List[Dict[str, str]],
    emotion_result: Dict[str, Any],
    need_result: Dict[str, Any],
    trajectory_result: Dict[str, Any],
) -> str:
    """
    Build prompt for need-aware empathetic response generation.
    """

    history_text = format_chat_history(chat_history)

    current_emotion = emotion_result.get("sentiment", "unknown")
    emotion_confidence = emotion_result.get("confidence", 0.0)
    emotion_reason = emotion_result.get("reason", "")
    sarcasm_detected = emotion_result.get("sarcasm_detected", False)

    user_need = need_result.get("user_need", "clarification")
    need_confidence = need_result.get("confidence", 0.0)
    need_reason = need_result.get("reason", "")
    recommended_strategy = need_result.get("recommended_strategy", "")

    trajectory = trajectory_result.get("trajectory", "unknown")
    trajectory_summary = trajectory_result.get("summary", "")

    prompt = f"""
You are an emotion- and need-aware empathetic dialogue assistant.

Your task is to generate a natural, supportive reply for the user based on:
1. the user's latest message,
2. the recent conversation history,
3. the user's current emotional state,
4. the user's detected support need,
5. the user's emotional trajectory across conversation turns.

Output rules:
1. Output plain text only.
2. Do not output JSON.
3. Do not use keys such as "response" or "analysis_note".
4. Do not include an analysis note in the final reply.
5. Do not mention internal modules, API, backend, prompts, or implementation details.
6. Do not use markdown tables.
7. Write 2 to 3 complete sentences.
8. Each sentence must be complete.
9. Do not end the reply with an unfinished phrase, preposition, conjunction, or comma.
10. The final sentence must be complete and useful.

Response rules:
1. Respond directly to the user's latest message.
2. Start with one short sentence of emotional validation.
3. Adapt the response to the detected user need.
4. If the need is emotional_validation, focus mainly on listening and validating.
5. If the need is reassurance, use calm and encouraging language.
6. If the need is practical_guidance, give 2 to 3 concrete suggestions after validation.
7. If the need is action_planning, provide a short step-by-step plan.
8. If the need is emotional_regulation, first help the user slow down, then give one grounding action.
9. If the need is clarification, ask one gentle follow-up question.
10. Do not only comfort the user when the detected need is practical_guidance or action_planning.
11. Keep the response concise, warm, useful, and complete.

Conversation History:
{history_text}

Latest User Message:
{user_text}

Current Emotion Analysis:
- emotion/sentiment: {current_emotion}
- confidence: {emotion_confidence}
- sarcasm_detected: {sarcasm_detected}
- reason: {emotion_reason}

Current User Need Analysis:
- user_need: {user_need}
- confidence: {need_confidence}
- reason: {need_reason}
- recommended_strategy: {recommended_strategy}

Emotional Trajectory:
- trajectory: {trajectory}
- summary: {trajectory_summary}

Now generate only the assistant's natural reply.
""".strip()

    return prompt


# =========================
# Reply Post-processing
# =========================

def build_rule_based_reply(
    user_text: str,
    need_result: Dict[str, Any],
) -> str:
    """
    Build a fallback reply based on detected user need.

    This is used when the model reply is incomplete, too vague,
    or does not match the detected support need.
    """

    user_need = str(need_result.get("user_need", "clarification")).lower().strip()

    if user_need == "practical_guidance":
        return (
            "It is understandable to feel unsure about what to do next. "
            "A good first step is to write down the main problem, then separate what you can control from what you cannot. "
            "After that, choose one small action you can take today and focus only on that."
        )

    if user_need == "action_planning":
        return (
            "It makes sense that you want a clearer plan. "
            "Start by identifying your main goal, then break it into three small steps: what to do now, what to do later today, and what to do tomorrow. "
            "The key is to make the next step small enough that you can actually begin."
        )

    if user_need == "emotional_regulation":
        return (
            "It sounds like things feel overwhelming right now. "
            "Before solving the problem, try to slow down and take a few steady breaths. "
            "Then focus on one immediate thing you can control in this moment."
        )

    if user_need == "reassurance":
        return (
            "It is understandable to feel worried right now. "
            "You do not need to solve everything at once. "
            "Try to focus on one manageable next step instead of the whole situation."
        )

    if user_need == "emotional_validation":
        return (
            "That sounds really difficult, and it makes sense that you would feel this way. "
            "You do not have to process everything immediately. "
            "I am here to help you talk through it step by step."
        )

    return (
        "I hear you. "
        "Could you tell me a little more about what happened or what part feels hardest right now?"
    )


def fix_incomplete_reply(
    reply: str,
    user_text: str = "",
    need_result: Dict[str, Any] | None = None,
) -> str:
    """
    Fix incomplete or too vague natural language replies.
    """

    reply = reply.strip()

    if need_result is None:
        need_result = {}

    if not reply:
        return build_rule_based_reply(user_text, need_result)

    incomplete_endings = (
        "and",
        "but",
        "or",
        "so",
        "because",
        "with",
        "to",
        "for",
        "about",
        "at",
        "in",
        "on",
        "of",
        "by",
        "from",
        "as",
    )

    clean_reply = reply.lower().strip()
    clean_reply_no_punct = clean_reply.rstrip(".,;:!?")

    if clean_reply_no_punct.endswith(incomplete_endings):
        return build_rule_based_reply(user_text, need_result)

    if reply[-1] not in ".!?":
        reply = reply + "."

    user_need = str(need_result.get("user_need", "")).lower().strip()

    guidance_needs = {
        "practical_guidance",
        "action_planning",
    }

    if user_need in guidance_needs:
        guidance_keywords = [
            "first step",
            "start by",
            "you can",
            "try to",
            "one option",
            "next step",
            "plan",
            "write down",
            "break",
            "choose",
            "focus",
        ]

        if not any(keyword in clean_reply for keyword in guidance_keywords):
            return build_rule_based_reply(user_text, need_result)

    return reply


# =========================
# Overall Need Summary
# =========================

def summarize_overall_need(turn_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize overall user support need across all turns.

    The latest non-clarification need is prioritized because user needs can shift
    during a multi-turn conversation.
    """

    if not turn_results:
        return {
            "overall_need": "unknown",
            "need_transition": [],
            "need_counts": {},
            "summary": "No turn results are available.",
        }

    needs = []

    for turn in turn_results:
        need_result = turn.get("need_result", {})
        need = str(need_result.get("user_need", "clarification")).lower().strip()
        needs.append(need)

    need_counts: Dict[str, int] = {}

    for need in needs:
        need_counts[need] = need_counts.get(need, 0) + 1

    non_clarification_needs = [
        need for need in needs
        if need != "clarification"
    ]

    if non_clarification_needs:
        overall_need = non_clarification_needs[-1]
    else:
        overall_need = "clarification"

    need_transition = needs

    summary = (
        f"The user's support need changes across turns as: "
        f"{' → '.join(need_transition)}. "
        f"The overall current support need appears to be {overall_need}."
    )

    return {
        "overall_need": overall_need,
        "need_transition": need_transition,
        "need_counts": need_counts,
        "summary": summary,
    }


# =========================
# Main Dialogue Turn Processor
# =========================

def process_dialogue_turn(
    user_text: str,
    chat_history: List[Dict[str, str]],
    emotion_states: List[Dict[str, Any]],
    top_k: int = 5,
    use_rag: bool = True,
    generate_reply: bool = True,
) -> Dict[str, Any]:
    """
    Process one user turn in the multi-turn empathetic dialogue system.

    This function is mainly used for Empathy Chat mode, where the user talks
    turn by turn and the assistant may generate a reply after each turn.
    """

    user_text = user_text.strip()

    if not user_text:
        raise ValueError("user_text cannot be empty.")

    emotion_result = analyze_sentiment(
        user_text=user_text,
        use_rag=use_rag,
        top_k=top_k,
    )

    need_result = detect_user_need(
        user_text=user_text,
        emotion_result=emotion_result,
        chat_history=chat_history,
        top_k=top_k,
        use_rag=use_rag,
    )

    current_state = build_emotion_state(
        turn_id=len(emotion_states) + 1,
        user_text=user_text,
        emotion_result=emotion_result,
        need_result=need_result,
    )

    updated_emotion_states = emotion_states + [current_state]

    trajectory_result = analyze_emotion_trajectory(
        emotion_states=updated_emotion_states,
    )

    reply = ""

    if generate_reply:
        prompt = build_empathetic_response_prompt(
            user_text=user_text,
            chat_history=chat_history,
            emotion_result=emotion_result,
            need_result=need_result,
            trajectory_result=trajectory_result,
        )

        reply = call_gemini_plain_text(prompt).strip()

        reply = fix_incomplete_reply(
            reply=reply,
            user_text=user_text,
            need_result=need_result,
        )

    return {
        "reply": reply,
        "emotion_result": emotion_result,
        "need_result": need_result,
        "current_state": current_state,
        "emotion_states": updated_emotion_states,
        "trajectory_result": trajectory_result,
        "use_rag": use_rag,
        "generate_reply": generate_reply,
    }


# =========================
# Full Multi-turn Tracking Processor
# =========================

def split_conversation_text(conversation_text: str) -> List[str]:
    """
    Split a multi-turn conversation text into user turns.

    Each non-empty line is treated as one user turn.
    """

    turns = [
        line.strip()
        for line in conversation_text.splitlines()
        if line.strip()
    ]

    return turns


def process_multi_turn_tracking(
    conversation_text: str,
    top_k: int = 5,
    use_rag: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a full multi-turn conversation at once.

    This function is designed for:
    - RAG Multi-turn Emotion Tracking
    - No-RAG Multi-turn Emotion Tracking

    Each non-empty line is treated as one user turn.
    """

    conversation_text = conversation_text.strip()

    if not conversation_text:
        raise ValueError("conversation_text cannot be empty.")

    user_turns = split_conversation_text(conversation_text)

    if not user_turns:
        raise ValueError("No valid user turns found in conversation_text.")

    chat_history: List[Dict[str, str]] = []
    emotion_states: List[Dict[str, Any]] = []
    turn_results: List[Dict[str, Any]] = []

    for turn_index, user_text in enumerate(user_turns, start=1):
        chat_history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        result = process_dialogue_turn(
            user_text=user_text,
            chat_history=chat_history,
            emotion_states=emotion_states,
            top_k=top_k,
            use_rag=use_rag,
            generate_reply=False,
        )

        emotion_states = result["emotion_states"]

        turn_results.append(
            {
                "turn": turn_index,
                "user_text": user_text,
                "emotion_result": result["emotion_result"],
                "need_result": result["need_result"],
                "current_state": result["current_state"],
                "trajectory_result": result["trajectory_result"],
            }
        )

    final_trajectory = analyze_emotion_trajectory(
        emotion_states=emotion_states,
    )

    overall_need_summary = summarize_overall_need(
        turn_results=turn_results,
    )

    return {
        "turn_results": turn_results,
        "emotion_states": emotion_states,
        "final_trajectory": final_trajectory,
        "overall_need_summary": overall_need_summary,
        "use_rag": use_rag,
        "num_turns": len(user_turns),
    }


# =========================
# Local Test
# =========================

if __name__ == "__main__":
    print("Dialogue Manager Test")
    print("1. Single-turn chat test")
    print("2. Full multi-turn tracking test")
    mode = input("Choose mode 1 or 2: ").strip()

    if mode == "2":
        sample_conversation = """I just broke up with my girlfriend. I feel terrible.
I don't know what to do now.
My life feels so empty.
I can't calm down. Everything feels too much."""

        try:
            result = process_multi_turn_tracking(
                conversation_text=sample_conversation,
                top_k=5,
                use_rag=True,
            )

            print("\nFinal Trajectory:")
            print(result["final_trajectory"])

            print("\nOverall Need Summary:")
            print(result["overall_need_summary"])

            print("\nTurn Results:")
            for turn in result["turn_results"]:
                print("-" * 60)
                print(f"Turn {turn['turn']}: {turn['user_text']}")
                print(f"Emotion: {turn['emotion_result'].get('sentiment')}")
                print(f"Need: {turn['need_result'].get('user_need')}")
                print(f"Trajectory: {turn['trajectory_result'].get('trajectory')}")

        except Exception as e:
            print(f"Error: {e}")

    else:
        chat_history = []
        emotion_states = []

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
                result = process_dialogue_turn(
                    user_text=user_input,
                    chat_history=chat_history,
                    emotion_states=emotion_states,
                    top_k=5,
                    use_rag=True,
                    generate_reply=True,
                )

                assistant_reply = result["reply"]
                emotion_states = result["emotion_states"]

                chat_history.append(
                    {
                        "role": "assistant",
                        "content": assistant_reply,
                    }
                )

                print("\nAssistant:")
                print(assistant_reply)

                print("\nTurn Analysis:")
                print(f"Use RAG: {result['use_rag']}")
                print(f"Emotion: {result['emotion_result'].get('sentiment')}")
                print(f"User Need: {result['need_result'].get('user_need')}")
                print(f"Trajectory: {result['trajectory_result'].get('trajectory')}")
                print(f"Trajectory Confidence: {result['trajectory_result'].get('trajectory_confidence')}")

                print("-" * 60)

            except Exception as e:
                print(f"Error: {e}")