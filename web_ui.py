import streamlit as st

from dialogue_manager import process_dialogue_turn, process_multi_turn_tracking


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Empathy Chat System",
    page_icon="💬",
    layout="wide"
)


# =========================
# Session State
# =========================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "emotion_states" not in st.session_state:
    st.session_state.emotion_states = []

if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False

if "use_rag" not in st.session_state:
    st.session_state.use_rag = True

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "RAG Multi-turn Emotion Tracking"

if "tracking_result_text" not in st.session_state:
    st.session_state.tracking_result_text = ""

if "tracking_raw_result" not in st.session_state:
    st.session_state.tracking_raw_result = None


# =========================
# Helper Functions
# =========================

def clear_chat_history():
    st.session_state.chat_messages = []
    st.session_state.emotion_states = []
    st.session_state.conversation_ended = False
    st.session_state.tracking_result_text = ""
    st.session_state.tracking_raw_result = None


def end_conversation():
    st.session_state.conversation_ended = True


def get_sentiment_badge(sentiment: str) -> str:
    sentiment = (sentiment or "unknown").lower().strip()

    if sentiment == "positive":
        return "🟢 Positive"
    if sentiment == "negative":
        return "🔴 Negative"
    if sentiment == "neutral":
        return "⚪ Neutral"
    if sentiment == "mixed":
        return "🟡 Mixed"

    return "⚫ Unknown"


def get_need_badge(user_need: str) -> str:
    user_need = (user_need or "clarification").lower().strip()

    labels = {
        "emotional_validation": "💙 Emotional Validation",
        "reassurance": "🌤️ Reassurance",
        "practical_guidance": "🧭 Practical Guidance",
        "emotional_regulation": "🧘 Emotional Regulation",
        "action_planning": "📝 Action Planning",
        "clarification": "❓ Clarification",
        "unknown": "⚫ Unknown",
    }

    return labels.get(user_need, "❓ Clarification")


def get_trajectory_badge(trajectory: str) -> str:
    trajectory = (trajectory or "unknown").lower().strip()

    labels = {
        "initial": "🚩 Initial",
        "improving": "📈 Improving",
        "planning_progress": "📝 Planning Progress",
        "worsening": "📉 Worsening",
        "persistent_negative": "🔴 Persistent Negative",
        "persistent_negative_escalating": "⚠️ Persistent Negative Escalating",
        "high_risk_worsening": "🚨 High-risk Worsening",
        "stable": "➖ Stable",
        "fluctuating": "〰️ Fluctuating",
        "unknown": "⚫ Unknown",
    }

    return labels.get(trajectory, "⚫ Unknown")


def format_turn_analysis(dialogue_result: dict) -> str:
    emotion_result = dialogue_result.get("emotion_result", {})
    need_result = dialogue_result.get("need_result", {})
    trajectory_result = dialogue_result.get("trajectory_result", {})

    emotion = emotion_result.get("sentiment", "unknown")
    emotion_confidence = emotion_result.get("confidence", 0.0)

    user_need = need_result.get("user_need", "clarification")
    need_confidence = need_result.get("confidence", 0.0)

    trajectory = trajectory_result.get("trajectory", "unknown")
    trajectory_confidence = trajectory_result.get("trajectory_confidence", 0.0)
    trajectory_summary = trajectory_result.get("summary", "")

    use_rag = dialogue_result.get("use_rag", True)
    rag_text = "Enabled" if use_rag else "Disabled"

    return f"""
### Turn Analysis

**RAG:** `{rag_text}`

**Current Emotion:** {get_sentiment_badge(emotion)}  
**Emotion Confidence:** `{emotion_confidence}`

**Detected User Need:** {get_need_badge(user_need)}  
**Need Confidence:** `{need_confidence}`

**Emotional Trajectory:** {get_trajectory_badge(trajectory)}  
**Trajectory Confidence:** `{trajectory_confidence}`

**Trajectory Summary:**  
{trajectory_summary}
"""


def format_multi_turn_tracking_result(result: dict) -> str:
    use_rag = result.get("use_rag", True)
    rag_text = "Enabled" if use_rag else "Disabled"

    final_trajectory = result.get("final_trajectory", {})
    trajectory = final_trajectory.get("trajectory", "unknown")
    trajectory_confidence = final_trajectory.get("trajectory_confidence", 0.0)
    trajectory_summary = final_trajectory.get("summary", "")

    overall_need_summary = result.get("overall_need_summary", {})
    overall_need = overall_need_summary.get("overall_need", "unknown")
    need_transition = overall_need_summary.get("need_transition", [])
    need_counts = overall_need_summary.get("need_counts", {})
    need_summary_text = overall_need_summary.get("summary", "")

    need_transition_text = " → ".join(need_transition) if need_transition else "unknown"

    if need_counts:
        need_counts_text = ", ".join(
            [f"{need}: {count}" for need, count in need_counts.items()]
        )
    else:
        need_counts_text = "unknown"

    num_turns = result.get("num_turns", 0)
    turn_results = result.get("turn_results", [])

    output = f"""
### Overall Multi-turn Emotion Tracking Result

**RAG:** `{rag_text}`  
**Number of Turns:** `{num_turns}`

**Overall Emotional Trajectory:** {get_trajectory_badge(trajectory)}  
**Trajectory Confidence:** `{trajectory_confidence}`

**Overall Summary:**  
{trajectory_summary}

---

### Overall User Need

**Overall Support Need:** {get_need_badge(overall_need)}

**Need Transition:**  
`{need_transition_text}`

**Need Counts:**  
`{need_counts_text}`

**Need Summary:**  
{need_summary_text}

---

### Turn-level Analysis
"""

    for i, turn in enumerate(turn_results, start=1):
        user_text = turn.get("user_text", "")

        emotion_result = turn.get("emotion_result", {})
        need_result = turn.get("need_result", {})
        state = turn.get("current_state", {})
        trajectory_result = turn.get("trajectory_result", {})

        emotion = emotion_result.get("sentiment", "unknown")
        emotion_confidence = emotion_result.get("confidence", 0.0)

        user_need = need_result.get("user_need", "clarification")
        need_confidence = need_result.get("confidence", 0.0)

        intensity = state.get("emotion_intensity", 0.0)
        turn_trajectory = trajectory_result.get("trajectory", "unknown")
        turn_trajectory_confidence = trajectory_result.get("trajectory_confidence", 0.0)

        output += f"""

#### Turn {i}

**User Text:**  
> {user_text}

**Emotion:** {get_sentiment_badge(emotion)}  
**Emotion Confidence:** `{emotion_confidence}`  
**Emotion Intensity:** `{intensity}`

**User Need:** {get_need_badge(user_need)}  
**Need Confidence:** `{need_confidence}`

**Trajectory After This Turn:** {get_trajectory_badge(turn_trajectory)}  
**Turn Trajectory Confidence:** `{turn_trajectory_confidence}`
"""

    return output


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.markdown("## 💬 Empathy Chat System")
    st.caption("Multi-turn emotion tracking and need-aware empathetic dialogue.")

    st.markdown("---")

    st.subheader("⚙️ Control Panel")

    mode_options = [
        "RAG Multi-turn Emotion Tracking",
        "No-RAG Multi-turn Emotion Tracking",
        "Empathy Chat",
    ]

    if st.session_state.analysis_mode not in mode_options:
        st.session_state.analysis_mode = "RAG Multi-turn Emotion Tracking"

    st.session_state.analysis_mode = st.radio(
        "Analysis Mode",
        options=mode_options,
        index=mode_options.index(st.session_state.analysis_mode)
    )

    st.session_state.use_rag = (
        st.session_state.analysis_mode != "No-RAG Multi-turn Emotion Tracking"
    )

    st.session_state.top_k = st.slider(
        "Top-K Retrieved Knowledge",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        disabled=not st.session_state.use_rag
    )

    st.markdown("---")

    if st.session_state.analysis_mode == "Empathy Chat":
        if st.button("🛑 End Conversation", use_container_width=True):
            end_conversation()
            st.rerun()

    if st.button("🗑️ Clear Results / History", use_container_width=True):
        clear_chat_history()
        st.rerun()

    st.markdown("---")

    st.subheader("📚 Knowledge Base")
    st.caption("Loaded files:")
    st.caption("- sentiment_lexicon.md")
    st.caption("- sarcasm_examples.md")
    st.caption("- user_needs.md")
    st.caption("- empathy_strategies.md")

    st.markdown("---")

    st.caption("Backend:")
    st.caption("- Gemini REST API")
    st.caption("- Local Markdown RAG")
    st.caption("- Emotion Analyzer")
    st.caption("- User Need Detector")
    st.caption("- Emotion Trajectory Tracker")


# =========================
# Main UI
# =========================

st.title("💬 Empathy Chat System")

tracking_modes = [
    "RAG Multi-turn Emotion Tracking",
    "No-RAG Multi-turn Emotion Tracking",
]

if st.session_state.analysis_mode == "RAG Multi-turn Emotion Tracking":
    st.info(
        "RAG Multi-turn Emotion Tracking Mode: Enter a full multi-turn conversation. "
        "The system uses the knowledge base to analyze each turn, estimate the overall emotional trajectory, "
        "and summarize the user's support need."
    )

elif st.session_state.analysis_mode == "No-RAG Multi-turn Emotion Tracking":
    st.warning(
        "No-RAG Multi-turn Emotion Tracking Mode: Enter a full multi-turn conversation. "
        "The system does not use the knowledge base and only analyzes the conversation text."
    )

else:
    if st.session_state.conversation_ended:
        st.error("Conversation ended. Please clear chat history to start a new conversation.")

        if st.button("🔄 Start New Conversation", use_container_width=True):
            clear_chat_history()
            st.rerun()
    else:
        st.success(
            "Empathy Chat Mode: The system tracks emotion, detects user needs, "
            "and generates need-aware empathetic responses turn by turn."
        )

        col1, col2 = st.columns([1, 4])

        with col1:
            if st.button("🛑 End Conversation", use_container_width=True):
                end_conversation()
                st.rerun()

        with col2:
            st.caption("Click End Conversation when the user wants to stop the dialogue.")


# =========================
# Example Inputs
# =========================

with st.expander("💡 Multi-turn Example Inputs", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sustained Positive Trajectory**")
        st.caption("Copy the full block into the text area.")

        st.code(
            """I feel proud because I finally finished my project.
My professor gave me some positive feedback.
I feel more motivated to keep improving it.
I want to plan my next steps and make it even better."""
        )

    with col2:
        st.markdown("**Sustained Negative Trajectory**")
        st.caption("Copy the full block into the text area.")

        st.code(
            """I just broke up with my girlfriend. I feel terrible.
I don't know what to do now.
My life feels so empty.
I can't calm down. Everything feels too much."""
        )


# =========================
# Tracking Modes: Full Multi-turn Input
# =========================

if st.session_state.analysis_mode in tracking_modes:
    st.markdown("### Enter a Multi-turn Conversation")
    st.caption("Each non-empty line will be treated as one user turn.")

    default_text = """I just broke up with my girlfriend. I feel terrible.
I don't know what to do now.
My life feels so empty.
I can't calm down. Everything feels too much."""

    conversation_text = st.text_area(
        "Conversation text",
        value=default_text,
        height=180,
        help="Enter one user message per line."
    )

    if st.button("🔍 Analyze Multi-turn Conversation", use_container_width=True):
        if not conversation_text.strip():
            st.warning("Conversation text cannot be empty.")
        else:
            try:
                with st.spinner("Analyzing the full multi-turn emotional trajectory..."):
                    tracking_result = process_multi_turn_tracking(
                        conversation_text=conversation_text,
                        top_k=st.session_state.top_k,
                        use_rag=st.session_state.use_rag,
                    )

                    st.session_state.tracking_raw_result = tracking_result
                    st.session_state.tracking_result_text = format_multi_turn_tracking_result(
                        tracking_result
                    )
                    st.session_state.emotion_states = tracking_result["emotion_states"]

            except Exception as e:
                st.error(f"System Backend Error: {e}")

    if st.session_state.tracking_result_text:
        st.markdown(st.session_state.tracking_result_text)

    if st.session_state.tracking_raw_result is not None:
        with st.expander("📈 Raw Emotion Trajectory States", expanded=False):
            st.json(st.session_state.tracking_raw_result.get("emotion_states", []))

        with st.expander("🧩 Raw Overall Need Summary", expanded=False):
            st.json(st.session_state.tracking_raw_result.get("overall_need_summary", {}))


# =========================
# Empathy Chat Mode: Turn-by-turn Input
# =========================

else:
    messages_to_render = st.session_state.chat_messages

    for message in messages_to_render:
        avatar_icon = "🧑" if message["role"] == "user" else "🤖"

        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    if st.session_state.emotion_states:
        with st.expander("📈 Emotion Trajectory States", expanded=False):
            st.json(st.session_state.emotion_states)

    if st.session_state.conversation_ended:
        st.chat_input(
            "Conversation ended. Clear chat history to start again.",
            disabled=True
        )
    else:
        user_text = st.chat_input("Talk to the empathy chat assistant...")

        if user_text:
            user_text = user_text.strip()

            if not user_text:
                st.warning("Input cannot be empty.")
            else:
                st.session_state.chat_messages.append(
                    {
                        "role": "user",
                        "content": user_text
                    }
                )

                with st.spinner("Generating empathy-aware response..."):
                    try:
                        dialogue_result = process_dialogue_turn(
                            user_text=user_text,
                            chat_history=st.session_state.chat_messages,
                            emotion_states=st.session_state.emotion_states,
                            top_k=st.session_state.top_k,
                            use_rag=True,
                            generate_reply=True,
                        )

                        st.session_state.emotion_states = dialogue_result["emotion_states"]

                        turn_analysis = format_turn_analysis(dialogue_result)

                        assistant_message = (
                            f"{dialogue_result['reply']}\n\n"
                            f"---\n\n"
                            f"{turn_analysis}"
                        )

                        st.session_state.chat_messages.append(
                            {
                                "role": "assistant",
                                "content": assistant_message
                            }
                        )

                        st.rerun()

                    except Exception as e:
                        st.error(f"System Backend Error: {e}")