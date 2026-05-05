from typing import Any, Dict, List


# =========================
# Base Emotion Score Mapping
# =========================

BASE_EMOTION_SCORE = {
    "positive": 1.0,
    "neutral": 0.0,
    "mixed": -0.35,
    "negative": -1.0,
    "unknown": 0.0,
}


# =========================
# Text-based Intensity and Risk Detection
# =========================

def estimate_emotional_intensity(
    user_text: str,
    emotion_result: Dict[str, Any],
    need_result: Dict[str, Any] | None = None,
) -> float:
    """
    Estimate emotional intensity for one user turn.

    Range:
    - 0.0 = no emotional intensity
    - 1.0 = very strong emotional intensity

    This is a lightweight rule-based estimator designed for trajectory tracking.
    It is not a clinical risk classifier.
    """

    text = (user_text or "").lower().strip()

    sentiment = str(emotion_result.get("sentiment", "unknown")).lower().strip()
    emotion_confidence = float(emotion_result.get("confidence", 0.0) or 0.0)

    user_need = "clarification"
    if need_result:
        user_need = str(need_result.get("user_need", "clarification")).lower().strip()

    if sentiment == "positive":
        intensity = 0.45
    elif sentiment == "neutral":
        intensity = 0.20
    elif sentiment == "mixed":
        intensity = 0.55
    elif sentiment == "negative":
        intensity = 0.65
    else:
        intensity = 0.30

    medium_negative_phrases = [
        "i feel bad",
        "i feel sad",
        "i am sad",
        "i feel lost",
        "i am worried",
        "i feel anxious",
        "i am scared",
        "i am confused",
        "i don't know what to do",
        "i do not know what to do",
        "my life is boring",
        "life is boring",
        "i feel bored",
        "i am bored",
    ]

    high_negative_phrases = [
        "i feel terrible",
        "i feel awful",
        "i feel horrible",
        "i am devastated",
        "i feel broken",
        "i can't take this",
        "i cannot take this",
        "everything feels too much",
        "i feel overwhelmed",
        "i feel hopeless",
        "i feel like giving up",
        "i wanna break up everything",
        "i want to break up everything",
        "i want to destroy everything",
        "i feel my life end",
        "my life is over",
    ]

    risk_phrases = [
        "i don't want to live",
        "i do not want to live",
        "i want to die",
        "i want to disappear",
        "my life is over",
        "i feel my life end",
        "life is meaningless",
        "i feel like giving up",
        "i can't go on",
        "i cannot go on",
    ]

    if any(phrase in text for phrase in risk_phrases):
        intensity = max(intensity, 1.00)
    elif any(phrase in text for phrase in high_negative_phrases):
        intensity = max(intensity, 0.90)
    elif any(phrase in text for phrase in medium_negative_phrases):
        intensity = max(intensity, 0.70)

    if user_need == "emotional_regulation":
        intensity = max(intensity, 0.85)
    elif user_need == "emotional_validation":
        intensity = max(intensity, 0.65)
    elif user_need == "reassurance":
        intensity = max(intensity, 0.60)

    if "!" in user_text:
        intensity += 0.05

    if text.count("very") >= 1 or text.count("so ") >= 1 or text.count("really") >= 1:
        intensity += 0.05

    if sentiment == "negative" and emotion_confidence >= 0.85:
        intensity += 0.03

    return round(max(0.0, min(intensity, 1.0)), 2)


def detect_risk_flag(user_text: str) -> bool:
    """
    Detect potentially high-risk emotional expressions.

    This does not diagnose the user.
    It only flags text that may require a more careful, safety-aware response.
    """

    text = (user_text or "").lower().strip()

    risk_phrases = [
        "i don't want to live",
        "i do not want to live",
        "i want to die",
        "i want to disappear",
        "my life is over",
        "i feel my life end",
        "life is meaningless",
        "i feel like giving up",
        "i can't go on",
        "i cannot go on",
    ]

    return any(phrase in text for phrase in risk_phrases)


def compute_emotion_score(sentiment: str, intensity: float) -> float:
    """
    Convert sentiment + intensity into a trajectory score.
    """

    sentiment = (sentiment or "unknown").lower().strip()

    if sentiment == "negative":
        return round(-1.0 * intensity, 2)

    if sentiment == "mixed":
        return round(-0.30 - 0.40 * intensity, 2)

    if sentiment == "positive":
        return round(0.40 + 0.60 * intensity, 2)

    if sentiment == "neutral":
        return 0.0

    return 0.0


# =========================
# Single-turn Emotion State
# =========================

def build_emotion_state(
    turn_id: int,
    user_text: str,
    emotion_result: Dict[str, Any],
    need_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a structured emotion state for one conversation turn.
    """

    sentiment = str(emotion_result.get("sentiment", "unknown")).lower().strip()
    user_need = str(need_result.get("user_need", "clarification")).lower().strip()

    intensity = estimate_emotional_intensity(
        user_text=user_text,
        emotion_result=emotion_result,
        need_result=need_result,
    )

    emotion_score = compute_emotion_score(
        sentiment=sentiment,
        intensity=intensity,
    )

    risk_flag = detect_risk_flag(user_text)

    return {
        "turn": turn_id,
        "user_text": user_text,
        "emotion": sentiment,
        "emotion_score": emotion_score,
        "emotion_intensity": intensity,
        "emotion_confidence": emotion_result.get("confidence", 0.0),
        "sarcasm_detected": emotion_result.get("sarcasm_detected", False),
        "risk_flag": risk_flag,
        "user_need": user_need,
        "need_confidence": need_result.get("confidence", 0.0),
        "need_reason": need_result.get("reason", ""),
        "recommended_strategy": need_result.get("recommended_strategy", ""),
    }


# =========================
# Trajectory Confidence
# =========================

def estimate_trajectory_confidence(
    trajectory: str,
    recent_states: List[Dict[str, Any]],
) -> float:
    """
    Estimate confidence for the emotional trajectory result.
    """

    if not recent_states:
        return 0.30

    emotion_confidences = [
        float(state.get("emotion_confidence", 0.0) or 0.0)
        for state in recent_states
    ]

    need_confidences = [
        float(state.get("need_confidence", 0.0) or 0.0)
        for state in recent_states
    ]

    avg_emotion_conf = sum(emotion_confidences) / len(emotion_confidences)
    avg_need_conf = sum(need_confidences) / len(need_confidences)

    base = (avg_emotion_conf + avg_need_conf) / 2

    if len(recent_states) == 1:
        base -= 0.10

    if len(recent_states) >= 3:
        base += 0.03

    if trajectory in {
        "persistent_negative",
        "persistent_negative_escalating",
        "high_risk_worsening",
        "worsening",
        "improving",
        "planning_progress",
    }:
        base += 0.05

    if trajectory == "fluctuating":
        base -= 0.03

    return round(max(0.30, min(base, 0.96)), 2)


# =========================
# Trajectory Analysis
# =========================

def analyze_emotion_trajectory(
    emotion_states: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Analyze the user's emotional trajectory across recent turns.

    Labels:
    - initial
    - improving
    - planning_progress
    - worsening
    - persistent_negative
    - persistent_negative_escalating
    - high_risk_worsening
    - stable
    - fluctuating
    - unknown
    """

    if not emotion_states:
        return {
            "trajectory": "unknown",
            "trajectory_confidence": 0.30,
            "summary": "No emotion states are available.",
            "recent_emotions": [],
            "recent_needs": [],
            "recent_scores": [],
            "recent_intensities": [],
            "risk_detected": False,
        }

    recent_states = emotion_states[-4:]
    current = recent_states[-1]

    recent_emotions = [
        state.get("emotion", "unknown")
        for state in recent_states
    ]

    recent_needs = [
        state.get("user_need", "clarification")
        for state in recent_states
    ]

    scores = [
        float(state.get("emotion_score", 0.0))
        for state in recent_states
    ]

    intensities = [
        float(state.get("emotion_intensity", 0.0))
        for state in recent_states
    ]

    risk_detected = any(
        bool(state.get("risk_flag", False))
        for state in recent_states
    )

    current_emotion = current.get("emotion", "unknown")
    current_need = current.get("user_need", "clarification")
    current_intensity = float(current.get("emotion_intensity", 0.0))
    current_score = float(current.get("emotion_score", 0.0))

    negative_count = sum(
        1 for emotion in recent_emotions
        if emotion == "negative"
    )

    # Single-turn case
    if len(emotion_states) == 1:
        if current.get("risk_flag", False):
            trajectory = "high_risk_worsening"
            summary = (
                "The user expresses a potentially high-risk negative emotional state. "
                "The assistant should respond with strong validation, calm language, "
                "and encourage reaching out to trusted people or professional support."
            )
        else:
            trajectory = "initial"
            summary = (
                f"Initial detected emotion is {current_emotion} with support need {current_need}."
            )

        trajectory_confidence = estimate_trajectory_confidence(
            trajectory=trajectory,
            recent_states=recent_states,
        )

        return {
            "trajectory": trajectory,
            "trajectory_confidence": trajectory_confidence,
            "summary": summary,
            "recent_emotions": recent_emotions,
            "recent_needs": recent_needs,
            "recent_scores": scores,
            "recent_intensities": intensities,
            "risk_detected": risk_detected,
        }

    previous = recent_states[-2]
    previous_score = float(previous.get("emotion_score", 0.0))
    previous_intensity = float(previous.get("emotion_intensity", 0.0))
    previous_need = previous.get("user_need", "clarification")
    previous_emotion = previous.get("emotion", "unknown")

    first_score = scores[0]
    last_score = scores[-1]
    first_intensity = intensities[0]
    last_intensity = intensities[-1]
    score_range = max(scores) - min(scores)

    planning_needs = {
        "action_planning",
        "practical_guidance",
    }

    planning_continuation = (
        current_emotion in {"neutral", "positive"}
        and previous_emotion in {"positive", "neutral", "mixed"}
        and previous_need in planning_needs
        and current_need in {"clarification", "action_planning", "practical_guidance"}
    )

    regulation_escalation = (
        current_emotion == "negative"
        and current_need == "emotional_regulation"
        and previous_need != "emotional_regulation"
    )

    intensity_escalation_from_previous = (
        current_emotion == "negative"
        and current_intensity > previous_intensity + 0.12
    )

    score_escalation_from_previous = (
        current_emotion == "negative"
        and current_score < previous_score - 0.12
    )

    worsening_from_window_start = (
        current_emotion == "negative"
        and (
            last_score < first_score - 0.18
            or last_intensity > first_intensity + 0.18
        )
    )

    improving_from_previous = (
        current_score > previous_score + 0.18
        and current_emotion in {"positive", "mixed", "neutral"}
    )

    improving_from_window_start = (
        last_score > first_score + 0.25
        and current_emotion in {"positive", "mixed", "neutral"}
    )

    # Trajectory priority order
    if current.get("risk_flag", False):
        trajectory = "high_risk_worsening"

    elif planning_continuation:
        trajectory = "planning_progress"

    elif regulation_escalation:
        trajectory = "persistent_negative_escalating"

    elif intensity_escalation_from_previous or score_escalation_from_previous:
        trajectory = "worsening"

    elif worsening_from_window_start:
        trajectory = "worsening"

    elif negative_count >= 2 and current_emotion == "negative":
        trajectory = "persistent_negative"

    elif improving_from_previous or improving_from_window_start:
        trajectory = "improving"

    elif score_range >= 1.20:
        trajectory = "fluctuating"

    else:
        trajectory = "stable"

    if trajectory == "high_risk_worsening":
        summary = (
            "The user's emotional trajectory shows a potentially high-risk worsening pattern. "
            "The latest message contains severe distress. The assistant should prioritize safety-aware support, "
            "validate the user's pain, and encourage reaching out to trusted people or professional help."
        )

    elif trajectory == "persistent_negative_escalating":
        summary = (
            "The user has shown repeated negative emotions, and the latest turn suggests stronger emotional "
            "intensity or a higher need for emotional regulation. The assistant should slow down, validate the "
            "feeling, and provide calm grounding support."
        )

    elif trajectory == "persistent_negative":
        summary = (
            "The user has shown negative emotion across multiple recent turns. The emotional state is not "
            "clearly improving yet, so the assistant should continue offering validation and gentle support."
        )

    elif trajectory == "worsening":
        summary = (
            "The user's emotional state appears to be worsening because the latest turn is more negative or more intense. "
            "The assistant should respond with stronger validation and careful support."
        )

    elif trajectory == "planning_progress":
        summary = (
            "The user appears to be refining a plan rather than becoming more negative. "
            "The assistant should treat this as planning progress and help turn the idea into concrete next steps."
        )

    elif trajectory == "improving":
        summary = (
            "The user's emotional state appears to be improving. The assistant can acknowledge the progress "
            "and help the user continue moving forward."
        )

    elif trajectory == "fluctuating":
        summary = (
            "The user's emotional state appears to be fluctuating across recent turns. The assistant should "
            "respond carefully and avoid assuming a single stable emotional direction."
        )

    else:
        summary = (
            f"The user's recent emotional trajectory appears to be stable. "
            f"The current emotion is {current_emotion}, the current intensity is {current_intensity}, "
            f"and the current support need is {current_need}."
        )

    trajectory_confidence = estimate_trajectory_confidence(
        trajectory=trajectory,
        recent_states=recent_states,
    )

    return {
        "trajectory": trajectory,
        "trajectory_confidence": trajectory_confidence,
        "summary": summary,
        "recent_emotions": recent_emotions,
        "recent_needs": recent_needs,
        "recent_scores": scores,
        "recent_intensities": intensities,
        "risk_detected": risk_detected,
    }


# =========================
# Display Formatting
# =========================

def format_trajectory_for_display(
    trajectory_result: Dict[str, Any],
) -> str:
    """
    Format trajectory result into readable text for UI or debugging.
    """

    trajectory = trajectory_result.get("trajectory", "unknown")
    trajectory_confidence = trajectory_result.get("trajectory_confidence", 0.0)
    summary = trajectory_result.get("summary", "")
    recent_emotions = trajectory_result.get("recent_emotions", [])
    recent_needs = trajectory_result.get("recent_needs", [])
    recent_scores = trajectory_result.get("recent_scores", [])
    recent_intensities = trajectory_result.get("recent_intensities", [])
    risk_detected = trajectory_result.get("risk_detected", False)

    output = f"""
Trajectory: {trajectory}

Trajectory confidence:
{trajectory_confidence}

Summary:
{summary}

Recent emotions:
{recent_emotions}

Recent needs:
{recent_needs}

Recent scores:
{recent_scores}

Recent intensities:
{recent_intensities}

Risk detected:
{risk_detected}
""".strip()

    return output


# =========================
# Local Test
# =========================

if __name__ == "__main__":
    sample_states: List[Dict[str, Any]] = []

    sample_data = [
        {
            "user_text": "I want to plan my next steps and make it even better.",
            "emotion_result": {
                "sentiment": "positive",
                "confidence": 0.89,
                "sarcasm_detected": False,
            },
            "need_result": {
                "user_need": "action_planning",
                "confidence": 0.95,
                "reason": "The user wants to plan next steps.",
                "recommended_strategy": "Help the user break goals into steps.",
            },
        },
        {
            "user_text": "Maybe on study",
            "emotion_result": {
                "sentiment": "neutral",
                "confidence": 0.69,
                "sarcasm_detected": False,
            },
            "need_result": {
                "user_need": "clarification",
                "confidence": 0.50,
                "reason": "The user gives a short planning direction.",
                "recommended_strategy": "Ask what study goal they want to focus on.",
            },
        },
    ]

    for i, item in enumerate(sample_data, start=1):
        state = build_emotion_state(
            turn_id=i,
            user_text=item["user_text"],
            emotion_result=item["emotion_result"],
            need_result=item["need_result"],
        )

        sample_states.append(state)

        result = analyze_emotion_trajectory(sample_states)

        print("=" * 60)
        print(f"After Turn {i}")
        print(format_trajectory_for_display(result))