import json
import os
import re
from typing import Final, Any, Dict, List

import requests

from retriever import SimpleMarkdownRetriever, RetrievedChunk
from prompt_template import build_sentiment_prompt


# =========================
# Gemini API Configuration
# =========================

# Recommended:
# Keep this empty and set GEMINI_API_KEY as an environment variable.
#
# Windows PowerShell:
# $env:GEMINI_API_KEY="your_new_api_key"
#
# macOS/Linux:
# export GEMINI_API_KEY="your_new_api_key"
GEMINI_API_KEY: Final[str] = "AIzaSyAConlML9BN8DeA4BV84BVBk-Yf8WO_On8"

GEMINI_API_BASE: Final[str] = "https://generativelanguage.googleapis.com/v1beta"

GEMINI_TEXT_MODEL: Final[str] = "gemini-2.5-flash"
GEMINI_VISION_MODEL: Final[str] = "gemini-2.5-flash"


# =========================
# Global Retriever Cache
# =========================

_retriever_instance: SimpleMarkdownRetriever | None = None


def get_retriever() -> SimpleMarkdownRetriever:
    global _retriever_instance

    if _retriever_instance is None:
        _retriever_instance = SimpleMarkdownRetriever(
            knowledge_base_dir="knowledge_base",
            chunk_size=350,
            chunk_overlap=60,
        )

    return _retriever_instance


# =========================
# Gemini REST API
# =========================

def get_api_key() -> str:
    """
    API key priority:
    1. GEMINI_API_KEY at the top of this file
    2. Environment variable GEMINI_API_KEY
    """

    api_key = GEMINI_API_KEY.strip() or os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "Gemini API key is missing. Please set GEMINI_API_KEY at the top "
            "of sentiment_analyzer.py or set environment variable GEMINI_API_KEY."
        )

    return api_key


def _post_gemini_request(
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    response_mime_type: str | None = None,
) -> str:
    """
    Shared Gemini REST caller.

    response_mime_type:
    - "application/json" for structured JSON tasks
    - None for plain natural-language dialogue generation
    """

    api_key = get_api_key()

    url = (
        f"{GEMINI_API_BASE}/models/"
        f"{GEMINI_TEXT_MODEL}:generateContent"
        f"?key={api_key}"
    )

    generation_config = {
        "temperature": temperature,
        "topP": 0.9,
        "maxOutputTokens": max_output_tokens,
    }

    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": generation_config,
    }

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API request failed.\n"
            f"Status code: {response.status_code}\n"
            f"Response: {response.text}"
        )

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected Gemini API response format.\n"
            f"Error: {e}\n"
            f"Response JSON: {json.dumps(data, indent=2, ensure_ascii=False)}"
        )


def call_gemini_text(prompt: str) -> str:
    """
    Call Gemini for JSON tasks.

    Use this for:
    - sentiment analysis
    - user need detection

    This function asks Gemini to return JSON.
    """

    return _post_gemini_request(
        prompt=prompt,
        temperature=0.4,
        max_output_tokens=1024,
        response_mime_type="application/json",
    )


def call_gemini_plain_text(prompt: str) -> str:
    """
    Call Gemini for natural dialogue replies.

    Use this for:
    - empathetic response generation

    This function does NOT request JSON output.
    """

    output = _post_gemini_request(
        prompt=prompt,
        temperature=0.4,
        max_output_tokens=512,
        response_mime_type=None,
    )

    return clean_plain_text_reply(output)


def clean_plain_text_reply(text: str) -> str:
    """
    Clean model output for natural dialogue.

    Sometimes the model may still return JSON-like text.
    This function extracts the response field if needed.
    """

    text = text.strip()

    # Try to parse if the model still returns JSON
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            if "response" in parsed:
                return str(parsed["response"]).strip()
            if "reply" in parsed:
                return str(parsed["reply"]).strip()
            if "message" in parsed:
                return str(parsed["message"]).strip()

    except json.JSONDecodeError:
        pass

    # Remove simple markdown fences
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # If it looks like {"response": "..."} but broken, extract response manually
    response_match = re.search(
        r'"response"\s*:\s*"([^"]+)"',
        text,
        flags=re.DOTALL
    )
    if response_match:
        return response_match.group(1).strip()

    reply_match = re.search(
        r'"reply"\s*:\s*"([^"]+)"',
        text,
        flags=re.DOTALL
    )
    if reply_match:
        return reply_match.group(1).strip()

    # Remove analysis note if model appended one
    text = re.sub(
        r'\(\s*Emotion-Need Analysis:.*?\)\s*$',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

    return text


# =========================
# JSON Parsing
# =========================

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from model output.

    This version is tolerant of incomplete JSON output.
    If the model returns a partially broken JSON, it tries to recover
    the key fields instead of crashing the whole system.
    """

    text = text.strip()

    # 1. Direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Remove markdown code fences
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Try to extract complete JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # 4. Fallback: manually extract common fields
    sentiment_match = re.search(r'"sentiment"\s*:\s*"([^"]+)"', text)
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', text, flags=re.DOTALL)
    sarcasm_match = re.search(
        r'"sarcasm_detected"\s*:\s*(true|false)',
        text,
        flags=re.IGNORECASE
    )

    sentiment = sentiment_match.group(1).strip() if sentiment_match else "neutral"

    try:
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
    except ValueError:
        confidence = 0.5

    reason = reason_match.group(1).strip() if reason_match else (
        "The model output was incomplete, so the system used a fallback sentiment result."
    )

    if sarcasm_match:
        sarcasm_detected = sarcasm_match.group(1).lower() == "true"
    else:
        sarcasm_detected = False

    evidence = []

    evidence_block_match = re.search(
        r'"evidence"\s*:\s*\[(.*?)\]',
        text,
        flags=re.DOTALL
    )

    if evidence_block_match:
        evidence_block = evidence_block_match.group(1)
        evidence = re.findall(r'"([^"]+)"', evidence_block)

    if not evidence:
        evidence = [
            "Fallback parsing was used because the model returned incomplete JSON."
        ]

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "reason": reason,
        "sarcasm_detected": sarcasm_detected,
        "evidence": evidence,
    }


# =========================
# Confidence Adjustment
# =========================

def adjust_confidence(
    sentiment: str,
    reason: str,
    evidence: list,
    sarcasm_detected: bool,
    retrieved_contexts: list,
) -> float:
    """
    Rule-based confidence adjustment.

    This avoids relying only on the LLM-generated confidence,
    because LLMs often output fixed values such as 0.95.
    """

    sentiment = (sentiment or "unknown").lower().strip()
    reason = reason or ""

    evidence_count = len(evidence) if isinstance(evidence, list) else 0
    retrieved_count = len(retrieved_contexts) if retrieved_contexts else 0
    reason_length = len(reason)

    if sentiment == "unknown":
        return 0.30

    if sentiment == "neutral":
        base = 0.62
    elif sentiment == "mixed":
        base = 0.72
    elif sentiment in {"positive", "negative"}:
        base = 0.76
    else:
        base = 0.50

    if evidence_count >= 4:
        base += 0.10
    elif evidence_count == 3:
        base += 0.07
    elif evidence_count == 2:
        base += 0.04
    elif evidence_count == 1:
        base += 0.00
    else:
        base -= 0.12

    if reason_length >= 120:
        base += 0.05
    elif reason_length >= 50:
        base += 0.03
    elif reason_length > 0:
        base += 0.01
    else:
        base -= 0.08

    if retrieved_count >= 5:
        base += 0.04
    elif retrieved_count >= 3:
        base += 0.03
    elif retrieved_count >= 1:
        base += 0.01
    else:
        base -= 0.05

    if sarcasm_detected:
        base -= 0.08

    if sentiment == "neutral":
        base = min(base, 0.76)

    if sentiment == "mixed":
        base = min(base, 0.86)

    if sarcasm_detected:
        base = min(base, 0.88)

    return round(max(0.30, min(base, 0.96)), 2)


# =========================
# Result Normalization
# =========================

def normalize_result(
    parsed_result: Dict[str, Any],
    retrieved_contexts: List[RetrievedChunk],
    raw_model_output: str,
) -> Dict[str, Any]:
    """
    Normalize model output into a stable dictionary format.

    The final confidence is recalculated by adjust_confidence()
    instead of directly trusting the LLM-generated confidence.
    """

    sentiment = str(parsed_result.get("sentiment", "unknown")).lower().strip()

    valid_sentiments = {"positive", "negative", "neutral", "mixed"}

    if sentiment not in valid_sentiments:
        sentiment = "unknown"

    reason = str(parsed_result.get("reason", "")).strip()

    sarcasm_detected = parsed_result.get("sarcasm_detected", False)

    if isinstance(sarcasm_detected, str):
        sarcasm_detected = sarcasm_detected.lower().strip() == "true"
    else:
        sarcasm_detected = bool(sarcasm_detected)

    evidence = parsed_result.get("evidence", [])

    if not isinstance(evidence, list):
        evidence = [str(evidence)]

    evidence = [str(item).strip() for item in evidence if str(item).strip()]

    confidence = adjust_confidence(
        sentiment=sentiment,
        reason=reason,
        evidence=evidence,
        sarcasm_detected=sarcasm_detected,
        retrieved_contexts=retrieved_contexts,
    )

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
        "sentiment": sentiment,
        "confidence": confidence,
        "model_confidence_original": parsed_result.get("confidence", None),
        "reason": reason,
        "sarcasm_detected": sarcasm_detected,
        "evidence": evidence,
        "retrieved_context": retrieved_context_list,
        "raw_model_output": raw_model_output,
    }


# =========================
# Main Public Function
# =========================

def analyze_sentiment(
    user_text: str,
    use_rag: bool = True,
    top_k: int = 5,
) -> Dict[str, Any]:
    user_text = user_text.strip()

    if not user_text:
        raise ValueError("user_text cannot be empty.")

    retrieved_contexts: List[RetrievedChunk] = []

    if use_rag:
        retriever = get_retriever()
        retrieved_contexts = retriever.retrieve(
            query=user_text,
            top_k=top_k,
        )

    prompt = build_sentiment_prompt(
        user_text=user_text,
        contexts=retrieved_contexts,
        use_rag=use_rag,
    )

    raw_output = call_gemini_text(prompt)

    parsed_result = extract_json_from_text(raw_output)

    final_result = normalize_result(
        parsed_result=parsed_result,
        retrieved_contexts=retrieved_contexts,
        raw_model_output=raw_output,
    )

    return final_result


# =========================
# Local Test
# =========================

if __name__ == "__main__":
    print("Sentiment Analyzer Test")
    print("Type 'exit' to quit.\n")

    while True:
        text = input("Enter text: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            print("Input cannot be empty.\n")
            continue

        try:
            result = analyze_sentiment(
                user_text=text,
                use_rag=True,
                top_k=5,
            )

            print("\nResult:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("-" * 60)

        except Exception as e:
            print(f"Error: {e}")