import json
from pathlib import Path
from typing import Any, Dict, List

from dialogue_manager import process_multi_turn_tracking


# =========================
# Config
# =========================

CASES_FILE = "rag_eval_questions.json"
TOP_K = 5
OUTPUT_MD_FILE = "rag_human_format_results.md"


# =========================
# Label Meanings
# =========================

TRAJECTORY_LABEL_MEANINGS = {
    "improving": "Emotion becomes more positive or more constructive across turns.",
    "planning_progress": "User moves toward concrete goals, planning, or next steps.",
    "stable": "Emotion remains mostly unchanged across turns.",
    "fluctuating": "Emotion changes back and forth without a clear direction.",
    "persistent_negative": "Negative emotion remains across multiple turns.",
    "persistent_negative_escalating": "Negative emotion remains and becomes stronger over time.",
    "worsening": "The latest turns show more distress than earlier turns.",
    "high_risk_worsening": "User expresses severe distress that may require safety-aware support.",
    "unknown": "The emotional trajectory cannot be clearly determined.",
}

USER_NEED_LABEL_MEANINGS = {
    "emotional_validation": "User mainly needs their feelings to be recognized and understood.",
    "reassurance": "User needs calming encouragement or confidence support.",
    "practical_guidance": "User wants concrete advice or a solution.",
    "emotional_regulation": "User needs help calming down before solving the problem.",
    "action_planning": "User wants structured next steps or a plan.",
    "clarification": "User need is unclear and requires a follow-up question.",
    "unknown": "The user need cannot be clearly determined.",
}


# =========================
# Loading Cases
# =========================

def load_cases(path: str = CASES_FILE) -> List[Dict[str, Any]]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the evaluation JSON file in the same folder as this script."
        )

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_case_id(case: Dict[str, Any], index: int) -> str:
    return str(case.get("case_id", f"C{index:02d}"))


def get_case_title(case: Dict[str, Any]) -> str:
    return str(case.get("title") or case.get("case_name") or "Untitled Case")


def get_conversation_text(case: Dict[str, Any]) -> str:
    if "conversation_text" in case:
        return str(case["conversation_text"]).strip()

    if "conversation" in case:
        conversation = case["conversation"]

        if isinstance(conversation, list):
            return "\n".join(str(turn).strip() for turn in conversation if str(turn).strip())

        return str(conversation).strip()

    raise KeyError("Each case must contain either 'conversation_text' or 'conversation'.")


def get_gold_trajectory(case: Dict[str, Any]) -> str:
    return str(case.get("gold_trajectory") or case.get("expected_trajectory") or "unknown")


def get_gold_need(case: Dict[str, Any]) -> str:
    return str(case.get("gold_need") or case.get("expected_need") or "unknown")


# =========================
# Matching
# =========================

def normalize_label(label: str) -> str:
    return str(label or "").strip().lower()


def expected_label_set(expected: str) -> set[str]:
    return {
        normalize_label(item)
        for item in str(expected).split("/")
        if normalize_label(item)
    }


def is_correct(predicted: str, expected: str) -> bool:
    return normalize_label(predicted) in expected_label_set(expected)


# =========================
# RAG-only Evaluation
# =========================

def summarize_rag_result(result: Dict[str, Any]) -> Dict[str, Any]:
    final_trajectory = result.get("final_trajectory", {})
    overall_need_summary = result.get("overall_need_summary", {})
    emotion_states = result.get("emotion_states", [])

    trajectory = final_trajectory.get("trajectory", "unknown")
    trajectory_confidence = float(final_trajectory.get("trajectory_confidence", 0.0) or 0.0)
    overall_need = overall_need_summary.get("overall_need", "unknown")

    need_confidences = [
        float(state.get("need_confidence", 0.0) or 0.0)
        for state in emotion_states
    ]

    if need_confidences:
        need_confidence = round(sum(need_confidences) / len(need_confidences), 4)
    else:
        need_confidence = 0.0

    need_transition = overall_need_summary.get("need_transition", [])

    return {
        "trajectory": trajectory,
        "trajectory_confidence": trajectory_confidence,
        "overall_need": overall_need,
        "need_confidence": need_confidence,
        "need_transition": need_transition,
        "trajectory_summary": final_trajectory.get("summary", ""),
        "need_summary": overall_need_summary.get("summary", ""),
    }


def run_single_case(case: Dict[str, Any], index: int, top_k: int = TOP_K) -> Dict[str, Any]:
    case_id = get_case_id(case, index)
    title = get_case_title(case)
    conversation_text = get_conversation_text(case)
    gold_trajectory = get_gold_trajectory(case)
    gold_need = get_gold_need(case)

    rag_result = process_multi_turn_tracking(
        conversation_text=conversation_text,
        top_k=top_k,
        use_rag=True,
    )

    rag_summary = summarize_rag_result(rag_result)

    return {
        "case_id": case_id,
        "title": title,
        "conversation_text": conversation_text,
        "gold_trajectory": gold_trajectory,
        "gold_need": gold_need,
        "rag": rag_summary,
        "trajectory_correct": is_correct(rag_summary["trajectory"], gold_trajectory),
        "need_correct": is_correct(rag_summary["overall_need"], gold_need),
    }


def run_rag_evaluation(cases: List[Dict[str, Any]], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    results = []
    total = len(cases)

    for index, case in enumerate(cases, start=1):
        case_id = get_case_id(case, index)
        title = get_case_title(case)

        print(f"Running RAG analysis {index}/{total}: {case_id} - {title}")

        try:
            results.append(
                run_single_case(
                    case=case,
                    index=index,
                    top_k=top_k,
                )
            )

        except Exception as e:
            results.append(
                {
                    "case_id": case_id,
                    "title": title,
                    "error": str(e),
                }
            )

    return results


# =========================
# Markdown Output
# =========================

def build_markdown_report(results: List[Dict[str, Any]]) -> str:
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    total = len(valid_results)
    trajectory_correct = sum(1 for r in valid_results if r["trajectory_correct"])
    need_correct = sum(1 for r in valid_results if r["need_correct"])

    trajectory_accuracy = round(trajectory_correct / total, 4) if total else 0.0
    need_accuracy = round(need_correct / total, 4) if total else 0.0

    md = []

    md.append("# RAG-only Annotation Results in Human Annotation Format\n\n")
    md.append(
        "This file reports RAG-only analysis results using a format similar to the simulated human annotation study. "
        "Each case is treated as one multi-turn conversation. The RAG system predicts the overall emotional trajectory and the main user support need.\n\n"
    )

    md.append("## Label Set\n\n")

    md.append("### Emotional Trajectory Labels\n\n")
    md.append("| Label | Meaning |\n")
    md.append("|---|---|\n")
    for label, meaning in TRAJECTORY_LABEL_MEANINGS.items():
        md.append(f"| `{label}` | {meaning} |\n")

    md.append("\n### User Need Labels\n\n")
    md.append("| Label | Meaning |\n")
    md.append("|---|---|\n")
    for label, meaning in USER_NEED_LABEL_MEANINGS.items():
        md.append(f"| `{label}` | {meaning} |\n")

    md.append("\n## Overall RAG Evaluation Summary\n\n")
    md.append("| Metric | Value |\n")
    md.append("|---|---:|\n")
    md.append(f"| Number of valid cases | {total} |\n")
    md.append(f"| Trajectory accuracy | {trajectory_accuracy} |\n")
    md.append(f"| Need accuracy | {need_accuracy} |\n")
    md.append(f"| Error cases | {len(error_results)} |\n")

    md.append("\n## Case-level RAG Summary\n\n")
    md.append("| Case | Title | Gold Trajectory | RAG Trajectory | Correct | Gold Need | RAG Need | Correct | Trajectory Confidence |\n")
    md.append("|---|---|---|---|---:|---|---|---:|---:|\n")

    for r in valid_results:
        rag = r["rag"]
        md.append(
            f"| {r['case_id']} | {r['title']} | `{r['gold_trajectory']}` | `{rag['trajectory']}` | "
            f"{r['trajectory_correct']} | `{r['gold_need']}` | `{rag['overall_need']}` | "
            f"{r['need_correct']} | {rag['trajectory_confidence']} |\n"
        )

    for r in valid_results:
        rag = r["rag"]
        trajectory_meaning = TRAJECTORY_LABEL_MEANINGS.get(
            normalize_label(rag["trajectory"]),
            "No label meaning available.",
        )

        md.append(f"\n---\n\n## {r['case_id']}: {r['title']}\n\n")

        md.append("### Multi-turn Conversation\n\n")
        turns = [
            line.strip()
            for line in r["conversation_text"].splitlines()
            if line.strip()
        ]

        for i, turn in enumerate(turns, start=1):
            md.append(f"{i}. User: {turn}\n")

        md.append("\n### Gold Answers\n\n")
        md.append(f"- Gold emotional trajectory: `{r['gold_trajectory']}`\n")
        md.append(f"- Gold user need: `{r['gold_need']}`\n")

        md.append("\n### RAG System Annotation\n\n")
        md.append("| Annotator | Trajectory Label | User Need Label | Confidence | Emotional Trajectory Label Meaning |\n")
        md.append("|---|---|---|---:|---|\n")
        md.append(
            f"| RAG_System | `{rag['trajectory']}` | `{rag['overall_need']}` | "
            f"{rag['trajectory_confidence']} | {trajectory_meaning} |\n"
        )

        md.append("\n### RAG Need Transition\n\n")
        if rag["need_transition"]:
            md.append(f"`{' -> '.join(rag['need_transition'])}`\n")
        else:
            md.append("`unknown`\n")

        md.append("\n### RAG Summaries\n\n")
        md.append(f"**Trajectory summary:** {rag['trajectory_summary']}\n\n")
        md.append(f"**Need summary:** {rag['need_summary']}\n")

    if error_results:
        md.append("\n---\n\n## Error Cases\n\n")
        md.append("| Case | Title | Error |\n")
        md.append("|---|---|---|\n")

        for r in error_results:
            md.append(f"| {r['case_id']} | {r['title']} | {r['error']} |\n")

    return "".join(md)


def save_markdown_report(results: List[Dict[str, Any]], output_path: str = OUTPUT_MD_FILE) -> None:
    report = build_markdown_report(results)
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"Saved RAG-only markdown report to: {output_path}")


# =========================
# Terminal Tables
# =========================

def print_summary_table(results: List[Dict[str, Any]]) -> None:
    valid = [r for r in results if "error" not in r]

    if not valid:
        print("No valid results.")
        return

    trajectory_acc = round(sum(r["trajectory_correct"] for r in valid) / len(valid), 4)
    need_acc = round(sum(r["need_correct"] for r in valid) / len(valid), 4)

    print("\n" + "=" * 100)
    print("RAG-only Evaluation Summary")
    print("=" * 100)
    print(f"{'Metric':<35} | {'Value':<20}")
    print("-" * 100)
    print(f"{'Number of Valid Cases':<35} | {len(valid):<20}")
    print(f"{'Trajectory Accuracy':<35} | {trajectory_acc:<20}")
    print(f"{'Need Accuracy':<35} | {need_acc:<20}")
    print("=" * 100)


def print_case_table(results: List[Dict[str, Any]]) -> None:
    valid = [r for r in results if "error" not in r]

    print("\n" + "=" * 160)
    print("RAG-only Case-level Results")
    print("=" * 160)
    print(
        f"{'Case':<6} | {'Gold Traj':<32} | {'RAG Traj':<28} | {'T Correct':<9} | "
        f"{'Gold Need':<24} | {'RAG Need':<24} | {'N Correct':<9} | {'Conf':<6}"
    )
    print("-" * 160)

    for r in valid:
        rag = r["rag"]
        print(
            f"{r['case_id']:<6} | {r['gold_trajectory']:<32} | {rag['trajectory']:<28} | "
            f"{str(r['trajectory_correct']):<9} | {r['gold_need']:<24} | "
            f"{rag['overall_need']:<24} | {str(r['need_correct']):<9} | "
            f"{rag['trajectory_confidence']:<6}"
        )

    print("=" * 160)


# =========================
# Main
# =========================

if __name__ == "__main__":
    cases = load_cases(CASES_FILE)

    print(f"Loaded {len(cases)} evaluation cases.")
    print("Running RAG-only evaluation...")

    results = run_rag_evaluation(
        cases=cases,
        top_k=TOP_K,
    )

    print_summary_table(results)
    print_case_table(results)
    save_markdown_report(results, OUTPUT_MD_FILE)
