import json
from pathlib import Path
from typing import Any, Dict, List

from dialogue_manager import process_multi_turn_tracking

CASES_FILE = "evaluation_cases_20.json"
TOP_K = 5


def load_cases(path: str = CASES_FILE) -> List[Dict[str, Any]]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put evaluation_cases_500.json in the same folder as evaluate_tracking.py."
        )

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    final_trajectory = result.get("final_trajectory", {})
    overall_need_summary = result.get("overall_need_summary", {})

    return {
        "trajectory": final_trajectory.get("trajectory", "unknown"),
        "trajectory_confidence": float(final_trajectory.get("trajectory_confidence", 0.0) or 0.0),
        "overall_need": overall_need_summary.get("overall_need", "unknown"),
    }


def run_single_case(case: Dict[str, Any], top_k: int = TOP_K) -> Dict[str, Any]:
    conversation = case["conversation"]

    no_rag_result = process_multi_turn_tracking(
        conversation_text=conversation,
        top_k=top_k,
        use_rag=False,
    )

    rag_result = process_multi_turn_tracking(
        conversation_text=conversation,
        top_k=top_k,
        use_rag=True,
    )

    no_rag = summarize_result(no_rag_result)
    rag = summarize_result(rag_result)

    return {
        "case_id": case.get("case_id", ""),
        "case_name": case["case_name"],
        "category": case.get("category", "unknown"),
        "expected_trajectory": case["expected_trajectory"],
        "expected_need": case["expected_need"],
        "no_rag": no_rag,
        "rag": rag,
        "no_rag_traj_correct": is_correct(no_rag["trajectory"], case["expected_trajectory"]),
        "rag_traj_correct": is_correct(rag["trajectory"], case["expected_trajectory"]),
        "no_rag_need_correct": is_correct(no_rag["overall_need"], case["expected_need"]),
        "rag_need_correct": is_correct(rag["overall_need"], case["expected_need"]),
    }


def run_evaluation(test_cases: List[Dict[str, Any]], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    results = []
    total = len(test_cases)

    for idx, case in enumerate(test_cases, start=1):
        print(f"Running {idx}/{total}: {case['case_name']}")

        try:
            results.append(run_single_case(case=case, top_k=top_k))
        except Exception as e:
            results.append({
                "case_id": case.get("case_id", ""),
                "case_name": case.get("case_name", "unknown"),
                "category": case.get("category", "unknown"),
                "error": str(e),
            })

    return results


def avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if "error" not in r]
    n = len(valid)

    if n == 0:
        return {
            "num_cases": 0,
            "no_rag_trajectory_accuracy": 0,
            "rag_trajectory_accuracy": 0,
            "no_rag_need_accuracy": 0,
            "rag_need_accuracy": 0,
            "no_rag_avg_conf_correct": 0,
            "no_rag_avg_conf_wrong": 0,
            "rag_avg_conf_correct": 0,
            "rag_avg_conf_wrong": 0,
        }

    no_rag_conf_correct = [r["no_rag"]["trajectory_confidence"] for r in valid if r["no_rag_traj_correct"]]
    no_rag_conf_wrong = [r["no_rag"]["trajectory_confidence"] for r in valid if not r["no_rag_traj_correct"]]
    rag_conf_correct = [r["rag"]["trajectory_confidence"] for r in valid if r["rag_traj_correct"]]
    rag_conf_wrong = [r["rag"]["trajectory_confidence"] for r in valid if not r["rag_traj_correct"]]

    return {
        "num_cases": n,
        "no_rag_trajectory_accuracy": round(sum(r["no_rag_traj_correct"] for r in valid) / n, 4),
        "rag_trajectory_accuracy": round(sum(r["rag_traj_correct"] for r in valid) / n, 4),
        "no_rag_need_accuracy": round(sum(r["no_rag_need_correct"] for r in valid) / n, 4),
        "rag_need_accuracy": round(sum(r["rag_need_correct"] for r in valid) / n, 4),
        "no_rag_avg_conf_correct": avg(no_rag_conf_correct),
        "no_rag_avg_conf_wrong": avg(no_rag_conf_wrong),
        "rag_avg_conf_correct": avg(rag_conf_correct),
        "rag_avg_conf_wrong": avg(rag_conf_wrong),
    }


def aggregate_by_category(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [r for r in results if "error" not in r]
    categories = sorted(set(r["category"] for r in valid))
    rows = []

    for category in categories:
        items = [r for r in valid if r["category"] == category]
        n = len(items)

        rows.append({
            "category": category,
            "cases": n,
            "no_rag_traj_acc": round(sum(r["no_rag_traj_correct"] for r in items) / n, 4),
            "rag_traj_acc": round(sum(r["rag_traj_correct"] for r in items) / n, 4),
            "no_rag_need_acc": round(sum(r["no_rag_need_correct"] for r in items) / n, 4),
            "rag_need_acc": round(sum(r["rag_need_correct"] for r in items) / n, 4),
        })

    return rows


def print_main_summary_table(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("Overall Evaluation Summary")
    print("=" * 100)
    print(f"{'Metric':<45} | {'No-RAG':<18} | {'RAG':<18}")
    print("-" * 100)
    print(f"{'Number of Cases':<45} | {metrics['num_cases']:<18} | {metrics['num_cases']:<18}")
    print(f"{'Trajectory Accuracy':<45} | {metrics['no_rag_trajectory_accuracy']:<18} | {metrics['rag_trajectory_accuracy']:<18}")
    print(f"{'Need Accuracy':<45} | {metrics['no_rag_need_accuracy']:<18} | {metrics['rag_need_accuracy']:<18}")
    print(f"{'Avg Confidence When Trajectory Correct':<45} | {metrics['no_rag_avg_conf_correct']:<18} | {metrics['rag_avg_conf_correct']:<18}")
    print(f"{'Avg Confidence When Trajectory Wrong':<45} | {metrics['no_rag_avg_conf_wrong']:<18} | {metrics['rag_avg_conf_wrong']:<18}")
    print("=" * 100)


def print_category_table(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 120)
    print("Accuracy by Category")
    print("=" * 120)
    print(
        f"{'Category':<32} | {'Cases':<8} | {'No-RAG Traj':<12} | {'RAG Traj':<12} | "
        f"{'No-RAG Need':<12} | {'RAG Need':<12}"
    )
    print("-" * 120)

    for row in rows:
        print(
            f"{row['category']:<32} | {row['cases']:<8} | "
            f"{row['no_rag_traj_acc']:<12} | {row['rag_traj_acc']:<12} | "
            f"{row['no_rag_need_acc']:<12} | {row['rag_need_acc']:<12}"
        )

    print("=" * 120)


def print_case_table(results: List[Dict[str, Any]], max_rows: int = 30) -> None:
    valid = [r for r in results if "error" not in r]

    print("\n" + "=" * 160)
    print(f"Case-level Comparison Table: First {min(max_rows, len(valid))} Cases")
    print("=" * 160)
    print(
        f"{'ID':<5} | {'Category':<28} | {'Expected Traj':<32} | {'No-RAG Traj':<24} | "
        f"{'RAG Traj':<24} | {'No-RAG Conf':<11} | {'RAG Conf':<8}"
    )
    print("-" * 160)

    for r in valid[:max_rows]:
        print(
            f"{str(r['case_id']):<5} | {r['category']:<28} | {r['expected_trajectory']:<32} | "
            f"{r['no_rag']['trajectory']:<24} | {r['rag']['trajectory']:<24} | "
            f"{r['no_rag']['trajectory_confidence']:<11} | {r['rag']['trajectory_confidence']:<8}"
        )

    print("=" * 160)


def print_error_table(results: List[Dict[str, Any]]) -> None:
    errors = [r for r in results if "error" in r]

    if not errors:
        return

    print("\n" + "=" * 100)
    print("Errors")
    print("=" * 100)
    print(f"{'Case':<40} | {'Error':<50}")
    print("-" * 100)

    for r in errors:
        print(f"{r.get('case_name', 'unknown'):<40} | {r['error']:<50}")

    print("=" * 100)


if __name__ == "__main__":
    cases = load_cases(CASES_FILE)
    print(f"Loaded {len(cases)} evaluation cases.")

    results = run_evaluation(test_cases=cases, top_k=TOP_K)

    metrics = aggregate_metrics(results)
    category_rows = aggregate_by_category(results)

    print_main_summary_table(metrics)
    print_category_table(category_rows)
    print_case_table(results, max_rows=30)
    print_error_table(results)
