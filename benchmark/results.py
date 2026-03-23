"""
results.py — Aggregate per-model results into summary.json.

Produces data/results/summary.json with:
  - Model rankings (Task A, A+RAG, B)
  - RAG vs no-RAG comparison (delta F1)
  - Average metrics across all cases
  - Per-task leaderboard

Usage:
    python3 results.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'pipeline'))
from config import LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "results.log")],
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "data" / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

MODELS = [
    "llama-3.3-70b",
    "gpt-oss-120b",
    "nemotron-120b",
    "qwen3-30b",
    "gemma-3-12b",
    "frequency-baseline",
]

TASK_LABELS = {
    "main": "Task C: Conditioned Recommendation Generation (no RAG)",
    "rag": "Task+RAG: Conditioned Recommendation Generation (with RAG)",
}


def load_model_results() -> dict:
    results = {}
    for model in MODELS:
        path = RESULTS_DIR / f"{model}.json"
        if path.exists():
            results[model] = json.loads(path.read_text())
        else:
            log.warning("No results file for %s", model)
    return results


def build_leaderboard(results: dict, task: str, metric: str = "mean_f1") -> list:
    """Return sorted list of (model, score) for a given task and metric."""
    scores = []
    for model, r in results.items():
        task_data = r.get("tasks", {}).get(task, {})
        if task_data.get("status"):
            # Missing or empty
            scores.append((model, None))
            continue
        rec_metrics = task_data.get("recommendations", {})
        score = rec_metrics.get(metric)
        scores.append((model, score))

    # Sort: available scores first (descending), unavailable last
    scores.sort(key=lambda x: (x[1] is None, -(x[1] or 0)))
    return scores


def build_rag_comparison(results: dict) -> list:
    """
    For each model, compute RAG delta F1 (C_RAG - C) for recommendations.
    Returns list of {model, f1_no_rag, f1_rag, delta, relative_improvement}.
    """
    comparison = []
    for model, r in results.items():
        tasks = r.get("tasks", {})
        a_data = tasks.get("main", {})
        rag_data = tasks.get("rag", {})

        a_f1 = (
            a_data.get("recommendations", {}).get("mean_f1")
            if not a_data.get("status")
            else None
        )
        rag_f1 = (
            rag_data.get("recommendations", {}).get("mean_f1")
            if not rag_data.get("status")
            else None
        )

        delta = None
        rel = None
        if a_f1 is not None and rag_f1 is not None:
            delta = round(rag_f1 - a_f1, 4)
            rel = round(delta / a_f1 * 100, 1) if a_f1 > 0 else None

        comparison.append(
            {
                "model": model,
                "f1_no_rag": a_f1,
                "f1_rag": rag_f1,
                "delta_f1": delta,
                "relative_improvement": rel,
            }
        )

    comparison.sort(key=lambda x: (x["delta_f1"] is None, -(x["delta_f1"] or -999)))
    return comparison


def build_task_b_summary(results: dict) -> list:
    """
    For Task B, summarise both recommendations and issues metrics.
    """
    rows = []
    for model, r in results.items():
        b_data = r.get("tasks", {}).get("B", {})
        if b_data.get("status"):
            rows.append({"model": model, "status": b_data["status"]})
            continue

        rec = b_data.get("recommendations", {})
        iss = b_data.get("issues", {})
        rows.append(
            {
                "model": model,
                "rec_mean_f1": rec.get("mean_f1"),
                "rec_micro_f1": rec.get("micro_f1"),
                "iss_mean_f1": iss.get("mean_f1"),
                "iss_micro_f1": iss.get("micro_f1"),
                "iss_los_match_rate": iss.get("mean_los_match_rate"),
                "case_count": b_data.get("case_count"),
            }
        )

    rows.sort(
        key=lambda x: (x.get("rec_mean_f1") is None, -(x.get("rec_mean_f1") or 0))
    )
    return rows


def print_summary(summary: dict) -> None:
    print("\n" + "═" * 65)
    print("  TIA BENCHMARK RESULTS — SUMMARY")
    print("═" * 65)

    # Task C ranking
    print("\n▶ Task C — Conditioned Recommendation Generation (no RAG)")
    print(
        f"  {'Model':<22} {'Loc+Type F1':>12} {'Type-only F1':>13} {'Prec':>7} {'Rec':>7}"
    )
    print("  " + "-" * 64)
    for model, r in sorted(
        [(m, r) for m, r in summary["all_models"].items()],
        key=lambda x: (
            -(
                x[1]
                .get("tasks", {})
                .get("main", {})
                .get("recommendations", {})
                .get("mean_f1")
                or 0
            )
        ),
    ):
        rd = r.get("tasks", {}).get("main", {}).get("recommendations", {})
        if not rd:
            print(f"  {model:<22} {'N/A':>12}")
            continue
        print(
            f"  {model:<22} {rd.get('mean_f1', 0):>12.3f}"
            f" {rd.get('type_only_mean_f1', 0):>13.3f}"
            f" {rd.get('mean_precision', 0):>7.3f}"
            f" {rd.get('mean_recall', 0):>7.3f}"
        )

    # Condition comparison table (main / fewshot / rag)
    print("\n▶ All conditions — F1_loc by model and experimental condition")
    print(f"  {'Model':<22} {'Zero-shot':>10} {'Few-shot':>10} {'RAG':>8}")
    print("  " + "-" * 54)
    for model in summary["models_evaluated"]:
        r = summary["all_models"][model]
        f_main = (
            r.get("tasks", {}).get("main", {}).get("recommendations", {}).get("mean_f1")
        )
        f_fewshot = (
            r.get("tasks", {})
            .get("fewshot", {})
            .get("recommendations", {})
            .get("mean_f1")
        )
        f_rag = (
            r.get("tasks", {}).get("rag", {}).get("recommendations", {}).get("mean_f1")
        )

        def fmt(v):
            return f"{v:.3f}" if v is not None else "N/A"

        print(f"  {model:<22} {fmt(f_main):>10} {fmt(f_fewshot):>10} {fmt(f_rag):>8}")

    print("\n" + "═" * 65 + "\n")


def run_results() -> None:
    results = load_model_results()
    if not results:
        log.error("No model results found in %s", RESULTS_DIR)
        return

    task_c_ranking = build_leaderboard(results, "main")
    task_fewshot_ranking = build_leaderboard(results, "fewshot")
    task_rag_ranking = build_leaderboard(results, "rag")
    rag_comparison = build_rag_comparison(results)

    def best(ranking):
        valid = [(m, s) for m, s in ranking if s is not None]
        return valid[0] if valid else (None, None)

    summary = {
        "models_evaluated": list(results.keys()),
        "task_c_ranking": [{"model": m, "rec_mean_f1": s} for m, s in task_c_ranking],
        "task_fewshot_ranking": [
            {"model": m, "rec_mean_f1": s} for m, s in task_fewshot_ranking
        ],
        "task_rag_ranking": [
            {"model": m, "rec_mean_f1": s} for m, s in task_rag_ranking
        ],
        "rag_comparison": rag_comparison,
        "best_model_task_c": {
            "model": best(task_c_ranking)[0],
            "f1": best(task_c_ranking)[1],
        },
        "best_model_task_fewshot": {
            "model": best(task_fewshot_ranking)[0],
            "f1": best(task_fewshot_ranking)[1],
        },
        "best_model_task_rag": {
            "model": best(task_rag_ranking)[0],
            "f1": best(task_rag_ranking)[1],
        },
        "all_models": results,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("Summary saved → %s", SUMMARY_PATH)
    print_summary(summary)


if __name__ == "__main__":
    run_results()
