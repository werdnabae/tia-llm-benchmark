"""
evaluate.py — Compute evaluation metrics for all model predictions.

Metrics:
  Recommendations (primary):
    - measure_type F1 (exact match on measure_type category)
    - precision, recall, count difference
  Issues:
    - location match F1 (fuzzy string match on intersection name)
    - LOS match (when both predicted and GT have LOS)

Saves per-model results to data/results/{model_key}.json.

Usage:
    python3 evaluate.py
    python3 evaluate.py --models llama kimi
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'pipeline'))
from config import LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "evaluate.log")],
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_PATH = DATA_DIR / "final" / "tia_dataset.jsonl"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "llama-3.3-70b",
    "gpt-oss-120b",
    "nemotron-120b",
    "qwen3-30b",
    "gemma-3-12b",
    "frequency-baseline",
]


# ── String matching helpers ────────────────────────────────────────────────────


def normalise_location(s: str) -> str:
    """Lowercase, strip punctuation, normalise whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[&/@\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def location_match(pred_loc: str, gt_loc: str, threshold: float = 0.5) -> bool:
    """
    Returns True if the two location strings share enough tokens to be considered
    the same intersection. Uses Jaccard similarity on word tokens.
    """
    p_tokens = set(normalise_location(pred_loc).split())
    g_tokens = set(normalise_location(gt_loc).split())
    if not p_tokens or not g_tokens:
        return False
    # Remove very common words
    stopwords = {
        "the",
        "and",
        "at",
        "of",
        "in",
        "on",
        "road",
        "street",
        "avenue",
        "boulevard",
        "drive",
        "way",
        "lane",
        "highway",
        "blvd",
        "ave",
        "st",
        "dr",
        "ln",
        "rd",
        "hwy",
        "not_specified",
    }
    p_tokens -= stopwords
    g_tokens -= stopwords
    if not p_tokens or not g_tokens:
        return False
    jaccard = len(p_tokens & g_tokens) / len(p_tokens | g_tokens)
    return jaccard >= threshold


# ── Recommendation matching ────────────────────────────────────────────────────

_TRIVIAL_LOCS = {"not_specified", "", "n/a", "project-wide", "site-wide", "various"}


def has_meaningful_location(loc: str) -> bool:
    """
    Returns True if the location string contains at least one meaningful
    (non-stopword) token that could identify a specific place.
    """
    if not loc or loc.lower().strip() in _TRIVIAL_LOCS:
        return False
    tokens = set(normalise_location(loc).split())
    stopwords = {
        "the",
        "and",
        "at",
        "of",
        "in",
        "on",
        "road",
        "street",
        "avenue",
        "boulevard",
        "drive",
        "way",
        "lane",
        "highway",
        "blvd",
        "ave",
        "st",
        "dr",
        "ln",
        "rd",
        "hwy",
        "not_specified",
        "project",
        "site",
        "wide",
        "intersection",
        "area",
        "corridor",
        "near",
        "between",
        "along",
    }
    meaningful = tokens - stopwords
    return len(meaningful) >= 1


def match_recommendations(predicted: list, ground_truth: list) -> dict:
    """
    Match predicted recommendations to GT using a two-tier approach:

    Tier 1 (strict) — both sides have a meaningful location:
        Require location_match(pred, gt) AND measure_type exact match.

    Tier 2 (loose) — either side lacks a specific location:
        Require only measure_type exact match (greedy, one-to-one).

    This mirrors how the issue evaluator works — location agreement is
    required when the information is present, but gracefully degrades
    when recommendations are project-wide (e.g. TDM programs).

    Also returns type_only_f1 (the old bag-of-types metric) so both
    can be reported side by side.
    """
    from collections import Counter

    matched_pred: set = set()
    tp_strict = 0

    for gt_item in ground_truth:
        gt_loc = str(gt_item.get("location", "not_specified"))
        gt_type = gt_item.get("measure_type", "other")
        gt_has_loc = has_meaningful_location(gt_loc)

        best_idx = None
        for i, pred_item in enumerate(predicted):
            if i in matched_pred:
                continue
            pred_loc = str(pred_item.get("location", "not_specified"))
            pred_type = pred_item.get("measure_type", "other")
            pred_has_loc = has_meaningful_location(pred_loc)

            if pred_type != gt_type:
                continue  # measure_type must always match

            if gt_has_loc and pred_has_loc:
                # Tier 1: both have locations — require location agreement
                if location_match(pred_loc, gt_loc):
                    best_idx = i
                    break
            else:
                # Tier 2: at least one lacks a specific location — type match alone
                best_idx = i
                break

        if best_idx is not None:
            tp_strict += 1
            matched_pred.add(best_idx)

    fp_strict = len(predicted) - tp_strict
    fn_strict = len(ground_truth) - tp_strict

    prec_strict = (
        tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) > 0 else 0.0
    )
    rec_strict = (
        tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) > 0 else 0.0
    )
    f1_strict = (
        2 * prec_strict * rec_strict / (prec_strict + rec_strict)
        if (prec_strict + rec_strict) > 0
        else 0.0
    )

    # ── Also compute the original type-only metric for comparison ────────────
    pred_counter = Counter(r.get("measure_type", "other") for r in predicted)
    gt_counter = Counter(r.get("measure_type", "other") for r in ground_truth)
    tp_type = sum((pred_counter & gt_counter).values())
    fp_type = sum(pred_counter.values()) - tp_type
    fn_type = sum(gt_counter.values()) - tp_type
    prec_t = tp_type / (tp_type + fp_type) if (tp_type + fp_type) > 0 else 0.0
    rec_t = tp_type / (tp_type + fn_type) if (tp_type + fn_type) > 0 else 0.0
    f1_type = 2 * prec_t * rec_t / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0.0

    return {
        # Primary metric: location + type (strict)
        "precision": round(prec_strict, 4),
        "recall": round(rec_strict, 4),
        "f1": round(f1_strict, 4),
        "tp": tp_strict,
        "fp": fp_strict,
        "fn": fn_strict,
        "pred_count": len(predicted),
        "gt_count": len(ground_truth),
        "count_diff": len(predicted) - len(ground_truth),
        # Secondary metric: type-only (original, for comparison)
        "type_only_f1": round(f1_type, 4),
        "type_only_precision": round(prec_t, 4),
        "type_only_recall": round(rec_t, 4),
        "pred_types": dict(pred_counter),
        "gt_types": dict(gt_counter),
    }


# ── Issue matching ─────────────────────────────────────────────────────────────


def match_issues(predicted: list, ground_truth: list) -> dict:
    """
    Match predicted issues to GT issues by location (fuzzy) and scenario.
    Returns precision, recall, F1 and LOS match rate.
    """
    if not predicted and not ground_truth:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "pred_count": 0,
            "gt_count": 0,
            "los_match_rate": None,
        }

    # Greedy matching: for each GT issue, find the best unmatched prediction
    matched_pred_indices = set()
    tp = 0
    los_matches = 0
    los_total = 0

    for gt_issue in ground_truth:
        gt_loc = str(gt_issue.get("location", ""))
        best_idx = None
        for i, pred_issue in enumerate(predicted):
            if i in matched_pred_indices:
                continue
            pred_loc = str(pred_issue.get("location", ""))
            if location_match(pred_loc, gt_loc):
                best_idx = i
                break

        if best_idx is not None:
            tp += 1
            matched_pred_indices.add(best_idx)
            # Check LOS agreement
            gt_los = gt_issue.get("los")
            pred_los = predicted[best_idx].get("los")
            if gt_los and pred_los:
                los_total += 1
                if gt_los == pred_los:
                    los_matches += 1

    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    los_match_rate = round(los_matches / los_total, 4) if los_total > 0 else None

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": len(predicted),
        "gt_count": len(ground_truth),
        "los_match_rate": los_match_rate,
    }


# ── Per-case evaluation ────────────────────────────────────────────────────────


def evaluate_model(model_key: str, records_by_id: dict) -> dict:
    """Evaluate all three tasks for one model. Returns full result dict."""
    task_dirs = {
        "main": DATA_DIR / "predictions_main" / model_key,
        "fewshot": DATA_DIR / "predictions_fewshot" / model_key,
        "rag": DATA_DIR / "predictions_rag" / model_key,
    }

    results = {
        "model": model_key,
        "tasks": {},
    }

    for task, pred_dir in task_dirs.items():
        if not pred_dir.exists():
            log.info("  %s Task %s: no predictions directory", model_key, task)
            results["tasks"][task] = {"status": "missing"}
            continue

        pred_files = sorted(pred_dir.glob("*.json"))
        if not pred_files:
            log.info("  %s Task %s: 0 predictions", model_key, task)
            results["tasks"][task] = {"status": "empty"}
            continue

        case_results = []
        for pf in pred_files:
            case_id = pf.stem
            record = records_by_id.get(case_id)
            if not record:
                continue

            pred_data = json.loads(pf.read_text())
            pred = pred_data.get("prediction", {})

            gt = record.get("ground_truth", {})

            case_result = {"case_id": case_id}

            # Recommendations
            pred_recs = pred.get("recommendations", []) or []
            gt_recs = gt.get("recommendations", []) or []
            rec_metrics = match_recommendations(pred_recs, gt_recs)
            case_result["recommendations"] = rec_metrics

            # No issue prediction in Task C (issues are given as input)
            # Keep the structure consistent by recording empty issue metrics
            case_result["issues"] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "pred_count": 0,
                "gt_count": 0,
                "los_match_rate": None,
            }

            case_results.append(case_result)

        if not case_results:
            results["tasks"][task] = {"status": "no_matches"}
            continue

        # Aggregate across cases
        agg = _aggregate_case_results(case_results, task)
        agg["case_count"] = len(case_results)
        agg["per_case"] = case_results
        results["tasks"][task] = agg
        log.info(
            "  %s Task %s: n=%d  rec_f1=%.3f  %s",
            model_key,
            task,
            len(case_results),
            agg.get("recommendations", {}).get("mean_f1", 0),
            f"iss_f1={agg.get('issues', {}).get('mean_f1', 0):.3f}"
            if task == "B"
            else "",
        )

    return results


def _aggregate_case_results(case_results: list, task: str) -> dict:
    """Compute mean metrics across cases."""
    agg: dict = {}

    # Recommendations — primary (location + type) and secondary (type-only)
    rec_metrics = [c["recommendations"] for c in case_results]
    agg["recommendations"] = {
        # Primary metric: location-aware (location match + type match)
        "mean_precision": round(_mean([m["precision"] for m in rec_metrics]), 4),
        "mean_recall": round(_mean([m["recall"] for m in rec_metrics]), 4),
        "mean_f1": round(_mean([m["f1"] for m in rec_metrics]), 4),
        "micro_f1": _micro_f1(rec_metrics),
        "total_tp": sum(m["tp"] for m in rec_metrics),
        "total_fp": sum(m["fp"] for m in rec_metrics),
        "total_fn": sum(m["fn"] for m in rec_metrics),
        "total_pred": sum(m["pred_count"] for m in rec_metrics),
        "total_gt": sum(m["gt_count"] for m in rec_metrics),
        "mean_count_diff": round(_mean([m["count_diff"] for m in rec_metrics]), 2),
        # Secondary metric: type-only (for comparison / ablation)
        "type_only_mean_f1": round(_mean([m["type_only_f1"] for m in rec_metrics]), 4),
        "type_only_mean_precision": round(
            _mean([m["type_only_precision"] for m in rec_metrics]), 4
        ),
        "type_only_mean_recall": round(
            _mean([m["type_only_recall"] for m in rec_metrics]), 4
        ),
    }

    # Issues (Task B only)
    if task == "B":
        iss_metrics = [c["issues"] for c in case_results]
        los_rates = [
            m["los_match_rate"] for m in iss_metrics if m["los_match_rate"] is not None
        ]
        agg["issues"] = {
            "mean_precision": round(_mean([m["precision"] for m in iss_metrics]), 4),
            "mean_recall": round(_mean([m["recall"] for m in iss_metrics]), 4),
            "mean_f1": round(_mean([m["f1"] for m in iss_metrics]), 4),
            "total_tp": sum(m["tp"] for m in iss_metrics),
            "total_fp": sum(m["fp"] for m in iss_metrics),
            "total_fn": sum(m["fn"] for m in iss_metrics),
            "total_pred": sum(m["pred_count"] for m in iss_metrics),
            "total_gt": sum(m["gt_count"] for m in iss_metrics),
            "mean_los_match_rate": round(_mean(los_rates), 4) if los_rates else None,
            "micro_f1": _micro_f1(iss_metrics),
        }

    return agg


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _micro_f1(metrics: list) -> float:
    """Micro-averaged F1 across all cases."""
    tp = sum(m.get("tp", 0) for m in metrics)
    fp = sum(m.get("fp", 0) for m in metrics)
    fn = sum(m.get("fn", 0) for m in metrics)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────


def run_evaluation(model_keys=None) -> None:
    if model_keys is None:
        model_keys = MODELS

    records = [
        json.loads(l) for l in DATASET_PATH.read_text().splitlines() if l.strip()
    ]
    records_by_id = {r["case_id"]: r for r in records}
    log.info("Loaded %d records", len(records))

    for model_key in model_keys:
        log.info("Evaluating model: %s", model_key)
        result = evaluate_model(model_key, records_by_id)
        out_path = RESULTS_DIR / f"{model_key}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log.info("  Saved → %s", out_path)

    log.info("Evaluation complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()
    run_evaluation(model_keys=args.models)
