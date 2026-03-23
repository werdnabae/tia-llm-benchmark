"""
baselines.py — Frequency baseline predictions for the TIA benchmark.

Frequency baseline: for each case, predict the most common measure_types
observed in the training set (all other cases, LOO).

Also runs LLM-as-judge evaluation: uses llama-3.3-70b to score each
model's predictions on a 1-5 scale for domain appropriateness.

Saves:
  data/predictions/frequency-baseline/{case_id}.json
  data/predictions_rag/frequency-baseline/{case_id}.json  (same — baseline has no RAG)
  data/predictions_full/frequency-baseline/{case_id}.json
  data/results/judge_scores.json
"""

import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / 'pipeline'))

_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from config import LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "baselines.log")],
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_PATH = DATA_DIR / "final" / "tia_dataset.jsonl"

CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
JUDGE_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
CF_BASE = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
)

VALID_MEASURE_TYPES = [
    "add_turn_lane",
    "signal_timing_optimization",
    "new_traffic_signal",
    "access_modification",
    "driveway_consolidation",
    "road_widening",
    "roundabout_installation",
    "pedestrian_improvement",
    "transit_improvement",
    "tdm_measure",
    "no_mitigation_required",
    "other",
]


# ── Frequency Baseline ─────────────────────────────────────────────────────────


def build_frequency_baseline(records: list) -> dict:
    """
    LOO frequency baseline.
    For each case, compute the most common measure_types from all other cases
    and predict the top-N most frequent ones (N = avg GT count for that project type).
    Returns {case_id: {"recommendations": [...]}} predictions.
    """
    predictions = {}

    for i, record in enumerate(records):
        case_id = record["case_id"]
        project_type = record["input"].get("project_type", "other")

        # Training set = all other cases
        train = [r for j, r in enumerate(records) if j != i]

        # Count measure_types across all training GT
        type_counts = Counter()
        for r in train:
            for rec in r["ground_truth"].get("recommendations") or []:
                mt = rec.get("measure_type", "other")
                type_counts[mt] += 1

        # Also count for same project type only
        same_type_counts = Counter()
        for r in train:
            if r["input"].get("project_type") == project_type:
                for rec in r["ground_truth"].get("recommendations") or []:
                    mt = rec.get("measure_type", "other")
                    same_type_counts[mt] += 1

        # Target N = average GT recommendation count across training set
        gt_counts = [
            len(r["ground_truth"].get("recommendations") or [])
            for r in train
            if r["ground_truth"].get("recommendations")
        ]
        target_n = max(1, round(sum(gt_counts) / len(gt_counts))) if gt_counts else 3

        # Use same-type counts if available, else global
        counts_to_use = (
            same_type_counts if sum(same_type_counts.values()) >= 5 else type_counts
        )

        # Predict top-N most frequent types
        top_types = [mt for mt, _ in counts_to_use.most_common(target_n)]
        recs = [
            {
                "location": "not_specified",
                "measure_type": mt,
                "description": f"Frequency baseline: {mt.replace('_', ' ')} (most common for {project_type})",
                "timing": "not_specified",
                "responsible_party": "not_specified",
            }
            for mt in top_types
        ]

        predictions[case_id] = {"recommendations": recs}
        log.debug("%s: predicting %d recs %s", case_id, len(recs), top_types)

    return predictions


def save_frequency_predictions(predictions: dict, records: list) -> None:
    """Save frequency baseline to Task C and C_RAG prediction directories."""
    task_c_ids = {
        r["case_id"]
        for r in records
        if r["ground_truth"].get("issues") and r["ground_truth"].get("recommendations")
    }

    for task, pred_dir_name in [("main", "predictions_main"), ("rag", "predictions_rag")]:
        out_dir = DATA_DIR / pred_dir_name / "frequency-baseline"
        out_dir.mkdir(parents=True, exist_ok=True)

        for record in records:
            case_id = record["case_id"]
            if case_id not in task_c_ids:
                continue  # Task C only runs on cases with issues

            pred = predictions[case_id]
            full_pred = pred  # baseline just predicts recommendations

            out = {
                "case_id": case_id,
                "model": "frequency-baseline",
                "task": task,
                "prediction": full_pred,
            }
            (out_dir / f"{case_id}.json").write_text(json.dumps(out, indent=2))

    log.info("Frequency baseline predictions saved.")


# ── LLM-as-Judge ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert transportation engineer reviewing AI-generated \
traffic impact analysis recommendations.
Evaluate the quality of the predicted recommendations against the project context.
Return ONLY valid JSON, no explanation."""

JUDGE_PROMPT = """PROJECT CONTEXT:
{input_json}

GROUND TRUTH RECOMMENDATIONS (from a real TIA report):
{gt_json}

PREDICTED RECOMMENDATIONS (AI-generated):
{pred_json}

Score the predicted recommendations on three dimensions (each 1-5):

1. domain_plausibility (1=nonsensical, 5=sounds like a real TIA):
   Would a licensed traffic engineer write recommendations like these?

2. location_specificity (1=completely generic, 5=specific to this project's context):
   Are the locations mentioned plausible given the project description?

3. gt_alignment (1=completely wrong types, 5=matches the GT measure types well):
   Do the predicted measure types match what the ground truth recommends?

Return ONLY:
{{
  "domain_plausibility": <1-5>,
  "location_specificity": <1-5>,
  "gt_alignment": <1-5>,
  "overall": <1-5>,
  "brief_rationale": "<one sentence>"
}}"""


def judge_predictions(
    model_key: str,
    records: list,
    task: str = "A",
    max_cases: int = 33,
) -> list:
    """
    Use llama-3.3-70b as judge to score a model's predictions.
    Returns list of {case_id, model, scores} dicts.
    """
    if task == "rag":
        pred_dir = DATA_DIR / "predictions_rag" / model_key
    else:
        pred_dir = DATA_DIR / "predictions_main" / model_key

    if not pred_dir.exists():
        log.warning("No predictions for %s task %s", model_key, task)
        return []

    results = []
    pred_files = sorted(pred_dir.glob("*.json"))[:max_cases]

    for pf in pred_files:
        case_id = pf.stem
        record = next((r for r in records if r["case_id"] == case_id), None)
        if not record:
            continue

        pred_data = json.loads(pf.read_text())
        pred = pred_data.get("prediction", {})

        # Build judge prompt
        input_str = json.dumps(record["input"], indent=2)[:2000]
        gt_str = json.dumps(
            record["ground_truth"].get("recommendations", [])[:5], indent=2
        )
        pred_str = json.dumps((pred.get("recommendations") or [])[:8], indent=2)

        user_msg = JUDGE_PROMPT.format(
            input_json=input_str,
            gt_json=gt_str,
            pred_json=pred_str,
        )

        log.info("Judging %s / %s / %s", model_key, task, case_id)
        try:
            resp = SESSION.post(
                f"{CF_BASE}/{JUDGE_MODEL}",
                json={
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
                timeout=60,
            )
            if resp.status_code != 200:
                log.warning(
                    "Judge HTTP %d for %s/%s", resp.status_code, model_key, case_id
                )
                continue

            data = resp.json()
            result = data.get("result", {})
            text = result.get("response") or ""
            if isinstance(text, dict):
                scores = text
            else:
                text = re.sub(r"^```(?:json)?\s*", "", str(text).strip())
                text = re.sub(r"\s*```$", "", text.strip())
                scores = json.loads(text)

            results.append(
                {
                    "case_id": case_id,
                    "model": model_key,
                    "task": task,
                    "scores": scores,
                    "pred_count": len(pred.get("recommendations") or []),
                    "gt_count": len(
                        record["ground_truth"].get("recommendations") or []
                    ),
                }
            )
        except Exception as e:
            log.warning("Judge failed for %s/%s: %s", model_key, case_id, e)

        time.sleep(0.3)

    return results


def run_judge_all(records: list, task: str = "main") -> dict:
    """Run LLM-as-judge for all models on Task C."""
    models = [
        "llama-3.3-70b",
        "gpt-oss-120b",
        "nemotron-120b",
        "qwen3-30b",
        "gemma-3-12b",
        "frequency-baseline",
    ]

    all_scores = {}
    for model_key in models:
        scores = judge_predictions(model_key, records, task=task)
        if scores:
            # Aggregate
            dims = [
                "domain_plausibility",
                "location_specificity",
                "gt_alignment",
                "overall",
            ]
            agg = {}
            for dim in dims:
                vals = [s["scores"].get(dim) for s in scores if s["scores"].get(dim)]
                agg[f"mean_{dim}"] = round(sum(vals) / len(vals), 2) if vals else None
            all_scores[model_key] = {
                "n_judged": len(scores),
                "aggregate": agg,
                "per_case": scores,
            }
            log.info(
                "Judge %s Task %s: overall=%.2f  plausibility=%.2f  specificity=%.2f",
                model_key,
                task,
                agg.get("mean_overall") or 0,
                agg.get("mean_domain_plausibility") or 0,
                agg.get("mean_location_specificity") or 0,
            )

    out_path = DATA_DIR / "results" / "judge_scores.json"
    out_path.write_text(json.dumps(all_scores, indent=2))
    log.info("Judge scores saved → %s", out_path)
    return all_scores


# ── Main ───────────────────────────────────────────────────────────────────────


def run_baselines() -> None:
    records = [
        json.loads(l) for l in DATASET_PATH.read_text().splitlines() if l.strip()
    ]
    log.info("Loaded %d records", len(records))

    # 1. Frequency baseline
    log.info("Building frequency baseline...")
    freq_preds = build_frequency_baseline(records)
    save_frequency_predictions(freq_preds, records)

    # 2. Evaluate frequency baseline
    from evaluate import run_evaluation

    run_evaluation(model_keys=["frequency-baseline"])

    # 3. LLM-as-judge for all models on Task C
    log.info("Running LLM-as-judge on Task C...")
    run_judge_all(records, task="main")

    log.info("Baselines complete.")


if __name__ == "__main__":
    run_baselines()
