"""
benchmark.py — Run Task C and Task C_RAG across all 5 models.

Primary task (C): given project input + identified failing intersections
                  (the LOS analysis results) → generate mitigation recommendations.
                  This reflects the actual practitioner workflow.

RAG variant (C_RAG): same as C but with top-8 retrieved chunks from similar
                     cases as additional context (LOO from 33 evaluation cases).

Models:
  llama-3.3-70b   @cf/meta/llama-3.3-70b-instruct-fp8-fast
  gpt-oss-120b    @cf/openai/gpt-oss-120b
  nemotron-120b   @cf/nvidia/nemotron-3-120b-a12b
  qwen3-30b       @cf/qwen/qwen3-30b-a3b-fp8
  gemma-3-12b     @cf/google/gemma-3-12b-it

Evaluated on the 17 cases that have both non-empty issues AND recommendations
in the ground truth (the full TIA subset).

Usage:
    python3 benchmark.py                        # all models, both tasks
    python3 benchmark.py --models llama gemma   # specific models
    python3 benchmark.py --tasks C              # Task C only
    python3 benchmark.py --overwrite            # re-run existing predictions
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
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
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "benchmark.log")],
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent.parent / "data"
EMBED_PATH = (
    DATA_DIR / "embeddings" / "chunks_eval.json"
)  # clean LOO corpus (33 cases only)
DATASET_PATH = DATA_DIR / "final" / "tia_dataset.jsonl"

CF_ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
CF_API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]
CF_BASE = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run"

EMBED_MODEL = "@cf/baai/bge-m3"
TOP_K = 8
MAX_TOKENS = 4096  # qwen3 uses ~2K tokens on internal reasoning before outputting JSON
TEMPERATURE = 0.1
REQUEST_TIMEOUT = 120  # kimi dropped; 120s is sufficient for remaining models

# Five models — kimi-k2.5 excluded (persistent timeout on long prompts)
MODELS = {
    "llama-3.3-70b": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "gpt-oss-120b": "@cf/openai/gpt-oss-120b",
    "nemotron-120b": "@cf/nvidia/nemotron-3-120b-a12b",
    "qwen3-30b": "@cf/qwen/qwen3-30b-a3b-fp8",
    "gemma-3-12b": "@cf/google/gemma-3-12b-it",
}

VALID_MEASURE_TYPES = {
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
}

SESSION = requests.Session()
SESSION.headers.update(
    {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
)


# ── Prompts ────────────────────────────────────────────────────────────────────

TASK_C_SYSTEM = """\
You are a transportation engineering consultant writing the mitigation section
of a Traffic Impact Analysis (TIA) report.
You have already run the LOS (Level of Service) analysis and identified which
intersections fail the acceptable standard under build conditions.
Your task is to prescribe the specific transportation improvements needed at
each failing intersection.
Return ONLY valid JSON — no explanation, no markdown fences."""

TASK_C_USER = """\
PROJECT CONTEXT:
{input_json}

TRAFFIC ANALYSIS RESULTS — FAILING INTERSECTIONS:
The LOS analysis identified the following intersections as deficient under
build conditions (they exceed the acceptable LOS standard):

{issues_json}

For each failing intersection above, prescribe the specific mitigation measure
a licensed traffic engineer would recommend. Also include any project-wide
demand management measures appropriate for this project type.

Return ONLY this JSON structure (no other text):
{{
  "recommendations": [
    {{
      "location": "<intersection or road segment — use the exact name from the failing intersections above>",
      "measure_type": "<one of: add_turn_lane|signal_timing_optimization|new_traffic_signal|access_modification|driveway_consolidation|road_widening|roundabout_installation|pedestrian_improvement|transit_improvement|tdm_measure|no_mitigation_required|other>",
      "description": "<specific mitigation action with engineering detail>",
      "timing": "<one of: prior_to_occupancy|phase_1|phase_2|background_improvement|not_specified>",
      "responsible_party": "<one of: applicant|city|state_dot|county|shared|not_specified>"
    }}
  ]
}}"""

TASK_C_RAG_USER = """\
SIMILAR PROJECT EXAMPLES (how comparable projects addressed their failing intersections):
{context}

PROJECT CONTEXT:
{input_json}

TRAFFIC ANALYSIS RESULTS — FAILING INTERSECTIONS:
{issues_json}

Using the similar project examples as guidance for the types of improvements
typically applied to failing intersections, prescribe mitigations for each
deficient intersection above.

Return ONLY this JSON structure (no other text):
{{
  "recommendations": [
    {{
      "location": "<intersection or road segment — use the exact name from the failing intersections above>",
      "measure_type": "<one of: add_turn_lane|signal_timing_optimization|new_traffic_signal|access_modification|driveway_consolidation|road_widening|roundabout_installation|pedestrian_improvement|transit_improvement|tdm_measure|no_mitigation_required|other>",
      "description": "<specific mitigation action with engineering detail>",
      "timing": "<one of: prior_to_occupancy|phase_1|phase_2|background_improvement|not_specified>",
      "responsible_party": "<one of: applicant|city|state_dot|county|shared|not_specified>"
    }}
  ]
}}"""

# ── Few-shot prompts ───────────────────────────────────────────────────────────
# True few-shot: shows 2-3 COMPLETE (input + issues → recommendations) demonstrations
# rather than arbitrary retrieved text chunks

TASK_C_FEWSHOT_USER = """\
Below are {n_examples} complete examples showing how a licensed traffic engineer
prescribes mitigations given a project description and failing intersections.
Study the examples carefully, then generate recommendations for the new project.

{demonstrations}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW PROJECT — generate recommendations for this:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT CONTEXT:
{input_json}

TRAFFIC ANALYSIS RESULTS — FAILING INTERSECTIONS:
{issues_json}

Return ONLY this JSON structure (no other text):
{{
  "recommendations": [
    {{
      "location": "<intersection or road segment — use the exact name from the failing intersections above>",
      "measure_type": "<one of: add_turn_lane|signal_timing_optimization|new_traffic_signal|access_modification|driveway_consolidation|road_widening|roundabout_installation|pedestrian_improvement|transit_improvement|tdm_measure|no_mitigation_required|other>",
      "description": "<specific mitigation action with engineering detail>",
      "timing": "<one of: prior_to_occupancy|phase_1|phase_2|background_improvement|not_specified>",
      "responsible_party": "<one of: applicant|city|state_dot|county|shared|not_specified>"
    }}
  ]
}}"""


# ── CF API ─────────────────────────────────────────────────────────────────────


def cf_generate(model_id: str, system: str, user: str, retries: int = 3):
    """Call CF Workers AI. Returns parsed dict or None."""
    url = f"{CF_BASE}/{model_id}"
    for attempt in range(1, retries + 2):
        try:
            resp = SESSION.post(
                url,
                json={
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 15))
                log.warning("Rate limited on %s, waiting %ds", model_id, wait)
                time.sleep(wait)
                continue
            if resp.status_code in (404, 403):
                log.error("Model %s unavailable: HTTP %d", model_id, resp.status_code)
                return None
            if resp.status_code != 200:
                log.warning(
                    "HTTP %d from %s (attempt %d)", resp.status_code, model_id, attempt
                )
                if attempt > retries:
                    return None
                time.sleep(2**attempt)
                continue

            data = resp.json()
            result = data.get("result", {})

            # Schema A (llama): result["response"] is already a parsed dict
            if isinstance(result.get("response"), dict):
                return result["response"]

            # Schema B (gemma): result["response"] is a string
            if isinstance(result.get("response"), str) and result["response"].strip():
                return parse_json_response(result["response"])

            # Schema C (gpt-oss, nemotron, qwen3): OpenAI-compatible choices
            choices = result.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content") or ""
                if content.strip():
                    return parse_json_response(content)

            for field in ("generated_text", "text"):
                val = result.get(field)
                if isinstance(val, str) and val.strip():
                    return parse_json_response(val)
                if isinstance(val, dict):
                    return val

            return None

        except requests.Timeout:
            log.warning("Timeout on %s attempt %d", model_id, attempt)
            if attempt > retries:
                return None
            time.sleep(2**attempt)
        except Exception as e:
            log.error("Error on %s attempt %d: %s", model_id, attempt, e)
            if attempt > retries:
                return None
            time.sleep(2)
    return None


def cf_embed(text: str, retries: int = 3):
    """Embed a text string using BGE-M3."""
    url = f"{CF_BASE}/{EMBED_MODEL}"
    for attempt in range(1, retries + 2):
        try:
            resp = SESSION.post(url, json={"text": text[:800]}, timeout=30)
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("Retry-After", 10)))
                continue
            if resp.status_code != 200:
                if attempt > retries:
                    return None
                time.sleep(2**attempt)
                continue
            result = resp.json().get("result", {})
            raw = result.get("data") or result.get("embeddings") or []
            if raw and isinstance(raw[0], list):
                return raw[0]
            if raw and isinstance(raw[0], float):
                return raw
            return None
        except Exception as e:
            if attempt > retries:
                return None
            time.sleep(2**attempt)
    return None


def parse_json_response(text: str):
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


# ── Cosine similarity ──────────────────────────────────────────────────────────


def cosine_sim(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if (na > 0 and nb > 0) else 0.0


# ── Input formatting ───────────────────────────────────────────────────────────


def format_input(record: dict) -> str:
    return json.dumps(record.get("input", {}), indent=2)


def format_issues(issues: list) -> str:
    """Format the GT issues as a clear numbered list for the Task C prompt."""
    if not issues:
        return "No intersections explicitly identified as deficient in the analysis."
    lines = []
    for i, iss in enumerate(issues, 1):
        loc = iss.get("location", "not_specified")
        scenario = iss.get("scenario", "not_specified").replace("_", " ")
        los = iss.get("los", "?")
        vc = iss.get("vc_ratio")
        delay = iss.get("delay_sec")
        std = iss.get("los_standard", "LOS D or better")
        dtype = (iss.get("deficiency_type") or "los_failure").replace("_", " ")

        detail_parts = [
            f"LOS {los}" if los else None,
            f"V/C = {vc:.2f}" if vc else None,
            f"delay = {delay:.1f}s" if delay else None,
        ]
        detail = ", ".join(p for p in detail_parts if p)

        lines.append(
            f"{i}. {loc}\n"
            f"   Scenario: {scenario}\n"
            f"   Analysis result: {detail}\n"
            f"   Standard: {std}\n"
            f"   Deficiency type: {dtype}"
        )
    return "\n\n".join(lines)


def format_context(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        case_id = chunk["case_id"]
        section = chunk["section_name"]
        text = chunk["text"][:800]
        parts.append(f"[Example {i} — {case_id} / {section}]\n{text}")
    return "\n\n---\n\n".join(parts)


def format_demonstrations(examples: list) -> str:
    """
    Format 2–3 complete (input, issues → recommendations) demonstrations
    for true few-shot prompting. Each example shows the full task.
    """
    parts = []
    for i, ex in enumerate(examples, 1):
        input_str = format_input(ex)[:1500]
        issues_str = format_issues(ex["ground_truth"].get("issues") or [])
        recs = ex["ground_truth"].get("recommendations") or []
        recs_json = json.dumps({"recommendations": recs}, indent=2)

        parts.append(
            f"━━ EXAMPLE {i} ━━\n"
            f"PROJECT CONTEXT:\n{input_str}\n\n"
            f"FAILING INTERSECTIONS:\n{issues_str}\n\n"
            f"TRAFFIC ENGINEER'S RECOMMENDATIONS:\n{recs_json}"
        )
    return "\n\n".join(parts)


def select_few_shot_examples(
    current_case_id: str,
    all_records: list,
    n: int = 3,
) -> list:
    """
    Select n demonstration examples for few-shot prompting.
    Prefers same project_type, then same state. Always excludes the current case.
    """
    pool = [
        r
        for r in all_records
        if r["case_id"] != current_case_id
        and r["ground_truth"].get("issues")
        and r["ground_truth"].get("recommendations")
    ]

    current = next((r for r in all_records if r["case_id"] == current_case_id), {})
    current_type = current.get("input", {}).get("project_type", "")
    current_state = (
        current.get("input", {}).get("location_context", {}).get("state", "")
    )

    # Sort: same type first, then same state, then everything else
    def rank(r):
        t = r["input"].get("project_type", "")
        s = r["input"].get("location_context", {}).get("state", "")
        return (0 if t == current_type else 1, 0 if s == current_state else 1)

    pool.sort(key=rank)
    return pool[:n]


def normalise_prediction(pred: dict) -> dict:
    recs = pred.get("recommendations", []) if pred else []
    out = []
    if isinstance(recs, list):
        for r in recs:
            if not isinstance(r, dict):
                continue
            mt = str(r.get("measure_type", "other")).lower().replace(" ", "_")
            if mt not in VALID_MEASURE_TYPES:
                mt = "other"
            out.append(
                {
                    "location": str(r.get("location", "not_specified"))[:100],
                    "measure_type": mt,
                    "description": str(r.get("description", ""))[:500],
                    "timing": str(r.get("timing", "not_specified")),
                    "responsible_party": str(
                        r.get("responsible_party", "not_specified")
                    ),
                }
            )
    return {"recommendations": out}


# ── Main runner ────────────────────────────────────────────────────────────────


def run_benchmark(model_keys=None, tasks=None, overwrite: bool = False) -> None:
    if model_keys is None:
        model_keys = list(MODELS.keys())
    if tasks is None:
        tasks = ["main", "rag"]

    # Load dataset — Task C only runs on cases with non-empty issues
    records = [
        json.loads(l) for l in DATASET_PATH.read_text().splitlines() if l.strip()
    ]
    task_c_records = [
        r
        for r in records
        if r["ground_truth"].get("issues") and r["ground_truth"].get("recommendations")
    ]
    log.info(
        "Dataset: %d total, %d Task C cases (have both issues and recs)",
        len(records),
        len(task_c_records),
    )

    # Load embeddings for RAG (clean LOO corpus — 33 evaluation cases only)
    embed_map: dict = {}
    if "rag" in tasks:
        if not EMBED_PATH.exists():
            log.error(
                "Clean embedding file not found at %s — run embed.py --eval-only first",
                EMBED_PATH,
            )
            sys.exit(1)
        for item in json.loads(EMBED_PATH.read_text()):
            embed_map[(item["case_id"], item["chunk_index"])] = item
        log.info("Loaded %d embeddings from evaluation corpus", len(embed_map))

    for model_key in model_keys:
        model_id = MODELS[model_key]
        log.info("\n══ Model: %s ══", model_key)

        for task in tasks:
            cases = task_c_records
            out_dir = DATA_DIR / f"predictions_{task}" / model_key
            out_dir.mkdir(parents=True, exist_ok=True)

            log.info("  Task %s: %d cases", task, len(cases))

            for record in cases:
                case_id = record["case_id"]
                out_path = out_dir / f"{case_id}.json"

                if out_path.exists() and not overwrite:
                    log.debug("  SKIP %s (exists)", case_id)
                    continue

                log.info("  %s | %s | %s", model_key, task, case_id)

                input_str = format_input(record)
                issues_str = format_issues(record["ground_truth"].get("issues") or [])

                if task == "main":
                    system = TASK_C_SYSTEM
                    user = TASK_C_USER.format(
                        input_json=input_str[:3000],
                        issues_json=issues_str,
                    )

                elif task == "fewshot":
                    examples = select_few_shot_examples(case_id, task_c_records, n=3)
                    demos = format_demonstrations(examples)
                    system = TASK_C_SYSTEM
                    user = TASK_C_FEWSHOT_USER.format(
                        n_examples=len(examples),
                        demonstrations=demos,
                        input_json=input_str[:2000],
                        issues_json=issues_str,
                    )

                else:  # rag
                    query_text = f"{input_str[:400]}\n{issues_str[:400]}"
                    query_vec = cf_embed(query_text)
                    if query_vec is None:
                        log.warning(
                            "  Embedding failed for %s — using zero-shot", case_id
                        )
                        system = TASK_C_SYSTEM
                        user = TASK_C_USER.format(
                            input_json=input_str[:3000],
                            issues_json=issues_str,
                        )
                    else:
                        candidates = [
                            v for v in embed_map.values() if v["case_id"] != case_id
                        ]
                        scored = sorted(
                            candidates,
                            key=lambda c: cosine_sim(query_vec, c["embedding"]),
                            reverse=True,
                        )[:TOP_K]
                        context = format_context(scored)
                        system = TASK_C_SYSTEM
                        user = TASK_C_RAG_USER.format(
                            context=context,
                            input_json=input_str[:3000],
                            issues_json=issues_str,
                        )

                raw_pred = cf_generate(model_id, system, user)
                if raw_pred is None:
                    log.warning(
                        "  %s returned None for %s/%s", model_key, task, case_id
                    )
                    pred = {"recommendations": []}
                else:
                    pred = normalise_prediction(raw_pred)

                out_path.write_text(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "model": model_key,
                            "task": task,
                            "prediction": pred,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )

                time.sleep(0.3)

    log.info("Benchmark complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TIA Benchmark — Task C")
    ap.add_argument(
        "--models", nargs="+", choices=list(MODELS.keys()), default=list(MODELS.keys())
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        choices=["main", "rag", "fewshot"],
        default=["main", "rag", "fewshot"],
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run_benchmark(model_keys=args.models, tasks=args.tasks, overwrite=args.overwrite)
