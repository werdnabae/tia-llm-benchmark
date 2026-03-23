"""
extract.py — Structured extraction via Cloudflare Workers AI.

For each parsed PDF (data/parsed/{case_id}_sections.json):
  - Sends input sections to the CF Workers AI model
  - Extracts structured INPUT fields (project context, intersections, trip gen)
  - Extracts structured GROUND TRUTH fields (issues, recommendations)
  - Writes data/split/{case_id}.json

Requires:
  CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in pipeline/.env or environment.

Usage:
  python3 extract.py                        # extract all parsed cases
  python3 extract.py --cases CEQ-CA-020128  # specific case(s)
  python3 extract.py --overwrite            # re-extract even if split file exists
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# ── Load .env ──────────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from config import PARSED_DIR, SPLIT_DIR, LOGS_DIR, REPORTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "extract.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Cloudflare Workers AI ──────────────────────────────────────────────────────
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CF_MODEL = os.environ.get("CF_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")

CATALOG = {r["case_id"]: r for r in REPORTS}

# ── Prompts ────────────────────────────────────────────────────────────────────

INPUT_PROMPT = """\
You are a transportation engineering data extractor.
Extract structured fields from the TIA / EIR Transportation Chapter text below.
Return ONLY valid JSON. Extract ONLY values explicitly stated in the text — use null for anything not found.

TEXT:
{text}

Return this exact JSON structure:
{{
  "project_type": "<residential_multifamily|residential_single_family|commercial_retail|commercial_office|mixed_use|industrial_warehouse|institutional|transit_oriented_development|hotel|medical|other>",
  "location_context": {{
    "city": "<city name or null>",
    "state": "<2-letter state code or null>",
    "area_type": "<urban_core|urban|suburban|rural|null>",
    "surrounding_land_use": "<brief description or null>",
    "nearest_major_intersection": "<intersection name or null>"
  }},
  "description": {{
    "narrative": "<2-4 sentence project summary>",
    "size_units": "<e.g. '450 residential units, 12,000 sq ft retail' or null>",
    "access_points": <integer or null>,
    "build_year": <integer or null>,
    "phases": <integer or null>,
    "trip_generation_summary": {{
      "am_peak_in":  <integer or null>,
      "am_peak_out": <integer or null>,
      "pm_peak_in":  <integer or null>,
      "pm_peak_out": <integer or null>,
      "daily_total": <integer or null>
    }}
  }},
  "known_conditions": {{
    "narrative": "<2-3 sentence summary of existing traffic conditions>",
    "study_intersections": [
      {{
        "name": "<intersection name>",
        "control_type": "<signalized|stop_sign|roundabout|yield>",
        "existing_am_los": "<A-F or null>",
        "existing_pm_los": "<A-F or null>",
        "existing_am_vc":  <float 0.0-2.0 or null>,
        "existing_pm_vc":  <float 0.0-2.0 or null>
      }}
    ],
    "background_growth_rate": <float e.g. 0.005 or null>,
    "no_build_horizon_year": <integer or null>
  }}
}}
"""

GT_PROMPT = """\
You are a transportation engineering data extractor.
Extract FINDINGS (deficient intersections) and RECOMMENDATIONS (mitigation measures) from the text below.
Return ONLY valid JSON. Extract ONLY items explicitly stated — never infer.

TEXT:
{text}

Return this exact JSON structure:
{{
  "issues": [
    {{
      "location": "<intersection or road segment name>",
      "scenario": "<am_peak_build|pm_peak_build|am_peak_no_build|pm_peak_no_build|cumulative|not_specified>",
      "los": "<A-F or null>",
      "vc_ratio": <float or null>,
      "delay_sec": <float or null>,
      "los_standard": "<e.g. LOS D or better or null>",
      "deficiency_type": "<los_failure|vc_exceeds_threshold|queue_spillback|sight_distance|access_conflict|null>"
    }}
  ],
  "recommendations": [
    {{
      "recommendation_id": "<R-01, R-02, ...>",
      "location": "<intersection or segment>",
      "measure_type": "<add_turn_lane|signal_timing_optimization|new_traffic_signal|access_modification|driveway_consolidation|road_widening|roundabout_installation|pedestrian_improvement|transit_improvement|tdm_measure|no_mitigation_required|other>",
      "description": "<verbatim or close paraphrase>",
      "addresses_issue": "<location this resolves or null>",
      "timing": "<prior_to_occupancy|phase_1|phase_2|background_improvement|not_specified>",
      "responsible_party": "<applicant|city|caltrans|state_dot|county|shared|not_specified>"
    }}
  ]
}}

Rules:
- issues[]: only intersections explicitly identified as failing the LOS standard
- recommendations[]: only measures explicitly stated in the text
- Empty arrays [] if nothing found
"""

# ── CF API call ────────────────────────────────────────────────────────────────


def _cf_url() -> str:
    return (
        f"https://api.cloudflare.com/client/v4/accounts"
        f"/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"
    )


def call_cf(prompt: str, retries: int = 2) -> str:
    """
    Call Cloudflare Workers AI. Returns the model's text response.
    Raises RuntimeError on failure after retries.
    """
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set. "
            "Add them to pipeline/.env"
        )

    for attempt in range(1, retries + 2):
        try:
            resp = requests.post(
                _cf_url(),
                headers={
                    "Authorization": f"Bearer {CF_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.1,
                },
                timeout=120,
            )
        except requests.exceptions.Timeout:
            log.warning("  CF timeout (attempt %d/%d)", attempt, retries + 1)
            if attempt > retries:
                raise RuntimeError("CF API timed out after retries")
            time.sleep(2**attempt)
            continue

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 10))
            log.warning("  CF rate limited — waiting %ds", wait)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            raise RuntimeError(f"CF API error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        result = data.get("result", {})
        text = (
            result.get("response")
            or result.get("text")
            or result.get("generated_text")
            or ""
        )
        if not text:
            raise RuntimeError(f"CF returned empty response: {data}")
        return text

    raise RuntimeError("CF API failed after all retries")


def parse_json_response(raw: str) -> dict:
    """
    Parse JSON from model response.
    Strips markdown code fences; attempts light salvage on truncated output.
    Raises ValueError if unparseable.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage a truncated object by closing it
        try:
            salvaged = text.rsplit(",", 1)[0] + "\n}}"
            return json.loads(salvaged)
        except Exception:
            pass
        raise ValueError(f"Unparseable JSON from model: {text[:200]}")


# ── Section text helpers ───────────────────────────────────────────────────────


def gather_text(sections: list, role: str) -> str:
    """Concatenate all section text with the given role."""
    parts = []
    for sec in sections:
        if sec.get("role") == role:
            txt = sec.get("text", "").strip()
            if txt:
                parts.append(f"=== {sec['section_name'].upper()} ===\n{txt}")
    return "\n\n".join(parts)


# Keywords that indicate transportation-relevant content in a large GT block
_TRANSPORT_KEYWORDS = re.compile(
    r"(?i)(level\s+of\s+service|LOS\s+[A-F]|V/C\s+ratio|intersection|"
    r"mitigation\s+measure|MM-TR|MM-TRANS|M-TR|"
    r"turn\s+lane|signal\s+timing|traffic\s+impact|"
    r"roadway|traffic\s+operation|peak\s+hour|"
    r"southbound|northbound|eastbound|westbound)"
)


def smart_gt_window(gt_text: str, max_chars: int = 12_000) -> str:
    """
    For long GT sections, find the most transportation-relevant window
    rather than blindly taking the first max_chars characters.

    Strategy:
      1. If the full text is short enough, return it as-is.
      2. Find the first strong transportation keyword hit.
      3. Return a window starting ~500 chars before that hit.
    """
    if len(gt_text) <= max_chars:
        return gt_text

    m = _TRANSPORT_KEYWORDS.search(gt_text)
    if m:
        start = max(0, m.start() - 500)
        return gt_text[start : start + max_chars]

    # Fallback: return the first window (preamble may contain useful context)
    return gt_text[:max_chars]


def resolve_catalog(case_id: str) -> dict:
    """Return catalog metadata for a case_id, falling back to download_log."""
    if case_id in CATALOG:
        return CATALOG[case_id]

    log_path = LOGS_DIR / "download_log.json"
    if log_path.exists():
        dl = json.loads(log_path.read_text())
        if case_id in dl:
            e = dl[case_id]
            return {
                "case_id": case_id,
                "title": e.get("project_title") or e.get("link_text", ""),
                "agency": e.get("agency", ""),
                "state": e.get("state", "CA"),
                "year": e.get("year"),
                "source_url": e.get("pdf_url", ""),
                "pdf_url": e.get("pdf_url"),
                "project_type": e.get("project_type", "mixed_use"),
            }

    return {"case_id": case_id, "state": "CA"}


# ── Per-case extraction ────────────────────────────────────────────────────────


def extract_case(case_id: str) -> dict:
    """
    Extract structured input + ground_truth for one case.
    Returns the complete split record dict.
    Raises RuntimeError if the CF API call fails.
    """
    parsed_path = PARSED_DIR / f"{case_id}_sections.json"
    if not parsed_path.exists():
        raise FileNotFoundError(f"No parsed file for {case_id}")

    parsed = json.loads(parsed_path.read_text())
    sections = parsed["sections"]
    catalog = resolve_catalog(case_id)

    # ── Gather text by role ───────────────────────────────────────────────────
    input_text = gather_text(sections, "input")
    if not input_text:
        # Fallback: use all non-GT sections so we at least have something
        input_text = (
            gather_text(
                [s for s in sections if s.get("role") != "ground_truth"], "other"
            )
            or gather_text(sections, "preamble")
            or ""
        )

    gt_text = gather_text(sections, "ground_truth")
    has_gt = bool(gt_text.strip())

    # ── INPUT extraction ──────────────────────────────────────────────────────
    log.info("%s: extracting INPUT via CF Workers AI", case_id)
    raw_input = call_cf(INPUT_PROMPT.format(text=input_text[:12_000]))
    input_data = parse_json_response(raw_input)

    # Patch in state/city from catalog if the model missed them
    loc = input_data.setdefault("location_context", {})
    if not loc.get("state") and catalog.get("state"):
        loc["state"] = catalog["state"]
    if not loc.get("city") and catalog.get("city"):
        loc["city"] = catalog["city"]

    # ── GROUND TRUTH extraction ───────────────────────────────────────────────
    gt_data = {"issues": [], "recommendations": []}
    if has_gt:
        log.info("%s: extracting GROUND TRUTH via CF Workers AI", case_id)
        raw_gt = call_cf(GT_PROMPT.format(text=smart_gt_window(gt_text, 12_000)))
        gt_data = parse_json_response(raw_gt)

    # ── Assemble record ───────────────────────────────────────────────────────
    return {
        "case_id": case_id,
        "input": input_data,
        "ground_truth": gt_data,
        "metadata": {
            "source_url": catalog.get("source_url"),
            "pdf_url": catalog.get("pdf_url"),
            "agency": catalog.get("agency"),
            "year": catalog.get("year"),
            "report_title": catalog.get("title"),
            "state": catalog.get("state", "CA"),
            "methodology": None,
            "los_standard": None,
            "study_intersections_count": len(
                input_data.get("known_conditions", {}).get("study_intersections") or []
            ),
            "pages": parsed.get("num_pages"),
            "extraction_method": "cf_workers_ai",
            "extraction_qa_passed": False,
            "exclude_from_retrieval": [case_id],
        },
        "_parse_coverage": parsed.get("coverage", {}),
        "_has_ground_truth": has_gt,
    }


# ── Batch runner ───────────────────────────────────────────────────────────────


def run_extraction(case_ids: Optional[list] = None, overwrite: bool = False) -> dict:
    """
    Extract all parsed cases (or the given subset).
    Skips cases that already have a split file unless overwrite=True.
    Returns a log dict: {case_id: {status, issues, recommendations}}.
    """
    if not CF_ACCOUNT_ID or not CF_API_TOKEN:
        log.error(
            "No Cloudflare credentials found.\n"
            "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in pipeline/.env"
        )
        sys.exit(1)

    log.info("CF Workers AI model: %s", CF_MODEL)

    parsed_files = sorted(PARSED_DIR.glob("*_sections.json"))
    if not parsed_files:
        log.warning("No parsed files in %s", PARSED_DIR)
        return {}

    results: dict = {}

    for pf in parsed_files:
        cid = pf.stem.replace("_sections", "")

        if case_ids and cid not in case_ids:
            continue

        out_path = SPLIT_DIR / f"{cid}.json"
        if out_path.exists() and not overwrite:
            log.info("SKIP %s (split file exists; use --overwrite to re-run)", cid)
            results[cid] = {"status": "skipped"}
            continue

        try:
            record = extract_case(cid)
        except FileNotFoundError as e:
            log.error("SKIP %s: %s", cid, e)
            results[cid] = {"status": "failed", "error": str(e)}
            continue
        except (RuntimeError, ValueError) as e:
            log.error("FAIL %s: %s", cid, e)
            results[cid] = {"status": "failed", "error": str(e)}
            continue

        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

        n_issues = len(record["ground_truth"].get("issues", []))
        n_recs = len(record["ground_truth"].get("recommendations", []))
        log.info(
            "  OK  %s  issues=%d  recs=%d  gt=%s",
            cid,
            n_issues,
            n_recs,
            record["_has_ground_truth"],
        )

        results[cid] = {
            "status": "success",
            "issues": n_issues,
            "recommendations": n_recs,
            "has_gt": record["_has_ground_truth"],
        }

    (LOGS_DIR / "extract_log.json").write_text(json.dumps(results, indent=2))

    ok = sum(1 for v in results.values() if v["status"] == "success")
    skipped = sum(1 for v in results.values() if v["status"] == "skipped")
    failed = sum(1 for v in results.values() if v["status"] == "failed")
    print(f"\n── Extraction: {ok} extracted  {skipped} skipped  {failed} failed ──────")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CF Workers AI extraction")
    ap.add_argument("--cases", nargs="+", help="Specific case IDs to extract")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if split file already exists",
    )
    args = ap.parse_args()
    run_extraction(case_ids=args.cases, overwrite=args.overwrite)
