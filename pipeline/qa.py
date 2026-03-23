"""
qa.py — Quality assurance checks on split records.

Checks per record:
  1. INPUT contains NO findings/recommendations text (data leakage test)
  2. GROUND TRUTH has at least one issue or recommendation
  3. All required schema fields are present and typed correctly
  4. Study intersections in known_conditions have valid LOS values
  5. Chunk alignment: each split record has a corresponding chunk file
  6. No duplicate case_ids in dataset

Applies full QA to all records but flags a random sample for manual review.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

from config import (
    SPLIT_DIR,
    CHUNKS_DIR,
    LOGS_DIR,
    FINAL_DIR,
    QA_SAMPLE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "qa.log"),
    ],
)
log = logging.getLogger(__name__)

# Patterns that must NOT appear in INPUT sections.
# These are specific to findings/recommendations language — NOT general EIR terminology.
# "mitigation" alone is excluded because it appears legitimately in EIR project
# descriptions (e.g. "the project will comply with adopted mitigation measures").
LEAKAGE_PATTERNS = [
    # Specific LOS failure language
    re.compile(r"(?i)\bfound\s+to\s+(operate|function)\s+at\s+los\s+[D-F]\b"),
    re.compile(r"(?i)\bwill\s+(fail|degrade|exceed)\s+(the\s+)?los\s+standard\b"),
    re.compile(r"(?i)\boperates?\s+at\s+los\s+[EF]\b"),
    # Specific physical mitigation language (would only appear in recommendations)
    re.compile(r"(?i)\bdedicated\s+(left|right).?turn\s+lane\b"),
    re.compile(r"(?i)\bsignal\s+timing\s+(plan|optimiz)\b"),
    re.compile(r"(?i)\badd\s+a\s+(northbound|southbound|eastbound|westbound)\b"),
]

VALID_LOS = {None, "A", "B", "C", "D", "E", "F"}
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
VALID_TIMINGS = {
    "prior_to_occupancy",
    "phase_1",
    "phase_2",
    "background_improvement",
    "not_specified",
}
VALID_AREA_TYPES = {"urban_core", "urban", "suburban", "rural", None}


def check_no_leakage(record: dict) -> list[str]:
    """Return list of leakage violations found in input fields."""
    errors = []
    input_str = json.dumps(record.get("input", {}))
    for pat in LEAKAGE_PATTERNS:
        if pat.search(input_str):
            errors.append(f"Leakage pattern detected: '{pat.pattern[:60]}'")
    return errors


def check_schema(record: dict) -> list[str]:
    """Validate required fields and types. Returns list of violations."""
    errors = []
    inp = record.get("input", {})
    gt = record.get("ground_truth", {})
    meta = record.get("metadata", {})

    # Top-level
    if not record.get("case_id"):
        errors.append("Missing case_id")

    # Input
    if not inp.get("project_type"):
        errors.append("input.project_type is missing")
    loc = inp.get("location_context", {})
    if not isinstance(loc, dict):
        errors.append("input.location_context must be dict")
    else:
        if loc.get("area_type") not in VALID_AREA_TYPES:
            errors.append(f"Invalid area_type: {loc.get('area_type')}")

    # Known conditions intersections
    for i, intr in enumerate(
        inp.get("known_conditions", {}).get("study_intersections", [])
    ):
        if not intr.get("name"):
            errors.append(f"Intersection {i}: missing name")
        if intr.get("existing_am_los") not in VALID_LOS:
            errors.append(
                f"Intersection {i}: invalid existing_am_los '{intr.get('existing_am_los')}'"
            )
        if intr.get("existing_pm_los") not in VALID_LOS:
            errors.append(
                f"Intersection {i}: invalid existing_pm_los '{intr.get('existing_pm_los')}'"
            )

    # Ground truth
    if not isinstance(gt.get("issues"), list):
        errors.append("ground_truth.issues must be a list")
    if not isinstance(gt.get("recommendations"), list):
        errors.append("ground_truth.recommendations must be a list")

    for i, issue in enumerate(gt.get("issues", [])):
        if not issue.get("location"):
            errors.append(f"Issue {i}: missing location")
        if issue.get("los") not in VALID_LOS:
            errors.append(f"Issue {i}: invalid los '{issue.get('los')}'")

    for i, rec in enumerate(gt.get("recommendations", [])):
        if not rec.get("location"):
            errors.append(f"Rec {i}: missing location")
        if rec.get("measure_type") not in VALID_MEASURE_TYPES:
            errors.append(f"Rec {i}: invalid measure_type '{rec.get('measure_type')}'")
        if rec.get("timing") and rec.get("timing") not in VALID_TIMINGS:
            errors.append(f"Rec {i}: invalid timing '{rec.get('timing')}'")

    # Metadata
    if not meta.get("agency"):
        errors.append("metadata.agency is missing")
    if not meta.get("year"):
        errors.append("metadata.year is missing")
    if not isinstance(meta.get("exclude_from_retrieval"), list):
        errors.append("metadata.exclude_from_retrieval must be a list")
    elif record.get("case_id") not in meta.get("exclude_from_retrieval", []):
        errors.append("case_id must appear in metadata.exclude_from_retrieval")

    return errors


def check_chunk_alignment(case_id: str) -> list[str]:
    """Check that a chunk file exists for this case."""
    chunk_path = CHUNKS_DIR / f"{case_id}.json"
    if not chunk_path.exists():
        return [f"Chunk file missing: {chunk_path}"]
    try:
        with open(chunk_path) as f:
            chunk_data = json.load(f)
        if not chunk_data.get("chunks"):
            return ["Chunk file has no chunks"]
    except Exception as e:
        return [f"Chunk file parse error: {e}"]
    return []


def check_ground_truth_completeness(record: dict) -> list[str]:
    """
    Warn (not error) if ground truth is entirely empty — these records
    need manual review before inclusion.
    """
    gt = record.get("ground_truth", {})
    if not gt.get("issues") and not gt.get("recommendations"):
        return [
            "WARN: ground_truth has no issues AND no recommendations — manual review required"
        ]
    return []


def qa_record(record: dict, is_sample: bool = False) -> dict:
    """Run all checks on one record. Returns QA result dict."""
    case_id = record.get("case_id", "UNKNOWN")
    errors = []
    warnings = []

    errors += check_no_leakage(record)
    errors += check_schema(record)
    errors += check_chunk_alignment(case_id)
    warnings += check_ground_truth_completeness(record)

    passed = len(errors) == 0
    return {
        "case_id": case_id,
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "requires_manual": is_sample or not passed,
    }


def run_qa(fix: bool = True) -> dict:
    """
    QA all split files.
    If fix=True, updates extraction_qa_passed in the split JSON.
    Returns full QA report.
    """
    split_files = sorted(SPLIT_DIR.glob("*.json"))
    if not split_files:
        log.warning("No split files found in %s", SPLIT_DIR)
        return {}

    all_case_ids = [f.stem for f in split_files]
    # Select random sample for manual review flag
    sample_n = max(1, int(len(all_case_ids) * QA_SAMPLE_RATE))
    manual_sample = set(random.sample(all_case_ids, sample_n))
    log.info(
        "QA sample (%d/%d): %s", sample_n, len(all_case_ids), sorted(manual_sample)
    )

    qa_report = {}
    passed_ids = []
    failed_ids = []
    duplicate_check = {}

    for sf in split_files:
        case_id = sf.stem
        with open(sf) as f:
            record = json.load(f)

        # Duplicate check
        cid = record.get("case_id", case_id)
        if cid in duplicate_check:
            log.error(
                "DUPLICATE case_id: %s (files: %s and %s)",
                cid,
                duplicate_check[cid],
                sf,
            )
        duplicate_check[cid] = str(sf)

        result = qa_record(record, is_sample=(case_id in manual_sample))
        qa_report[case_id] = result

        if result["passed"]:
            passed_ids.append(case_id)
        else:
            failed_ids.append(case_id)
            log.warning("FAIL  %s: %s", case_id, result["errors"])

        # Update qa_passed flag in split file
        if fix:
            record["metadata"]["extraction_qa_passed"] = result["passed"]
            with open(sf, "w") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

    # Save QA report
    report = {
        "total": len(all_case_ids),
        "passed": len(passed_ids),
        "failed": len(failed_ids),
        "pass_rate_pct": round(len(passed_ids) / len(all_case_ids) * 100, 1),
        "manual_sample": sorted(manual_sample),
        "passed_ids": sorted(passed_ids),
        "failed_ids": sorted(failed_ids),
        "details": qa_report,
    }

    qa_path = LOGS_DIR / "qa_report.json"
    with open(qa_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n── QA Summary ────────────────────────────────────────────")
    print(f"  Total cases:  {report['total']}")
    print(f"  Passed:       {report['passed']}  ({report['pass_rate_pct']}%)")
    print(f"  Failed:       {report['failed']}")
    print(f"  Manual review: {sorted(manual_sample)}")
    print(f"  Report:       {qa_path}")

    return report


if __name__ == "__main__":
    run_qa()
