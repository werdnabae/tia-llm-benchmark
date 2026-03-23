"""
build_dataset.py — Assemble the final tia_dataset.jsonl from validated split files.

Only includes records that passed QA (extraction_qa_passed = True),
UNLESS override=True is passed (includes all with a warning flag).

Output: /data/final/tia_dataset.jsonl
        /data/final/tia_dataset_index.json   (lightweight index for quick lookup)
"""

import json
import logging
from datetime import date
from pathlib import Path

from config import SPLIT_DIR, FINAL_DIR, LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "build.log"),
    ],
)
log = logging.getLogger(__name__)

FINAL_JSONL = FINAL_DIR / "tia_dataset.jsonl"
FINAL_INDEX = FINAL_DIR / "tia_dataset_index.json"

# Canonical field order for each JSONL record
FIELD_ORDER = ["case_id", "input", "ground_truth", "metadata"]


def clean_record(record: dict) -> dict:
    """
    Return a clean copy of the record with:
      - Only canonical top-level fields
      - Private/debug fields (_parse_coverage, _has_ground_truth) removed
      - build_date added to metadata
    """
    clean = {k: record[k] for k in FIELD_ORDER if k in record}
    clean["metadata"] = dict(clean.get("metadata", {}))
    clean["metadata"]["build_date"] = str(date.today())
    return clean


def build_dataset(include_failed_qa: bool = False) -> dict:
    """
    Read all split files, filter by QA, write final JSONL.
    Returns a summary dict.
    """
    split_files = sorted(SPLIT_DIR.glob("*.json"))
    if not split_files:
        log.warning("No split files found in %s", SPLIT_DIR)
        return {}

    records = []
    skipped_qa = []
    skipped_no_gt = []

    for sf in split_files:
        with open(sf) as f:
            record = json.load(f)

        case_id = record.get("case_id", sf.stem)
        qa_ok = record.get("metadata", {}).get("extraction_qa_passed", False)
        has_gt = bool(
            record.get("ground_truth", {}).get("issues")
            or record.get("ground_truth", {}).get("recommendations")
        )

        if not qa_ok and not include_failed_qa:
            log.info("SKIP  %s (QA failed)", case_id)
            skipped_qa.append(case_id)
            continue

        if not has_gt:
            log.warning("SKIP  %s (no ground truth extracted)", case_id)
            skipped_no_gt.append(case_id)
            continue

        records.append(clean_record(record))
        log.info("ADD   %s", case_id)

    if not records:
        log.error("No records to write!")
        return {"error": "empty dataset"}

    # Write JSONL
    with open(FINAL_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), FINAL_JSONL)

    # Write lightweight index
    index = {
        "created": str(date.today()),
        "total_records": len(records),
        "entries": [
            {
                "case_id": r["case_id"],
                "state": r["metadata"].get("state"),
                "agency": r["metadata"].get("agency"),
                "year": r["metadata"].get("year"),
                "issues": len(r["ground_truth"].get("issues", [])),
                "recs": len(r["ground_truth"].get("recommendations", [])),
                "exclude_from_retrieval": r["metadata"].get(
                    "exclude_from_retrieval", []
                ),
            }
            for r in records
        ],
    }
    with open(FINAL_INDEX, "w") as f:
        json.dump(index, f, indent=2)

    summary = {
        "total_included": len(records),
        "skipped_qa_failed": len(skipped_qa),
        "skipped_no_gt": len(skipped_no_gt),
        "output_jsonl": str(FINAL_JSONL),
        "output_index": str(FINAL_INDEX),
        "states": sorted(
            {r["metadata"].get("state") for r in records if r["metadata"].get("state")}
        ),
        "years": sorted(
            {r["metadata"].get("year") for r in records if r["metadata"].get("year")}
        ),
        "avg_issues": round(
            sum(len(r["ground_truth"].get("issues", [])) for r in records)
            / len(records),
            2,
        ),
        "avg_recs": round(
            sum(len(r["ground_truth"].get("recommendations", [])) for r in records)
            / len(records),
            2,
        ),
    }

    print(f"\n── Build Summary ─────────────────────────────────────────")
    print(f"  Included:   {summary['total_included']} records")
    print(f"  Skipped QA: {summary['skipped_qa_failed']}")
    print(f"  Skipped GT: {summary['skipped_no_gt']}")
    print(f"  States:     {summary['states']}")
    print(f"  Output:     {FINAL_JSONL}")

    return summary


if __name__ == "__main__":
    build_dataset()
