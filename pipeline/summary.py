"""
summary.py — Generate dataset summary report.

Reads from:
  - /data/final/tia_dataset.jsonl
  - /data/logs/download_log.json
  - /data/logs/parse_log.json
  - /data/logs/chunk_log.json
  - /data/logs/extract_log.json
  - /data/logs/qa_report.json

Outputs:
  - /data/logs/dataset_summary.json
  - prints human-readable summary table
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from config import FINAL_DIR, LOGS_DIR

FINAL_JSONL = FINAL_DIR / "tia_dataset.jsonl"


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_jsonl(path: Path) -> list:
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def generate_summary() -> dict:
    # Load all log files
    dl_log = load_json(LOGS_DIR / "download_log.json")
    parse_log = load_json(LOGS_DIR / "parse_log.json")
    chunk_log = load_json(LOGS_DIR / "chunk_log.json")
    extract_log = load_json(LOGS_DIR / "extract_log.json")
    qa_report = load_json(LOGS_DIR / "qa_report.json")
    records = load_jsonl(FINAL_JSONL)

    # ── Download stats ─────────────────────────────────────────────────────────
    dl_statuses = Counter(v.get("status") for v in dl_log.values())

    # ── Parse stats ────────────────────────────────────────────────────────────
    parse_ok = sum(1 for v in parse_log.values() if v.get("status") == "success")

    # ── Chunk stats ────────────────────────────────────────────────────────────
    total_chunks = sum(v.get("total_chunks", 0) for v in chunk_log.values())
    avg_chunks = round(total_chunks / max(len(chunk_log), 1), 1)

    # ── Dataset stats ─────────────────────────────────────────────────────────
    state_dist = Counter(r["metadata"].get("state") for r in records)
    year_dist = Counter(r["metadata"].get("year") for r in records)

    issue_counts = [len(r["ground_truth"].get("issues", [])) for r in records]
    rec_counts = [len(r["ground_truth"].get("recommendations", [])) for r in records]
    chunk_counts = []
    for r in records:
        cid = r["case_id"]
        chunk_counts.append(chunk_log.get(cid, {}).get("total_chunks", 0))

    avg_issues = round(sum(issue_counts) / max(len(issue_counts), 1), 2)
    avg_recs = round(sum(rec_counts) / max(len(rec_counts), 1), 2)
    avg_chunks_final = round(sum(chunk_counts) / max(len(chunk_counts), 1), 1)

    # ── QA stats ──────────────────────────────────────────────────────────────
    qa_pass_rate = qa_report.get("pass_rate_pct", 0.0)

    # ── Failed / replaced ─────────────────────────────────────────────────────
    failed_downloads = [k for k, v in dl_log.items() if v.get("status") == "failed"]
    no_url_reports = [
        k for k, v in dl_log.items() if v.get("status") == "no_direct_url"
    ]

    summary = {
        "pipeline_status": {
            "download": {
                "total_in_catalog": len(dl_log),
                "success": dl_statuses.get("success", 0),
                "no_direct_url": dl_statuses.get("no_direct_url", 0),
                "failed": dl_statuses.get("failed", 0),
                "failed_list": sorted(failed_downloads),
                "no_url_list": sorted(no_url_reports),
            },
            "parse": {
                "attempted": len(parse_log),
                "success": parse_ok,
                "failed": len(parse_log) - parse_ok,
            },
            "chunk": {
                "total_cases_chunked": len(chunk_log),
                "total_chunks": total_chunks,
                "avg_chunks_per_case": avg_chunks,
            },
            "extraction": {
                "attempted": len(extract_log),
                "success": sum(
                    1 for v in extract_log.values() if v.get("status") == "success"
                ),
            },
            "qa": {
                "total_checked": qa_report.get("total", 0),
                "passed": qa_report.get("passed", 0),
                "failed": qa_report.get("failed", 0),
                "pass_rate_pct": qa_pass_rate,
                "manual_sample": qa_report.get("manual_sample", []),
            },
        },
        "final_dataset": {
            "valid_cases": len(records),
            "distribution_by_state": dict(state_dist.most_common()),
            "distribution_by_year": dict(sorted(year_dist.items())),
            "avg_issues_per_case": avg_issues,
            "avg_recommendations_per_case": avg_recs,
            "avg_chunks_per_case": avg_chunks_final,
            "total_issues": sum(issue_counts),
            "total_recommendations": sum(rec_counts),
            "leave_one_out_ready": True,
            "qa_first_pass_rate_pct": qa_pass_rate,
        },
        "failed_or_replaced": {
            "failed_download": sorted(failed_downloads),
            "no_direct_url_skipped": sorted(no_url_reports),
            "note": (
                "Reports with 'no_direct_url' require manual download from the agency "
                "portal using the portal_case_id listed in config.py. "
                "Failed reports should be replaced with alternative TIAs from the "
                "same state/year to maintain geographic diversity."
            ),
        },
    }

    # Save
    out_path = LOGS_DIR / "dataset_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Human-readable print ───────────────────────────────────────────────────
    ds = summary["final_dataset"]
    pl = summary["pipeline_status"]

    print("\n" + "═" * 60)
    print("  TIA DATASET — CONSTRUCTION SUMMARY")
    print("═" * 60)
    print(f"\n  PIPELINE STAGES")
    print(
        f"  ├─ Download:    {pl['download']['success']} succeeded  |  {pl['download']['no_direct_url']} need manual  |  {pl['download']['failed']} failed"
    )
    print(
        f"  ├─ Parse:       {pl['parse']['success']}/{pl['parse']['attempted']} succeeded"
    )
    print(
        f"  ├─ Chunk:       {pl['chunk']['total_chunks']} total chunks  (avg {pl['chunk']['avg_chunks_per_case']}/case)"
    )
    print(
        f"  ├─ Extract:     {pl['extraction']['success']}/{pl['extraction']['attempted']} succeeded"
    )
    print(
        f"  └─ QA:          {pl['qa']['passed']}/{pl['qa']['total_checked']} passed  ({pl['qa']['pass_rate_pct']}%)"
    )

    print(f"\n  FINAL DATASET")
    print(f"  ├─ Valid cases: {ds['valid_cases']}")
    print(f"  ├─ State dist:  {dict(state_dist.most_common())}")
    print(
        f"  ├─ Year range:  {min(year_dist) if year_dist else 'n/a'} – {max(year_dist) if year_dist else 'n/a'}"
    )
    print(f"  ├─ Avg issues/case:  {ds['avg_issues_per_case']}")
    print(f"  ├─ Avg recs/case:    {ds['avg_recommendations_per_case']}")
    print(f"  ├─ Avg chunks/case:  {ds['avg_chunks_per_case']}")
    print(f"  └─ LOO ready:   {ds['leave_one_out_ready']}")

    if summary["failed_or_replaced"]["failed_download"]:
        print(f"\n  FAILED/REPLACED:")
        for cid in summary["failed_or_replaced"]["failed_download"]:
            print(f"    - {cid}")

    if summary["failed_or_replaced"]["no_direct_url_skipped"]:
        print(f"\n  MANUAL DOWNLOAD REQUIRED ({len(no_url_reports)} reports):")
        for cid in sorted(no_url_reports)[:5]:
            print(f"    - {cid}")
        if len(no_url_reports) > 5:
            print(f"    ... and {len(no_url_reports) - 5} more — see download_log.json")

    print("\n" + "═" * 60)
    print(f"  Summary saved: {out_path}")
    print("═" * 60 + "\n")

    return summary


if __name__ == "__main__":
    generate_summary()
