"""
parse.py — Extract text from PDFs and detect section boundaries.
Handles both standalone TIA reports and EIR transportation chapter PDFs.
Output: /data/parsed/{case_id}_sections.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pdfplumber

from config import (
    RAW_DIR,
    PARSED_DIR,
    LOGS_DIR,
    SECTION_PATTERNS,
    INPUT_SECTIONS,
    GROUND_TRUTH_SECTIONS,
)

# ── EIR-specific supplemental section patterns ────────────────────────────────
# These catch headers used in California CEQA EIR transportation chapters
# that differ from standalone TIA headers.
EIR_EXTRA_PATTERNS = [
    # EIR section headers like "IV.D Transportation and Circulation"
    ("study_area", r"(?i)^\s*[IVX]+\.[A-Z]\s+(Transportation|Traffic|Circulation)"),
    # Common EIR subsection for existing conditions
    (
        "existing_conditions",
        r"(?i)^\s*(Existing\s+(Transportation\s+)?(Conditions?|Network|System|LOS)"
        r"|Baseline\s+(Transportation|Traffic|Conditions?)"
        r"|Existing\s+Roadway\s+Conditions?)",
    ),
    # Future no-build / background
    (
        "no_build_conditions",
        r"(?i)^\s*(Future\s+No.?Build"
        r"|Background\s+(Traffic\s+)?Conditions?"
        r"|Ambient\s+Growth"
        r"|No.?Project\s+(Alternative\s+)?Conditions?)",
    ),
    # Trip gen in EIR format
    (
        "trip_generation",
        r"(?i)^\s*(Project\s+Trip\s+Generation"
        r"|Estimated\s+Trip\s+Generation"
        r"|Traffic\s+Generation)",
    ),
    # Trip distribution
    (
        "trip_distribution",
        r"(?i)^\s*(Trip\s+Distribution\s+(and\s+(Traffic\s+)?Assignment)?"
        r"|Traffic\s+Distribution\s+and\s+Assignment"
        r"|Distribution\s+and\s+Assignment)",
    ),
    # Future build / with-project
    (
        "future_build",
        r"(?i)^\s*(Future\s+(Build|With.?Project)\s+Conditions?"
        r"|With.?Project\s+(Traffic\s+)?Conditions?"
        r"|Build\s+Out\s+Conditions?"
        r"|Project.?Generated\s+Traffic\s+Impacts?)",
    ),
    # Findings — most common EIR variations
    (
        "findings",
        r"(?i)^\s*(Intersection\s+Level\s+of\s+Service"
        r"|Level\s+of\s+Service\s+(Analysis|Results|Summary|Evaluation|Impacts?)"
        r"|LOS\s+(Analysis|Results|Summary|Impacts?)"
        r"|Traffic\s+(Operations?|Impacts?|Analysis\s+Results)"
        r"|Cumulative\s+Traffic\s+Impacts?"
        r"|Roadway\s+(Segment|Level\s+of\s+Service)\s+(Analysis|Results)"
        r"|Signalized\s+Intersection\s+(Analysis|LOS)"
        r"|Peak.Hour\s+(LOS|Level\s+of\s+Service))",
    ),
    # Mitigation — EIR often has longer heading
    (
        "mitigation",
        r"(?i)^\s*(Mitigation\s+Measure[s\s]"
        r"|Transportation\s+Mitigation"
        r"|Traffic\s+Improvements?"
        r"|Transportation\s+Improvements?"
        r"|Recommended\s+(Improvements?|Mitigation)"
        r"|Proposed\s+(Mitigations?|Improvements?)"
        r"|Required\s+Improvements?"
        r"|TDM\s+(Program|Measures?))",
    ),
]

# Merge extra patterns into a compiled list — EIR patterns take precedence
_ALL_PATTERNS = [
    (name, re.compile(pattern, re.MULTILINE)) for name, pattern in EIR_EXTRA_PATTERNS
] + [(name, re.compile(pattern, re.MULTILINE)) for name, pattern in SECTION_PATTERNS]


# Page cap by file size (in bytes)
def page_cap_for_size(size_bytes: int) -> int:
    mb = size_bytes / 1_048_576
    if mb < 5:
        return 999  # parse all pages
    elif mb < 15:
        return 200
    elif mb < 25:
        return 150
    else:
        return 120


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "parse.log"),
    ],
)
log = logging.getLogger(__name__)


def classify_line(line: str) -> Optional[str]:
    """Return section name if the line is a section heading, else None.
    EIR-specific patterns are tried first."""
    stripped = line.strip()
    if not stripped or len(stripped) > 160:
        return None
    for section_name, pattern in _ALL_PATTERNS:
        if pattern.match(stripped):
            return section_name
    return None


def extract_text_with_pdfplumber(pdf_path: Path) -> list[dict]:
    """
    Open a PDF with pdfplumber.  Apply a size-based page cap so large
    EIR volumes don't time out.
    Returns list of {page_number, text} dicts.
    """
    pages = []
    size_bytes = pdf_path.stat().st_size
    cap = page_cap_for_size(size_bytes)
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total = len(pdf.pages)
            read_to = min(total, cap)
            for i, page in enumerate(pdf.pages[:read_to], start=1):
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                pages.append({"page_number": i, "text": text})
            if total > read_to:
                log.info(
                    "  Page cap applied: read %d/%d pages (%.0f MB)",
                    read_to,
                    total,
                    size_bytes / 1_048_576,
                )
    except Exception as e:
        log.error("pdfplumber failed on %s: %s", pdf_path.name, e)
    return pages


def stitch_into_sections(pages: list[dict]) -> list[dict]:
    """
    Walk through all page text line by line.
    When a section-heading line is found, close the current section and
    open a new one.  Returns list of section dicts.
    """
    sections = []
    current_section = "preamble"  # text before first detected heading
    current_text_buf = []

    def flush():
        text = "\n".join(current_text_buf).strip()
        if text:
            sections.append(
                {
                    "section_name": current_section,
                    "text": text,
                    "page_start": _page_start,
                    "page_end": _page_end,
                }
            )

    _page_start = 1
    _page_end = 1

    for page_dict in pages:
        page_no = page_dict["page_number"]
        for line in page_dict["text"].split("\n"):
            detected = classify_line(line)
            if detected:
                # Save accumulated text under current section
                _page_end = page_no
                flush()
                current_section = detected
                current_text_buf = []
                _page_start = page_no
            else:
                current_text_buf.append(line)
                _page_end = page_no

    # Flush the last section
    flush()

    # Collapse consecutive chunks of the same section (can happen with
    # appendices that repeat headers)
    merged = []
    for sec in sections:
        if merged and merged[-1]["section_name"] == sec["section_name"]:
            merged[-1]["text"] += "\n" + sec["text"]
            merged[-1]["page_end"] = sec["page_end"]
        else:
            merged.append(sec)

    # Label INPUT vs GROUND_TRUTH vs OTHER
    for sec in merged:
        name = sec["section_name"]
        if name in INPUT_SECTIONS:
            sec["role"] = "input"
        elif name in GROUND_TRUTH_SECTIONS:
            sec["role"] = "ground_truth"
        else:
            sec["role"] = "other"

    return merged


def parse_pdf(case_id: str, pdf_path: Path) -> Optional[dict]:
    """
    Full parse of one PDF.  Returns the section dict, or None on failure.
    """
    log.info("Parsing %s  [%s]", case_id, pdf_path.name)
    pages = extract_text_with_pdfplumber(pdf_path)
    if not pages:
        log.error("No pages extracted from %s", pdf_path.name)
        return None

    sections = stitch_into_sections(pages)
    if not sections:
        log.error("No sections detected for %s", case_id)
        return None

    # Report coverage
    found = {s["section_name"] for s in sections}
    input_found = found & INPUT_SECTIONS
    gt_found = found & GROUND_TRUTH_SECTIONS
    log.info(
        "  %s: %d sections | input=%d/%d | gt=%d/%d | pages=%d",
        case_id,
        len(sections),
        len(input_found),
        len(INPUT_SECTIONS),
        len(gt_found),
        len(GROUND_TRUTH_SECTIONS),
        len(pages),
    )

    return {
        "case_id": case_id,
        "pdf_path": str(pdf_path),
        "num_pages": len(pages),
        "sections": sections,
        "coverage": {
            "input_sections_found": sorted(input_found),
            "ground_truth_found": sorted(gt_found),
            "all_detected": sorted(found),
            "input_coverage_pct": round(
                len(input_found) / len(INPUT_SECTIONS) * 100, 1
            ),
            "ground_truth_coverage_pct": round(
                len(gt_found) / len(GROUND_TRUTH_SECTIONS) * 100, 1
            ),
        },
    }


def run_parsing(case_ids=None) -> dict:
    """
    Parse all downloaded PDFs.  Saves per-case JSON to data/parsed/.
    Returns dict of {case_id: parse_result}.
    """
    parse_log = {}

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDFs found in %s", RAW_DIR)
        return parse_log

    for pdf_path in pdf_files:
        case_id = pdf_path.stem  # filename without .pdf
        if case_ids and case_id not in case_ids:
            continue

        result = parse_pdf(case_id, pdf_path)
        if result is None:
            parse_log[case_id] = {"status": "failed"}
            continue

        out_path = PARSED_DIR / f"{case_id}_sections.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        parse_log[case_id] = {
            "status": "success",
            "out_path": str(out_path),
            "coverage": result["coverage"],
        }

    # Save summary
    with open(LOGS_DIR / "parse_log.json", "w") as f:
        json.dump(parse_log, f, indent=2)

    ok = sum(1 for v in parse_log.values() if v.get("status") == "success")
    total = len(parse_log)
    print(f"\n── Parse Summary: {ok}/{total} succeeded ──────────────────")
    return parse_log


if __name__ == "__main__":
    run_parsing()
