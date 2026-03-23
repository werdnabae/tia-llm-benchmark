"""
chunk.py — Section-aware chunking of parsed TIA sections.

Rules:
  - NO arbitrary token chunking; every chunk is a complete semantic unit
  - Sections <= CHUNK_MAX_CHARS → single chunk
  - Sections > CHUNK_MAX_CHARS → split at paragraph (blank-line) boundaries,
    then at sentence boundary if a paragraph is still too large
  - Each chunk carries: case_id, section_name, subsection_name, chunk_index, text
  - Ground-truth sections are chunked separately and flagged

Output: /data/chunks/{case_id}.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from config import PARSED_DIR, CHUNKS_DIR, LOGS_DIR, CHUNK_MAX_CHARS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "chunk.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Subsection heading detection ───────────────────────────────────────────────
# Matches numbered or titled subsections like "3.1 Existing Intersection LOS"
_SUBSECTION_RE = re.compile(
    r"^(\d+[\.\d]*\s+.{5,60}|[A-Z][A-Za-z\s\-/]{4,59})\s*$",
    re.MULTILINE,
)

# Sentence boundary (rough; avoids splitting on abbreviations like "Fig.")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def detect_subsection(text: str) -> Optional[str]:
    """Return the first subsection heading found in text, or None."""
    m = _SUBSECTION_RE.search(text[:300])
    return m.group(1).strip() if m else None


def split_by_paragraphs(text: str) -> list[str]:
    """Split text at blank lines."""
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def split_long_paragraph(para: str, max_chars: int) -> list[str]:
    """
    If a single paragraph exceeds max_chars, split at sentence boundaries.
    As a last resort, hard-split at max_chars.
    """
    if len(para) <= max_chars:
        return [para]
    sentences = _SENTENCE_RE.split(para)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip()
    if current:
        chunks.append(current)
    # Hard split if still too large
    final = []
    for c in chunks:
        while len(c) > max_chars:
            final.append(c[:max_chars])
            c = c[max_chars:]
        if c:
            final.append(c)
    return final


def chunk_section(
    case_id: str,
    section_name: str,
    role: str,
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[dict]:
    """
    Produce a list of chunk dicts for one section.
    """
    chunks = []

    if len(text) <= max_chars:
        subsection = detect_subsection(text)
        chunks.append(
            {
                "case_id": case_id,
                "section_name": section_name,
                "subsection_name": subsection,
                "role": role,
                "chunk_index": 0,
                "text": text,
                "char_count": len(text),
            }
        )
        return chunks

    # Split into paragraphs first
    paragraphs = split_by_paragraphs(text)
    current_buf = []
    current_len = 0
    chunk_idx = 0
    subsection = detect_subsection(text)  # section-level fallback

    def flush_buf():
        nonlocal chunk_idx, current_buf, current_len
        joined = "\n\n".join(current_buf).strip()
        if joined:
            local_subsec = detect_subsection(joined) or subsection
            chunks.append(
                {
                    "case_id": case_id,
                    "section_name": section_name,
                    "subsection_name": local_subsec,
                    "role": role,
                    "chunk_index": chunk_idx,
                    "text": joined,
                    "char_count": len(joined),
                }
            )
            chunk_idx += 1
        current_buf = []
        current_len = 0

    for para in paragraphs:
        # A paragraph may itself be very long
        sub_paras = split_long_paragraph(para, max_chars)
        for sp in sub_paras:
            if current_len + len(sp) + 2 > max_chars and current_buf:
                flush_buf()
            current_buf.append(sp)
            current_len += len(sp) + 2

    flush_buf()

    return chunks


def chunk_case(case_id: str) -> Optional[list[dict]]:
    """Load parsed sections and produce all chunks for one case."""
    parsed_path = PARSED_DIR / f"{case_id}_sections.json"
    if not parsed_path.exists():
        log.warning("Parsed file not found: %s", parsed_path)
        return None

    with open(parsed_path) as f:
        parsed = json.load(f)

    all_chunks = []
    for sec in parsed["sections"]:
        sec_name = sec["section_name"]
        role = sec.get("role", "other")
        text = sec.get("text", "").strip()
        if not text:
            continue
        c = chunk_section(case_id, sec_name, role, text)
        all_chunks.extend(c)

    log.info(
        "%s: %d sections → %d chunks",
        case_id,
        len(parsed["sections"]),
        len(all_chunks),
    )
    return all_chunks


def run_chunking(case_ids=None) -> dict:
    """Chunk all parsed files. Saves per-case JSON to data/chunks/."""
    chunk_log = {}

    parsed_files = sorted(PARSED_DIR.glob("*_sections.json"))
    if not parsed_files:
        log.warning("No parsed files found in %s", PARSED_DIR)
        return chunk_log

    for pf in parsed_files:
        case_id = pf.stem.replace("_sections", "")
        if case_ids and case_id not in case_ids:
            continue

        chunks = chunk_case(case_id)
        if chunks is None:
            chunk_log[case_id] = {"status": "failed"}
            continue

        out_path = CHUNKS_DIR / f"{case_id}.json"
        with open(out_path, "w") as f:
            json.dump(
                {"case_id": case_id, "chunks": chunks}, f, indent=2, ensure_ascii=False
            )

        input_chunks = [c for c in chunks if c["role"] == "input"]
        gt_chunks = [c for c in chunks if c["role"] == "ground_truth"]

        chunk_log[case_id] = {
            "status": "success",
            "total_chunks": len(chunks),
            "input_chunks": len(input_chunks),
            "gt_chunks": len(gt_chunks),
            "other_chunks": len(chunks) - len(input_chunks) - len(gt_chunks),
        }

    with open(LOGS_DIR / "chunk_log.json", "w") as f:
        json.dump(chunk_log, f, indent=2)

    ok = sum(1 for v in chunk_log.values() if v.get("status") == "success")
    print(f"\n── Chunk Summary: {ok}/{len(chunk_log)} succeeded ──────────────────")
    return chunk_log


if __name__ == "__main__":
    run_chunking()
