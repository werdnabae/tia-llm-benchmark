"""
ceqanet_scraper.py — Download Transportation appendices from the California
State CEQA Clearinghouse (ceqanet.opr.ca.gov).

Strategy:
  1. Query /Search?ReceivedDate=YYYY-MM-DD for ~50 sampled dates across 2018-2023
  2. Filter rows with type "EIR" (Draft EIR / Supplemental EIR)
  3. For each EIR project page, extract Transportation appendix attachment links
  4. Download PDFs within the 1–30 MB size range
  5. Map to catalog case IDs; update download_log.json

All pages are server-rendered HTML; no JavaScript required.
"""

import argparse
import json
import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import random

import requests
from bs4 import BeautifulSoup

import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "ceqanet.log"),
    ],
)
log = logging.getLogger(__name__)

BASE = "https://ceqanet.opr.ca.gov"
DATE_SEARCH = f"{BASE}/Search"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# EIR document type codes we want
EIR_TYPES = {"EIR", "SBE", "SEA", "SIR", "SIS", "REV", "RIR"}

# Transport attachment title patterns
TRANSPORT_RE = re.compile(
    r"(?i)(traffic\s+impact|traffic\s+study|traffic\s+analysis|"
    r"\btia\b|transportation\s+impact|transportation\s+assessment|"
    r"transportation\s+study|transportation\s+analysis|"
    r"level\s+of\s+service|intersection.*analysis|"
    r"trip\s+gen|local\s+transportation\s+analysis|"
    r"traffic\s+engineering|traffic\s+report)",
)
SKIP_RE = re.compile(
    r"(?i)(noise|air.qual|bio|cultural|historic|shadow|wind|fiscal|"
    r"geo|hydro|hazard|visual|aesthet|mmrp|nop|nod|notice\.|newspaper|"
    r"scoping|comment\s+letter|appendi[cx].*(noise|air|bio|cult|hist))",
)

MIN_SIZE_KB = 800  # ignore cover pages / brief appendices
MAX_SIZE_KB = 28_000  # ignore oversized raw-data volumes

# Project type keywords
PTYPE_MAP = [
    (re.compile(r"(?i)(mixed.use|mixed use)"), "mixed_use"),
    (
        re.compile(r"(?i)(residential|apartment|multifam|housing|dwelling)"),
        "residential_multifamily",
    ),
    (re.compile(r"(?i)(single.family|sfr)"), "residential_single_family"),
    (re.compile(r"(?i)(retail|commercial|shopping|center)"), "commercial_retail"),
    (re.compile(r"(?i)(office|campus|corporat)"), "commercial_office"),
    (
        re.compile(r"(?i)(warehouse|industr|logistic|distribut|business.park)"),
        "industrial_warehouse",
    ),
    (re.compile(r"(?i)(hotel|hospitality|resort)"), "hotel"),
    (re.compile(r"(?i)(hospital|medical|clinic|health)"), "medical"),
    (re.compile(r"(?i)(school|universit|college)"), "institutional"),
    (
        re.compile(r"(?i)(transit|tod|station\s+area|rail)"),
        "transit_oriented_development",
    ),
]


def guess_project_type(title: str, description: str = "") -> str:
    text = (title + " " + description).lower()
    for pattern, ptype in PTYPE_MAP:
        if pattern.search(text):
            return ptype
    return "mixed_use"


def sample_dates(year_start: int = 2018, year_end: int = 2023) -> list[str]:
    """
    Generate dates to check. Uses every end-of-month date (EIR filings cluster
    around month-end deadlines) plus the 15th of each month.
    This is fully deterministic — no random sampling — so re-running always
    finds new dates not yet checked.
    """
    dates = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            # Last day of month
            if month == 12:
                last = date(year, month, 31)
            else:
                last = date(year, month + 1, 1) - timedelta(days=1)
            dates.append(last.isoformat())
            # 15th of month
            dates.append(date(year, month, 15).isoformat())
            # Also end of each quarter (high-volume filing dates)
            if month in (3, 6, 9, 12):
                dates.append(date(year, month, 1).isoformat())
    return dates


def fetch_html(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 200:
                return BeautifulSoup(r.text, "lxml")
            log.warning("  HTTP %d: %s", r.status_code, url[-60:])
        except Exception as e:
            log.warning("  Fetch error (attempt %d): %s", attempt + 1, e)
        time.sleep(1.0)
    return None


def get_eir_rows_for_date(date_str: str) -> list[dict]:
    """
    Fetch /Search?ReceivedDate=YYYY-MM-DD and return EIR-type rows.
    Each row: {sch, doc_url, title, agency, doc_type}
    """
    url = f"{DATE_SEARCH}?ReceivedDate={date_str}"
    soup = fetch_html(url)
    if soup is None:
        return []

    rows = soup.select("table tr")
    results = []
    for row in rows[1:]:  # skip header
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        doc_type = cells[1].get_text(strip=True).upper()
        if doc_type not in EIR_TYPES:
            continue

        sch_link = cells[0].find("a")
        doc_link = cells[1].find("a")
        if not sch_link:
            continue

        sch = sch_link.get_text(strip=True)
        sch_href = str(sch_link.get("href", ""))
        doc_href = str(doc_link.get("href", "")) if doc_link else sch_href
        title = cells[-1].get_text(strip=True)
        agency = cells[2].get_text(strip=True) if len(cells) > 2 else ""

        # Prefer the specific document URL (e.g. /2022060070/2)
        doc_url = BASE + doc_href if doc_href.startswith("/") else doc_href
        results.append(
            {
                "sch": sch,
                "doc_url": doc_url,
                "title": title,
                "agency": agency,
                "doc_type": doc_type,
                "date": date_str,
            }
        )
    return results


def get_transport_attachments(eir: dict) -> list[dict]:
    """
    Fetch the EIR project page and extract Transportation-labeled attachments.
    """
    soup = fetch_html(eir["doc_url"])
    if soup is None:
        return []

    page_text = soup.get_text(" ", strip=True)

    # Extract city
    city = ""
    m = re.search(
        r"Cities\s+([\w\s,]+?)(?:Counties|Regions|Zip|Cross|$)", page_text, re.I
    )
    if m:
        city = m.group(1).strip().split(",")[0].strip()

    # Extract year
    year = int(eir["date"][:4])

    # Extract description snippet
    description = ""
    m = re.search(
        r"Document Description\s+(.{50,400}?)(?:\s{3,}|Lead Agency|Contact)",
        page_text,
        re.S | re.I,
    )
    if m:
        description = m.group(1).strip()

    # Find attachment links
    attachments = []
    for a_tag in soup.find_all("a", href=True):
        href = str(a_tag["href"])
        if "/Attachment/" not in href:
            continue

        # Get link text + surrounding context
        raw_text = a_tag.get_text(strip=True)
        if not raw_text or len(raw_text) < 4:
            container = (
                a_tag.find_parent("td") or a_tag.find_parent("li") or a_tag.parent
            )
            raw_text = container.get_text(strip=True)[:150] if container else ""

        title_for_check = raw_text

        if not TRANSPORT_RE.search(title_for_check):
            continue
        if SKIP_RE.search(title_for_check):
            continue

        # Try to get file size
        size_kb = None
        container = a_tag.find_parent("td") or a_tag.find_parent("li") or a_tag.parent
        if container:
            m_size = re.search(r"([\d,]+)\s*K", container.get_text(), re.I)
            if m_size:
                size_kb = int(m_size.group(1).replace(",", ""))

        full_url = BASE + href if href.startswith("/") else href

        if size_kb is not None and (size_kb < MIN_SIZE_KB or size_kb > MAX_SIZE_KB):
            log.debug("  SKIP attachment %s (%s KB)", raw_text[:40], size_kb)
            continue

        attachments.append(
            {
                "sch": eir["sch"],
                "project_title": eir["title"],
                "agency": eir["agency"],
                "city": city,
                "year": year,
                "description": description,
                "attach_title": raw_text[:120],
                "url": full_url,
                "size_kb": size_kb,
                "project_type": guess_project_type(eir["title"], description),
            }
        )

    return attachments


def download_pdf(url: str, out_path: Path, max_mb: float = 28.0) -> Optional[int]:
    if out_path.exists() and out_path.stat().st_size > 50_000:
        return out_path.stat().st_size  # already downloaded

    try:
        r = SESSION.get(url, timeout=90, stream=True)
        if r.status_code != 200:
            return None
        data = b""
        for chunk in r.iter_content(65536):
            data += chunk
            if len(data) > max_mb * 1_048_576:
                log.warning("  ABORT — exceeded %.0f MB cap", max_mb)
                return None
        if data[:5] != b"%PDF-" or len(data) < 50_000:
            return None
        out_path.write_bytes(data)
        return len(data)
    except Exception as e:
        log.error("  Download error: %s", e)
        return None


def run_ceqanet(
    max_downloads: int = 40, year_start: int = 2018, year_end: int = 2023
) -> dict:

    log_path = LOGS_DIR / "download_log.json"
    dl_log = json.loads(log_path.read_text()) if log_path.exists() else {}

    random.seed(42)
    dates = sample_dates(year_start, year_end)
    log.info("Sampling %d dates across %d–%d", len(dates), year_start, year_end)

    # ── Collect EIR candidates ─────────────────────────────────────────────────
    all_eirs: list[dict] = []
    seen_sch: set = set()
    dates_checked = 0

    for date_str in dates:
        if len(all_eirs) >= max_downloads * 6:
            break
        if dates_checked >= 120:  # hard cap on date requests
            break

        rows = get_eir_rows_for_date(date_str)
        new_eirs = [r for r in rows if r["sch"] not in seen_sch]
        for r in new_eirs:
            seen_sch.add(r["sch"])
        all_eirs.extend(new_eirs)
        dates_checked += 1

        if new_eirs:
            log.info(
                "Date %s: %d EIR docs (total so far: %d)",
                date_str,
                len(new_eirs),
                len(all_eirs),
            )
        time.sleep(0.6)

    log.info("Total unique EIR projects collected: %d", len(all_eirs))

    # ── Extract transport attachments ──────────────────────────────────────────
    candidates: list[dict] = []
    seen_attach_sch: set = set()

    for eir in all_eirs:
        if len(candidates) >= max_downloads * 4:
            break

        log.info("Scanning %s — %s", eir["sch"], eir["title"][:60])
        attaches = get_transport_attachments(eir)
        for a in attaches:
            if a["sch"] not in seen_attach_sch:
                seen_attach_sch.add(a["sch"])
                candidates.append(a)
                log.info(
                    "  ✓ Transport attachment: %s (%.0f KB)",
                    a["attach_title"][:50],
                    a["size_kb"] if a["size_kb"] else 0,
                )
        time.sleep(0.8)

    log.info("Transport attachment candidates: %d", len(candidates))

    # ── Download ───────────────────────────────────────────────────────────────
    # Sort by size preference: 2–15 MB first
    def size_rank(c):
        kb = c.get("size_kb") or 5000
        return 0 if 2000 <= kb <= 15000 else (1 if kb < 2000 else 2)

    candidates.sort(key=size_rank)
    downloaded = 0

    for idx, cand in enumerate(candidates[:max_downloads]):
        sch_short = cand["sch"][-6:]
        case_id = f"CEQ-CA-{sch_short}"
        pdf_path = RAW_DIR / f"{case_id}.pdf"

        log.info(
            "GET [%d/%d] %s  %s",
            idx + 1,
            min(max_downloads, len(candidates)),
            case_id,
            cand["url"][-70:],
        )

        nbytes = download_pdf(cand["url"], pdf_path)
        if nbytes:
            log.info("  ✓ %.1f MB", nbytes / 1_048_576)
            dl_log[case_id] = {
                "status": "success",
                "source": "ceqanet",
                "sch_number": cand["sch"],
                "project_title": cand["project_title"],
                "agency": cand["agency"],
                "city": cand["city"],
                "year": cand["year"],
                "attach_title": cand["attach_title"],
                "pdf_url": cand["url"],
                "bytes": nbytes,
                "project_type": cand["project_type"],
                "state": "CA",
                "access_date": str(date.today()),
            }
            downloaded += 1
        else:
            log.warning("  FAIL  %s", cand["url"][-60:])
            dl_log[case_id] = {
                "status": "failed",
                "pdf_url": cand["url"],
                "access_date": str(date.today()),
            }
        time.sleep(1.0)

    log_path.write_text(json.dumps(dl_log, indent=2))

    # Save manifest of all candidates found
    manifest_path = LOGS_DIR / "ceqanet_manifest.json"
    manifest_path.write_text(json.dumps(candidates, indent=2))

    print(f"\n── CEQAnet Summary ──────────────────────────────────────────")
    print(f"  Dates checked:     {dates_checked}")
    print(f"  EIR projects:      {len(all_eirs)}")
    print(f"  Transport cands:   {len(candidates)}")
    print(f"  Downloaded:        {downloaded}")
    return dl_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CEQAnet Transportation Appendix Scraper"
    )
    parser.add_argument("--max-downloads", type=int, default=40)
    parser.add_argument("--year-start", type=int, default=2018)
    parser.add_argument("--year-end", type=int, default=2023)
    args = parser.parse_args()
    run_ceqanet(
        max_downloads=args.max_downloads,
        year_start=args.year_start,
        year_end=args.year_end,
    )
