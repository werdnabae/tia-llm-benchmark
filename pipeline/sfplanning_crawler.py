"""
sfplanning_crawler.py — Crawl all 16 pages of the SF Planning EIR/NegDec catalog
and download transportation chapter PDFs hosted on sfplanning.s3.amazonaws.com
and sfmea.sfplanning.org / archives.sfplanning.org.

No authentication required.  All links are direct CDN URLs.

Usage:
  python3 sfplanning_crawler.py [--max-downloads N]
"""

import argparse
import json
import logging
import re
import time
from datetime import date
from pathlib import Path
from typing import Optional

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
        logging.FileHandler(LOGS_DIR / "sfplanning_crawl.log"),
    ],
)
log = logging.getLogger(__name__)

CATALOG_BASE = "https://sfplanning.org"
# Category 212 = EIRs and Negative Declarations
CATALOG_URL = (
    f"{CATALOG_BASE}/environmental-review-documents"
    "?field_environmental_review_categ_target_id=212"
)
TOTAL_PAGES = 16

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,*/*",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# Direct CDN patterns — confirmed accessible
DIRECT_CDN_RE = re.compile(
    r"https?://"
    r"(sfplanning\.s3\.amazonaws\.com/sfmea/"
    r"|sfmea\.sfplanning\.org/"
    r"|archives\.sfplanning\.org/documents/"
    r"|default\.sfplanning\.org/MEA/)",
    re.I,
)

# Transport-specific filename / link text patterns
TRANSPORT_LINK_RE = re.compile(
    r"(?i)(transport|traffic|tia|circulation|iv\.?d|4[.-]?d|trip\s*gen|"
    r"travel\s*demand|vehicular|multimodal|pedestrian.*circulation)",
)

# Things to skip (non-transport chapters)
SKIP_LINK_RE = re.compile(
    r"(?i)(noise|air.qual|bio|cultural|historic|shadow|wind|fiscal|land.use|"
    r"util|energy|agri|paleonto|geo|hydro|hazard|visual|aesthet|mmrp|nop|nod|"
    r"notice\.of|newspaper|scoping|comments)",
)

# Also grab whole DEIRs for specific SF projects in our catalog
KNOWN_SF_DEIRS = [
    # One Oak Street (1500 Market) — confirmed direct S3 link
    (
        "SFP-2009-0159",
        "https://sfplanning.s3.amazonaws.com/sfmea/2009.0159E_DEIR.pdf",
        "One Oak Street / 1500 Market Street DEIR",
        2017,
    ),
    # Visitacion Valley EIR parts with transportation
    (
        "SFP-2006-1308-t",
        "https://sfplanning.s3.amazonaws.com/sfmea/2006.1308E_VisValley_DEIR_Pt4.pdf",
        "Visitacion Valley DEIR Part 4 (Transportation)",
        2009,
    ),
    # Market & Octavia — Ch 4 (transportation)
    (
        "SFP-2003-0347-t",
        "https://sfmea.sfplanning.org/2003.0347E_Market_Octavia_Neighborhood_Plan_TOC_Ch.4.pdf",
        "Market & Octavia Neighborhood Plan Ch.4 Transportation",
        2008,
    ),
    # Western SoMa Ch 4
    (
        "SFP-2008-0877-t",
        "http://archives.sfplanning.org/documents/9037-2008.0877E_Chapter%204.pdf",
        "Western SoMa EIR Chapter 4 Transportation",
        2012,
    ),
    # Transit Center District FEIR Vol 2 (Ch IV.F–IV.S, includes transportation)
    (
        "SFP-2007-0558-t",
        "http://sfmea.sfplanning.org/2007.0558E_FEIR2.pdf",
        "Transit Center District Plan FEIR Volume 2",
        2012,
    ),
]


def fetch_html(url: str) -> Optional[BeautifulSoup]:
    try:
        r = SESSION.get(url, timeout=30)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "lxml")
        log.warning("  HTTP %d: %s", r.status_code, url)
    except Exception as e:
        log.warning("  Fetch error %s: %s", url, e)
    return None


def extract_transport_links(soup: BeautifulSoup) -> list[dict]:
    """Return list of {url, link_text} for transport CDN PDFs on this catalog page."""
    found = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if not DIRECT_CDN_RE.match(href):
            continue
        text = a.get_text(strip=True)
        # Use surrounding context if link text is just "PDF" or empty
        if not text or len(text) < 5:
            parent = a.find_parent(["li", "td", "p"])
            text = parent.get_text(" ", strip=True)[:120] if parent else ""

        if SKIP_LINK_RE.search(text):
            continue
        if TRANSPORT_LINK_RE.search(text) or TRANSPORT_LINK_RE.search(href):
            found.append({"url": href, "link_text": text})
    return found


def extract_case_year(soup: BeautifulSoup, url: str) -> tuple[str, int]:
    """Try to extract a SF Planning case number and year from the page."""
    case_num = ""
    m = re.search(r"(\d{4}\.\d{4}[A-Z]+)", url)
    if m:
        case_num = m.group(1)
    year = 2020
    m = re.search(r"/(20\d{2})\b", url)
    if m:
        year = int(m.group(1))
    return case_num, year


def download_pdf(
    url: str, out_path: Path, max_bytes: int = 30_000_000
) -> Optional[int]:
    """Download a PDF. Returns bytes written or None."""
    if out_path.exists() and out_path.stat().st_size > 50_000:
        return out_path.stat().st_size  # already have it

    try:
        r = SESSION.get(url, timeout=90, stream=True)
        if r.status_code != 200:
            log.warning("  HTTP %d: %s", r.status_code, url)
            return None
        data = b""
        for chunk in r.iter_content(65536):
            data += chunk
            if len(data) > max_bytes:
                log.warning("  ABORT %s — size cap hit", url[-60:])
                return None
        if data[:5] != b"%PDF-" or len(data) < 50_000:
            return None
        out_path.write_bytes(data)
        return len(data)
    except Exception as e:
        log.error("  Download error %s: %s", url, e)
        return None


def run_sfplanning(max_downloads: int = 20) -> dict:
    log_path = LOGS_DIR / "download_log.json"
    dl_log = json.loads(log_path.read_text()) if log_path.exists() else {}

    all_links: list[dict] = []

    # ── 1. Crawl all 16 catalog pages ─────────────────────────────────────────
    for page_num in range(TOTAL_PAGES):
        url = f"{CATALOG_URL}&page={page_num}"
        log.info("Crawling SF Planning catalog page %d …", page_num)
        soup = fetch_html(url)
        if soup is None:
            log.warning("  Could not fetch page %d", page_num)
            continue
        links = extract_transport_links(soup)
        log.info("  Found %d transport CDN links on page %d", len(links), page_num)
        for lk in links:
            lk["catalog_page"] = page_num
        all_links.extend(links)
        time.sleep(0.8)

    # ── 2. Add known hard-coded known-good DEIRs ──────────────────────────────
    for case_id, url, title, year in KNOWN_SF_DEIRS:
        all_links.append(
            {
                "url": url,
                "link_text": title,
                "case_id": case_id,
                "year": year,
                "catalog_page": -1,
            }
        )

    # Deduplicate by URL
    seen_urls: set = set()
    unique_links = []
    for lk in all_links:
        if lk["url"] not in seen_urls:
            seen_urls.add(lk["url"])
            unique_links.append(lk)

    log.info("Total unique transport CDN links: %d", len(unique_links))

    # ── 3. Download ────────────────────────────────────────────────────────────
    downloaded = 0
    for idx, lk in enumerate(unique_links):
        if downloaded >= max_downloads:
            break

        # Derive case_id from URL
        case_id = lk.get("case_id")
        if not case_id:
            m = re.search(r"(\d{4}[\.\-]\d{4,}[A-Z]*)", lk["url"])
            slug = m.group(1).replace(".", "-") if m else f"p{idx:03d}"
            case_id = f"SFP-CA-{slug}"

        pdf_path = RAW_DIR / f"{case_id}.pdf"
        log.info("GET [%d] %s  %s", idx + 1, case_id, lk["url"][-70:])

        nbytes = download_pdf(lk["url"], pdf_path)
        if nbytes:
            log.info("  ✓ %s  %.1f MB", case_id, nbytes / 1_048_576)
            dl_log[case_id] = {
                "status": "success",
                "source": "sfplanning_catalog",
                "link_text": lk["link_text"],
                "pdf_url": lk["url"],
                "bytes": nbytes,
                "year": lk.get("year", 2020),
                "agency": "San Francisco Planning Department",
                "state": "CA",
                "access_date": str(date.today()),
            }
            downloaded += 1
        else:
            dl_log[case_id] = {
                "status": "failed",
                "pdf_url": lk["url"],
                "access_date": str(date.today()),
            }
        time.sleep(0.6)

    log_path.write_text(json.dumps(dl_log, indent=2))

    print(f"\n── SF Planning Crawler Summary ──────────────────────────────")
    print(f"  Unique links found: {len(unique_links)}")
    print(f"  Downloaded:         {downloaded}")
    return dl_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SF Planning EIR Catalog Crawler")
    parser.add_argument("--max-downloads", type=int, default=20)
    args = parser.parse_args()
    run_sfplanning(max_downloads=args.max_downloads)
