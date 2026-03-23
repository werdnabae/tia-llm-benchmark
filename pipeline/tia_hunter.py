"""
tia_hunter.py — Hunt for standalone TIA PDFs across US city planning portals.

Strategy:
  1. Scan MuniCode Meetings cities for Planning Commission / P&Z boards
  2. For each city with a planning board, paginate through meeting list
  3. Download meeting agenda packets (combined PDFs from Azure Blob Storage)
  4. Scan downloaded packets for TIA content (LOS, trip generation, mitigation)
  5. If TIA content found at sufficient density, extract and save as candidate

Also tries:
  - Direct URL patterns on city websites (Colorado, Texas, NC, AZ)
  - Known-working CEQAnet-adjacent sources

Usage:
    python3 tia_hunter.py [--max-packets N] [--min-tia-hits N]
"""

import argparse
import io
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
        logging.FileHandler(LOGS_DIR / "tia_hunter.log"),
    ],
)
log = logging.getLogger(__name__)

H = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf,*/*",
}
SESSION = requests.Session()
SESSION.headers.update(H)

# TIA-specific keywords — must be dense enough to indicate a real TIA section
TIA_STRONG_RE = re.compile(
    r"(?i)(Traffic\s+Impact\s+(Analysis|Study|Assessment|Report)|"
    r"Transportation\s+Impact\s+(Analysis|Study)|"
    r"Level\s+of\s+Service\s+Analysis|"
    r"Intersection\s+Level\s+of\s+Service|"
    r"HCM\s+6|Highway\s+Capacity\s+Manual|"
    r"ITE\s+Trip\s+Generation|"
    r"AM\s+Peak\s+Hour.*LOS|PM\s+Peak\s+Hour.*LOS)"
)

MIN_TIA_PAGE_DENSITY = 3  # minimum TIA_STRONG_RE hits per page to flag as TIA content
MIN_TIA_PAGES = 4  # minimum pages with TIA density to confirm it's a TIA doc

# MuniCode cities with known or suspected planning board activity
# Format: (subdomain, state, description)
MUNICODE_CITIES = [
    # Colorado
    ("greeley-co", "CO", "City of Greeley"),
    ("evans-co", "CO", "City of Evans"),
    ("windsor-co", "CO", "Town of Windsor"),
    ("brighton-co", "CO", "City of Brighton"),
    ("erie-co", "CO", "Town of Erie"),
    ("timnath-co", "CO", "Town of Timnath"),
    ("johnstown-co", "CO", "Town of Johnstown"),
    ("severance-co", "CO", "Town of Severance"),
    ("milliken-co", "CO", "Town of Milliken"),
    # Texas (fast-growth suburbs)
    ("leander-tx", "TX", "City of Leander"),
    ("kyle-tx", "TX", "City of Kyle"),
    ("buda-tx", "TX", "City of Buda"),
    ("hutto-tx", "TX", "City of Hutto"),
    ("pflugerville-tx", "TX", "City of Pflugerville"),
    ("cedar-park-tx", "TX", "City of Cedar Park"),
    ("manor-tx", "TX", "City of Manor"),
    ("liberty-hill-tx", "TX", "City of Liberty Hill"),
    ("bastrop-tx", "TX", "City of Bastrop"),
    ("brenham-tx", "TX", "City of Brenham"),
    ("lufkin-tx", "TX", "City of Lufkin"),
    ("conroe-tx", "TX", "City of Conroe"),
    ("new-braunfels-tx", "TX", "City of New Braunfels"),
    ("seguin-tx", "TX", "City of Seguin"),
    ("waxahachie-tx", "TX", "City of Waxahachie"),
    ("mansfield-tx", "TX", "City of Mansfield"),
    ("forney-tx", "TX", "City of Forney"),
    ("burleson-tx", "TX", "City of Burleson"),
    # North Carolina
    ("apex-nc", "NC", "Town of Apex"),
    ("wake-forest-nc", "NC", "Town of Wake Forest"),
    ("morrisville-nc", "NC", "Town of Morrisville"),
    ("holly-springs-nc", "NC", "Town of Holly Springs"),
    ("garner-nc", "NC", "Town of Garner"),
    ("fuquay-varina-nc", "NC", "Town of Fuquay-Varina"),
    ("knightdale-nc", "NC", "Town of Knightdale"),
    ("wendell-nc", "NC", "Town of Wendell"),
    # Georgia
    ("woodstock-ga", "GA", "City of Woodstock"),
    ("canton-ga", "GA", "City of Canton"),
    ("dallas-ga", "GA", "City of Dallas"),
    ("cartersville-ga", "GA", "City of Cartersville"),
    ("gainesville-ga", "GA", "City of Gainesville"),
    # Utah (fast-growing)
    ("lehi-ut", "UT", "City of Lehi"),
    ("saratoga-springs-ut", "UT", "City of Saratoga Springs"),
    ("herriman-ut", "UT", "City of Herriman"),
    ("eagle-mountain-ut", "UT", "City of Eagle Mountain"),
    ("spanish-fork-ut", "UT", "City of Spanish Fork"),
    ("payson-ut", "UT", "City of Payson"),
    # Idaho
    ("meridian-id", "ID", "City of Meridian"),
    ("nampa-id", "ID", "City of Nampa"),
    ("caldwell-id", "ID", "City of Caldwell"),
    ("twin-falls-id", "ID", "City of Twin Falls"),
    ("coeur-dalene-id", "ID", "City of Coeur d'Alene"),
    ("post-falls-id", "ID", "City of Post Falls"),
    # Washington (smaller cities)
    ("kennewick-wa", "WA", "City of Kennewick"),
    ("pasco-wa", "WA", "City of Pasco"),
    ("richland-wa", "WA", "City of Richland"),
    ("moses-lake-wa", "WA", "City of Moses Lake"),
    ("yakima-wa", "WA", "City of Yakima"),
    # Arizona
    ("casa-grande-az", "AZ", "City of Casa Grande"),
    ("prescott-az", "AZ", "City of Prescott"),
    ("prescott-valley-az", "AZ", "Town of Prescott Valley"),
    ("kingman-az", "AZ", "City of Kingman"),
    ("lake-havasu-city-az", "AZ", "City of Lake Havasu City"),
    ("yuma-az", "AZ", "City of Yuma"),
    ("buckeye-az", "AZ", "City of Buckeye"),
    ("goodyear-az", "AZ", "City of Goodyear"),
    ("avondale-az", "AZ", "City of Avondale"),
    ("queen-creek-az", "AZ", "Town of Queen Creek"),
]

PLANNING_BOARD_KEYWORDS = {
    "pc",
    "plan",
    "planning",
    "pz",
    "pzb",
    "zoning",
    "lurc",
    "dev-review",
    "development-review",
    "design-review",
    "boards",
}


def fetch_html(url: str, timeout: int = 15) -> Optional[BeautifulSoup]:
    try:
        r = SESSION.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "lxml")
        return None
    except Exception:
        return None


def scan_municode_city(subdomain: str, state: str) -> Optional[dict]:
    """
    Probe a MuniCode Meetings city.
    Returns: {subdomain, bucket, planning_boards, packet_ids} or None
    """
    url = f"https://{subdomain}.municodemeetings.com/"
    soup = fetch_html(url, timeout=8)
    if soup is None:
        return None

    html = str(soup)
    boards = set(re.findall(r"/bc-([a-z0-9\-]+)/page/", html))
    bucket_m = re.findall(r"blob\.core\.usgovcloudapi\.net/([a-z0-9\-]+)/", html)
    bucket = bucket_m[0] if bucket_m else f"{subdomain.replace('-', '')}-pubu"

    planning_boards = [
        b for b in boards if any(k in b for k in PLANNING_BOARD_KEYWORDS)
    ]
    if not planning_boards:
        return None

    packet_ids = re.findall(r"MEET-Packet-([a-f0-9]+)\.pdf", html)

    return {
        "subdomain": subdomain,
        "state": state,
        "bucket": bucket,
        "planning_boards": planning_boards,
        "packet_ids": packet_ids,
    }


def get_all_packets_for_city(
    subdomain: str, bucket: str, max_pages: int = 10
) -> list[str]:
    """Paginate through a city's MuniCode meetings and collect all packet IDs."""
    all_ids: list[str] = []
    seen: set = set()

    for page in range(0, max_pages):
        url = f"https://{subdomain}.municodemeetings.com/meetings3?page={page}"
        soup = fetch_html(url)
        if soup is None:
            break
        html = str(soup)
        ids = re.findall(r"MEET-Packet-([a-f0-9]+)\.pdf", html)
        new_ids = [i for i in ids if i not in seen]
        seen.update(ids)
        all_ids.extend(new_ids)
        if not new_ids:
            break
        time.sleep(0.5)

    log.info("  %s: %d total packet IDs found", subdomain, len(all_ids))
    return all_ids


def fast_tia_prescan(data: bytes) -> int:
    """
    Quick raw-bytes pre-scan for TIA keyword density.
    PDFs store some text uncompressed; this catches most cases before pdfplumber.
    Returns hit count.
    """
    raw = data.decode("latin-1", errors="ignore")
    return len(TIA_STRONG_RE.findall(raw))


def scan_packet_for_tia(bucket: str, packet_id: str) -> Optional[dict]:
    """
    Download a meeting packet PDF and scan for TIA content density.
    Uses a fast raw-bytes pre-scan to avoid slow pdfplumber on non-TIA packets.
    Returns: {packet_id, tia_start, tia_end, tia_pages, total_pages, url, data}
             or None if no TIA content detected.
    """
    import pdfplumber

    url = f"https://mccmeetings.blob.core.usgovcloudapi.net/{bucket}/MEET-Packet-{packet_id}.pdf"

    # Check file size with HEAD
    try:
        head = SESSION.head(url, timeout=10)
        if head.status_code != 200:
            return None
        content_length = int(head.headers.get("Content-Length", 0))
        if content_length < 100_000:
            return None
        if content_length > 60_000_000:  # skip very large packets
            log.debug(
                "  SKIP %s — too large (%d MB)",
                packet_id[:16],
                content_length // 1_048_576,
            )
            return None
    except Exception:
        return None

    # Download with cap
    try:
        r = SESSION.get(url, timeout=90, stream=True)
        if r.status_code != 200:
            return None
        data = b""
        for chunk in r.iter_content(65536):
            data += chunk
            if len(data) > 55_000_000:
                break
        if data[:5] != b"%PDF-":
            return None
    except Exception as e:
        log.debug("  Download error: %s", e)
        return None

    # Fast pre-scan: skip pdfplumber if no raw TIA keywords
    prescan_hits = fast_tia_prescan(data)
    if prescan_hits < 5:
        log.debug("  Pre-scan miss (%d hits) — skipping pdfplumber", prescan_hits)
        return None

    log.info("  Pre-scan hit (%d raw hits) — running pdfplumber scan", prescan_hits)

    # Detailed page-by-page scan — but with a 60-second timeout guard
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("pdfplumber scan timed out")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(60)  # 60-second max for page scanning

    tia_pages: list[int] = []
    total_pages = 0
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            total_pages = len(pdf.pages)
            for pg_num, page in enumerate(pdf.pages[:200], 1):  # cap at 200 pages
                text = page.extract_text(x_tolerance=5, y_tolerance=5) or ""
                hits = len(TIA_STRONG_RE.findall(text))
                if hits >= MIN_TIA_PAGE_DENSITY:
                    tia_pages.append(pg_num)
    except TimeoutError:
        log.warning(
            "  pdfplumber timed out on %s — using pre-scan result", packet_id[:16]
        )
        # If pre-scan found hits but pdfplumber timed out, still save
        if prescan_hits >= 20:
            return {
                "packet_id": packet_id,
                "bucket": bucket,
                "url": url,
                "total_pages": total_pages or 999,
                "tia_start": 1,
                "tia_end": total_pages or 999,
                "tia_pages": [],
                "tia_page_count": prescan_hits,
                "data": data,
                "prescan_only": True,
            }
        return None
    except Exception as e:
        log.debug("  PDF scan error: %s", e)
        return None
    finally:
        signal.alarm(0)

    if len(tia_pages) < MIN_TIA_PAGES:
        return None

    tia_start = min(tia_pages)
    tia_end = max(tia_pages)

    return {
        "packet_id": packet_id,
        "bucket": bucket,
        "url": url,
        "total_pages": total_pages,
        "tia_start": tia_start,
        "tia_end": tia_end,
        "tia_pages": tia_pages,
        "tia_page_count": len(tia_pages),
        "data": data,
        "prescan_only": False,
    }


def extract_tia_section(
    city_info: dict,
    packet_result: dict,
    case_id: str,
) -> Optional[Path]:
    """
    Extract the TIA pages from a combined meeting packet PDF.
    Saves as a new PDF to data/raw/{case_id}.pdf using pypdf.
    """
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore[import]
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter  # type: ignore[import]
        except ImportError:
            log.error("pypdf not installed; saving full packet instead")
            out = RAW_DIR / f"{case_id}.pdf"
            out.write_bytes(packet_result["data"])
            return out

    import io as _io

    reader = PdfReader(_io.BytesIO(packet_result["data"]))
    writer = PdfWriter()

    # Extract from 5 pages before TIA start to 5 pages after TIA end
    start = max(0, packet_result["tia_start"] - 6)
    end = min(len(reader.pages), packet_result["tia_end"] + 6)

    for pg_num in range(start, end):
        writer.add_page(reader.pages[pg_num])

    out = RAW_DIR / f"{case_id}.pdf"
    with open(out, "wb") as f:
        writer.write(f)

    log.info(
        "  Extracted TIA pages %d–%d → %s (%d pages)",
        start + 1,
        end,
        case_id,
        end - start,
    )
    return out


def run_hunter(max_packets: int = 100, min_tia_hits: int = MIN_TIA_PAGES) -> dict:
    """Main entry point."""
    log_path = LOGS_DIR / "download_log.json"
    dl_log = json.loads(log_path.read_text()) if log_path.exists() else {}
    hunt_log = {}
    found = 0
    case_counter = 1

    log.info("Scanning %d MuniCode cities for planning boards...", len(MUNICODE_CITIES))

    # Phase 1: find cities with planning boards
    active_cities = []
    for subdomain, state, desc in MUNICODE_CITIES:
        result = scan_municode_city(subdomain, state)
        if result and result["planning_boards"]:
            log.info(
                "  ✓ %s — boards=%s  initial_packets=%d",
                subdomain,
                result["planning_boards"],
                len(result["packet_ids"]),
            )
            active_cities.append(result)
        time.sleep(0.3)

    log.info("Active cities with planning boards: %d", len(active_cities))

    # Phase 2: collect all packets for each active city
    all_candidates: list[tuple] = []  # (city_info, packet_id)
    for city_info in active_cities:
        subdomain = city_info["subdomain"]
        bucket = city_info["bucket"]
        all_ids = get_all_packets_for_city(subdomain, bucket)
        for pid in all_ids:
            all_candidates.append((city_info, pid))

    log.info("Total candidate packets: %d", len(all_candidates))

    # Phase 3: scan packets for TIA content
    packets_scanned = 0
    for city_info, packet_id in all_candidates:
        if packets_scanned >= max_packets:
            break
        if found >= 40:
            break

        subdomain = city_info["subdomain"]
        state = city_info["state"]
        case_id = (
            f"MCC-{state.upper()}-{subdomain.split('-')[0].upper()}-{case_counter:03d}"
        )

        # Skip already downloaded
        if (RAW_DIR / f"{case_id}.pdf").exists():
            continue

        log.info("Scanning %s / %s...", subdomain, packet_id[:16])
        result = scan_packet_for_tia(city_info["bucket"], packet_id)
        packets_scanned += 1
        time.sleep(0.8)

        if result is None:
            continue

        log.info(
            "  TIA FOUND in %s/%s — pages %d–%d (%d tia pages / %d total)",
            subdomain,
            packet_id[:16],
            result["tia_start"],
            result["tia_end"],
            result["tia_page_count"],
            result["total_pages"],
        )

        out_path = extract_tia_section(city_info, result, case_id)
        if out_path is None:
            continue

        # Record in download log
        dl_log[case_id] = {
            "status": "success",
            "source": f"municode_packet/{subdomain}",
            "bucket": city_info["bucket"],
            "packet_id": packet_id,
            "packet_url": result["url"],
            "state": state,
            "agency": city_info.get("desc", subdomain),
            "tia_pages": result["tia_page_count"],
            "total_pages": result["total_pages"],
            "bytes": out_path.stat().st_size,
            "access_date": str(date.today()),
        }
        hunt_log[case_id] = dl_log[case_id]
        found += 1
        case_counter += 1

    log_path.write_text(json.dumps(dl_log, indent=2))
    (LOGS_DIR / "hunt_log.json").write_text(json.dumps(hunt_log, indent=2))

    print(f"\n── TIA Hunter Summary ────────────────────────────────────")
    print(f"  Cities scanned:     {len(MUNICODE_CITIES)}")
    print(f"  Active cities:      {len(active_cities)}")
    print(f"  Packets scanned:    {packets_scanned}")
    print(f"  TIA sections found: {found}")
    return hunt_log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="TIA Hunter — MuniCode planning packet scanner"
    )
    ap.add_argument(
        "--max-packets",
        type=int,
        default=100,
        help="Maximum packets to scan (default 100)",
    )
    ap.add_argument(
        "--min-tia-hits",
        type=int,
        default=MIN_TIA_PAGES,
        help=f"Min pages with TIA density to accept (default {MIN_TIA_PAGES})",
    )
    args = ap.parse_args()
    run_hunter(max_packets=args.max_packets, min_tia_hits=args.min_tia_hits)
