"""
mepa_scraper.py — Direct API scraper for Massachusetts Environmental Policy Act (MEPA)
                   Environmental Monitor.

The MEPA portal exposes a REST API that allows programmatic access to all EIR
project submissions and their attached documents (no authentication required —
uses a public API key embedded in the Angular frontend).

Strategy:
  1. Get all published Environmental Monitor editions from 2015–2022
  2. For each edition, get the list of EIR (Draft/Final EIR) project submissions
  3. Filter for development project types (residential, commercial, mixed-use, etc.)
  4. For each matching project, get its attachment list
  5. Find transportation / traffic study appendix PDFs
  6. Download and save to data/raw/MEPA-{EEA_NUMBER}.pdf

The resulting PDFs are Massachusetts MEPA transportation chapters — similar structure
to California EIR transportation chapters: existing conditions → trip generation →
LOS analysis → findings → mitigation measures.

Usage:
    python3 mepa_scraper.py [--years 2018 2019 2020 2021 2022] [--max-downloads 40]
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

import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_DIR, LOGS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "mepa_scraper.log"),
    ],
)
log = logging.getLogger(__name__)

# ── MEPA API constants ─────────────────────────────────────────────────────────
DATA_API = "https://t7i6mic1h4.execute-api.us-east-1.amazonaws.com/PROD/V1.0.0"
API_KEY = "ZyygCR4t0y8gKbqSbbuUO6g4GrfcGRMF9QRplY4m"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "x-api-key": API_KEY,
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Origin": "https://eeaonline.eea.state.ma.us",
        "Referer": "https://eeaonline.eea.state.ma.us/",
    }
)


_last_warmup = 0.0


def warm_session(force: bool = False):
    """
    Visit the MEPA portal to establish a server-side IP session.
    The MEPA API download endpoint (api/Attachment/{fileServiceId}) requires
    an active session. The session expires after ~15 minutes of inactivity,
    so we re-warm every 10 minutes automatically.
    """
    global _last_warmup
    import time as _time

    now = _time.time()
    if not force and (now - _last_warmup) < 600:  # re-warm every 10 min
        return
    try:
        SESSION.get(
            "https://eeaonline.eea.state.ma.us/EEA/MEPA-eMonitor/home", timeout=20
        )
        SESSION.get(f"{DATA_API}/api/ProjectType", timeout=10)
        _last_warmup = _time.time()
        log.info("MEPA session warmed up.")
    except Exception as e:
        log.warning("Session warm-up failed (proceeding anyway): %s", e)


# Development project name patterns — must match to keep
DEV_RE = re.compile(
    r"(?i)(residential|commercial|mixed.use|hotel|office|retail|apartment|"
    r"condominium|development|housing|school|university|medical|hospital|"
    r"campus|laboratory|lab|warehouse|distribution|transit|TOD|"
    r"redevelopment|biotech|life.scienc)"
)

# Transportation study filename patterns
TRANSPORT_FILE_RE = re.compile(
    r"(?i)(transport|traffic|TIA|TIS|trip.gen|circulation|"
    r"level.of.service|LOS|VMT|travel.demand|vehicular)"
)

# Skip attachments with these patterns in filename
SKIP_FILE_RE = re.compile(
    r"(?i)(noise|air.qual|bio|cultural|historic|shadow|wind|fiscal|"
    r"geo|hydro|hazard|visual|aesthet|mmrp|notice|comment|response|"
    r"scoping|meeting|presentation|errata|addendum)"
)

MIN_TRANSPORT_KB = 200  # minimum file size for a real transport study
MAX_TRANSPORT_KB = 30_000  # skip very large files


def api_get(endpoint: str, params=None):
    url = f"{DATA_API}{endpoint}"
    try:
        r = SESSION.get(url, params=params or {}, timeout=15)
        if r.status_code == 200:
            return r.json()
        log.debug("API %d: %s", r.status_code, url[:80])
        return None
    except Exception as e:
        log.debug("API error: %s", e)
        return None


def get_publications(year: int) -> list[dict]:
    """Get all published Environmental Monitor editions for a given year."""
    result = api_get(
        "/api/Publishing/publication", {"year": year, "statusFilter": "Published"}
    )
    if not result:
        return []
    return result


def get_pub_items(publishing_id: str) -> list[dict]:
    """
    Get ALL project submissions in a specific edition (all subtypes).
    Returns deduplicated list of publication history items.
    """
    all_items = []
    # Include all submission types — development projects appear under any of these
    for subtype in ("ENF", "EENF", "DEIR", "FEIR", "Single EIR", "EIR", "NPC"):
        result = api_get(
            "/api/PublicationHistory",
            {
                "publishingId": publishing_id,
                "subType": subtype,
                "type": "ProjSubmitted",
            },
        )
        if result and isinstance(result, dict):
            all_items.extend(result.get("list", []))

    seen = set()
    unique = []
    for item in all_items:
        key = item.get("eeaNumber", "") or item.get("publicationHistoryId", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def get_pub_history_attachments(publication_history_id: str) -> list[dict]:
    """Get attachments for a publication history item (ENF/EENF-style)."""
    result = api_get(
        f"/api/Attachment/ListByPublicationHistoryId/{publication_history_id}"
    )
    return result if isinstance(result, list) else []


def get_submittal_attachments(submittal_id: str) -> list[dict]:
    """Get attachments from a submittal record (DEIR/FEIR-style)."""
    result = api_get(f"/api/Submittal/{submittal_id}")
    if isinstance(result, dict):
        return result.get("attachments", [])
    return []


def get_project_all_attachments(project_name: str) -> list[dict]:
    """
    Get all attachments across ALL submittals of a project found by name.
    Tries both pubHistory and submittal attachment endpoints.
    """
    all_atts = []
    # Find project via Project/search
    result = api_get(
        "/api/Project/search", {"projectName": project_name, "pageSize": 3}
    )
    if not result or not isinstance(result, dict):
        return []
    for proj in result.get("list", []):
        for sub in proj.get("submittals", []):
            sub_id = sub.get("submittalId", "")
            sub_type = sub.get("submittalType", "")
            if not sub_id:
                continue
            atts = get_submittal_attachments(sub_id)
            for att in atts:
                att["_submittalType"] = sub_type
                all_atts.append(att)
    return all_atts


def find_transport_attachment(attachments: list[dict]) -> Optional[dict]:
    """
    Among a project's attachments, find the one most likely to be a
    transportation study. Returns the best candidate or None.
    Also accepts large combined ENF documents (may contain transport analysis).
    """
    candidates = []
    large_fallbacks = []

    for att in attachments:
        fname = att.get("fileName", "")
        size_kb = att.get("size", 0) // 1024

        if SKIP_FILE_RE.search(fname):
            continue
        if size_kb < MIN_TRANSPORT_KB or size_kb > MAX_TRANSPORT_KB:
            continue

        if TRANSPORT_FILE_RE.search(fname):
            candidates.append(att)
        elif (
            size_kb >= 2000
        ):  # large document — likely combined ENF with transport chapter
            large_fallbacks.append(att)

    if candidates:
        return max(candidates, key=lambda a: a.get("size", 0))
    if large_fallbacks:
        # Fall back to largest combined document (likely contains transport chapter)
        return max(large_fallbacks, key=lambda a: a.get("size", 0))
    return None


def build_download_url(att: dict) -> Optional[str]:
    """
    Construct the PDF download URL for an attachment.
    The MEPA API exposes attachments via the fileServiceId.
    Possible URL patterns (try in order):
      1. /api/Attachment/{attachmentId}           → direct download
      2. /api/Attachment/download/{fileServiceId} → via service ID
      3. /api/Attachment/{attachmentId}/download
    """
    att_id = att.get("attachmentId", "")
    fid = att.get("fileServiceId", "")
    fname = att.get("fileName", "")

    candidates = []
    if att_id:
        candidates += [
            f"{DATA_API}/api/Attachment/{att_id}",
            f"{DATA_API}/api/Attachment/{att_id}/download",
        ]
    if fid:
        candidates += [
            f"{DATA_API}/api/Attachment/download/{fid}",
            f"{DATA_API}/api/FileStorage/{fid}",
        ]
    return candidates[0] if candidates else None


def try_download_attachment(att: dict, out_path: Path) -> Optional[int]:
    """Download an attachment PDF using the confirmed working URL pattern."""
    fid = att.get("fileServiceId", "")
    # Confirmed working pattern: /api/Attachment/{fileServiceId}
    urls_to_try = []
    if fid:
        urls_to_try.append(f"{DATA_API}/api/Attachment/{fid}")
    # Fallback: try attachmentId as well
    att_id = att.get("attachmentId", "")
    if att_id:
        urls_to_try.append(f"{DATA_API}/api/Attachment/{att_id}")

    for url in urls_to_try:
        try:
            r = SESSION.get(url, timeout=60, stream=True)
            if r.status_code != 200:
                continue
            data = b""
            for chunk in r.iter_content(65536):
                data += chunk
                if len(data) > 35_000_000:
                    break
            if data[:5] == b"%PDF-" and len(data) > 50_000:
                out_path.write_bytes(data)
                return len(data)
            # Might be JSON with a URL field
            try:
                resp_json = json.loads(data)
                if isinstance(resp_json, dict) and resp_json.get("url"):
                    dl_url = resp_json["url"]
                    r2 = SESSION.get(dl_url, timeout=90, stream=True)
                    if r2.status_code == 200:
                        pdf_data = r2.content
                        if pdf_data[:5] == b"%PDF-":
                            out_path.write_bytes(pdf_data)
                            return len(pdf_data)
            except Exception:
                pass
        except Exception:
            pass

    return None


# ── Main scraper ───────────────────────────────────────────────────────────────


def run_mepa(
    years=None,
    max_downloads: int = 40,
) -> dict:
    if years is None:
        years = list(range(2015, 2023))  # 2015–2022 (pre-VMT-shift)

    log_path = LOGS_DIR / "download_log.json"
    dl_log = json.loads(log_path.read_text()) if log_path.exists() else {}

    # Warm up the session before attempting any downloads
    warm_session()

    found = 0
    total_eir = 0
    total_dev = 0

    for year in years:
        log.info("Year %d: fetching publications...", year)
        pubs = get_publications(year)
        log.info("  %d editions found", len(pubs))

        for pub in pubs:
            pub_id = pub.get("publishingId", "")
            vol_iss = pub.get("volumeIssue", "?")
            if not pub_id:
                continue

            # Re-warm session every 10 minutes to prevent expiry mid-scan
            warm_session()

            all_items = get_pub_items(pub_id)
            total_eir += len(all_items)

            dev_items = [
                item for item in all_items if DEV_RE.search(item.get("projectName", ""))
            ]
            total_dev += len(dev_items)

            if dev_items:
                log.info(
                    "  Vol.%s: %d items, %d development projects",
                    vol_iss,
                    len(all_items),
                    len(dev_items),
                )

            for item in dev_items:
                if found >= max_downloads:
                    break

                eea_num = item.get("eeaNumber", "?")
                name = item.get("projectName", "")
                location = item.get("location", "")
                hist_id = item.get("publicationHistoryId", "")
                case_id = f"MEPA-MA-{eea_num}"

                # Skip already downloaded
                pdf_path = RAW_DIR / f"{case_id}.pdf"
                if pdf_path.exists() and pdf_path.stat().st_size > 50_000:
                    log.info("  SKIP %s (exists)", case_id)
                    found += 1
                    continue

                # Get attachments from publication history
                attachments = get_pub_history_attachments(hist_id) if hist_id else []

                # Also get attachments from project submittals
                submittal_atts = get_project_all_attachments(name)
                all_attachments = attachments + submittal_atts
                time.sleep(0.3)

                transport_att = find_transport_attachment(all_attachments)
                if not transport_att:
                    log.debug(
                        "  EEA#%s — no transport attachment found (%d total attachments)",
                        eea_num,
                        len(all_attachments),
                    )
                    continue

                fname = transport_att.get("fileName", "")
                size_kb = transport_att.get("size", 0) // 1024

                log.info("  EEA#%s — %s  [%s]", eea_num, name[:55], location)
                log.info("    Transport attachment: %s (%d KB)", fname[:60], size_kb)

                # Try to download
                nbytes = try_download_attachment(transport_att, pdf_path)
                if nbytes:
                    log.info("    ✓ Saved %s  (%.1f MB)", case_id, nbytes / 1_048_576)
                    dl_log[case_id] = {
                        "status": "success",
                        "source": "mepa",
                        "eeaNumber": eea_num,
                        "project_title": name,
                        "location": location,
                        "agency": "MEPA / Massachusetts EEA",
                        "state": "MA",
                        "year": year,
                        "attach_title": fname,
                        "bytes": nbytes,
                        "access_date": str(date.today()),
                    }
                    found += 1
                else:
                    log.warning("    ✗ Download failed for EEA#%s", eea_num)
                    dl_log[case_id] = {
                        "status": "failed",
                        "source": "mepa",
                        "eeaNumber": eea_num,
                        "project_title": name,
                        "access_date": str(date.today()),
                    }

                time.sleep(0.8)

            if found >= max_downloads:
                break

    log_path.write_text(json.dumps(dl_log, indent=2))

    print(f"\n── MEPA Scraper Summary ─────────────────────────────────")
    print(f"  Years scanned:       {years}")
    print(f"  Total EIR items:     {total_eir}")
    print(f"  Development EIR:     {total_dev}")
    print(f"  PDFs downloaded:     {found}")
    return dl_log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MEPA Transportation Study Scraper")
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2015, 2023)),
        help="Years to scan (default: 2015-2022)",
    )
    ap.add_argument("--max-downloads", type=int, default=40)
    args = ap.parse_args()
    run_mepa(years=args.years, max_downloads=args.max_downloads)
