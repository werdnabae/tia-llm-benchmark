"""
config.py — Report catalog and pipeline configuration for TIA dataset construction.
"""

from pathlib import Path

# ── Base paths ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"
SPLIT_DIR = DATA_DIR / "split"
FINAL_DIR = DATA_DIR / "final"
LOGS_DIR = DATA_DIR / "logs"

for _d in [RAW_DIR, PARSED_DIR, CHUNKS_DIR, SPLIT_DIR, FINAL_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Downloader settings ────────────────────────────────────────────────────────
DOWNLOAD_TIMEOUT = 60  # seconds
DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}
MIN_PDF_BYTES = 20_000  # anything smaller is likely an error page

# ── Section detection ──────────────────────────────────────────────────────────
# Ordered list; first match wins for each page's heading
SECTION_PATTERNS = [
    ("executive_summary", r"(?i)^\s*(executive\s+summary|summary)\s*$"),
    (
        "project_description",
        r"(?i)^\s*(project\s+description|introduction|project\s+overview|background)\s*$",
    ),
    (
        "study_area",
        r"(?i)^\s*(study\s+area|scope\s+of\s+(study|analysis)|study\s+scope)\s*$",
    ),
    (
        "existing_conditions",
        r"(?i)^\s*(existing\s+(traffic\s+)?conditions?|existing\s+roadway|baseline\s+conditions?)\s*$",
    ),
    (
        "no_build_conditions",
        r"(?i)^\s*(no.?build|background\s+(traffic\s+)?conditions?|future\s+no.?build)\s*$",
    ),
    ("trip_generation", r"(?i)^\s*(trip\s+generation)\s*$"),
    (
        "trip_distribution",
        r"(?i)^\s*(trip\s+distribution|traffic\s+distribution\s+and\s+assignment)\s*$",
    ),
    ("traffic_assignment", r"(?i)^\s*(traffic\s+assignment)\s*$"),
    (
        "future_build",
        r"(?i)^\s*(future\s+(build\s+)?conditions?|build\s+(out\s+)?conditions?|with.?project\s+conditions?)\s*$",
    ),
    (
        "findings",
        r"(?i)^\s*(findings|results|level\s+of\s+service\s+(analysis|results|summary|evaluation)|los\s+(results|summary|analysis))\s*$",
    ),
    (
        "mitigation",
        r"(?i)^\s*(mitigation(\s+measures?)?|recommendations?|proposed\s+(improvements?|mitigation)|improvement\s+measures?|traffic\s+improvements?)\s*$",
    ),
    ("appendix", r"(?i)^\s*(appendix|appendices)\s*[A-Z]?\s*[\-–:]?"),
]

# Sections that belong to INPUT (no ground-truth leakage allowed)
INPUT_SECTIONS = {
    "executive_summary",
    "project_description",
    "study_area",
    "existing_conditions",
    "no_build_conditions",
    "trip_generation",
    "trip_distribution",
    "traffic_assignment",
    "future_build",
}

# Sections held out as GROUND TRUTH
GROUND_TRUTH_SECTIONS = {"findings", "mitigation"}

# ── Chunking settings ──────────────────────────────────────────────────────────
CHUNK_MAX_CHARS = 3000  # soft max per chunk; split at paragraph boundary
CHUNK_OVERLAP = 0  # section-aware chunking; no token overlap needed

# ── QA settings ───────────────────────────────────────────────────────────────
QA_SAMPLE_RATE = 0.20  # fraction of cases to QA manually

# ─────────────────────────────────────────────────────────────────────────────
# REPORT CATALOG
# Each entry has: case_id, title, agency, state, year, source_url,
#                 pdf_url (direct link or None), portal_case_id
# pdf_url priority:  direct link > None (portal search required)
# ─────────────────────────────────────────────────────────────────────────────
REPORTS = [
    # ── California ─────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-CA-001",
        "title": "Mission Rock Mixed-Use Development Traffic Impact Study",
        "agency": "San Francisco Planning Department",
        "state": "CA",
        "year": 2019,
        "source_url": "https://sfplanning.org/projects/mission-rock",
        "pdf_url": None,  # CEQA ENV-2016-002899; portal PDF lookup required
        "portal_case_id": "ENV-2016-002899",
    },
    {
        "case_id": "TIA-CA-002",
        "title": "Central SoMa Plan Transportation Impact Study",
        "agency": "San Francisco Planning Department",
        "state": "CA",
        "year": 2018,
        "source_url": "https://sfplanning.org/central-soma-plan",
        "pdf_url": None,  # Case 2013.0856 EIR
        "portal_case_id": "2013.0856",
    },
    {
        "case_id": "TIA-CA-003",
        "title": "Pier 70 Mixed-Use Development Traffic Study",
        "agency": "San Francisco Planning Department",
        "state": "CA",
        "year": 2017,
        "source_url": "https://sfplanning.org/projects/pier-70",
        "pdf_url": None,
        "portal_case_id": "ENV-2014-001272",
    },
    {
        "case_id": "TIA-CA-004",
        "title": "Diridon Station Area Plan Traffic Impact Study",
        "agency": "City of San Jose Planning Dept",
        "state": "CA",
        "year": 2018,
        "source_url": "https://www.sanjoseca.gov/your-government/departments-offices/planning-building-code-enforcement/planning-division/projects-of-public-interest/diridon-station-area-plan",
        "pdf_url": None,
        "portal_case_id": "GP14-014",
    },
    # ── Texas ───────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-TX-001",
        "title": "Domain Northside Phase 2 Traffic Impact Analysis",
        "agency": "City of Austin Transportation Dept",
        "state": "TX",
        "year": 2018,
        "source_url": "https://austintexas.gov/department/development-services",
        "pdf_url": None,
        "portal_case_id": "SP-2017-0387C",
    },
    {
        "case_id": "TIA-TX-002",
        "title": "East Riverside Corridor Mixed-Use Traffic Impact Analysis",
        "agency": "City of Austin Transportation Dept",
        "state": "TX",
        "year": 2021,
        "source_url": "https://austintexas.gov/department/development-services",
        "pdf_url": None,
        "portal_case_id": "SP-2020-0461D",
    },
    # ── Virginia ────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-VA-001",
        "title": "Amazon HQ2 Pentagon City Traffic Impact Analysis",
        "agency": "Arlington County Dept of Community Planning",
        "state": "VA",
        "year": 2019,
        "source_url": "https://www.arlingtonva.us/Government/Projects/Project-Details-Page/id/99",
        "pdf_url": None,
        "portal_case_id": "SP-439",
    },
    {
        "case_id": "TIA-VA-002",
        "title": "Tysons Corner Mixed-Use Redevelopment Traffic Impact Analysis",
        "agency": "Fairfax County Dept of Planning and Zoning",
        "state": "VA",
        "year": 2018,
        "source_url": "https://www.fairfaxcounty.gov/dpz/rezoning/rz2016hm021",
        "pdf_url": None,
        "portal_case_id": "RZ-2016-HM-021",
    },
    # ── Maryland ────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-MD-001",
        "title": "White Flint Sector Plan Traffic Impact Study",
        "agency": "Montgomery County Planning Dept",
        "state": "MD",
        "year": 2016,
        "source_url": "https://montgomeryplanning.org/planning/communities/white-flint",
        "pdf_url": None,
        "portal_case_id": "WF2-2016",
    },
    {
        "case_id": "TIA-MD-002",
        "title": "Shady Grove Science and Technology Sector Plan TIA",
        "agency": "Montgomery County Planning Dept",
        "state": "MD",
        "year": 2018,
        "source_url": "https://montgomeryplanning.org/planning/communities/shady-grove",
        "pdf_url": None,
        "portal_case_id": "TIS-2018-00442",
    },
    # ── North Carolina ──────────────────────────────────────────────────────────
    {
        "case_id": "TIA-NC-001",
        "title": "Brier Creek Commons Expansion Traffic Impact Analysis",
        "agency": "NCDOT / City of Raleigh",
        "state": "NC",
        "year": 2018,
        "source_url": "https://connect.ncdot.gov/projects/planning/Pages/Traffic-Impact-Analysis.aspx",
        "pdf_url": None,
        "portal_case_id": "NCDOT-TIA-2018-03",
    },
    {
        "case_id": "TIA-NC-002",
        "title": "RDU Innovation District Traffic Impact Analysis",
        "agency": "NCDOT / Town of Morrisville",
        "state": "NC",
        "year": 2020,
        "source_url": "https://connect.ncdot.gov/projects/planning/Pages/Traffic-Impact-Analysis.aspx",
        "pdf_url": None,
        "portal_case_id": "NCDOT-TIA-2020-07",
    },
    # ── Georgia ─────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-GA-001",
        "title": "The Battery Atlanta (Braves Stadium) Traffic Impact Analysis",
        "agency": "Cobb County Community Development",
        "state": "GA",
        "year": 2015,
        "source_url": "https://www.cobbcounty.org/planning",
        "pdf_url": None,
        "portal_case_id": "RZ-15-001",
    },
    {
        "case_id": "TIA-GA-002",
        "title": "Midtown Atlanta BeltLine Westside Park TIA",
        "agency": "City of Atlanta Dept of City Planning",
        "state": "GA",
        "year": 2017,
        "source_url": "https://www.atlantaga.gov/government/departments/city-planning",
        "pdf_url": None,
        "portal_case_id": "Z-17-94",
    },
    # ── Florida ─────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-FL-001",
        "title": "Midtown Miami Residential Tower Traffic Impact Study",
        "agency": "Miami-Dade Dept of Transportation and Public Works",
        "state": "FL",
        "year": 2019,
        "source_url": "https://www.miamidade.gov/global/transportation/home.page",
        "pdf_url": None,
        "portal_case_id": "2019-0234",
    },
    {
        "case_id": "TIA-FL-002",
        "title": "Midtown Tampa Mixed-Use Development Traffic Impact Analysis",
        "agency": "City of Tampa / Hillsborough County MPO",
        "state": "FL",
        "year": 2019,
        "source_url": "https://www.tampa.gov/planning",
        "pdf_url": None,
        "portal_case_id": "DRO-2019-000312",
    },
    # ── Illinois ────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-IL-001",
        "title": "Lincoln Yards Planned Development Traffic Impact Analysis",
        "agency": "City of Chicago Dept of Planning and Development",
        "state": "IL",
        "year": 2019,
        "source_url": "https://www.chicago.gov/city/en/depts/dcd/supp_info/lincoln-yards.html",
        "pdf_url": None,
        "portal_case_id": "PD-1173",
    },
    # ── Washington ──────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-WA-001",
        "title": "South Lake Union Phase 3 Development Traffic Impact Analysis",
        "agency": "City of Seattle SDOT",
        "state": "WA",
        "year": 2019,
        "source_url": "https://www.seattle.gov/transportation",
        "pdf_url": None,
        "portal_case_id": "SEP-19-0045",
    },
    {
        "case_id": "TIA-WA-002",
        "title": "Spring District Mixed-Use Phase 2 Traffic Impact Analysis",
        "agency": "City of Bellevue",
        "state": "WA",
        "year": 2020,
        "source_url": "https://www.bellevuewa.gov/city-government/departments/planning-and-development",
        "pdf_url": None,
        "portal_case_id": "LUA-20-0019",
    },
    # ── Oregon ──────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-OR-001",
        "title": "Pearl District Mixed-Use Tower Traffic Impact Analysis",
        "agency": "City of Portland Bureau of Transportation",
        "state": "OR",
        "year": 2018,
        "source_url": "https://www.portland.gov/transportation",
        "pdf_url": None,
        "portal_case_id": "LU-18-229453",
    },
    # ── Massachusetts ───────────────────────────────────────────────────────────
    {
        "case_id": "TIA-MA-001",
        "title": "Seaport Square Phase 2 Transportation Impact Study",
        "agency": "Boston Planning and Development Agency",
        "state": "MA",
        "year": 2018,
        "source_url": "https://www.bostonplans.org/projects/development-projects/seaport-square",
        "pdf_url": None,
        "portal_case_id": "Art80-2017-0234",
    },
    {
        "case_id": "TIA-MA-002",
        "title": "Assembly Row Phase 3 Traffic Impact Study",
        "agency": "City of Somerville Office of Planning and Zoning",
        "state": "MA",
        "year": 2020,
        "source_url": "https://www.somervillema.gov/departments/ospcd/planning-and-zoning",
        "pdf_url": None,
        "portal_case_id": "SP-2020-04",
    },
    # ── Colorado ────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-CO-001",
        "title": "Colfax Avenue Corridor Intensification Traffic Impact Study",
        "agency": "City and County of Denver Community Planning",
        "state": "CO",
        "year": 2019,
        "source_url": "https://www.denvergov.org/Government/Departments/Community-Planning-and-Development",
        "pdf_url": None,
        "portal_case_id": "2019-PLAN-001524",
    },
    # ── Tennessee ───────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-TN-001",
        "title": "Nashville Yards Mixed-Use Development Traffic Impact Analysis",
        "agency": "Metro Nashville Planning Dept",
        "state": "TN",
        "year": 2018,
        "source_url": "https://www.nashville.gov/departments/planning",
        "pdf_url": None,
        "portal_case_id": "2018-041Z",
    },
    # ── Arizona ─────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-AZ-001",
        "title": "Tempe Town Lake Mixed-Use Development Traffic Impact Study",
        "agency": "City of Tempe Transportation Division",
        "state": "AZ",
        "year": 2019,
        "source_url": "https://www.tempe.gov/government/transportation",
        "pdf_url": None,
        "portal_case_id": "TIA-2019-TTL-01",
    },
    # ── Minnesota ───────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-MN-001",
        "title": "Mall of America Expansion Phase 2 Traffic Impact Analysis",
        "agency": "City of Bloomington Engineering Dept",
        "state": "MN",
        "year": 2019,
        "source_url": "https://www.bloomingtonmn.gov/transportation",
        "pdf_url": None,
        "portal_case_id": "MOA-EIS-2019",
    },
    # ── Ohio ────────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-OH-001",
        "title": "Short North Infill Development Traffic Impact Analysis",
        "agency": "City of Columbus Development",
        "state": "OH",
        "year": 2020,
        "source_url": "https://www.columbus.gov/development/",
        "pdf_url": None,
        "portal_case_id": "ZO-2020-00456",
    },
    # ── Nevada ──────────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-NV-001",
        "title": "Las Vegas Convention Center Expansion Traffic Impact Analysis",
        "agency": "Clark County Public Works",
        "state": "NV",
        "year": 2018,
        "source_url": "https://www.clarkcountynv.gov/government/departments/public_works",
        "pdf_url": None,
        "portal_case_id": "TIA-LVCC-2018",
    },
    # ── New Jersey ──────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-NJ-001",
        "title": "Hudson Yards North Bergen Development Traffic Impact Analysis",
        "agency": "NJ Dept of Transportation / Bergen County Planning",
        "state": "NJ",
        "year": 2020,
        "source_url": "https://www.state.nj.us/transportation/",
        "pdf_url": None,
        "portal_case_id": "NJDOT-TIA-2020-HY",
    },
    # ── Pennsylvania ────────────────────────────────────────────────────────────
    {
        "case_id": "TIA-PA-001",
        "title": "Philadelphia Navy Yard Mixed-Use Phase 3 TIA",
        "agency": "Philadelphia City Planning Commission",
        "state": "PA",
        "year": 2020,
        "source_url": "https://phdcnavyyard.org/",
        "pdf_url": None,
        "portal_case_id": "PCPC-TIA-2020-NY3",
    },
]
