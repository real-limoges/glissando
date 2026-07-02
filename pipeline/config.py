"""Central configuration for the fire-perimeter data pipeline.

Every tunable, pinned URL, and path lives here so stages stay declarative.
All data-dependent values (layer name, CAUSE table, climdiv layout, GSOM
columns/units, coarse threshold) were verified against the real fire25_1 /
NOAA files on 2026-07-02; decisions and evidence live in NOTES.md.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# All pipeline I/O lives under DATA_ROOT so the smoke test can redirect the
# whole tree to a temp directory with one env var.
DATA_ROOT = Path(os.environ.get("PIPELINE_DATA_ROOT", str(PROJECT_ROOT)))

RAW_DIR = DATA_ROOT / "data" / "raw"
CLIMDIV_RAW_DIR = Path(os.environ.get("PIPELINE_CLIMDIV_DIR", str(RAW_DIR / "climdiv")))
INTERIM_DIR = DATA_ROOT / "data" / "interim"
PROCESSED_DIR = DATA_ROOT / "data" / "processed"
ARTIFACTS_DIR = DATA_ROOT / "artifacts"
MANIFEST_PATH = RAW_DIR / "MANIFEST.json"

# ---------------------------------------------------------------------------
# Source: CAL FIRE FRAP fire perimeters, pinned release fire25_1
# ---------------------------------------------------------------------------
FRAP_RELEASE = "fire25_1"

# The official fire25_1.gdb.zip lives on www.fire.ca.gov behind a WAF that
# blocks this environment (403 for both local curl and server-side fetchers),
# so the pinned source is the CNRA hub file-geodatabase export of CAL FIRE's
# AGOL service "California Fire Perimeters (all)" (item c3c10388..., layer 0),
# which carries the fire25_1 release data (verified: 23,334 features, max
# YEAR_ = 2025, FRAP column set). See NOTES.md D15 for the trade-offs
# (service CRS is Web Mercator; the snapshot checksum in MANIFEST.json is the
# real pin because hub exports are regenerated server-side).
FRAP_GDB_URL: str | None = (
    "https://gis.data.cnra.ca.gov/api/download/v1/items/"
    "c3c10388e3b24cec8a954ba10458039d/filegdb?layers=0"
)

# Local filename for the raw download; PIPELINE_FRAP_RAW lets the smoke test
# substitute a synthetic GeoJSON without touching the download path.
FRAP_RAW_PATH = Path(os.environ.get("PIPELINE_FRAP_RAW", str(RAW_DIR / f"{FRAP_RELEASE}.gdb.zip")))

# Layer inside the gdb holding wildfire perimeters (verified 2026-07-02: the
# hub export contains exactly this single layer; the prescribed-burn layer of
# the official gdb is not part of item c3c10388/layer 0).
FRAP_GDB_LAYER: str | None = "California_Fire_Perimeters__all_"

# ---------------------------------------------------------------------------
# Source: NOAA nClimDiv division-month climate (PDSI, temperature, precip)
# ---------------------------------------------------------------------------
CLIMDIV_BASE_URL = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"

# nClimDiv publishes one fixed-width file per element, with a version+procdate
# suffix that changes monthly; old versions are not retained upstream, so the
# MANIFEST.json checksums are the real pin. s06 discovers files in
# CLIMDIV_RAW_DIR by these prefixes, so smoke fixtures and real downloads
# parse identically.
CLIMDIV_ELEMENTS = {
    # prefix -> (output column, missing-value sentinel, element code expected
    # in chars 5-6 of each record ID) — sentinels and codes verified
    # 2026-07-02 against the real files: pcpn=01 -9.99, tavg=02 -99.90,
    # pdsi=05 -99.99.
    "climdiv-pdsidv": ("pdsi", -99.99, "05"),
    "climdiv-tmpcdv": ("tavg_degf", -99.90, "02"),
    "climdiv-pcpndv": ("precip_in", -9.99, "01"),
}
# Pinned 2026-07-02 (procdate.txt = 20260604). NCEI replaces these monthly and
# does not retain old versions; MANIFEST.json checksums pin the snapshot.
CLIMDIV_PINNED_FILES: dict[str, str | None] = {
    "climdiv-pdsidv": "climdiv-pdsidv-v1.0.0-20260604",
    "climdiv-tmpcdv": "climdiv-tmpcdv-v1.0.0-20260604",
    "climdiv-pcpndv": "climdiv-pcpndv-v1.0.0-20260604",
}

# nClimDiv state code for California (not FIPS).
CLIMDIV_STATE_CODE_CA = 4

# Climate-division boundary polygons (same NCEI directory).
DIVISIONS_URL = CLIMDIV_BASE_URL + "CONUS_CLIMATE_DIVISIONS.shp.zip"
DIVISIONS_RAW_PATH = Path(
    os.environ.get("PIPELINE_DIVISIONS_RAW", str(RAW_DIR / "CONUS_CLIMATE_DIVISIONS.shp.zip"))
)

# ---------------------------------------------------------------------------
# Source: NOAA GSOM station wind (AWND) via the NCEI Access Data Service
# ---------------------------------------------------------------------------
# Two-step pull (the data service requires explicit stations; a bounding box
# alone is rejected with 400 "A station is required"):
#   1. search service enumerates GSOM stations with AWND in a CA bounding box
#      (bbox order: north,west,south,east) -> pinned gsom_stations.json;
#   2. data service fetches AWND per chunk of stations -> concatenated into
#      one pinned gsom_ca_awnd.csv (columns verified 2026-07-02: STATION,
#      LATITUDE, LONGITUDE, ELEVATION, DATE "YYYY-MM", AWND).
GSOM_SEARCH_URL = "https://www.ncei.noaa.gov/access/services/search/v1/data"
GSOM_SEARCH_PARAMS = {
    "dataset": "global-summary-of-the-month",
    "dataTypes": "AWND",
    "bbox": "42.1,-124.6,32.4,-113.9",
    "limit": "1000",
}
GSOM_API_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
GSOM_PARAMS = {
    "dataset": "global-summary-of-the-month",
    "dataTypes": "AWND",
    "startDate": "1980-01-01",
    "endDate": "2025-12-31",
    "format": "csv",
    "includeStationLocation": "true",
}
GSOM_STATIONS_CHUNK = 40
GSOM_STATIONS_RAW_PATH = RAW_DIR / "gsom_stations.json"
GSOM_RAW_PATH = Path(os.environ.get("PIPELINE_GSOM_RAW", str(RAW_DIR / "gsom_ca_awnd.csv")))

# ---------------------------------------------------------------------------
# Processing parameters
# ---------------------------------------------------------------------------
# Native FRAP CRS (NAD83 / California Albers, meters). All geometry metrics
# are computed here and the artifact geometry is stored in this CRS.
CRS_ALBERS = "EPSG:3310"
CRS_WGS84 = "EPSG:4326"

# Perimeters digitized below this vertex density are flagged coarse_geometry.
# Calibrated 2026-07-02 against the real fire25_1 distribution
# (data/processed/s04_vertex_density.csv): median 27.4 vertices/km, p05 5.9,
# p01 3.17. 3.0 ≈ the empirical 1st percentile and corresponds to an average
# vertex spacing coarser than ~333 m; flags 197/23205 fires (0.85%) — mostly
# pre-1950 digitizations and a few crude modern polygons. See NOTES.md D11.
COARSE_VERTICES_PER_KM = 3.0

# A fire whose representative point falls outside every division polygon is
# assigned the nearest division if within this distance, else left null.
DIVISION_NEAREST_MAX_KM = 25.0

# FRAP CAUSE code table, verified 2026-07-02 against the official "Wildland
# Fire Perimeter Metadata" PDF (AGOL item a31aa1efe1d6466f8530b501c30ab00a,
# CAUSE domain, display values verbatim).
CAUSE_CODES = {
    1: "Lightning",
    2: "Equipment Use",
    3: "Smoking",
    4: "Campfire",
    5: "Debris",
    6: "Railroad",
    7: "Arson",
    8: "Playing with fire",
    9: "Miscellaneous",
    10: "Vehicle",
    11: "Electrical Power",
    12: "Firefighter Training",
    13: "Non-Firefighter Training",
    14: "Unknown/Unidentified",
    15: "Structure",
    16: "Aircraft",
    17: "Volcanic",
    18: "Escaped Prescribed Burn",
    19: "Illegal Alien Campfire",
}

# Deterministic parquet writing.
PARQUET_COMPRESSION = "snappy"

ARTIFACT_PATH = ARTIFACTS_DIR / "fires_enriched.parquet"
QC_REPORT_PATH = PROCESSED_DIR / "s09_qc_report.md"
