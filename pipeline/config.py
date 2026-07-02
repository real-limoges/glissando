"""Central configuration for the fire-perimeter data pipeline.

Every tunable, pinned URL, and path lives here so stages stay declarative.
Values marked PROVISIONAL were chosen before the real data was inspectable
(network policy blocked the data hosts); they must be re-verified against the
actual fire25_1 gdb / NOAA files and the decision recorded in NOTES.md.
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

# TBD: the exact download URL for the fire25_1 file-geodatabase zip could not
# be resolved yet — the session network policy denies CONNECT to
# frap.fire.ca.gov / www.fire.ca.gov / gis.data.cnra.ca.gov / *.arcgis.com
# (403 from the egress gateway, see NOTES.md). download.py fails loudly while
# this is None. Expected shape:
#   https://frap.fire.ca.gov/media/<hash>/fire25_1.gdb.zip
FRAP_GDB_URL: str | None = None

# Local filename for the raw download; PIPELINE_FRAP_RAW lets the smoke test
# substitute a synthetic GeoJSON without touching the download path.
FRAP_RAW_PATH = Path(os.environ.get("PIPELINE_FRAP_RAW", str(RAW_DIR / f"{FRAP_RELEASE}.gdb.zip")))

# Layer inside the gdb holding wildfire perimeters. None = autodetect the
# unique layer whose name contains "firep" (PROVISIONAL: expected name
# "firep25_1"; the gdb also carries prescribed-burn layers like "rxburn25_1").
FRAP_GDB_LAYER: str | None = None

# ---------------------------------------------------------------------------
# Source: NOAA nClimDiv division-month climate (PDSI, temperature, precip)
# ---------------------------------------------------------------------------
CLIMDIV_BASE_URL = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"

# nClimDiv publishes one fixed-width file per element, with a version+procdate
# suffix that changes monthly and old versions are not retained. TBD: pin the
# exact filenames once the host is reachable (read procdate.txt, then set e.g.
# "climdiv-pdsidv-v1.0.0-20250605"). s06 discovers files in CLIMDIV_RAW_DIR by
# these prefixes, so smoke fixtures and real downloads parse identically.
CLIMDIV_ELEMENTS = {
    # prefix -> (output column, missing-value sentinel)  [sentinels PROVISIONAL,
    # from the climdiv README: pcpn -9.99, tavg -99.90, pdsi -99.99]
    "climdiv-pdsidv": ("pdsi", -99.99),
    "climdiv-tmpcdv": ("tavg_degf", -99.90),
    "climdiv-pcpndv": ("precip_in", -9.99),
}
CLIMDIV_PINNED_FILES: dict[str, str | None] = {
    "climdiv-pdsidv": None,  # TBD (blocked network)
    "climdiv-tmpcdv": None,  # TBD (blocked network)
    "climdiv-pcpndv": None,  # TBD (blocked network)
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
GSOM_API_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
# PROVISIONAL: boundingBox is (north, west, south, east) covering California;
# parameter support and the exact CSV column set must be verified when the
# host is reachable. GSOM AWND units are documented as meters/second
# (PROVISIONAL — verify against the GSOM documentation table).
GSOM_PARAMS = {
    "dataset": "global-summary-of-the-month",
    "dataTypes": "AWND",
    "boundingBox": "42.1,-124.6,32.4,-113.9",
    "startDate": "1980-01-01",
    "endDate": "2025-12-31",
    "format": "csv",
    "includeStationLocation": "true",
}
GSOM_RAW_PATH = Path(os.environ.get("PIPELINE_GSOM_RAW", str(RAW_DIR / "gsom_ca_awnd.csv")))

# ---------------------------------------------------------------------------
# Processing parameters
# ---------------------------------------------------------------------------
# Native FRAP CRS (NAD83 / California Albers, meters). All geometry metrics
# are computed here and the artifact geometry is stored in this CRS.
CRS_ALBERS = "EPSG:3310"
CRS_WGS84 = "EPSG:4326"

# Perimeters digitized below this vertex density are flagged coarse_geometry.
# PROVISIONAL placeholder — calibrate against the real vertex-density
# distribution (s04 writes data/processed/s04_vertex_density.csv) and record
# the chosen value + rationale in NOTES.md and PROVENANCE.md.
COARSE_VERTICES_PER_KM = 0.5

# A fire whose representative point falls outside every division polygon is
# assigned the nearest division if within this distance, else left null.
DIVISION_NEAREST_MAX_KM = 25.0

# FRAP CAUSE code table. PROVISIONAL: transcribed from FRAP metadata for
# earlier releases; re-verify against the fire25_1 metadata before release.
CAUSE_CODES = {
    1: "Lightning",
    2: "Equipment Use",
    3: "Smoking",
    4: "Campfire",
    5: "Debris",
    6: "Railroad",
    7: "Arson",
    8: "Playing with Fire",
    9: "Miscellaneous",
    10: "Vehicle",
    11: "Powerline",
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
