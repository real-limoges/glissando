# SCHEMA — artifacts/fires_enriched.parquet

GeoParquet, geometry in **EPSG:3310** (NAD83 / California Albers, meters),
snappy compression, one row per deduplicated fire perimeter, sorted by
`fire_id`. Values marked **TBD** are filled from the first real build's QC
report (`data/processed/s09_qc_report.md`); the network policy blocked the
data hosts when this skeleton was written (see NOTES.md).

- rows: **TBD**
- year range: **TBD**
- artifact sha256: **TBD**
- FRAP release: `fire25_1`

## Identity

| column | type | description |
|---|---|---|
| `fire_id` | string | Deterministic ID: first 12 hex chars of sha256 over `year\|fire_name_norm\|alarm_date\|src_row`. Unique; stable within release fire25_1. |
| `src_row` | int64 | 0-based row position in the raw FRAP layer (provenance back-reference). |

## Fire attributes (from FRAP, cleaned)

| column | type | description |
|---|---|---|
| `year` | Int64 | Fire year as recorded by FRAP (`YEAR_`). |
| `state` | string | State code, predominantly `CA`. |
| `agency` | string | Reporting agency code. |
| `unit_id` | string | Responsibility-area unit. |
| `fire_name` | string | Fire name as recorded. |
| `fire_name_norm` | string | Uppercased, whitespace-collapsed name (dedup key component). |
| `inc_num` | string | Incident number. |
| `irwin_id` | string | IRWIN incident GUID where present. |
| `complex_name` | string | Complex name, if part of a complex. |
| `complex_id` | string | Complex ID, if part of a complex. |
| `alarm_date` | date | Ignition/alarm date (null when unrecorded). |
| `cont_date` | date | Containment date (null when unrecorded). |
| `alarm_year` | Int64 | Year of `alarm_date` (join key). |
| `alarm_month` | Int64 | Month of `alarm_date` (join key). |
| `cause_code` | Int64 | FRAP cause code (table PROVISIONAL — NOTES.md D6). |
| `cause_desc` | string | Decoded cause; null for codes outside the table. |
| `collection_method` | Int64/string | FRAP `C_METHOD` — perimeter collection method code. **TBD: verify dtype on real data.** |
| `objective_code` | Int64/string | FRAP `OBJECTIVE` code. **TBD: verify dtype on real data.** |
| `gis_acres` | float64 | GIS-computed acreage as published by FRAP. |

## Geometry & metrics (computed, EPSG:3310)

| column | type | description |
|---|---|---|
| `geometry` | (Multi)Polygon | Perimeter, `make_valid`-repaired. |
| `area_km2` | float64 | Polygon area / 10⁶. |
| `perimeter_km` | float64 | Boundary length / 10³. |
| `n_vertices` | int32 | Total coordinate count. |
| `vertices_per_km` | float64 | `n_vertices / perimeter_km`; digitization density. |
| `coarse_geometry` | bool | `vertices_per_km < COARSE_VERTICES_PER_KM` (threshold **TBD**, currently 0.5 placeholder). |
| `centroid_lon` / `centroid_lat` | float64 | Centroid in EPSG:4326. |

## Climate covariates (division-month of the alarm date; null when no division or alarm date)

| column | type | description |
|---|---|---|
| `division` | Int64 | nClimDiv California climate division (1–7). |
| `division_assigned_nearest` | bool | True when assigned via ≤25 km nearest-polygon fallback. |
| `pdsi` | float64 | Palmer Drought Severity Index (unitless). |
| `tavg_degf` | float64 | Division mean temperature, °F. |
| `precip_in` | float64 | Division precipitation, inches. |
| `awnd_ms` | float64 | Mean GSOM station wind speed in the division, m/s (units **TBD** — verify). |
| `awnd_n_stations` | Int64 | Stations contributing to `awnd_ms`. |

## Null / coverage summary

**TBD** — copy from `data/processed/s09_qc_report.md` after the first real
build (null rates per column, climate coverage among joinable fires, flag
counts).
