# PROVENANCE — fires_enriched.parquet

Where every byte of `artifacts/fires_enriched.parquet` comes from and how to
reproduce it. Fields marked **TBD** await the first real `make download` /
`make all` run — the session network policy currently denies the data hosts
(see NOTES.md).

## Reproduction

```sh
pip install -r requirements.txt
make download   # raw snapshot -> data/raw/ + MANIFEST.json checksums
make all        # stages s01-s09 -> artifacts/fires_enriched.parquet
make smoke      # synthetic end-to-end + determinism check (no network)
```

Reproducibility contract: given the same raw snapshot (verified against
`data/raw/MANIFEST.json`), `make all` produces a byte-identical artifact.
Upstream sources drift (NCEI overwrites nClimDiv files monthly; the GSOM API
is live), so the checksummed snapshot — not the URLs — defines the dataset.

- artifact sha256: **TBD**
- artifact rows: **TBD**
- built with: Python 3.11, pinned package versions in `requirements.txt`

## Source 1 — CAL FIRE FRAP fire perimeters (pinned release `fire25_1`)

- What: statewide historical wildfire perimeters maintained by CAL FIRE's
  Fire and Resource Assessment Program (FRAP); file geodatabase release
  `fire25_1`, wildfire layer (expected `firep25_1`; prescribed burns excluded).
- URL: **TBD** (`pipeline/config.py:FRAP_GDB_URL` — frap.fire.ca.gov blocked
  by network policy at build time)
- sha256: **TBD** (see `data/raw/MANIFEST.json` after download)
- retrieved: **TBD**
- License/terms: **TBD** (record the statement published with the release)
- Processing: s01 extract → s02 clean (column normalization, date parsing,
  CAUSE decoding, `make_valid` repair, exact-duplicate drop) → s03
  near-duplicate collapse (NOTES.md D5) → s04 geometry metrics.

## Source 2 — NOAA nClimDiv division-month climate

- What: monthly PDSI (`climdiv-pdsidv`), average temperature
  (`climdiv-tmpcdv`, °F), and precipitation (`climdiv-pcpndv`, inches) for
  California's 7 climate divisions (nClimDiv state code 04).
- Base URL: https://www.ncei.noaa.gov/pub/data/cirs/climdiv/
- Pinned filenames: **TBD** (versioned suffix changes monthly; old versions
  are not retained upstream — the checksums in MANIFEST.json are the pin)
- sha256 / retrieved: **TBD**
- Processing: s06 fixed-width parse, CA only, element missing-value sentinels
  → division-month table.

## Source 3 — NOAA climate-division boundaries

- What: CONUS climate-division polygons used to assign fires (representative
  point, 25 km nearest fallback) and GSOM stations to divisions.
- URL: https://www.ncei.noaa.gov/pub/data/cirs/climdiv/CONUS_CLIMATE_DIVISIONS.shp.zip
- sha256 / retrieved: **TBD**
- Processing: s05 (fires), reused by s07 (stations).

## Source 4 — NOAA GSOM station wind (AWND)

- What: Global Summary of the Month average wind speed per station, pulled
  over a California bounding box (42.1, −124.6, 32.4, −113.9),
  1980-01-01 – 2025-12-31, via the NCEI Access Data Service
  (https://www.ncei.noaa.gov/access/services/data/v1).
- Units: meters/second (**TBD — verify against GSOM documentation**)
- sha256 / retrieved: **TBD**
- Processing: s07 station→division assignment, division-month mean AWND +
  contributing-station count; s08 left-joins climate onto fires by
  (division, alarm year, alarm month).

## Known caveats

- Fires without an alarm date or an assignable division carry null climate
  covariates (counted in `data/processed/s09_qc_report.md`).
- `coarse_geometry` flags low-vertex-density perimeters; threshold
  calibration: **TBD** (NOTES.md D11).
- Deduplication is heuristic (name/date keyed); complex-fire perimeters
  reported both per-fire and as a complex may survive as separate rows.
- Climate values are division-scale; they are context, not at-fire weather.
