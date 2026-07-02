# PROVENANCE — fires_enriched.parquet

Where every byte of `artifacts/fires_enriched.parquet` comes from and how to
reproduce it.

## Reproduction

```sh
pip install -r requirements.txt
make download   # raw snapshot -> data/raw/ + MANIFEST.json checksums
make all        # stages s01-s09 -> artifacts/fires_enriched.parquet
make smoke      # synthetic end-to-end + determinism check (no network)
```

Reproducibility contract: given the same raw snapshot (each stage verifies its
inputs against `data/raw/MANIFEST.json` before reading), `make all` produces a
byte-identical artifact. Verified 2026-07-02: two clean rebuilds from the
pinned snapshot both produced the sha256 below. Upstream sources drift (NCEI
overwrites nClimDiv files monthly; hub exports and the GSOM API are live), so
the checksummed snapshot — not the URLs — defines the dataset; `download.py`
refuses to silently re-pin a drifted file.

- artifact sha256: `128675030cad1030acd725b96c489a0b43f02fea837e14557e67617a55394cd5`
- artifact rows: 23,205 (fires 1878–2025)
- built with: Python 3.11.15, pinned package versions in `requirements.txt`
- full per-file checksums: `data/raw/MANIFEST.json` (committed)
- QC detail: `data/processed/s09_qc_report.md` (committed)

## Source 1 — CAL FIRE FRAP fire perimeters (release `fire25_1`)

- What: statewide historical wildland fire perimeters maintained by CAL FIRE's
  Fire and Resource Assessment Program (FRAP); annual release fire25_1
  (published April 2026, adds 516 fires from the 2025 season). 23,334 raw
  perimeters, single layer `California_Fire_Perimeters__all_`, EPSG:3857.
- URL: `https://gis.data.cnra.ca.gov/api/download/v1/items/c3c10388e3b24cec8a954ba10458039d/filegdb?layers=0`
  — the CNRA hub file-geodatabase export of CAL FIRE's AGOL item
  `c3c10388e3b24cec8a954ba10458039d` ("California Fire Perimeters (all)",
  layer 0), which carries the fire25_1 release data. **Deviation from the
  official file:** the canonical `fire25_1.gdb.zip` on fire.ca.gov sits behind
  a WAF that blocks this build environment, so the hub export is the pinned
  source instead. Consequences: geometry arrives in Web Mercator (EPSG:3857,
  the hosted-service CRS) rather than FRAP-native EPSG:3310 and is reprojected
  in s02 (sub-meter round-trip effect), and the export omits the prescribed-
  burn layer (unused here anyway). The hub regenerates exports server-side, so
  only the manifest sha256 pins the snapshot.
- sha256: `1bc7e1ef21014f5048631631b79d9db03e298201cf0d2bf9cbc4e5ed8e0bd68a`
  (45,719,838 bytes), retrieved 2026-07-02T16:06:59Z.
- License/terms: per the official "Wildland Fire Perimeter Metadata" PDF
  (AGOL item `a31aa1efe1d6466f8530b501c30ab00a`): no restrictions on
  distribution; users must cite CAL FIRE (Department of Forestry and Fire
  Protection) as the original source and denote modifications; the State
  provides the data without warranty.
- Processing: s01 extract → s02 clean (column normalization, midnight-UTC
  date-stamp parsing, CAUSE decoding per the official domain table,
  `make_valid` repair of 408 invalid geometries, reprojection to EPSG:3310)
  → s03 near-duplicate collapse (129 rows removed; rule in NOTES.md D5)
  → s04 geometry metrics.

## Source 2 — NOAA nClimDiv division-month climate

- What: monthly PDSI (`climdiv-pdsidv`), average temperature
  (`climdiv-tmpcdv`, °F), and precipitation (`climdiv-pcpndv`, inches) for
  California's 7 climate divisions (nClimDiv state code 04), 1895–present.
- Base URL: https://www.ncei.noaa.gov/pub/data/cirs/climdiv/
- Pinned filenames (procdate 20260604; NCEI replaces these monthly and does
  not retain old versions — the manifest checksums are the real pin):
  - `climdiv-pdsidv-v1.0.0-20260604` — sha256 `db6c4e40…a3e101`
  - `climdiv-tmpcdv-v1.0.0-20260604` — sha256 `b5eaaa08…9cdc792`
  - `climdiv-pcpndv-v1.0.0-20260604` — sha256 `cd127848…8afbe5109`
  (full hashes in MANIFEST.json; all retrieved 2026-07-02)
- Processing: s06 fixed-width parse (element codes validated: pdsi=05,
  tavg=02, pcpn=01), CA only, element sentinels (−99.99 / −99.90 / −9.99)
  → 11,088 division-months, 1895–2026.

## Source 3 — NOAA climate-division boundaries

- What: CONUS climate-division polygons used to assign fires (representative
  point in EPSG:3310, `intersects`, 25 km nearest fallback) and GSOM stations
  to divisions. CA identified via the `CLIMDIV` attribute (401–407).
- URL: https://www.ncei.noaa.gov/pub/data/cirs/climdiv/CONUS_CLIMATE_DIVISIONS.shp.zip
- sha256: `edcfadfd390f41eafb8cd4e112d3ead7444a25e505f43fa29107f0c64516aeb3`,
  retrieved 2026-07-02T16:03:53Z.
- Processing: s05 (fires; 42 assigned via the nearest fallback, 1 left null —
  the 2002 BISCUIT fire, whose representative point lies in Oregon), reused
  by s07 (stations; 19 of 118 stations fall outside CA divisions + 25 km and
  are excluded).

## Source 4 — NOAA GSOM station wind (AWND)

- What: Global Summary of the Month average wind speed. Two-step pull via
  NCEI Access services: (1) search service enumerates the 118 GSOM stations
  reporting AWND in a California bounding box (42.1, −124.6, 32.4, −113.9) —
  pinned as `gsom_stations.json`; (2) data service pulls AWND for those
  stations, 1980-01-01 – 2025-12-31, in 3 chunks of ≤40 stations,
  concatenated to `gsom_ca_awnd.csv` (44,461 station-months).
- Units: meters/second — the Access Data Service default (metric); confirmed
  against the GSOM documentation ("miles per hour or meters per second
  depending on user specification") and the value distribution
  (mean 2.9 m/s, max 8.3 m/s).
- sha256: stations `246dc8c2…5ed9f300`, data `45512cf4…439ef49` (full hashes
  in MANIFEST.json; retrieved 2026-07-02).
- Processing: s07 station→division assignment, division-month mean AWND +
  contributing-station count (3,413 division-months); s08 left-joins all
  climate onto fires by (division, alarm year, alarm month).

## Known caveats

- 5,384 fires (23.2%) have no alarm date and 1 fire has no division; they
  carry null climate covariates (never dropped). Among joinable fires,
  nClimDiv coverage is 100%; AWND coverage is 64.2% because the GSOM pull
  starts at 1980 — all pre-1980 fires have null `awnd_ms`.
- `coarse_geometry` flags 197 fires (0.85%) below 3.0 vertices/km
  (calibration: NOTES.md D11; distribution: `data/processed/s04_vertex_density.csv`).
- Deduplication is heuristic (year + normalized name + alarm date); FRAP
  itself warns duplicates may remain, and complex fires reported both
  per-fire and as a complex may survive as separate rows.
- FRAP's own use limitation applies: the record is incomplete (missing and
  over-generalized perimeters, especially pre-1950); use care in statistical
  analysis.
- `gis_acres` is FRAP's published value and is not recomputed: it matches the
  geometry almost everywhere (median gis_acres / geometry-acres = 1.0000,
  p01 = p99 = 1.0), but 9 fires >10 acres disagree with their geometry by
  more than 25% (worst: DOOLITTLE 2023, published 16.6 ac vs ~2.4 ac of
  polygon) — upstream inconsistencies, kept as published. Use `area_km2` for
  geometry-derived size.
- Climate values are division-scale monthly context, not at-fire weather.
