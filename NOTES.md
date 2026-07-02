# NOTES — fire-perimeter data pipeline decision log

Running log for the reproducible pipeline that produces
`artifacts/fires_enriched.parquet`: CAL FIRE FRAP fire perimeters (pinned
release **fire25_1**) cleaned and deduplicated, joined with NOAA climate
covariates (PDSI / temperature / precipitation by nClimDiv division-month;
wind from GSOM stations). Dataset + docs only — no analysis.

Newest decisions at the bottom; the last section is always the exact state of
the most recent session.

---

## Rebuild notice (2026-07-02)

An earlier session built stages s01–s09, the Makefile targets, the smoke test,
and doc skeletons, but its work was **never pushed** — the remote branch
contained only `main`'s history, and the ephemeral container that held the
work is gone (along with the original version of this file). This session
rebuilt everything from scratch from the task description. Decisions below are
therefore re-made, not recovered; anything data-dependent is flagged
PROVISIONAL until the real files are inspectable.

## Decisions

- **D1 — Layout.** Pipeline is a plain Python package `pipeline/` at the repo
  root (this repo otherwise hosts the `glissando` Rust crate; the two share
  nothing but the Makefile). Stages are `s01`–`s09`, each runnable as
  `python -m pipeline.sXX_name`, chained by `pipeline/run_all.py`. All I/O
  under `data/{raw,interim,processed}` and `artifacts/`, redirectable via
  `PIPELINE_DATA_ROOT` (how the smoke test isolates itself).
- **D2 — Pinned inputs, checksummed snapshot.** `make download` writes
  `data/raw/MANIFEST.json` (url, sha256, bytes, retrieved_at per file). NCEI
  overwrites nClimDiv files monthly and the GSOM API is live, so the
  *snapshot*, not the URL, defines the dataset; re-downloads may differ and
  the manifest is committed to record what was used.
- **D3 — CRS.** Everything is computed and stored in EPSG:3310 (NAD83 /
  California Albers, meters — FRAP's native CRS, equal-area). Centroids are
  additionally exported as lon/lat (EPSG:4326) columns.
- **D4 — Cleaning (s02).** Raw FRAP columns map to canonical snake_case via a
  tolerant COLMAP (handles the `YEAR_` trailing-underscore style); required
  columns missing ⇒ hard failure, unrecognized raw columns dropped with a log
  line, optional canonical columns absent from the source created as all-null,
  and every optional column force-cast to pandas `string` (even
  present-but-all-null ones) so the output schema is source-stable. Invalid geometries repaired with
  `make_valid` (keeping only the polygonal part); null/empty geometries and
  exact duplicate rows dropped. PROVISIONAL: COLMAP was written from FRAP docs
  for earlier releases, not the actual fire25_1 schema.
- **D5 — Deduplication (s03).** Near-duplicates are rows sharing
  (year, normalized fire_name, alarm_date), with non-empty name and non-null
  date; keep the largest `gis_acres`, ties broken by geometry area then lowest
  `src_row` (deterministic). Rows missing the key fields are never grouped.
- **D6 — CAUSE codes.** `config.CAUSE_CODES` (1–19) transcribed from FRAP
  metadata of earlier releases; codes outside the table keep the code and get
  null `cause_desc` plus a warning. PROVISIONAL until checked against
  fire25_1 metadata.
- **D7 — Division assignment (s05).** Fire → division by representative point
  within CA division polygons (nClimDiv state code 4); misses fall back to
  nearest division within 25 km (`division_assigned_nearest=True`), else null
  division. Same helper reused for GSOM stations.
- **D8 — nClimDiv parsing (s06).** Fixed-width: 10-char ID (state 2, division
  2, element 2, year 4) + 12 monthly values; units left native (tavg °F,
  precip inches, PDSI unitless); element-specific missing sentinels
  (pdsi −99.99, tavg −99.90, pcpn −9.99). PROVISIONAL: layout and sentinels
  from the climdiv README, unverified against real files.
- **D9 — GSOM wind (s07).** AWND per station-month from an NCEI Access Data
  Service CSV over a CA bounding box, 1980-01-01–2025-12-31; stations mapped
  to divisions as in D7; covariate is the division-month *mean* AWND plus the
  contributing-station count. PROVISIONAL: exact API parameter support, CSV
  column names, and AWND units (assumed m/s) unverified.
- **D10 — Join semantics (s08).** Covariate month = alarm month. Left joins
  only; fires without division or alarm_date keep null covariates and are
  never dropped.
- **D11 — Coarse-geometry flag (s04).** `vertices_per_km` (vertex count over
  perimeter length) below `COARSE_VERTICES_PER_KM` ⇒ `coarse_geometry`.
  Threshold currently a 0.5 placeholder; s04 writes
  `data/processed/s04_vertex_density.csv` (quantiles) to calibrate against
  the real distribution — TBD, document the chosen value here and in
  PROVENANCE.md.
- **D12 — Determinism.** `fire_id` = first 12 hex of sha256 over
  `year|fire_name_norm|alarm_date|src_row` (`src_row` = position in the raw
  layer, stable within a pinned release). Final table sorted by `fire_id`,
  fixed column order, snappy parquet, no embedded timestamps. The smoke test
  runs the pipeline twice and asserts byte-identical artifacts; the same
  double-run check must be repeated on real data.
- **D13 — What gets committed.** Code, docs, fixtures, `MANIFEST.json`, the
  QC report, and `s04_vertex_density.csv` (calibration evidence); raw/interim
  data and the parquet artifact are gitignored (artifact is distributed
  out-of-band; its sha256 lives in the QC report).
- **D14 — Manifest is enforced, not advisory.** Stages verify each raw input's
  sha256 against `MANIFEST.json` before reading it (files without an entry —
  e.g. smoke fixtures — are exempt), and `download.py` refuses to silently
  re-pin: a drifted local file or drifted upstream content is a hard error
  telling the operator to delete the manifest entry deliberately.

## Fresh-context audit (2026-07-02)

A subagent with no build context audited the three commits (code, smoke
rigor, determinism, docs). Verdict: no high-severity findings; determinism
independently confirmed. All 6 medium + 11 low findings were fixed:
schema stability for all-null string columns (→ D4), zero-climate-coverage
runs now fail in s08 instead of shipping an all-null artifact, manifest
verification + no-silent-re-pin (→ D14), climdiv element-code validation and
empty-CA guard, `intersects` instead of `within` for boundary points in s05,
bare `assert`s replaced with raises, tz-aware date warning in s02, GSOM
manifest entry records the full query URL, download errors are collected
rather than aborting on the first, the vacuous December-sentinel smoke
assertion was replaced with a real fixture fire (THETA) plus new fixtures for
null-division (IOTA), s02 exact-dup drop (ZETA twin), and an out-of-state
poison-value GSOM station, `.gitignore` now actually commits the QC report
D13 promised, and SCHEMA.md's date types / null policy / code-column dtypes
were corrected.

## Session state (end of 2026-07-02 session)

**Done and verified this session:**

- Rebuilt from scratch: `pipeline/` (config, util, download, s01–s09,
  run_all), Makefile targets (`download`, `all`, `smoke`, `pipeline-clean`),
  synthetic fixtures under `tests/fixtures/pipeline/`, smoke runner
  `scripts/smoke_pipeline.py`, `requirements.txt` (pinned), this file,
  PROVENANCE.md, artifacts/SCHEMA.md.
- `make smoke` is green: 8-row artifact; assertions cover near-dedup, exact
  dedup, name normalization, bow-tie repair, null-geometry drop, coarse flag,
  nearest-division fallback, beyond-fallback null division, climate join
  values, December sentinel month → null covariates, out-of-state station
  exclusion, and string-typing of all-null optional columns; two full runs
  produce byte-identical artifacts.
- Fresh-context audit run; all findings fixed (see audit section above) and
  re-verified by the strengthened smoke test.
- `python -m pipeline.download` fails loudly (exit 1) with actionable errors
  while URLs are unresolved / network is blocked — by design.

**Blocked — network egress policy.** The session's egress gateway answers
403 (CONNECT denied) for every data host: `www.ncei.noaa.gov`,
`gis.data.cnra.ca.gov`, `data.ca.gov`, `www.fire.ca.gov`, `frap.fire.ca.gov`,
`services.arcgis.com`. Verified via curl and the agent-proxy status endpoint
at session start (2026-07-02). PyPI was reachable (proxy-exempt), so
dependencies install fine. **The network policy for this environment must
allow those hosts before the remaining work can proceed.**

**Remaining work, in order (unchanged from the original plan):**

1. Resolve and hardcode the fire25_1 gdb URL (`config.FRAP_GDB_URL`) and the
   versioned nClimDiv filenames (`config.CLIMDIV_PINNED_FILES`); then
   `make download` (writes `data/raw/MANIFEST.json`).
2. Inspect the real gdb schema / climdiv files / GSOM CSV; fix the
   PROVISIONAL assumptions (D4 COLMAP, D6 cause table, D8 layout+sentinels,
   D9 GSOM columns+units, s05 boundary-shapefile attribute names).
3. `make all`; calibrate `COARSE_VERTICES_PER_KM` from
   `data/processed/s04_vertex_density.csv` and document it (D11).
4. Rebuild twice; confirm byte-identical artifact sha256.
5. Fill every TBD in PROVENANCE.md and artifacts/SCHEMA.md from the QC report
   and the manifest.
6. Fresh-context subagent audit of dataset + docs; fix findings; update this
   file; commit each logical step; push.
