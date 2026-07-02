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
  and every optional column force-cast (strings to pandas `string`, code
  columns to `Int64`, even present-but-all-null ones) so the output schema is
  source-stable. Invalid geometries repaired with `make_valid` (keeping only
  the polygonal part); null/empty geometries and exact duplicate rows
  dropped. VERIFIED 2026-07-02 against the real data: COLMAP matches the
  fire25_1 column set exactly; `collection_method`/`objective_code` are int16
  domain codes; extras `GlobalID`/`SHAPE_Length`/`SHAPE_Area` drop as
  unrecognized; 408 invalid geometries repaired, zero null-geometry or
  exact-duplicate rows in the real source.
- **D5 — Deduplication (s03).** Near-duplicates are rows sharing
  (year, normalized fire_name, alarm_date), with non-empty name and non-null
  date; keep the largest `gis_acres`, ties broken by geometry area then lowest
  `src_row` (deterministic). Rows missing the key fields are never grouped.
- **D6 — CAUSE codes.** `config.CAUSE_CODES` (1–19) VERIFIED 2026-07-02
  against the official "Wildland Fire Perimeter Metadata" PDF (AGOL item
  a31aa1efe1d6466f8530b501c30ab00a); two corrections from the provisional
  table: 8 = "Playing with fire", 11 = "Electrical Power". Codes outside the
  table keep the code and get null `cause_desc` plus a warning (none occur in
  fire25_1). C_METHOD and OBJECTIVE domains documented in SCHEMA.md.
- **D7 — Division assignment (s05).** Fire → division by representative point
  within CA division polygons (nClimDiv state code 4); misses fall back to
  nearest division within 25 km (`division_assigned_nearest=True`), else null
  division. Same helper reused for GSOM stations.
- **D8 — nClimDiv parsing (s06).** Fixed-width: 10-char ID (state 2, division
  2, element 2, year 4) + 12 monthly values; units left native (tavg °F,
  precip inches, PDSI unitless); element-specific missing sentinels
  (pdsi −99.99, tavg −99.90, pcpn −9.99). VERIFIED 2026-07-02 against the
  real files: layout, CA state code 04, divisions 01–07, element codes
  (05/02/01, now validated per record), and sentinels all confirmed.
- **D9 — GSOM wind (s07).** AWND per station-month, 1980-01-01–2025-12-31;
  stations mapped to divisions as in D7; covariate is the division-month
  *mean* AWND plus the contributing-station count. VERIFIED/REVISED
  2026-07-02: the data service rejects bounding boxes ("A station is
  required"), so the pull is two-step — search service enumerates the 118
  AWND stations in the CA bbox (pinned `gsom_stations.json`), then chunked
  data pulls concatenate into one pinned CSV. Columns confirmed (STATION,
  LATITUDE, LONGITUDE, ELEVATION, DATE, AWND); units confirmed m/s (metric
  service default; GSOM docs + value distribution, mean 2.9 max 8.3).
- **D10 — Join semantics (s08).** Covariate month = alarm month. Left joins
  only; fires without division or alarm_date keep null covariates and are
  never dropped.
- **D11 — Coarse-geometry flag (s04).** `vertices_per_km` (vertex count over
  perimeter length) below `COARSE_VERTICES_PER_KM` ⇒ `coarse_geometry`.
  CALIBRATED 2026-07-02 to **3.0**: the real distribution (committed as
  `data/processed/s04_vertex_density.csv`) has median 27.4, p05 5.9,
  p01 3.17; 3.0 ≈ the empirical 1st percentile and reads as "average vertex
  spacing coarser than ~333 m". Flags 197/23,205 fires (0.85%) — the tail is
  dominated by pre-1950 digitizations plus a few crude modern polygons
  (e.g. FIREBAUGH 2024: 7 vertices over a 9.6 km perimeter). Rejected
  alternatives: 0.5/1.0 flag almost nothing real; p05 (≈5.9) starts flagging
  perimeters that are visibly fine at fire scale.
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
- **D15 — FRAP source is the CNRA hub gdb export, not the official file.**
  The canonical `fire25_1.gdb.zip` on fire.ca.gov is behind a WAF that 403s
  this environment (browser UAs and server-side fetchers included; the old
  frap.fire.ca.gov pages now redirect there). The pinned source is therefore
  the hub filegdb export of AGOL item c3c10388e3b24cec8a954ba10458039d
  layer 0 ("California Fire Perimeters (all)"), verified to carry the
  fire25_1 release data (23,334 features, max YEAR_ 2025, FRAP schema).
  Consequences documented in PROVENANCE.md: source CRS is EPSG:3857
  (reprojected in s02), no prescribed-burn layer, export regenerated
  server-side (download.py polls through the hub's "Pending" state and
  validates the zip magic; the manifest sha256 is the pin). Revisit if the
  official file becomes reachable.
- **D16 — Date semantics.** fire25_1 stores ALARM_DATE/CONT_DATE as
  midnight-UTC instants (verified: every non-null value is 00:00:00 UTC).
  s02 strips the timezone to recover the recorded calendar date and warns if
  a future release ever carries non-midnight times.

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

## Session state (end of 2026-07-02 session, part 2 — network opened)

**The dataset is built.** After the user opened the egress policy, this
session completed every remaining step:

- Verified host access (note: www.fire.ca.gov / the relocated FRAP pages are
  still WAF-blocked at the application level — hence D15).
- Resolved and pinned all sources; `make download` wrote
  `data/raw/MANIFEST.json` (7 files: hub gdb export, 3 nClimDiv element
  files @ 20260604, division boundaries, GSOM station list + AWND CSV).
- Verified every provisional assumption against real data (D4, D6, D8, D9,
  D15, D16 updated above); fixed s01 layer detection, s02 code-column types
  and NaT-safe midnight check, GSOM two-step download, hub Pending polling,
  zip-magic validation.
- `make all`: 23,334 raw → 23,205 final rows (129 near-dups removed; 408
  invalid geometries repaired; 1 division-null fire = 2002 BISCUIT, rep.
  point in Oregon). Climate coverage among joinable fires: nClimDiv 100%,
  AWND 64.2% (GSOM window starts 1980).
- Calibrated `COARSE_VERTICES_PER_KM` = 3.0 (D11) — flags 197 fires (0.85%).
- Determinism verified on real data: two clean rebuilds → identical
  sha256 `128675030cad1030acd725b96c489a0b43f02fea837e14557e67617a55394cd5`.
- PROVENANCE.md and artifacts/SCHEMA.md fully populated (no TBDs left);
  QC report + vertex-density CSV committed.
- `make smoke` still green after all changes.

**Second fresh-context audit (dataset + docs).** A no-context subagent
recomputed every checkable claim: checksums, all null counts, dedup
arithmetic (23,334 → 23,205 = exactly 129), flag counts, coverage
percentages, parquet schema vs SCHEMA.md, fire_id formula on a sample, and
plausibility spot-checks (CAMP/DIXIE/AUGUST COMPLEX/PALISADES present once;
metrics reproduce from geometry byte-exactly). Everything reproduced. Four
findings, all fixed: SCHEMA.md listed a phantom `MX` state value (actual:
CA 23,188 / NV 11 / OR 4 / AZ 2); stale PROVISIONAL/TBD code comments
contradicted the verified decision log; two empty-string IDs
(irwin_id/complex_id) now normalized to null in s02 — this changed the
artifact hash to the value above; and a PROVENANCE caveat was added for the
9 fires whose published gis_acres disagrees with their geometry by >25%.
Also noted, no action: the 1958 DAM fire has alarm_date 1957-07-02
(alarm_year ≠ year; climate joins on alarm month per D10, as documented).

**Remaining / follow-ups (none blocking):**

- The artifact parquet itself is gitignored; distribute out-of-band and
  verify against the sha256 above.
- If the official fire25_1.gdb.zip ever becomes reachable, consider re-basing
  on it (D15) — expect geometry deltas from the 3857→3310 round-trip.
