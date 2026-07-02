"""s09: finalize the artifact and write the QC report.

- mints fire_id: first 12 hex chars of sha256 over
  "year|fire_name_norm|alarm_date|src_row" — deterministic for a pinned
  release, unique because src_row is;
- fixes column order (FINAL_COLUMNS), sorts rows by fire_id, writes
  artifacts/fires_enriched.parquet (GeoParquet, EPSG:3310, snappy);
- writes data/processed/s09_qc_report.md: per-stage row counts, null rates,
  ranges, coverage, and the artifact sha256.
"""

from __future__ import annotations

import hashlib
import sys

import geopandas as gpd

from pipeline import config, util
from pipeline.s01_extract import OUT_PATH_NAME as S01_NAME
from pipeline.s02_clean import OUT_PATH_NAME as S02_NAME
from pipeline.s03_dedup import OUT_PATH_NAME as S03_NAME
from pipeline.s08_join import OUT_PATH_NAME as S08_NAME

STAGE = "s09_finalize"

FINAL_COLUMNS = [
    "fire_id", "year", "state", "agency", "unit_id",
    "fire_name", "fire_name_norm", "inc_num", "irwin_id",
    "complex_name", "complex_id",
    "alarm_date", "cont_date", "alarm_year", "alarm_month",
    "cause_code", "cause_desc", "collection_method", "objective_code",
    "gis_acres", "area_km2", "perimeter_km",
    "n_vertices", "vertices_per_km", "coarse_geometry",
    "centroid_lon", "centroid_lat",
    "division", "division_assigned_nearest",
    "pdsi", "tavg_degf", "precip_in", "awnd_ms", "awnd_n_stations",
    "src_row", "geometry",
]


def mint_fire_id(row) -> str:
    key = f"{row['year']}|{row['fire_name_norm']}|{row['alarm_date']}|{row['src_row']}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def main() -> int:
    gdf = gpd.read_parquet(config.INTERIM_DIR / S08_NAME)
    gdf["fire_id"] = gdf.apply(mint_fire_id, axis=1).astype("string")
    if gdf["fire_id"].duplicated().any():
        raise ValueError("fire_id collision — key fields are not unique")

    missing = [c for c in FINAL_COLUMNS if c not in gdf.columns]
    if missing:
        raise ValueError(f"expected final columns missing: {missing}")
    extra = [c for c in gdf.columns if c not in FINAL_COLUMNS]
    if extra:
        util.log(STAGE, f"dropping non-final columns: {extra}")
    gdf = gdf[FINAL_COLUMNS].sort_values("fire_id").reset_index(drop=True)

    util.write_geoparquet(gdf, config.ARTIFACT_PATH)
    artifact_sha = util.sha256_file(config.ARTIFACT_PATH)
    util.log(STAGE, f"artifact: {config.ARTIFACT_PATH} rows={len(gdf)} sha256={artifact_sha}")

    _write_qc_report(gdf, artifact_sha)
    util.log(STAGE, f"QC report: {config.QC_REPORT_PATH}")
    return 0


def _write_qc_report(gdf: gpd.GeoDataFrame, artifact_sha: str) -> None:
    n = len(gdf)
    stage_rows = {
        "s01 raw": util.parquet_num_rows(config.INTERIM_DIR / S01_NAME),
        "s02 cleaned": util.parquet_num_rows(config.INTERIM_DIR / S02_NAME),
        "s03 deduplicated": util.parquet_num_rows(config.INTERIM_DIR / S03_NAME),
        "s09 final": n,
    }
    lines = [
        "# s09 QC report — fires_enriched.parquet",
        "",
        f"- FRAP release: `{config.FRAP_RELEASE}`",
        f"- artifact: `{config.ARTIFACT_PATH.name}`",
        f"- artifact sha256: `{artifact_sha}`",
        f"- rows: {n}",
        f"- CRS: {config.CRS_ALBERS}",
        f"- coarse-geometry threshold (vertices/km): {config.COARSE_VERTICES_PER_KM}",
        "",
        "## Row counts by stage",
        "",
        "| stage | rows |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in stage_rows.items()],
        "",
        "## Field summary",
        "",
        "| column | nulls | null % | min | max |",
        "|---|---|---|---|---|",
    ]
    for col in gdf.columns:
        if col == "geometry":
            continue
        s = gdf[col]
        nulls = int(s.isna().sum())
        try:
            mn, mx = s.dropna().min(), s.dropna().max()
        except TypeError:
            mn = mx = ""
        lines.append(f"| {col} | {nulls} | {nulls / n:.1%} | {mn} | {mx} |")

    joinable = gdf["division"].notna() & gdf["alarm_date"].notna()
    lines += [
        "",
        "## Climate coverage (among fires with a division and an alarm date)",
        "",
        f"- joinable fires: {int(joinable.sum())} / {n}",
    ]
    for col in ("pdsi", "tavg_degf", "precip_in", "awnd_ms"):
        cov = gdf.loc[joinable, col].notna().mean() if joinable.any() else float("nan")
        lines.append(f"- {col}: {cov:.1%}")
    lines += [
        "",
        "## Flags",
        "",
        f"- coarse_geometry: {int(gdf['coarse_geometry'].sum())} ({gdf['coarse_geometry'].mean():.1%})",
        f"- division_assigned_nearest: {int(gdf['division_assigned_nearest'].sum())}",
        f"- null division: {int(gdf['division'].isna().sum())}",
        f"- null alarm_date: {int(gdf['alarm_date'].isna().sum())}",
        "",
    ]
    config.QC_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.QC_REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
