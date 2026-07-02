"""s02: normalize columns/types and repair geometry.

- maps raw FRAP column names to canonical snake_case names (tolerant of the
  trailing-underscore style, e.g. YEAR_);
- parses dates, coerces types, attaches cause_desc from the CAUSE code table;
- reprojects to EPSG:3310, repairs invalid geometries with make_valid, drops
  null/empty geometries;
- drops rows that are exact duplicates (all attributes + geometry).

The COLMAP below is PROVISIONAL — written from FRAP documentation for earlier
releases before fire25_1 was inspectable. s02 fails loudly if any *required*
canonical column is missing so schema drift is caught, and logs raw columns it
did not recognize.
"""

from __future__ import annotations

import sys

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon

from pipeline import config, util
from pipeline.s01_extract import OUT_PATH_NAME as S01_NAME

STAGE = "s02_clean"
OUT_PATH_NAME = "s02_clean.parquet"

# raw name (upper, underscores stripped from the ends) -> canonical name
COLMAP = {
    "YEAR": "year",
    "STATE": "state",
    "AGENCY": "agency",
    "UNIT_ID": "unit_id",
    "FIRE_NAME": "fire_name",
    "INC_NUM": "inc_num",
    "IRWINID": "irwin_id",
    "ALARM_DATE": "alarm_date",
    "CONT_DATE": "cont_date",
    "CAUSE": "cause_code",
    "C_METHOD": "collection_method",
    "OBJECTIVE": "objective_code",
    "GIS_ACRES": "gis_acres",
    "COMPLEX_NAME": "complex_name",
    "COMPLEX_ID": "complex_id",
    "FIRE_NUM": "fire_num",
    "COMMENTS": "comments",
    "DECADES": "decades",
}
REQUIRED = {"year", "fire_name", "alarm_date", "cause_code", "gis_acres"}
# Canonical columns carried forward; anything else raw is dropped (logged).
KEEP = [
    "year", "state", "agency", "unit_id", "fire_name", "inc_num", "irwin_id",
    "alarm_date", "cont_date", "cause_code", "cause_desc", "collection_method",
    "objective_code", "gis_acres", "complex_name", "complex_id", "src_row",
]
# Every KEEP column not otherwise typed below is coerced to pandas "string"
# (even when present-but-all-null) so the artifact schema never depends on
# which optional fields a release happens to populate. collection_method and
# objective_code are numeric codes in FRAP docs but stored as string until the
# real dtype is confirmed (PROVISIONAL).
STRING_COLS = [
    "state", "agency", "unit_id", "fire_name", "inc_num", "irwin_id",
    "collection_method", "objective_code", "complex_name", "complex_id",
]


def main() -> int:
    gdf = gpd.read_parquet(config.INTERIM_DIR / S01_NAME)
    n0 = len(gdf)

    rename, unrecognized = {}, []
    for col in gdf.columns:
        if col in ("geometry", "src_row"):
            continue
        key = col.upper().strip("_")
        if key in COLMAP:
            rename[col] = COLMAP[key]
        else:
            unrecognized.append(col)
    gdf = gdf.rename(columns=rename)
    if unrecognized:
        util.log(STAGE, f"unrecognized raw columns dropped: {unrecognized}")
    recognized_dropped = sorted(set(rename.values()) - set(KEEP))
    if recognized_dropped:
        util.log(STAGE, f"recognized columns intentionally not carried forward: {recognized_dropped}")
    missing = REQUIRED - set(gdf.columns)
    if missing:
        raise ValueError(
            f"required canonical columns missing after rename: {sorted(missing)}. "
            "The raw schema differs from COLMAP assumptions — update s02_clean.py."
        )

    # Types. Dates in FRAP gdbs are datetime already; be tolerant of strings.
    gdf["year"] = pd.to_numeric(gdf["year"], errors="coerce").astype("Int64")
    for c in ("alarm_date", "cont_date"):
        if c in gdf.columns:
            parsed = pd.to_datetime(gdf[c], errors="coerce")
            if getattr(parsed.dt, "tz", None) is not None:
                # PROVISIONAL: dropping tz keeps the stored wall-clock time; if
                # the real gdb turns out to carry tz-aware instants, decide the
                # local-date policy explicitly (NOTES.md) before trusting this.
                util.log(STAGE, f"WARNING: {c} is timezone-aware ({parsed.dt.tz}); "
                                "stripping tz without conversion")
                parsed = parsed.dt.tz_localize(None)
            gdf[c] = parsed.dt.normalize()
    gdf["cause_code"] = pd.to_numeric(gdf["cause_code"], errors="coerce").astype("Int64")
    gdf["gis_acres"] = pd.to_numeric(gdf["gis_acres"], errors="coerce").astype("float64")
    for c in STRING_COLS:
        if c in gdf.columns:
            gdf[c] = gdf[c].astype("string")
    gdf["cause_desc"] = gdf["cause_code"].map(config.CAUSE_CODES).astype("string")
    n_unknown_cause = int((gdf["cause_desc"].isna() & gdf["cause_code"].notna()).sum())
    if n_unknown_cause:
        util.log(STAGE, f"WARNING: {n_unknown_cause} rows have cause_code outside "
                        "the CAUSE_CODES table (left cause_desc null)")

    # Geometry: CRS, validity, emptiness.
    if gdf.crs is None:
        util.log(STAGE, f"WARNING: raw layer has no CRS; assuming {config.CRS_ALBERS}")
        gdf = gdf.set_crs(config.CRS_ALBERS)
    gdf = gdf.to_crs(config.CRS_ALBERS)
    null_geom = gdf.geometry.isna() | gdf.geometry.is_empty
    if null_geom.any():
        util.log(STAGE, f"dropped {int(null_geom.sum())} rows with null/empty geometry")
        gdf = gdf[~null_geom]
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        util.log(STAGE, f"repaired {int(invalid.sum())} invalid geometries with make_valid")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
    # make_valid can return GeometryCollections; keep only the polygonal part.
    gdf["geometry"] = gdf.geometry.apply(_polygonal_part)
    still_empty = gdf.geometry.isna() | gdf.geometry.is_empty
    if still_empty.any():
        util.log(STAGE, f"dropped {int(still_empty.sum())} rows non-polygonal after repair")
        gdf = gdf[~still_empty]

    # Optional canonical columns absent from the raw schema still exist in the
    # output (all-null, string-typed) so the final schema is source-stable.
    absent = [c for c in KEEP if c not in gdf.columns]
    for c in absent:
        gdf[c] = pd.Series(pd.NA, index=gdf.index, dtype="string")
    if absent:
        util.log(STAGE, f"optional columns absent in raw source, created as null: {absent}")
    gdf = gdf[KEEP + ["geometry"]]

    # Exact duplicates (identical attributes and geometry, ignoring src_row).
    attr_cols = [c for c in gdf.columns if c not in ("geometry", "src_row")]
    dup = gdf.assign(_wkb=gdf.geometry.to_wkb()).duplicated(subset=attr_cols + ["_wkb"])
    if dup.any():
        util.log(STAGE, f"dropped {int(dup.sum())} exact duplicate rows")
        gdf = gdf[~dup.values]

    gdf = gdf.sort_values("src_row").reset_index(drop=True)
    util.log(STAGE, f"{n0} -> {len(gdf)} rows")
    util.write_geoparquet(gdf, config.INTERIM_DIR / OUT_PATH_NAME)
    return 0


def _polygonal_part(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    if not polys:
        return None
    flat = []
    for g in polys:
        flat.extend(g.geoms if isinstance(g, MultiPolygon) else [g])
    return MultiPolygon(flat) if len(flat) > 1 else flat[0]


if __name__ == "__main__":
    sys.exit(main())
