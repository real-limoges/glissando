"""s03: deduplicate near-duplicate fire records.

FRAP perimeters carry the same fire more than once when multiple agencies
submitted perimeters. Rule (recorded in NOTES.md): rows sharing
(year, normalized fire_name, alarm_date) — with a non-empty name and a
non-null alarm_date — are one fire; keep the record with the largest
gis_acres, tie-broken by largest geometry area then lowest src_row, so the
choice is deterministic. Rows missing name or alarm_date are never grouped.
"""

from __future__ import annotations

import sys

import geopandas as gpd
import pandas as pd

from pipeline import config, util
from pipeline.s02_clean import OUT_PATH_NAME as S02_NAME

STAGE = "s03_dedup"
OUT_PATH_NAME = "s03_dedup.parquet"


def normalize_name(s):
    return s.str.upper().str.strip().str.replace(r"\s+", " ", regex=True)


def main() -> int:
    gdf = gpd.read_parquet(config.INTERIM_DIR / S02_NAME)
    n0 = len(gdf)

    gdf["fire_name_norm"] = normalize_name(gdf["fire_name"]).fillna("")
    groupable = (gdf["fire_name_norm"] != "") & gdf["alarm_date"].notna() & gdf["year"].notna()

    g = gdf[groupable].copy()
    g["_area"] = g.geometry.area
    g = g.sort_values(
        ["year", "fire_name_norm", "alarm_date", "gis_acres", "_area", "src_row"],
        ascending=[True, True, True, False, False, True],
        na_position="last",
    )
    kept = g.drop_duplicates(subset=["year", "fire_name_norm", "alarm_date"], keep="first")
    n_dupes = len(g) - len(kept)

    out = gpd.GeoDataFrame(
        pd.concat([kept.drop(columns=["_area"]), gdf[~groupable]]),
        crs=gdf.crs,
    )
    out = out.sort_values("src_row").reset_index(drop=True)
    util.log(STAGE, f"{n0} -> {len(out)} rows ({n_dupes} near-duplicates removed, "
                    f"{int((~groupable).sum())} rows ungroupable and kept as-is)")
    util.write_geoparquet(out, config.INTERIM_DIR / OUT_PATH_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
