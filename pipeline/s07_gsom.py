"""s07: aggregate GSOM station wind (AWND) to division-months.

Reads the GSOM CSV snapshot (station, month, lat/lon, AWND), assigns each
station to a California climate division with the same point-in-polygon +
nearest fallback used for fires (s05.assign_division), and averages AWND per
(division, year, month), keeping the contributing-station count.

PROVISIONAL assumptions to verify against the real download: CSV columns are
STATION, DATE (YYYY-MM), LATITUDE, LONGITUDE, AWND; AWND is meters/second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from pipeline import config, util
from pipeline.s05_divisions import assign_division, load_ca_divisions

STAGE = "s07_gsom"
OUT_PATH_NAME = "s07_gsom_wind.parquet"

REQUIRED = {"STATION", "DATE", "LATITUDE", "LONGITUDE", "AWND"}


def main() -> int:
    path = Path(config.GSOM_RAW_PATH)
    if not path.exists():
        raise FileNotFoundError(f"GSOM snapshot missing: {path} (run `make download`)")
    util.verify_against_manifest(path)
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"GSOM CSV missing expected columns {sorted(missing)}; "
                         f"has {list(df.columns)} — update s07_gsom.py")
    n0 = len(df)

    df["AWND"] = pd.to_numeric(df["AWND"], errors="coerce")
    df = df.dropna(subset=["AWND", "LATITUDE", "LONGITUDE"])
    date = df["DATE"].astype(str).str.extract(r"^(\d{4})-(\d{2})")
    df["year"] = pd.to_numeric(date[0], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(date[1], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "month"])
    util.log(STAGE, f"{n0} raw rows -> {len(df)} usable AWND station-months, "
                    f"{df['STATION'].nunique()} stations")

    stations = df.groupby("STATION")[["LONGITUDE", "LATITUDE"]].first()
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(stations["LONGITUDE"], stations["LATITUDE"]),
        index=stations.index, crs=config.CRS_WGS84,
    ).to_crs(config.CRS_ALBERS)
    assigned = assign_division(pts, load_ca_divisions())
    stations["division"] = assigned["division"]
    n_unassigned = int(stations["division"].isna().sum())
    if n_unassigned:
        util.log(STAGE, f"{n_unassigned} stations outside CA divisions "
                        f"(+{config.DIVISION_NEAREST_MAX_KM} km buffer) excluded")

    df = df.merge(stations["division"].dropna().astype("Int64"),
                  left_on="STATION", right_index=True, how="inner")
    agg = (
        df.groupby(["division", "year", "month"], as_index=False)
        .agg(awnd_ms=("AWND", "mean"), awnd_n_stations=("STATION", "nunique"))
        .astype({"division": "Int64", "awnd_n_stations": "Int64"})
        .sort_values(["division", "year", "month"])
        .reset_index(drop=True)
    )
    util.log(STAGE, f"{len(agg)} division-month wind rows")
    util.write_parquet(agg, config.INTERIM_DIR / OUT_PATH_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
