"""s05: assign each fire to a NOAA nClimDiv climate division.

Reads the CONUS climate-division polygons, keeps California (nClimDiv state
code 4), and assigns each fire the division containing its representative
point. Fires whose point falls outside every polygon (coastal slivers, CRS
edge effects) get the nearest division if within DIVISION_NEAREST_MAX_KM,
recorded via division_assigned_nearest; otherwise division stays null.

The boundary-file attribute schema is detected tolerantly (PROVISIONAL until
verified against the real shapefile): first a CLIMDIV column (state*100+div),
else STATE_CODE/CD_2DIG style pairs, else lowercase state_code/division as
used by the smoke fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from pipeline import config, util
from pipeline.s04_geom_metrics import OUT_PATH_NAME as S04_NAME

STAGE = "s05_divisions"
OUT_PATH_NAME = "s05_divisions.parquet"


def load_ca_divisions() -> gpd.GeoDataFrame:
    """California division polygons in EPSG:3310 with a `division` Int64 column."""
    raw = Path(config.DIVISIONS_RAW_PATH)
    if not raw.exists():
        raise FileNotFoundError(f"division boundaries missing: {raw} (run `make download`)")
    util.verify_against_manifest(raw)
    src = f"zip://{raw}" if raw.suffix == ".zip" else str(raw)
    div = gpd.read_file(src)
    cols = {c.upper(): c for c in div.columns}

    if "CLIMDIV" in cols:
        code = pd.to_numeric(div[cols["CLIMDIV"]], errors="raise").astype(int)
        div["state_code"], div["division"] = code // 100, code % 100
    elif "STATE_CODE" in cols and "CD_2DIG" in cols:
        div["state_code"] = pd.to_numeric(div[cols["STATE_CODE"]], errors="raise").astype(int)
        div["division"] = pd.to_numeric(div[cols["CD_2DIG"]], errors="raise").astype(int)
    elif "state_code" in div.columns and "division" in div.columns:
        pass
    else:
        raise ValueError(
            f"cannot locate division identifiers in boundary columns {list(div.columns)}; "
            "update s05_divisions.load_ca_divisions"
        )

    ca = div[div["state_code"] == config.CLIMDIV_STATE_CODE_CA]
    if ca.empty:
        raise ValueError(f"no divisions with state_code == {config.CLIMDIV_STATE_CODE_CA}")
    ca = ca[["division", "geometry"]].copy()
    ca["division"] = ca["division"].astype("Int64")
    if ca.crs is None:
        ca = ca.set_crs(config.CRS_WGS84)
    return ca.to_crs(config.CRS_ALBERS)


def assign_division(points: gpd.GeoDataFrame, divisions: gpd.GeoDataFrame) -> pd.DataFrame:
    """Map a point GeoDataFrame (EPSG:3310) to division / division_assigned_nearest."""
    # "intersects" (not "within") so a point exactly on a shared boundary
    # still counts as inside rather than falling through to the nearest-
    # division fallback; such a point matches both polygons, so keep the
    # lowest division deterministically.
    joined = gpd.sjoin(points, divisions, how="left", predicate="intersects")
    joined = joined.sort_values("division").groupby(level=0).first()
    out = pd.DataFrame(index=points.index)
    out["division"] = joined["division"].astype("Int64")
    out["division_assigned_nearest"] = False

    unmatched = out["division"].isna()
    if unmatched.any():
        near = gpd.sjoin_nearest(
            points[unmatched], divisions, how="left",
            max_distance=config.DIVISION_NEAREST_MAX_KM * 1000.0,
            distance_col="_dist",
        )
        near = near.sort_values(["_dist", "division"]).groupby(level=0).first()
        got = near["division"].notna()
        out.loc[near.index[got], "division"] = near.loc[got, "division"].astype("Int64")
        out.loc[near.index[got], "division_assigned_nearest"] = True
    return out


def main() -> int:
    gdf = gpd.read_parquet(config.INTERIM_DIR / S04_NAME)
    divisions = load_ca_divisions()
    util.log(STAGE, f"CA divisions loaded: {sorted(divisions['division'].tolist())}")

    pts = gpd.GeoDataFrame(geometry=gdf.geometry.representative_point(), crs=gdf.crs)
    assigned = assign_division(pts, divisions)
    gdf["division"] = assigned["division"]
    gdf["division_assigned_nearest"] = assigned["division_assigned_nearest"]

    n_null = int(gdf["division"].isna().sum())
    n_near = int(gdf["division_assigned_nearest"].sum())
    util.log(STAGE, f"{len(gdf)} rows; division null={n_null}, via-nearest={n_near}")
    util.write_geoparquet(gdf, config.INTERIM_DIR / OUT_PATH_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
