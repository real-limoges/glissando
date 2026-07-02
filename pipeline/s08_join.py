"""s08: join climate covariates onto fires by (division, alarm year, month).

The covariate month is the fire's alarm month. Fires without an alarm_date or
a division keep null covariates (never dropped). Joins are left joins; the
stage logs coverage so the QC report can surface it.
"""

from __future__ import annotations

import sys

import geopandas as gpd
import pandas as pd

from pipeline import config, util
from pipeline.s05_divisions import OUT_PATH_NAME as S05_NAME
from pipeline.s06_climdiv import OUT_PATH_NAME as S06_NAME
from pipeline.s07_gsom import OUT_PATH_NAME as S07_NAME

STAGE = "s08_join"
OUT_PATH_NAME = "s08_enriched.parquet"

CLIMDIV_COLS = ["pdsi", "tavg_degf", "precip_in"]
GSOM_COLS = ["awnd_ms", "awnd_n_stations"]


def main() -> int:
    fires = gpd.read_parquet(config.INTERIM_DIR / S05_NAME)
    climdiv = pd.read_parquet(config.INTERIM_DIR / S06_NAME)
    gsom = pd.read_parquet(config.INTERIM_DIR / S07_NAME)

    fires["alarm_year"] = fires["alarm_date"].dt.year.astype("Int64")
    fires["alarm_month"] = fires["alarm_date"].dt.month.astype("Int64")

    merged = fires.merge(
        climdiv.rename(columns={"year": "alarm_year", "month": "alarm_month"}),
        on=["division", "alarm_year", "alarm_month"], how="left",
    ).merge(
        gsom.rename(columns={"year": "alarm_year", "month": "alarm_month"}),
        on=["division", "alarm_year", "alarm_month"], how="left",
    )
    if len(merged) != len(fires):
        raise ValueError(
            f"join fanned out ({len(fires)} -> {len(merged)} rows): "
            "climate tables are not unique per division-month"
        )
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=fires.crs)

    joinable = merged["division"].notna() & merged["alarm_date"].notna()
    for col in CLIMDIV_COLS + GSOM_COLS:
        cov = merged.loc[joinable, col].notna().mean() if joinable.any() else float("nan")
        util.log(STAGE, f"coverage among joinable fires: {col} = {cov:.1%}")
    # Zero coverage with joinable fires means the join keys are broken (wrong
    # state code, format drift, ...) — that must fail, not ship an all-null
    # artifact with exit code 0.
    if joinable.any():
        for cols, source in ((CLIMDIV_COLS, "nClimDiv"), (GSOM_COLS, "GSOM")):
            if all(merged.loc[joinable, c].isna().all() for c in cols):
                raise ValueError(
                    f"no joinable fire received any {source} covariate — "
                    "division/year/month join keys look broken"
                )
    util.log(STAGE, f"{len(merged)} rows; joinable={int(joinable.sum())}, "
                    f"unjoinable (null division or alarm_date)={int((~joinable).sum())}")
    util.write_geoparquet(merged, config.INTERIM_DIR / OUT_PATH_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
