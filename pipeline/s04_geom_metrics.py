"""s04: geometry metrics and the coarse-digitization flag.

Computed in EPSG:3310 (meters): area_km2, perimeter_km, n_vertices,
vertices_per_km, coarse_geometry (vertices_per_km < COARSE_VERTICES_PER_KM),
centroid_lon/centroid_lat (WGS84). Also writes the vertex-density quantile
table (data/processed/s04_vertex_density.csv) used to calibrate the
COARSE_VERTICES_PER_KM threshold.
"""

from __future__ import annotations

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from pipeline import config, util
from pipeline.s03_dedup import OUT_PATH_NAME as S03_NAME

STAGE = "s04_geom_metrics"
OUT_PATH_NAME = "s04_geom_metrics.parquet"


def main() -> int:
    gdf = gpd.read_parquet(config.INTERIM_DIR / S03_NAME)
    expected_epsg = int(config.CRS_ALBERS.split(":")[1])
    assert gdf.crs is not None and gdf.crs.to_epsg() == expected_epsg, \
        f"expected {config.CRS_ALBERS}, got {gdf.crs}"

    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf["perimeter_km"] = gdf.geometry.length / 1e3
    gdf["n_vertices"] = shapely.get_num_coordinates(gdf.geometry.values)
    gdf["vertices_per_km"] = np.where(
        gdf["perimeter_km"] > 0, gdf["n_vertices"] / gdf["perimeter_km"], np.nan
    )
    gdf["coarse_geometry"] = gdf["vertices_per_km"] < config.COARSE_VERTICES_PER_KM

    centroids = gdf.geometry.centroid.to_crs(config.CRS_WGS84)
    gdf["centroid_lon"] = centroids.x
    gdf["centroid_lat"] = centroids.y

    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    dist = pd.DataFrame({
        "quantile": qs,
        "vertices_per_km": gdf["vertices_per_km"].quantile(qs).values,
    })
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dist.to_csv(config.PROCESSED_DIR / "s04_vertex_density.csv", index=False)

    util.log(STAGE, f"{len(gdf)} rows; vertices_per_km median="
                    f"{gdf['vertices_per_km'].median():.2f}; "
                    f"coarse={int(gdf['coarse_geometry'].sum())} "
                    f"at threshold {config.COARSE_VERTICES_PER_KM}")
    util.write_geoparquet(gdf, config.INTERIM_DIR / OUT_PATH_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
