"""s01: extract the wildfire-perimeter layer from the raw FRAP source.

Reads the pinned fire25_1 file-geodatabase zip (or, under the smoke test, a
synthetic GeoJSON substituted via PIPELINE_FRAP_RAW) and writes it untouched
to interim GeoParquet, adding only src_row — the 0-based position in the raw
layer, which is stable within a pinned release and anchors determinism
downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pyogrio

from pipeline import config, util

STAGE = "s01_extract"
OUT_PATH_NAME = "s01_firep_raw.parquet"


def _resolve_source() -> tuple[str, str | None]:
    """Return (path readable by pyogrio, layer name or None)."""
    raw = Path(config.FRAP_RAW_PATH)
    if not raw.exists():
        raise FileNotFoundError(f"raw FRAP source missing: {raw} (run `make download`)")
    util.verify_against_manifest(raw)
    if raw.suffix != ".zip":
        return str(raw), None
    src = f"zip://{raw}"
    layers = [str(name) for name, _ in pyogrio.list_layers(src)]
    util.log(STAGE, f"layers in {raw.name}: {layers}")
    if config.FRAP_GDB_LAYER is not None:
        if config.FRAP_GDB_LAYER not in layers:
            raise ValueError(f"pinned layer {config.FRAP_GDB_LAYER!r} not in {layers}")
        return src, config.FRAP_GDB_LAYER
    if len(layers) == 1:
        return src, layers[0]
    candidates = [name for name in layers if "firep" in name.lower()]
    if len(candidates) != 1:
        raise ValueError(
            f"cannot autodetect perimeter layer, candidates={candidates}; "
            "pin config.FRAP_GDB_LAYER"
        )
    return src, candidates[0]


def main() -> int:
    src, layer = _resolve_source()
    gdf = gpd.read_file(src, layer=layer) if layer else gpd.read_file(src)
    gdf["src_row"] = range(len(gdf))
    util.log(STAGE, f"read {len(gdf)} rows, columns={list(gdf.columns)}, crs={gdf.crs}")
    util.write_geoparquet(gdf, config.INTERIM_DIR / OUT_PATH_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
