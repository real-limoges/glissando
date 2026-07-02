"""Smoke test: run the full pipeline (s01-s09) on synthetic fixtures, assert
the artifact's contents, and verify determinism by running twice and comparing
artifact hashes. No network access required. Exercised by `make smoke`.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "pipeline"


def run_pipeline(data_root: Path) -> Path:
    env = os.environ.copy()
    env.update({
        "PIPELINE_DATA_ROOT": str(data_root),
        "PIPELINE_FRAP_RAW": str(FIX / "firep_synthetic.geojson"),
        "PIPELINE_DIVISIONS_RAW": str(FIX / "divisions_synthetic.geojson"),
        "PIPELINE_GSOM_RAW": str(FIX / "gsom_synthetic.csv"),
        "PIPELINE_CLIMDIV_DIR": str(FIX / "climdiv"),
    })
    subprocess.run([sys.executable, "-m", "pipeline.run_all"], cwd=ROOT, env=env, check=True)
    return data_root / "artifacts" / "fires_enriched.parquet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pipeline-smoke-") as tmp:
        art1 = run_pipeline(Path(tmp) / "run1")
        art2 = run_pipeline(Path(tmp) / "run2")

        h1, h2 = sha256(art1), sha256(art2)
        assert h1 == h2, f"artifact not deterministic: {h1} != {h2}"

        gdf = gpd.read_parquet(art1)
        row = lambda name: gdf[gdf["fire_name_norm"] == name].iloc[0]

        # EPSILON dropped (null geometry), one ALPHA dropped (near-duplicate,
        # s03), one ZETA dropped (exact duplicate, s02).
        assert len(gdf) == 8, f"expected 8 rows, got {len(gdf)}"
        assert set(gdf["fire_name_norm"]) == {
            "ALPHA", "BETA", "GAMMA", "DELTA", "ZETA", "ETA", "THETA", "IOTA"
        }
        assert (gdf["fire_name_norm"] == "ZETA").sum() == 1, "s02 exact-dup drop failed"
        assert gdf["fire_id"].is_unique and gdf["fire_id"].is_monotonic_increasing

        alpha = row("ALPHA")
        assert alpha["gis_acres"] == 1000.0, "dedup kept the wrong ALPHA record"
        assert alpha["division"] == 1 and not alpha["division_assigned_nearest"]
        # climdiv fixture pattern: pdsi = div + month/100; tavg = 60 + month; pcpn = month/10
        assert math.isclose(alpha["pdsi"], 1.07), alpha["pdsi"]
        assert math.isclose(alpha["tavg_degf"], 67.0), alpha["tavg_degf"]
        assert math.isclose(alpha["precip_in"], 0.7), alpha["precip_in"]
        # GSOM fixture: division-1 stations report 3.0 and 5.0 in 2001-07;
        # this also proves the out-of-state station's poison value (100.0 in
        # 2001-07) was excluded by s07.
        assert math.isclose(alpha["awnd_ms"], 4.0), alpha["awnd_ms"]
        assert alpha["awnd_n_stations"] == 2

        beta = row("BETA")  # " beta " normalized
        assert beta["division"] == 2
        assert math.isclose(beta["pdsi"], 2.08), beta["pdsi"]
        assert math.isclose(beta["awnd_ms"], 7.5), beta["awnd_ms"]

        assert bool(row("GAMMA")["coarse_geometry"]) is True
        assert bool(alpha["coarse_geometry"]) is False

        delta = row("DELTA")  # bow-tie repaired to valid polygonal geometry
        assert delta.geometry.is_valid and delta.geometry.area > 0

        zeta = row("ZETA")  # no alarm_date -> null covariates, still present
        assert zeta["alarm_date"] is None or str(zeta["alarm_date"]) == "NaT"
        assert all(map(math.isnan, (zeta["pdsi"], zeta["tavg_degf"], zeta["precip_in"], zeta["awnd_ms"])))

        eta = row("ETA")  # offshore -> nearest-division fallback
        assert eta["division"] == 1 and bool(eta["division_assigned_nearest"]) is True

        theta = row("THETA")  # December alarm: the climdiv sentinel month
        assert theta["division"] == 1 and theta["alarm_month"] == 12
        assert all(map(math.isnan, (theta["pdsi"], theta["tavg_degf"],
                                    theta["precip_in"], theta["awnd_ms"]))), \
            "sentinel-month values must be null"

        iota = row("IOTA")  # far offshore: beyond the nearest fallback
        assert iota["division"] is None or str(iota["division"]) == "<NA>"
        assert bool(iota["division_assigned_nearest"]) is False
        assert math.isnan(iota["pdsi"]) and math.isnan(iota["awnd_ms"])

        # Source-stable schema: optional string columns stay string-typed even
        # when entirely null in the source (fixture leaves irwin_id/complex_*
        # unpopulated).
        schema = pq.read_schema(art1)
        for c in ("irwin_id", "complex_name", "complex_id", "state", "agency"):
            assert schema.field(c).type == "string", f"{c}: {schema.field(c).type}"

        qc = Path(tmp) / "run1" / "data" / "processed" / "s09_qc_report.md"
        assert qc.exists() and "Row counts by stage" in qc.read_text()

    print(f"SMOKE OK: {len(gdf)} rows, deterministic artifact sha256={h1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
