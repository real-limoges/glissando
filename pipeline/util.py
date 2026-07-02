"""Shared helpers: hashing, manifest bookkeeping, deterministic parquet I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pyarrow.parquet as pq

from pipeline import config


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text())
    return {"release": config.FRAP_RELEASE, "files": {}}


def verify_against_manifest(path: Path) -> None:
    """Verify a raw input against its MANIFEST.json pin before using it.

    Matching is by filename; inputs without a manifest entry (e.g. smoke-test
    fixtures, or a manifest that doesn't exist yet) are skipped — the manifest
    only constrains files it has pinned.
    """
    if not config.MANIFEST_PATH.exists():
        return
    for name, entry in load_manifest()["files"].items():
        if entry["filename"] == path.name:
            actual = sha256_file(path)
            if actual != entry["sha256"]:
                raise ValueError(
                    f"{path} does not match its MANIFEST.json pin "
                    f"({name}: expected {entry['sha256']}, got {actual}); "
                    "raw snapshot has drifted or is corrupted — re-download "
                    "deliberately (delete the manifest entry first) instead of "
                    "building from it"
                )
            return


def record_in_manifest(name: str, path: Path, url: str, retrieved_at: str) -> None:
    manifest = load_manifest()
    manifest["files"][name] = {
        "filename": path.name,
        "url": url,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "retrieved_at": retrieved_at,
    }
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_geoparquet(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write a GeoParquet file deterministically (fixed compression, no
    embedded timestamps; callers are responsible for stable row/column order)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path, compression=config.PARQUET_COMPRESSION, index=False)


def write_parquet(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=config.PARQUET_COMPRESSION, index=False)


def parquet_num_rows(path: Path) -> int:
    return pq.read_metadata(path).num_rows


def log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)
