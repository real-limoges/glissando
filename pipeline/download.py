"""Download all raw inputs and record checksums in data/raw/MANIFEST.json.

Raw files are the pinned snapshot the rest of the pipeline reproduces from:
NCEI overwrites nClimDiv files monthly and the GSOM API reflects live data,
so re-downloading later is NOT guaranteed to be byte-identical — the manifest
checksums define the dataset. Re-runs skip files that already exist and match
the manifest.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from pipeline import config, util

STAGE = "download"


def _fetch(url: str, dest: Path, params: dict | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    util.log(STAGE, f"GET {url} -> {dest}")
    with requests.get(url, params=params, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def _download(name: str, url: str, dest: Path, params: dict | None = None) -> None:
    manifest = util.load_manifest()
    entry = manifest["files"].get(name)
    if dest.exists() and entry and entry["sha256"] == util.sha256_file(dest):
        util.log(STAGE, f"{name}: already present and matches manifest, skipping")
        return
    _fetch(url, dest, params=params)
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    util.record_in_manifest(name, dest, url, retrieved_at)
    util.log(STAGE, f"{name}: sha256={util.sha256_file(dest)} bytes={dest.stat().st_size}")


def main() -> int:
    errors: list[str] = []

    # 1. FRAP fire perimeters (pinned release).
    if config.FRAP_GDB_URL is None:
        errors.append(
            "config.FRAP_GDB_URL is unresolved (TBD). Resolve the fire25_1 "
            "gdb zip URL on frap.fire.ca.gov and hardcode it in "
            "pipeline/config.py. As of the last session the egress policy "
            "denied frap.fire.ca.gov / www.fire.ca.gov / gis.data.cnra.ca.gov "
            "/ *.arcgis.com with 403 — see NOTES.md."
        )
    else:
        _download("frap_gdb", config.FRAP_GDB_URL, Path(config.FRAP_RAW_PATH))

    # 2. nClimDiv element files (pinned versioned filenames).
    unpinned = [k for k, v in config.CLIMDIV_PINNED_FILES.items() if v is None]
    if unpinned:
        errors.append(
            f"config.CLIMDIV_PINNED_FILES has unpinned entries {unpinned} (TBD). "
            f"List {config.CLIMDIV_BASE_URL} (procdate.txt gives the current "
            "suffix) and pin the exact versioned filenames."
        )
    else:
        for prefix, filename in config.CLIMDIV_PINNED_FILES.items():
            _download(prefix, config.CLIMDIV_BASE_URL + filename,
                      Path(config.CLIMDIV_RAW_DIR) / filename)

    # 3. Climate-division boundary polygons.
    _download_safely("divisions_shp", config.DIVISIONS_URL,
                     Path(config.DIVISIONS_RAW_PATH), errors)

    # 4. GSOM monthly wind for a California bounding box.
    _download_safely("gsom_awnd", config.GSOM_API_URL, Path(config.GSOM_RAW_PATH),
                     errors, params=config.GSOM_PARAMS)

    if errors:
        for e in errors:
            util.log(STAGE, f"ERROR: {e}")
        return 1
    util.log(STAGE, f"manifest written: {config.MANIFEST_PATH}")
    return 0


def _download_safely(name: str, url: str, dest: Path, errors: list[str],
                     params: dict | None = None) -> None:
    try:
        _download(name, url, dest, params=params)
    except requests.RequestException as exc:
        errors.append(f"{name}: download failed: {exc}")


if __name__ == "__main__":
    sys.exit(main())
