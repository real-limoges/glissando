"""Download all raw inputs and record checksums in data/raw/MANIFEST.json.

Raw files are the pinned snapshot the rest of the pipeline reproduces from:
NCEI overwrites nClimDiv files monthly and the GSOM API reflects live data,
so re-downloading later is NOT guaranteed to be byte-identical — the manifest
checksums define the dataset. Re-runs skip files that already exist and match
the manifest.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from pipeline import config, util

STAGE = "download"

# The CNRA hub regenerates stale exports on request, answering
# {"status": "Pending"} until the file is ready.
HUB_PENDING_RETRIES = 40
HUB_PENDING_WAIT_S = 15


def _fetch(url: str, dest: Path, params: dict | None = None) -> str:
    """Download url -> dest atomically; return the full effective URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    full_url = requests.Request("GET", url, params=params).prepare().url
    util.log(STAGE, f"GET {full_url} -> {dest}")
    with requests.get(url, params=params, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    _validate_download(tmp, dest.name)
    tmp.rename(dest)
    return full_url


def _validate_download(tmp: Path, name: str) -> None:
    """Reject obviously-wrong payloads before they can be manifest-pinned."""
    head = open(tmp, "rb").read(4)
    if name.endswith(".zip") and head[:2] != b"PK":
        body = open(tmp, "rb").read(300)
        tmp.unlink()
        raise ValueError(f"{name}: expected a zip, got {body!r}")


def _download(name: str, url: str, dest: Path, params: dict | None = None) -> None:
    """Fetch one raw input, honoring the manifest pin.

    - pinned + file matches: skip.
    - pinned + file differs: error (never silently re-pin a drifted snapshot).
    - pinned + file missing: re-fetch, error if upstream no longer matches.
    - unpinned: fetch and record the initial pin.
    """
    entry = util.load_manifest()["files"].get(name)
    if entry and dest.exists():
        actual = util.sha256_file(dest)
        if actual == entry["sha256"]:
            util.log(STAGE, f"{name}: already present and matches manifest, skipping")
            return
        raise ValueError(
            f"{name}: {dest} does not match its MANIFEST.json pin (expected "
            f"{entry['sha256']}, got {actual}). Delete the manifest entry first "
            "if re-pinning is intended."
        )
    full_url = _fetch(url, dest, params=params)
    sha = util.sha256_file(dest)
    if entry and sha != entry["sha256"]:
        raise ValueError(
            f"{name}: upstream content has drifted from the pinned snapshot "
            f"(expected {entry['sha256']}, downloaded {sha}). The pinned dataset "
            "can no longer be reproduced from this URL; delete the manifest "
            "entry to re-pin deliberately."
        )
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    util.record_in_manifest(name, dest, full_url, retrieved_at)
    util.log(STAGE, f"{name}: sha256={sha} bytes={dest.stat().st_size}")


def _wait_for_hub_export(url: str) -> None:
    """Block until the CNRA hub export is generated (it 302s when ready)."""
    for _ in range(HUB_PENDING_RETRIES):
        r = requests.get(url, allow_redirects=False, timeout=120)
        r.raise_for_status()
        if r.is_redirect or not r.content.lstrip().startswith(b"{"):
            return
        status = json.loads(r.content).get("status")
        util.log(STAGE, f"hub export status={status!r}; waiting {HUB_PENDING_WAIT_S}s")
        time.sleep(HUB_PENDING_WAIT_S)
    raise ValueError(f"hub export still pending after "
                     f"{HUB_PENDING_RETRIES * HUB_PENDING_WAIT_S}s: {url}")


def main() -> int:
    errors: list[str] = []

    # 1. FRAP fire perimeters (pinned release).
    if config.FRAP_GDB_URL is None:
        errors.append(
            "config.FRAP_GDB_URL is unresolved (TBD). Resolve the fire25_1 "
            "gdb zip URL and hardcode it in pipeline/config.py."
        )
    else:
        dest = Path(config.FRAP_RAW_PATH)
        entry = util.load_manifest()["files"].get("frap_gdb")
        try:
            if not (entry and dest.exists()):
                _wait_for_hub_export(config.FRAP_GDB_URL)
            _download("frap_gdb", config.FRAP_GDB_URL, dest)
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"frap_gdb: {exc}")

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
            _download_safely(prefix, config.CLIMDIV_BASE_URL + filename,
                             Path(config.CLIMDIV_RAW_DIR) / filename, errors)

    # 3. Climate-division boundary polygons.
    _download_safely("divisions_shp", config.DIVISIONS_URL,
                     Path(config.DIVISIONS_RAW_PATH), errors)

    # 4. GSOM monthly wind: station search, then chunked data pulls.
    try:
        _download_gsom()
    except (requests.RequestException, ValueError) as exc:
        errors.append(f"gsom: {exc}")

    if errors:
        for e in errors:
            util.log(STAGE, f"ERROR: {e}")
        return 1
    util.log(STAGE, f"manifest written: {config.MANIFEST_PATH}")
    return 0


def _download_gsom() -> None:
    """Station search -> pinned station list -> chunked AWND pulls -> one CSV.

    Station IDs are sorted and chunk boundaries are fixed, so the assembled
    CSV is a deterministic function of the pinned station list plus the (live)
    data service responses; once assembled it is pinned like any raw file.
    """
    _download("gsom_stations", config.GSOM_SEARCH_URL,
              Path(config.GSOM_STATIONS_RAW_PATH), params=config.GSOM_SEARCH_PARAMS)
    search = json.loads(Path(config.GSOM_STATIONS_RAW_PATH).read_text())
    # Result names look like "USW00093193.csv".
    ids = sorted({r["name"].removesuffix(".csv") for r in search["results"]})
    if not ids:
        raise ValueError("GSOM station search returned no stations")
    util.log(STAGE, f"gsom: {len(ids)} AWND stations in bbox")

    dest = Path(config.GSOM_RAW_PATH)
    entry = util.load_manifest()["files"].get("gsom_awnd")
    if entry and dest.exists() and entry["sha256"] == util.sha256_file(dest):
        util.log(STAGE, "gsom_awnd: already present and matches manifest, skipping")
        return
    if entry:
        raise ValueError(
            f"gsom_awnd: {dest} is missing or does not match its MANIFEST.json "
            "pin; the data service is live so a re-pull cannot reproduce the "
            "snapshot. Delete the manifest entry to re-pin deliberately."
        )

    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as out:
        for i in range(0, len(ids), config.GSOM_STATIONS_CHUNK):
            chunk = ids[i:i + config.GSOM_STATIONS_CHUNK]
            params = dict(config.GSOM_PARAMS, stations=",".join(chunk))
            r = requests.get(config.GSOM_API_URL, params=params, timeout=600)
            r.raise_for_status()
            body = r.content
            if i > 0:  # drop the repeated header line
                body = body.split(b"\n", 1)[1]
            out.write(body if body.endswith(b"\n") else body + b"\n")
            util.log(STAGE, f"gsom: stations {i + 1}-{i + len(chunk)} fetched")
    tmp.rename(dest)
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    util.record_in_manifest(
        "gsom_awnd", dest,
        requests.Request("GET", config.GSOM_API_URL,
                         params=dict(config.GSOM_PARAMS, stations="<pinned station list>"),
                         ).prepare().url,
        retrieved_at,
    )
    util.log(STAGE, f"gsom_awnd: sha256={util.sha256_file(dest)} bytes={dest.stat().st_size}")


def _download_safely(name: str, url: str, dest: Path, errors: list[str],
                     params: dict | None = None) -> None:
    """Collect failures instead of aborting so one run reports every problem."""
    try:
        _download(name, url, dest, params=params)
    except (requests.RequestException, ValueError) as exc:
        errors.append(f"{name}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
