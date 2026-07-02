"""s06: parse nClimDiv fixed-width element files into division-month rows.

Record layout (per the climdiv README, PROVISIONAL until checked against the
real files): a 10-char ID — state(2) division(2) element(2) year(4) — followed
by 12 whitespace-separated monthly values. Units are native nClimDiv units:
tavg °F, precip inches, PDSI unitless. Element-specific missing-value
sentinels come from config.CLIMDIV_ELEMENTS.

Files are discovered in CLIMDIV_RAW_DIR by element prefix so pinned real
downloads and smoke fixtures parse through the same code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config, util

STAGE = "s06_climdiv"
OUT_PATH_NAME = "s06_climdiv_monthly.parquet"


def _find_file(prefix: str) -> Path:
    d = Path(config.CLIMDIV_RAW_DIR)
    matches = sorted(d.glob(prefix + "*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {prefix}* in {d}, found {[m.name for m in matches]} "
            "(run `make download`)"
        )
    return matches[0]


def parse_element(path: Path, column: str, sentinel: float, element_code: str) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ident, values = line[:10], line[10:].split()
        if len(values) != 12:
            raise ValueError(f"{path.name}: expected 12 monthly values, got {len(values)}: {line!r}")
        if ident[4:6] != element_code:
            raise ValueError(
                f"{path.name}: record element code {ident[4:6]!r} != expected "
                f"{element_code!r} for {column} — wrong file pinned for this element?"
            )
        state, division, year = int(ident[0:2]), int(ident[2:4]), int(ident[6:10])
        if state != config.CLIMDIV_STATE_CODE_CA:
            continue
        for month, v in enumerate(values, start=1):
            rows.append((division, year, month, float(v)))
    if not rows:
        raise ValueError(
            f"{path.name}: no records for state code {config.CLIMDIV_STATE_CODE_CA} "
            "(California) — file layout or state-code assumption is wrong"
        )
    df = pd.DataFrame(rows, columns=["division", "year", "month", column])
    # Sentinels mark missing months (e.g. the current partial year).
    df.loc[np.isclose(df[column], sentinel), column] = np.nan
    return df


def main() -> int:
    merged: pd.DataFrame | None = None
    for prefix, (column, sentinel, element_code) in config.CLIMDIV_ELEMENTS.items():
        path = _find_file(prefix)
        util.verify_against_manifest(path)
        df = parse_element(path, column, sentinel, element_code)
        util.log(STAGE, f"{path.name}: {len(df)} CA division-months, "
                        f"{int(df[column].isna().sum())} missing")
        merged = df if merged is None else merged.merge(
            df, on=["division", "year", "month"], how="outer"
        )

    merged = merged.astype({"division": "Int64", "year": "Int64", "month": "Int64"})
    merged = merged.sort_values(["division", "year", "month"]).reset_index(drop=True)
    util.log(STAGE, f"{len(merged)} division-month rows, "
                    f"years {merged['year'].min()}-{merged['year'].max()}, "
                    f"divisions {sorted(merged['division'].dropna().unique().tolist())}")
    util.write_parquet(merged, config.INTERIM_DIR / OUT_PATH_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
