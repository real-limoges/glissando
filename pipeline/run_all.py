"""Run stages s01-s09 in order (downloads are separate: `make download`)."""

from __future__ import annotations

import sys

from pipeline import (
    s01_extract, s02_clean, s03_dedup, s04_geom_metrics, s05_divisions,
    s06_climdiv, s07_gsom, s08_join, s09_finalize,
)

STAGES = [
    s01_extract, s02_clean, s03_dedup, s04_geom_metrics, s05_divisions,
    s06_climdiv, s07_gsom, s08_join, s09_finalize,
]


def main() -> int:
    for mod in STAGES:
        print(f"=== {mod.__name__} ===", flush=True)
        rc = mod.main()
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
