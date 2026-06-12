#!/usr/bin/env python3
"""
Phase 0 ocat spike report — Guide 3 §2.

Orchestrates the full spike and prints a comparison table:
  - Candidate A (3 independent Binomial/logit threshold models) vs mgcv ocat(R=4)
  - Both vs the true DGP probabilities (to contextualise the approximation gap)

Steps:
  1. Generate train/test parquets via gen_ocat_spike.py
  2. Build (unless --no-build) and run the Rust spike binary (spike_ocat)
  3. Run the R/mgcv reference script (fit_ocat_mgcv.R)
  4. Load results, compute metrics, print table

Usage:
  cd benchmark
  uv run python spike_ocat_report.py --output-dir /tmp/ocat_spike
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent


def run(cmd: list, label: str) -> None:
    print(f"\n[{label}] $ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[{label}] FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)


def relative_error(cand: np.ndarray, ref: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-element |cand - ref| / max(|ref|, eps)."""
    return np.abs(cand - ref) / np.maximum(np.abs(ref), eps)


def print_error_block(label: str, cand: np.ndarray, ref: np.ndarray) -> None:
    """cand and ref are (n, 4) arrays."""
    err = relative_error(cand, ref)   # (n, 4)
    flat = err.ravel()
    per_row_max = err.max(axis=1)
    print(f"\n  {'─'*58}")
    print(f"  {label}")
    print(f"  {'─'*58}")
    print(f"  Per-cell rel. error (all n×4 cells):")
    print(f"    P50  = {np.percentile(flat,  50):.4f}")
    print(f"    P95  = {np.percentile(flat,  95):.4f}")
    print(f"    P99  = {np.percentile(flat,  99):.4f}")
    print(f"    max  = {flat.max():.4f}")
    print(f"  Per-row max rel. error:")
    print(f"    P50  = {np.percentile(per_row_max, 50):.4f}")
    print(f"    P95  = {np.percentile(per_row_max, 95):.4f}")
    print(f"    P99  = {np.percentile(per_row_max, 99):.4f}")
    print(f"    max  = {per_row_max.max():.4f}")


def argmax_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.argmax(a, axis=1) == np.argmax(b, axis=1)).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--n-train",  type=int, default=1500)
    parser.add_argument("--n-test",   type=int, default=500)
    parser.add_argument(
        "--no-build", action="store_true",
        help="Skip cargo build (binary already compiled)",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    train_parquet = out / "ocat_train.parquet"
    test_parquet  = out / "ocat_test.parquet"
    rust_json     = out / "rust_ocat.json"
    mgcv_json     = out / "mgcv_ocat.json"

    # ── 1. Generate data ──────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Guide 3 — Phase 0 ocat spike                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    run(
        [
            sys.executable, HERE / "gen_ocat_spike.py",
            "--output-dir", out,
            "--seed",    str(args.seed),
            "--n-train", str(args.n_train),
            "--n-test",  str(args.n_test),
        ],
        "gen",
    )

    # ── 2. Rust spike binary ──────────────────────────────────────────────────
    if not args.no_build:
        run(
            [
                "cargo", "build",
                "-p", "glissando_benchmark",
                "--bin", "spike_ocat",
                "--release",
            ],
            "cargo",
        )

    rust_bin = REPO_ROOT / "target" / "release" / "spike_ocat"
    if not rust_bin.exists():
        rust_bin = REPO_ROOT / "target" / "debug" / "spike_ocat"
        if not rust_bin.exists():
            print(f"spike_ocat binary not found; tried release and debug", file=sys.stderr)
            sys.exit(1)

    run(
        [rust_bin, "--train", train_parquet, "--test", test_parquet, "--output", rust_json],
        "rust",
    )

    # ── 3. R/mgcv reference ───────────────────────────────────────────────────
    run(
        [
            "Rscript", HERE / "fit_ocat_mgcv.R",
            "--train",  train_parquet,
            "--test",   test_parquet,
            "--output", mgcv_json,
        ],
        "mgcv",
    )

    # ── 4. Load results ───────────────────────────────────────────────────────
    with open(rust_json) as f:
        rust = json.load(f)
    with open(mgcv_json) as f:
        mgcv_res = json.load(f)

    if rust.get("error"):
        print(f"\nRust spike FAILED: {rust['error']}", file=sys.stderr)
        sys.exit(1)
    if mgcv_res.get("error") not in (None, "NA", "null"):
        print(f"\nmgcv spike FAILED: {mgcv_res['error']}", file=sys.stderr)
        sys.exit(1)

    cand_probs = np.array(rust["probs"], dtype=float)        # (n, 4)
    mgcv_probs = np.array(mgcv_res["probs"], dtype=float)    # (n, 4)

    # True DGP probs from test parquet
    test_df = pl.read_parquet(test_parquet)
    true_probs = np.column_stack(
        [test_df[c].to_numpy() for c in ["p1", "p2", "p3", "p4"]]
    )  # (n, 4)

    n_test    = cand_probs.shape[0]
    n_viol    = int(rust["n_violations"])
    converged = rust["models_converged"]

    # ── 5. Print report ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  RESULTS  —  n_test={n_test}")
    print(f"{'═'*60}")
    print(f"\n  Rust fit time:  {rust['fit_time_ms']:.0f} ms")
    print(f"  mgcv fit time:  {mgcv_res['fit_time_ms']:.0f} ms")
    print(f"  Models converged (le1 / le2 / le3): {converged}")
    print(f"\n  Monotonicity violations (μ₁≤μ₂≤μ₃ broken):")
    print(f"    {n_viol} / {n_test}  ({100.0 * n_viol / n_test:.1f}%)")
    print( "    (rows where independent fits are not monotone;")
    print( "     negative raw probs before clip-and-renorm)")

    # ── Error tables ──────────────────────────────────────────────────────────
    print_error_block(
        "Candidate A (binomial/threshold)  vs  mgcv ocat(R=4)",
        cand_probs, mgcv_probs,
    )
    print(f"  Argmax agreement vs mgcv: {argmax_agreement(cand_probs, mgcv_probs):.1%}")

    print_error_block(
        "Candidate A (binomial/threshold)  vs  TRUE probs",
        cand_probs, true_probs,
    )
    print(f"  Argmax agreement vs true: {argmax_agreement(cand_probs, true_probs):.1%}")

    print_error_block(
        "mgcv ocat(R=4)  vs  TRUE probs  [context: mgcv's own error]",
        mgcv_probs, true_probs,
    )
    print(f"  Argmax agreement vs true: {argmax_agreement(mgcv_probs, true_probs):.1%}")

    print(f"\n{'═'*60}")
    print("  DECISION GATE (Guide 3 §2)")
    print("  Review the 'Candidate A vs mgcv' block above.")
    print("  Pass → adopt Candidate A for B3; close guides 3 / 4 / 5.")
    print("  Fail → proceed to Phase 1 (full ocat family; Guide 3 §3–§7).")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
