#!/usr/bin/env python3
"""
ocat benchmark — compares glissando Ocat(R=4) against mgcv::ocat(R=4).

Log-likelihood cross-check (intercept-only model)
  Fits `Ocat(R=4)` with an intercept-only formula in both glissando and mgcv
  (method="ML") on the same training data.  Because no smoothing is involved,
  both optimizers converge to the same MLE; the training log-likelihoods must
  agree to within 0.01.

Probability-matrix comparison (P-spline smooth model)
  Fits `mu ~ 1 + s(x1) + s(x2)` and compares the (n_test × 4)
  category-probability matrices.  Smoothing parameters are selected
  independently (glissando: REML; mgcv: REML), so some divergence is expected;
  the benchmark reports P50/P95/max relative error and argmax agreement.
  Target: P95 relative error < 10 %.

Usage:
  cd benchmark
  uv run python ocat_benchmark.py --output-dir /tmp/ocat_bench
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent

LOGLIK_TOLERANCE = 0.01   # absolute difference in total log-likelihood
PROB_P95_TARGET  = 0.10   # P95 relative error on (n × 4) probability matrix
ARGMAX_TARGET    = 0.90   # minimum argmax agreement rate


def run(cmd: list, label: str) -> None:
    print(f"\n[{label}] $ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[{label}] FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)


def relative_error(cand: np.ndarray, ref: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.abs(cand - ref) / np.maximum(np.abs(ref), eps)


def argmax_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.argmax(a, axis=1) == np.argmax(b, axis=1)).mean())


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def print_separator(title: str = "") -> None:
    line = "═" * 62
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-test",  type=int, default=500)
    parser.add_argument("--no-build", action="store_true",
                        help="Skip cargo build (binaries already compiled)")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    train_parquet = out / "ocat_train.parquet"
    test_parquet  = out / "ocat_test.parquet"

    # ── Generate data ─────────────────────────────────────────────────────────
    print_separator("ocat benchmark  (glissando vs mgcv)")
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

    # ── Build ─────────────────────────────────────────────────────────────────
    if not args.no_build:
        run(
            ["cargo", "build", "-p", "glissando_benchmark", "--bin", "fit_ocat", "--release"],
            "cargo",
        )

    rust_bin = REPO_ROOT / "target" / "release" / "fit_ocat"
    if not rust_bin.exists():
        rust_bin = REPO_ROOT / "target" / "debug" / "fit_ocat"
        if not rust_bin.exists():
            print("fit_ocat binary not found", file=sys.stderr)
            sys.exit(1)

    # ── Log-likelihood cross-check (intercept-only) ───────────────────────────
    print_separator("Log-likelihood cross-check (intercept-only, method=ML)")

    rust_icpt_json = out / "glissando_ocat_intercept.json"
    mgcv_icpt_json = out / "mgcv_ocat_intercept.json"

    run(
        [rust_bin,
         "--train", train_parquet, "--test", test_parquet,
         "--output", rust_icpt_json,
         "--intercept-only"],
        "glissando-intercept",
    )
    run(
        ["Rscript", HERE / "fit_ocat_mgcv.R",
         "--train",  train_parquet,
         "--test",   test_parquet,
         "--output", mgcv_icpt_json,
         "--intercept-only"],
        "mgcv-intercept",
    )

    rust_icpt = load_json(rust_icpt_json)
    mgcv_icpt = load_json(mgcv_icpt_json)

    if rust_icpt.get("error"):
        print(f"glissando intercept-only FAILED: {rust_icpt['error']}", file=sys.stderr)
        sys.exit(1)
    if mgcv_icpt.get("error") not in (None, "NA"):
        print(f"mgcv intercept-only FAILED: {mgcv_icpt['error']}", file=sys.stderr)
        sys.exit(1)

    ll_rust = float(rust_icpt["loglik_train"])
    ll_mgcv = float(mgcv_icpt["loglik_train"])
    ll_diff = abs(ll_rust - ll_mgcv)

    print(f"\n  n_train          : {rust_icpt['n_train']}")
    print(f"  glissando loglik : {ll_rust:.6f}")
    print(f"  mgcv loglik      : {ll_mgcv:.6f}")
    print(f"  |Δ loglik|       : {ll_diff:.6f}  (tolerance: {LOGLIK_TOLERANCE})")
    print(f"  glissando conv.  : {rust_icpt['converged']}")
    print(f"  mgcv converged   : {mgcv_icpt['converged']}")

    loglik_pass = ll_diff <= LOGLIK_TOLERANCE
    print(f"\n  {'✓ PASS' if loglik_pass else '✗ FAIL'}  "
          f"|Δ loglik| = {ll_diff:.6f} {'≤' if loglik_pass else '>'} {LOGLIK_TOLERANCE}")

    # ── Probability-matrix comparison (P-spline smooth) ───────────────────────
    print_separator("Probability-matrix comparison (P-spline smooth model)")

    rust_smooth_json = out / "glissando_ocat_smooth.json"
    mgcv_smooth_json = out / "mgcv_ocat_smooth.json"

    run(
        [rust_bin,
         "--train", train_parquet, "--test", test_parquet,
         "--output", rust_smooth_json],
        "glissando-smooth",
    )
    run(
        ["Rscript", HERE / "fit_ocat_mgcv.R",
         "--train",  train_parquet,
         "--test",   test_parquet,
         "--output", mgcv_smooth_json],
        "mgcv-smooth",
    )

    rust_smooth = load_json(rust_smooth_json)
    mgcv_smooth = load_json(mgcv_smooth_json)

    if rust_smooth.get("error"):
        print(f"glissando smooth FAILED: {rust_smooth['error']}", file=sys.stderr)
        sys.exit(1)
    if mgcv_smooth.get("error") not in (None, "NA"):
        print(f"mgcv smooth FAILED: {mgcv_smooth['error']}", file=sys.stderr)
        sys.exit(1)

    rust_probs = np.array(rust_smooth["probs"], dtype=float)  # (n, 4)
    mgcv_probs = np.array(mgcv_smooth["probs"], dtype=float)  # (n, 4)

    test_df    = pl.read_parquet(test_parquet)
    true_probs = np.column_stack([test_df[c].to_numpy() for c in ["p1", "p2", "p3", "p4"]])

    err_vs_mgcv = relative_error(rust_probs, mgcv_probs)
    flat_vm     = err_vs_mgcv.ravel()
    row_max_vm  = err_vs_mgcv.max(axis=1)

    err_vs_true = relative_error(rust_probs, true_probs)
    flat_vt     = err_vs_true.ravel()

    print(f"\n  n_test               : {rust_probs.shape[0]}")
    print(f"  glissando conv.      : {rust_smooth['converged']}  ({rust_smooth['fit_time_ms']:.0f} ms)")
    print(f"  mgcv converged       : {mgcv_smooth['converged']}  ({mgcv_smooth['fit_time_ms']:.0f} ms)")
    print(f"  glissando loglik(tr) : {rust_smooth['loglik_train']:.4f}")
    print(f"  mgcv loglik(tr)      : {mgcv_smooth['loglik_train']:.4f}")

    print(f"\n  glissando vs mgcv — per-cell relative error on (n × 4) probs:")
    print(f"    P50 = {np.percentile(flat_vm, 50):.4f}")
    print(f"    P95 = {np.percentile(flat_vm, 95):.4f}  (target < {PROB_P95_TARGET:.2f})")
    print(f"    max = {flat_vm.max():.4f}")
    print(f"  Per-row max P95 = {np.percentile(row_max_vm, 95):.4f}")
    print(f"  Argmax agreement vs mgcv : {argmax_agreement(rust_probs, mgcv_probs):.1%}"
          f"  (target ≥ {ARGMAX_TARGET:.0%})")

    print(f"\n  glissando vs TRUE probs — per-cell relative error:")
    print(f"    P50 = {np.percentile(flat_vt, 50):.4f}")
    print(f"    P95 = {np.percentile(flat_vt, 95):.4f}")
    print(f"    max = {flat_vt.max():.4f}")
    print(f"  Argmax agreement vs true : {argmax_agreement(rust_probs, true_probs):.1%}")

    p95_vm     = float(np.percentile(flat_vm, 95))
    agree      = argmax_agreement(rust_probs, mgcv_probs)
    smooth_pass = (p95_vm < PROB_P95_TARGET) and (agree >= ARGMAX_TARGET)
    print(f"\n  {'✓ PASS' if smooth_pass else '✗ FAIL'}  P95={p95_vm:.4f} argmax={agree:.1%}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_separator("Summary")
    print(f"  [{'✓' if loglik_pass  else '✗'}] loglik matches mgcv (|Δ| ≤ {LOGLIK_TOLERANCE})  →  {ll_diff:.6f}")
    print(f"  [{'✓' if smooth_pass else '✗'}] P95 rel-err < {PROB_P95_TARGET:.0%}  →  {p95_vm:.4f}")
    print(f"  [{'✓' if smooth_pass else '✗'}] Argmax agreement ≥ {ARGMAX_TARGET:.0%}  →  {agree:.1%}")
    print_separator()

    if not (loglik_pass and smooth_pass):
        sys.exit(1)


if __name__ == "__main__":
    main()
