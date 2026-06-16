#!/usr/bin/env python3
"""Glissando vs mgcv comparison orchestrator.

Generates synthetic data per scenario, optionally invokes the Rust glissando
binary and the R/mgcv script, and merges per-scenario fits into
`comparison_summary.json` — the file `tests/mgcv_reference.rs` validates.

Two phases (controlled by `--generate-only`):
  1. data generation: writes parquet files into `output_dir` from a fixed seed.
  2. fitting + comparison: dispatches each parquet to the Rust binary
     (`--rust-binary`) and the R script (`--r-script`), merges the results.

Scenarios are registered below with metadata declaring whether they're
smooth (looser tolerance in the Rust test) and whether mgcv natively supports
them.  Student-t is compared against mgcv's `scat()` scaled-t family.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl


@dataclass(frozen=True)
class Scenario:
    name: str
    smooth: bool
    mgcv_capable: bool
    n_obs_override: Optional[int]
    generate: Callable[[np.random.Generator, int], dict[str, np.ndarray]]


# ─── Data generators ─────────────────────────────────────────────────────────

def gen_gaussian_linear(rng, n):
    x = np.linspace(0, 10, n)
    y = 2.0 + 0.5 * x + rng.normal(0.0, 1.0, n)
    return {"y": y, "x": x}


def gen_gaussian_heteroskedastic(rng, n):
    x = np.linspace(0, 5, n)
    sigma = np.exp(-1.0 + 0.3 * x)
    y = 2.0 + 0.5 * x + rng.normal(0.0, sigma, n)
    return {"y": y, "x": x}


def gen_gaussian_smooth(rng, n):
    x = np.linspace(0, 4 * np.pi, n)
    y = np.sin(x) + rng.normal(0.0, 0.3, n)
    return {"y": y, "x": x}


def gen_gaussian_sigma_smooth(rng, n):
    # Constant mean; a full sine period of structure in log σ. The scale-smooth
    # analogue of gen_gaussian_smooth, for the gaulss comparison.
    x = np.linspace(0, 1, n)
    log_sigma = -0.7 + 0.8 * np.sin(2 * np.pi * x)
    y = 2.0 + rng.normal(0.0, np.exp(log_sigma), n)
    return {"y": y, "x": x}


def gen_gaussian_multiple(rng, n):
    x1 = rng.uniform(0, 5, n)
    x2 = rng.uniform(0, 5, n)
    x3 = rng.uniform(0, 5, n)
    y = 1.0 + 0.5 * x1 + 0.3 * x2 - 0.2 * x3 + rng.normal(0.0, 1.0, n)
    return {"y": y, "x1": x1, "x2": x2, "x3": x3}


def gen_gaussian_quadratic(rng, n):
    x = np.linspace(-2, 2, n)
    y = 1.0 + 0.5 * x + 0.5 * x ** 2 + rng.normal(0.0, 0.5, n)
    return {"y": y, "x": x}


def gen_poisson_linear(rng, n):
    x = np.linspace(0, 4, n)
    mu = np.exp(0.5 + 0.3 * x)
    y = rng.poisson(mu).astype(float)
    return {"y": y, "x": x}


def gen_poisson_smooth(rng, n):
    x = np.linspace(0, 2 * np.pi, n)
    # +1 keeps μ comfortably positive across the full range.
    mu = np.exp(np.sin(x) + 1.0)
    y = rng.poisson(mu).astype(float)
    return {"y": y, "x": x}


def gen_binomial_linear(rng, n):
    # Bernoulli logistic: p = logistic(−1 + 0.8·x), x ∈ [0, 4].
    x = np.linspace(0, 4, n)
    p = 1.0 / (1.0 + np.exp(-(-1.0 + 0.8 * x)))
    y = rng.binomial(1, p).astype(float)
    return {"y": y, "x": x}


def gen_binomial_smooth(rng, n):
    # Logistic smooth: p = logistic(0.5·sin(x)), x ∈ [0, 2π].
    x = np.linspace(0, 2 * np.pi, n)
    p = 1.0 / (1.0 + np.exp(-0.5 * np.sin(x)))
    y = rng.binomial(1, p).astype(float)
    return {"y": y, "x": x}


def _gamma_sample(rng, mu, sigma):
    # Glissando parameterisation: shape = 1/σ², scale = μσ².
    shape = 1.0 / (sigma * sigma)
    scale = mu * sigma * sigma
    return rng.gamma(shape, scale)


def gen_gamma_linear(rng, n):
    x = np.linspace(0, 4, n)
    mu = np.exp(0.5 + 0.3 * x)
    y = _gamma_sample(rng, mu, sigma=0.5)
    return {"y": y, "x": x}


def gen_gamma_smooth(rng, n):
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(np.sin(x) + 1.5)
    y = _gamma_sample(rng, mu, sigma=0.5)
    return {"y": y, "x": x}


def gen_gamma_sigma_smooth(rng, n):
    # Constant mean, smooth CV (sigma); compared against mgcv gammals.
    x = np.linspace(0, 1, n)
    log_sigma = -1.0 + 0.6 * np.sin(2 * np.pi * x)   # CV range ~0.37 .. 1.0
    mu = np.exp(1.5)                                   # constant mean ~4.5
    sigma = np.exp(log_sigma)
    y = _gamma_sample(rng, mu, sigma)
    return {"y": y, "x": x}


def gen_studentt_linear(rng, n):
    x = np.linspace(0, 4, n)
    mu = 2.0 + 0.5 * x
    y = mu + rng.standard_t(df=5.0, size=n)
    return {"y": y, "x": x}


def gen_studentt_smooth(rng, n):
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.sin(x) + 2.0
    y = mu + rng.standard_t(df=5.0, size=n)
    return {"y": y, "x": x}


def _negbin_sample(rng, mu, sigma):
    # NB2 parameterisation: r = 1/σ, p = r/(r+μ).
    r = 1.0 / sigma
    p = r / (r + mu)
    return rng.negative_binomial(r, p).astype(float)


def gen_negative_binomial_linear(rng, n):
    x = np.linspace(0, 4, n)
    mu = np.exp(0.5 + 0.3 * x)
    y = _negbin_sample(rng, mu, sigma=0.5)
    return {"y": y, "x": x}


def gen_negative_binomial_smooth(rng, n):
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(np.sin(x) + 1.5)
    y = _negbin_sample(rng, mu, sigma=0.5)
    return {"y": y, "x": x}


def gen_beta_linear(rng, n):
    x = np.linspace(0, 4, n)
    eta = -2.0 + 0.5 * x
    mu = 1.0 / (1.0 + np.exp(-eta))
    phi = 10.0
    alpha = mu * phi
    beta = (1.0 - mu) * phi
    y = rng.beta(alpha, beta)
    return {"y": y, "x": x}


def gen_beta_smooth(rng, n):
    # Logistic-smooth mean, constant precision.
    x = np.linspace(0, 2 * np.pi, n)
    mu = 1.0 / (1.0 + np.exp(-0.5 * np.sin(x)))
    phi = 10.0
    alpha = mu * phi
    beta = (1.0 - mu) * phi
    y = rng.beta(alpha, beta)
    return {"y": y, "x": x}


def gen_tensor_smooth(rng, n):
    # 2D Gaussian: mu = sin(x1) + 0.4·x2² + 0.5, small noise.
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 1, n)
    mu = np.sin(x1) + 0.4 * x2 ** 2 + 0.5
    y = rng.normal(mu, scale=0.3)
    return {"y": y, "x1": x1, "x2": x2}


def gen_random_effect(rng, n):
    # 10 groups with random intercepts; also a linear covariate x.
    n_groups = 10
    group_effects = rng.normal(0.0, 1.0, n_groups)
    g = rng.integers(0, n_groups, n).astype(float)
    x = rng.uniform(0, 3, n)
    mu = 2.0 + 0.5 * x + group_effects[g.astype(int)]
    y = rng.normal(mu, scale=0.5)
    return {"y": y, "x": x, "g": g}


def _listing_weights(rng, n):
    """Simulate listing-level weights: 1 / n_obs_per_listing.

    Groups n rows into listings of size 1–4; each row's weight is the
    reciprocal of its listing's size so every listing contributes equally.
    """
    weights = np.ones(n)
    i = 0
    while i < n:
        group_size = rng.integers(1, 5)  # 1..4 rows per listing
        end = min(i + group_size, n)
        w = 1.0 / (end - i)
        weights[i:end] = w
        i = end
    return weights


def gen_b1_weighted(rng, n):
    """B1: Gaussian with five smooths + binary dummy, weights=1/n_obs_per_listing."""
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 1, n)
    x3 = rng.uniform(-1, 1, n)
    x4 = rng.uniform(0, 4, n)
    x5 = rng.uniform(0, 3, n)
    d1 = (rng.random(n) > 0.5).astype(float)
    mu = (
        np.sin(x1)
        + 0.4 * x2 ** 2
        + 0.3 * x3
        + 0.2 * np.cos(x4)
        + 0.1 * x5
        + 0.5 * d1
        + 2.0
    )
    y = rng.normal(mu, scale=0.5)
    weights = _listing_weights(rng, n)
    return {"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "d1": d1, "weights": weights}


def gen_b2_weighted(rng, n):
    """B2: StudentT with four smooths, weights=1/n_obs_per_listing."""
    x1 = rng.uniform(0, 2 * np.pi, n)
    x2 = rng.uniform(0, 1, n)
    x3 = rng.uniform(-1, 1, n)
    x4 = rng.uniform(0, 4, n)
    mu = np.sin(x1) + 0.4 * x2 ** 2 + 0.3 * x3 + 0.2 * np.cos(x4) + 2.0
    y = mu + rng.standard_t(df=5.0, size=n)
    weights = _listing_weights(rng, n)
    return {"y": y, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "weights": weights}


# ─── Scenario registry ────────────────────────────────────────────────────────
# IMPORTANT: Append new scenarios to the END only. Per-scenario seeds are
# spawned in iteration order (XOR with hash(name)), so inserting in the middle
# would change seeds for every subsequent scenario and invalidate stored results.

SCENARIOS: list[Scenario] = [
    # ── Original scenarios (seeds stable) ─────────────────────────────────
    Scenario("gaussian_linear",          False, True,  None,   gen_gaussian_linear),
    Scenario("gaussian_heteroskedastic", False, False, None,   gen_gaussian_heteroskedastic),
    Scenario("gaussian_smooth",          True,  True,  None,   gen_gaussian_smooth),
    Scenario("gaussian_multiple",        False, True,  None,   gen_gaussian_multiple),
    Scenario("gaussian_large",           False, True,  10_000, gen_gaussian_linear),
    Scenario("gaussian_quadratic",       True,  True,  None,   gen_gaussian_quadratic),
    Scenario("poisson_linear",           False, True,  None,   gen_poisson_linear),
    Scenario("poisson_smooth",           True,  True,  None,   gen_poisson_smooth),
    Scenario("gamma_linear",             False, True,  None,   gen_gamma_linear),
    Scenario("gamma_smooth",             True,  True,  None,   gen_gamma_smooth),
    # Student-t: primary oracle is gamlss TF() (same RS algorithm); mgcv_capable=True
    # keeps scat() as a loose mu-only cross-method sanity check.
    Scenario("studentt_linear",          False, True,  None,   gen_studentt_linear),
    Scenario("studentt_smooth",          True,  True,  None,   gen_studentt_smooth),
    Scenario("negative_binomial_linear", False, True,  None,   gen_negative_binomial_linear),
    Scenario("negative_binomial_smooth", True,  True,  None,   gen_negative_binomial_smooth),
    Scenario("beta_linear",              False, True,  None,   gen_beta_linear),
    Scenario("gaussian_sigma_smooth",    True,  True,  None,   gen_gaussian_sigma_smooth),
    # B1: Gaussian + prior weights; mgcv-capable via gam(..., weights=w).
    Scenario("b1_weighted_gaussian",     True,  True,  None,   gen_b1_weighted),
    # B2: StudentT + prior weights; mgcv uses scat() for the location mean.
    Scenario("b2_weighted_studentt",     True,  True,  None,   gen_b2_weighted),

    # ── New scenarios appended below (seeds unchanged above) ───────────────
    # Binomial (Bernoulli logistic).
    Scenario("binomial_linear",          False, True,  None,   gen_binomial_linear),
    Scenario("binomial_smooth",          True,  True,  None,   gen_binomial_smooth),
    # Beta smooth on μ.
    Scenario("beta_smooth",              True,  True,  None,   gen_beta_smooth),
    # Gamma location-scale: constant mean, smooth CV; compared via gammals.
    Scenario("gamma_sigma_smooth",       True,  True,  None,   gen_gamma_sigma_smooth),
    # 2D tensor-product smooth.
    Scenario("tensor_smooth",            True,  True,  None,   gen_tensor_smooth),
    # Random effects: linear + group intercepts; compared via s(g, bs="re").
    Scenario("random_effect",            True,  True,  None,   gen_random_effect),
    # CR spline; compared via s(x, bs="cr") which uses the same quantile knots.
    Scenario("gaussian_cr_smooth",       True,  True,  None,   gen_gaussian_smooth),
]


# ─── Orchestration ────────────────────────────────────────────────────────────

def write_parquet(data: dict[str, np.ndarray], path: Path) -> None:
    df = pl.DataFrame({k: pl.Series(k, v.tolist(), dtype=pl.Float64) for k, v in data.items()})
    df.write_parquet(path)


def run_subprocess(cmd: list[str], output_path: Path, label: str) -> Optional[dict]:
    """Execute `cmd` and return the JSON it wrote to `output_path`, or None on failure."""
    if output_path.exists():
        output_path.unlink()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[{label}] failed: {e.stderr.strip()}", flush=True)
        return None
    if not output_path.exists():
        print(f"[{label}] produced no output", flush=True)
        return None
    return json.loads(output_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rust-binary", type=Path, default=None)
    parser.add_argument("--r-script", type=Path, default=None)
    parser.add_argument(
        "--gamlss-script", type=Path, default=None,
        help="R/gamlss script (fit_gamlss.R); the correct like-for-like oracle for "
             "StudentT scenarios (same RS algorithm + μ/σ/ν parameterization).",
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Subset of scenario names to run (default: all)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.scenarios) if args.scenarios else {s.name for s in SCENARIOS}

    # 1. Data generation. Per-scenario sub-RNG so adding scenarios doesn't perturb others.
    base = np.random.SeedSequence(args.seed)
    for scenario in SCENARIOS:
        if scenario.name not in selected:
            continue
        n = scenario.n_obs_override or args.n_obs
        sub_seed = base.spawn(1)[0].generate_state(1)[0]
        # zlib.crc32 is deterministic across Python sessions; hash() is not
        # (PYTHONHASHSEED randomization would give different data every run).
        name_hash = zlib.crc32(scenario.name.encode()) & 0xFFFFFFFF
        sub_rng = np.random.default_rng(int(sub_seed) ^ name_hash)
        data = scenario.generate(sub_rng, n)
        path = args.output_dir / f"data_{scenario.name}.parquet"
        write_parquet(data, path)
        print(f"[gen] {scenario.name}: n={n} → {path.name}", flush=True)

    if args.generate_only:
        return

    # 2. Fit + merge.
    summary_scenarios = []
    for scenario in SCENARIOS:
        if scenario.name not in selected:
            continue

        data_path = args.output_dir / f"data_{scenario.name}.parquet"
        rust_result = None
        mgcv_result = None
        gamlss_result = None

        if args.rust_binary and args.rust_binary.exists():
            output = args.output_dir / f"rust_{scenario.name}.json"
            rust_result = run_subprocess(
                [
                    str(args.rust_binary),
                    "--data", str(data_path),
                    "--scenario", scenario.name,
                    "--output", str(output),
                ],
                output,
                f"rust:{scenario.name}",
            )

        if args.r_script and args.r_script.exists() and scenario.mgcv_capable:
            output = args.output_dir / f"mgcv_{scenario.name}.json"
            mgcv_result = run_subprocess(
                [
                    "Rscript", str(args.r_script),
                    "--data", str(data_path),
                    "--scenario", scenario.name,
                    "--output", str(output),
                ],
                output,
                f"mgcv:{scenario.name}",
            )

        # gamlss TF() is the correct like-for-like reference for StudentT: same RS
        # algorithm and same (μ, σ, ν) parameterization, so it exposes σ/ν/EDF/SE
        # that mgcv's scat() cannot. mgcv scat() is retained above as a loose,
        # mu-only cross-method (independent-algorithm) sanity check.
        if (
            args.gamlss_script
            and args.gamlss_script.exists()
            and "studentt" in scenario.name
        ):
            output = args.output_dir / f"gamlss_{scenario.name}.json"
            gamlss_result = run_subprocess(
                [
                    "Rscript", str(args.gamlss_script),
                    "--data", str(data_path),
                    "--scenario", scenario.name,
                    "--output", str(output),
                ],
                output,
                f"gamlss:{scenario.name}",
            )

        summary_scenarios.append({
            "name": scenario.name,
            "smooth": scenario.smooth,
            "glissando": rust_result,
            "mgcv": mgcv_result,
            "gamlss": gamlss_result,
        })

    summary = {
        "n_obs": args.n_obs,
        "seed": args.seed,
        "scenarios": summary_scenarios,
    }
    out_path = args.output_dir / "comparison_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
