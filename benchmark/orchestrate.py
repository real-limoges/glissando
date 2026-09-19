#!/usr/bin/env python3
"""Glissando vs mgcv comparison orchestrator.

Generates synthetic data per scenario, optionally invokes the Rust glissando
binary and the R/mgcv script, and merges per-scenario fits into
`comparison_summary.json`, the file `tests/mgcv_reference.rs` validates.

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
import os
import subprocess
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # Per-scenario replicate cap (bounds nightly cost for expensive scenarios);
    # None uses the run-wide --reps. Only ever reduces the count.
    reps_override: Optional[int] = None

    def reps(self, run_reps: int) -> int:
        return min(run_reps, self.reps_override) if self.reps_override else run_reps


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
    # Constant mean, with a full sine period of structure in log σ. This is the
    # scale-smooth cousin of gen_gaussian_smooth, for the gaulss comparison.
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
    # the +1 keeps μ safely positive across the whole range.
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
    # Glissando parameterization: shape = 1/σ², scale = μσ².
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
    # NB2 parameterization: r = 1/σ, p = r/(r+μ).
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
    # 10 groups, each with a random intercept, plus a linear covariate x.
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
# IMPORTANT: append new scenarios to the END, nowhere else. Per-scenario seeds
# get spawned in iteration order (XOR with hash(name)), so a middle insert
# reshuffles the seed for every scenario after it and invalidates stored results.

SCENARIOS: list[Scenario] = [
    # ── Original scenarios (seeds stable) ─────────────────────────────────
    Scenario("gaussian_linear",          False, True,  None,   gen_gaussian_linear),
    Scenario("gaussian_heteroskedastic", False, True,  None,   gen_gaussian_heteroskedastic),
    Scenario("gaussian_smooth",          True,  True,  None,   gen_gaussian_smooth),
    Scenario("gaussian_multiple",        False, True,  None,   gen_gaussian_multiple),
    Scenario("gaussian_large",           False, True,  10_000, gen_gaussian_linear, reps_override=5),
    Scenario("gaussian_quadratic",       True,  True,  None,   gen_gaussian_quadratic),
    Scenario("poisson_linear",           False, True,  None,   gen_poisson_linear),
    Scenario("poisson_smooth",           True,  True,  None,   gen_poisson_smooth),
    Scenario("gamma_linear",             False, True,  None,   gen_gamma_linear),
    Scenario("gamma_smooth",             True,  True,  None,   gen_gamma_smooth),
    # Student-t: the real oracle is gamlss TF() (same RS algorithm). mgcv_capable=True
    # just keeps scat() around as a loose, mu-only cross-method sanity check.
    Scenario("studentt_linear",          False, True,  None,   gen_studentt_linear),
    Scenario("studentt_smooth",          True,  True,  None,   gen_studentt_smooth),
    Scenario("negative_binomial_linear", False, True,  None,   gen_negative_binomial_linear),
    Scenario("negative_binomial_smooth", True,  True,  None,   gen_negative_binomial_smooth),
    Scenario("beta_linear",              False, True,  None,   gen_beta_linear),
    Scenario("gaussian_sigma_smooth",    True,  True,  None,   gen_gaussian_sigma_smooth),
    # B1: Gaussian + prior weights; mgcv can do it via gam(..., weights=w).
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


def rep_salt(k: int) -> int:
    # Rep 0 → salt 0, so rep 0 reproduces the historical single-seed data.
    # crc32 for cross-session stability (hash() is randomized by PYTHONHASHSEED).
    return 0 if k == 0 else (zlib.crc32(f"__rep{k}__".encode()) & 0xFFFFFFFF)


def scenario_rng(sub_seed: int, name: str, rep: int) -> np.random.Generator:
    # sub_seed is spawned once per scenario in iteration order (see main); folding
    # the name hash and rep salt in here keeps every rep-0 draw independent.
    name_hash = zlib.crc32(name.encode()) & 0xFFFFFFFF
    return np.random.default_rng(int(sub_seed) ^ name_hash ^ rep_salt(rep))


# Per-fit wall-clock budget. A well-behaved fit is done in seconds; a hung
# solver (or R session) shouldn't get to stall the whole run. Kill it and mark
# the scenario failed instead.
FIT_TIMEOUT_S = 600


def run_subprocess(cmd: list[str], output_path: Path, label: str) -> Optional[dict]:
    """Execute `cmd` and return the JSON it wrote to `output_path`, or None on failure."""
    if output_path.exists():
        output_path.unlink()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=FIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"[{label}] timed out after {FIT_TIMEOUT_S}s", flush=True)
        return None
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
    parser.add_argument("--reps", type=int, default=25, help="Replicates per scenario")
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="Parallel fit workers (default: os.cpu_count())",
    )
    parser.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Subset of scenario names to run (default: all)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.scenarios) if args.scenarios else {s.name for s in SCENARIOS}
    active = [s for s in SCENARIOS if s.name in selected]

    def data_path(name: str, rep: int) -> Path:
        return args.output_dir / f"data_{name}_rep{rep}.parquet"

    # 1. Data generation. sub_seed is spawned once per scenario in iteration
    # order (preserving historical seeds); rep 0 reproduces the old single draw.
    base = np.random.SeedSequence(args.seed)
    for scenario in active:
        n = scenario.n_obs_override or args.n_obs
        sub_seed = base.spawn(1)[0].generate_state(1)[0]
        reps = scenario.reps(args.reps)
        for k in range(reps):
            data = scenario.generate(scenario_rng(int(sub_seed), scenario.name, k), n)
            write_parquet(data, data_path(scenario.name, k))
        print(f"[gen] {scenario.name}: n={n} × {reps} reps", flush=True)

    if args.generate_only:
        return

    # 2. Fit. One independent subprocess per (scenario, rep, engine); run them
    # through a thread pool since each writes its own JSON.
    def engine_cmd(engine: str, scenario: Scenario, rep: int, out: Path) -> Optional[list[str]]:
        common = ["--data", str(data_path(scenario.name, rep)),
                  "--scenario", scenario.name, "--output", str(out)]
        if engine == "glissando" and args.rust_binary and args.rust_binary.exists():
            return [str(args.rust_binary), *common]
        if engine == "mgcv" and args.r_script and args.r_script.exists() and scenario.mgcv_capable:
            return ["Rscript", str(args.r_script), *common]
        if engine == "gamlss" and args.gamlss_script and args.gamlss_script.exists() and "studentt" in scenario.name:
            return ["Rscript", str(args.gamlss_script), *common]
        return None

    jobs = []
    for scenario in active:
        for rep in range(scenario.reps(args.reps)):
            for engine in ("glissando", "mgcv", "gamlss"):
                out = args.output_dir / f"{engine}_{scenario.name}_rep{rep}.json"
                cmd = engine_cmd(engine, scenario, rep, out)
                if cmd is not None:
                    jobs.append((scenario.name, rep, engine, cmd, out))

    results: dict[tuple[str, int, str], Optional[dict]] = {}
    with ThreadPoolExecutor(max_workers=args.jobs or os.cpu_count()) as pool:
        futures = {
            pool.submit(run_subprocess, cmd, out, f"{engine}:{name}#{rep}"): (name, rep, engine)
            for name, rep, engine, cmd, out in jobs
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    # 3. Merge into schema v2: per scenario, a list of per-rep engine fits.
    summary_scenarios = []
    for scenario in active:
        reps = [
            {
                "rep": rep,
                "glissando": results.get((scenario.name, rep, "glissando")),
                "mgcv": results.get((scenario.name, rep, "mgcv")),
                "gamlss": results.get((scenario.name, rep, "gamlss")),
            }
            for rep in range(scenario.reps(args.reps))
        ]
        summary_scenarios.append({"name": scenario.name, "smooth": scenario.smooth, "reps": reps})

    summary = {
        "version": 2,
        "n_obs": args.n_obs,
        "seed": args.seed,
        "reps": args.reps,
        "scenarios": summary_scenarios,
    }
    out_path = args.output_dir / "comparison_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
