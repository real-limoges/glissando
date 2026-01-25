#!/usr/bin/env python3
"""
GAMLSS Comparison Framework: Python Orchestrator

Generates synthetic data with known parameters, coordinates fitting in
R (mgcv) and Rust (gamlss_rs), and produces comparison reports.
"""

import subprocess
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class ScenarioConfig:
    """Configuration for a single test scenario."""
    name: str
    n_obs: int
    distribution: str           # gaussian, poisson, gamma, studentt
    mu_formula: str             # linear, smooth, or tensor
    sigma_formula: str          # intercept or linear
    true_params: dict           # ground truth for validation


@dataclass
class FitResult:
    """Standardized output from any fitting implementation."""
    implementation: str
    scenario: str
    converged: bool
    iterations: int
    fit_time_ms: float
    coefficients: dict          # param_name -> list of coeffs
    fitted_mu: list             # fitted values for mu
    fitted_sigma: list          # fitted values for sigma (if applicable)
    edf: dict                   # effective degrees of freedom per param
    log_likelihood: Optional[float]
    aic: Optional[float]
    error: Optional[str]


def generate_gaussian_linear(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate data from: y ~ N(mu, sigma^2)
    where mu = 2 + 3*x and sigma = 1.5
    """
    rng = np.random.default_rng(seed)
    
    x = rng.uniform(0, 10, n)
    
    true_intercept = 2.0
    true_slope = 3.0
    true_sigma = 1.5
    
    mu = true_intercept + true_slope * x
    y = rng.normal(mu, true_sigma)
    
    df = pl.DataFrame({
        "x": x,
        "y": y
    })
    
    true_params = {
        "mu_intercept": true_intercept,
        "mu_slope": true_slope,
        "sigma": true_sigma,
        "log_sigma": np.log(true_sigma)
    }
    
    return df, true_params


def generate_gaussian_heteroskedastic(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate heteroskedastic data where variance increases with x.
    mu = 5 + 2*x
    log(sigma) = -0.5 + 0.15*x  (sigma increases with x)
    """
    rng = np.random.default_rng(seed)
    
    x = rng.uniform(0, 10, n)
    
    true_mu_int, true_mu_slope = 5.0, 2.0
    true_log_sig_int, true_log_sig_slope = -0.5, 0.15
    
    mu = true_mu_int + true_mu_slope * x
    log_sigma = true_log_sig_int + true_log_sig_slope * x
    sigma = np.exp(log_sigma)
    
    y = rng.normal(mu, sigma)
    
    df = pl.DataFrame({"x": x, "y": y})
    
    true_params = {
        "mu_intercept": true_mu_int,
        "mu_slope": true_mu_slope,
        "log_sigma_intercept": true_log_sig_int,
        "log_sigma_slope": true_log_sig_slope
    }
    
    return df, true_params


def generate_gaussian_smooth(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate data with a smooth nonlinear mean function.
    mu = sin(x) + 0.5*x
    sigma = 0.5
    """
    rng = np.random.default_rng(seed)
    
    x = rng.uniform(0, 2 * np.pi, n)
    true_sigma = 0.5
    
    mu = np.sin(x) + 0.5 * x
    y = rng.normal(mu, true_sigma)
    
    df = pl.DataFrame({"x": x, "y": y})
    
    # For smooth functions we store the function form rather than coeffs
    true_params = {
        "mu_function": "sin(x) + 0.5*x",
        "sigma": true_sigma
    }
    
    return df, true_params


def generate_poisson_linear(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate Poisson data with log link.
    log(mu) = 0.5 + 0.3*x
    """
    rng = np.random.default_rng(seed)
    
    x = rng.uniform(0, 5, n)
    
    true_intercept = 0.5
    true_slope = 0.3
    
    log_mu = true_intercept + true_slope * x
    mu = np.exp(log_mu)
    y = rng.poisson(mu).astype(float)
    
    df = pl.DataFrame({"x": x, "y": y})
    
    true_params = {
        "log_mu_intercept": true_intercept,
        "log_mu_slope": true_slope
    }
    
    return df, true_params


def generate_gamma_linear(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate Gamma data.
    log(mu) = 1.0 + 0.5*x (mu ranges roughly 2.7 to 12)
    sigma = 0.5 (coefficient of variation)
    """
    rng = np.random.default_rng(seed)
    
    x = rng.uniform(0, 3, n)
    
    true_log_mu_int = 1.0
    true_log_mu_slope = 0.5
    true_sigma = 0.5  # CV
    
    log_mu = true_log_mu_int + true_log_mu_slope * x
    mu = np.exp(log_mu)
    
    # Gamma parameterization: shape = 1/sigma^2, scale = mu * sigma^2
    shape = 1.0 / (true_sigma ** 2)
    scale = mu * (true_sigma ** 2)
    
    y = rng.gamma(shape, scale)
    
    df = pl.DataFrame({"x": x, "y": y})
    
    true_params = {
        "log_mu_intercept": true_log_mu_int,
        "log_mu_slope": true_log_mu_slope,
        "sigma": true_sigma
    }
    
    return df, true_params


def generate_studentt_linear(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate Student-t data with heavier tails.
    mu = 3 + 2*x
    sigma = 1.0
    nu = 5 (degrees of freedom)
    """
    rng = np.random.default_rng(seed)
    
    x = rng.uniform(0, 10, n)
    
    true_mu_int, true_mu_slope = 3.0, 2.0
    true_sigma = 1.0
    true_nu = 5.0
    
    mu = true_mu_int + true_mu_slope * x
    
    # Generate t-distributed errors, scale by sigma
    t_errors = rng.standard_t(true_nu, n)
    y = mu + true_sigma * t_errors
    
    df = pl.DataFrame({"x": x, "y": y})
    
    true_params = {
        "mu_intercept": true_mu_int,
        "mu_slope": true_mu_slope,
        "sigma": true_sigma,
        "nu": true_nu
    }
    
    return df, true_params


def generate_negative_binomial_linear(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate overdispersed count data from Negative Binomial.
    log(mu) = 1.0 + 0.3*x
    sigma = 0.5 (overdispersion parameter)
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 5, n)

    true_log_mu_int, true_log_mu_slope = 1.0, 0.3
    true_sigma = 0.5  # overdispersion: Var = mu + mu^2 * sigma^2

    log_mu = true_log_mu_int + true_log_mu_slope * x
    mu = np.exp(log_mu)

    # NB parameterization: r = 1/sigma^2, p = r/(r+mu)
    r = 1.0 / (true_sigma ** 2)
    p = r / (r + mu)
    y = rng.negative_binomial(r, p)

    df = pl.DataFrame({"x": x, "y": y.astype(float)})

    true_params = {
        "log_mu_intercept": true_log_mu_int,
        "log_mu_slope": true_log_mu_slope,
        "sigma": true_sigma
    }

    return df, true_params


def generate_beta_linear(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate proportions data from Beta distribution.
    logit(mu) = -1 + 0.5*x  (mu ranges ~0.27 to ~0.73 for x in [0,2])
    phi = 10 (precision parameter)
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 2, n)

    true_logit_mu_int, true_logit_mu_slope = -1.0, 1.0
    true_phi = 10.0  # precision (higher = less variance)

    logit_mu = true_logit_mu_int + true_logit_mu_slope * x
    mu = 1.0 / (1.0 + np.exp(-logit_mu))  # inverse logit

    # Beta parameterization: a = mu*phi, b = (1-mu)*phi
    a = mu * true_phi
    b = (1 - mu) * true_phi
    y = rng.beta(a, b)

    # Clamp to avoid exact 0 or 1
    y = np.clip(y, 1e-6, 1 - 1e-6)

    df = pl.DataFrame({"x": x, "y": y})

    true_params = {
        "logit_mu_intercept": true_logit_mu_int,
        "logit_mu_slope": true_logit_mu_slope,
        "phi": true_phi,
        "log_phi": np.log(true_phi)
    }

    return df, true_params


def generate_poisson_smooth(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate count data with nonlinear mean function.
    log(mu) = sin(x) + 0.5  (oscillating rate)
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 2 * np.pi, n)

    log_mu = np.sin(x) + 0.5
    mu = np.exp(log_mu)
    y = rng.poisson(mu)

    df = pl.DataFrame({"x": x, "y": y.astype(float)})

    true_params = {"functional_form": "sin(x) + 0.5"}

    return df, true_params


def generate_gamma_smooth(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate positive continuous data with smooth mean.
    log(mu) = 1 + 0.5*sin(2*x) (smooth oscillation)
    sigma = 0.3 (CV)
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 2 * np.pi, n)

    true_sigma = 0.3
    log_mu = 1.0 + 0.5 * np.sin(2 * x)
    mu = np.exp(log_mu)

    # Gamma: shape = 1/sigma^2, scale = mu * sigma^2
    shape = 1.0 / (true_sigma ** 2)
    scale = mu * (true_sigma ** 2)
    y = rng.gamma(shape, scale)

    df = pl.DataFrame({"x": x, "y": y})

    true_params = {"functional_form": "1 + 0.5*sin(2*x)", "sigma": true_sigma}

    return df, true_params


def generate_gaussian_multiple(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Generate data with multiple predictors.
    mu = 1 + 2*x1 - 1.5*x2 + 0.5*x3
    sigma = 1.0
    """
    rng = np.random.default_rng(seed)

    x1 = rng.uniform(0, 5, n)
    x2 = rng.uniform(0, 5, n)
    x3 = rng.uniform(0, 5, n)

    true_int, true_b1, true_b2, true_b3 = 1.0, 2.0, -1.5, 0.5
    true_sigma = 1.0

    mu = true_int + true_b1 * x1 + true_b2 * x2 + true_b3 * x3
    y = rng.normal(mu, true_sigma)

    df = pl.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})

    true_params = {
        "intercept": true_int,
        "beta_x1": true_b1,
        "beta_x2": true_b2,
        "beta_x3": true_b3,
        "sigma": true_sigma
    }

    return df, true_params


def generate_gaussian_large(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Large sample Gaussian linear (tests scalability).
    Same as gaussian_linear but will be called with n=5000.
    """
    # Use 5x the default n for scalability testing
    return generate_gaussian_linear(n * 5, seed)


def generate_studentt_smooth(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Student-t with smooth mean function.
    mu = 2*sin(x), sigma = 0.5, nu = 5
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 2 * np.pi, n)

    true_sigma = 0.5
    true_nu = 5.0

    mu = 2 * np.sin(x)
    t_errors = rng.standard_t(true_nu, n)
    y = mu + true_sigma * t_errors

    df = pl.DataFrame({"x": x, "y": y})

    true_params = {
        "functional_form": "2*sin(x)",
        "sigma": true_sigma,
        "nu": true_nu
    }

    return df, true_params


def generate_negative_binomial_smooth(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Overdispersed counts with smooth mean.
    log(mu) = 1 + cos(x), sigma = 0.5
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 2 * np.pi, n)

    true_sigma = 0.5
    log_mu = 1.0 + np.cos(x)
    mu = np.exp(log_mu)

    r = 1.0 / (true_sigma ** 2)
    p = r / (r + mu)
    y = rng.negative_binomial(r, p)

    df = pl.DataFrame({"x": x, "y": y.astype(float)})

    true_params = {"functional_form": "1 + cos(x)", "sigma": true_sigma}

    return df, true_params


def generate_gaussian_quadratic(n: int, seed: int = 42) -> tuple[pl.DataFrame, dict]:
    """
    Gaussian with quadratic mean (tests if smooth can capture polynomial).
    mu = 1 + 0.5*x - 0.05*x^2
    sigma = 0.5
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 10, n)

    true_sigma = 0.5
    mu = 1.0 + 0.5 * x - 0.05 * x**2
    y = rng.normal(mu, true_sigma)

    df = pl.DataFrame({"x": x, "y": y})

    true_params = {"functional_form": "1 + 0.5*x - 0.05*x^2", "sigma": true_sigma}

    return df, true_params


SCENARIOS = {
    "gaussian_linear": generate_gaussian_linear,
    "gaussian_heteroskedastic": generate_gaussian_heteroskedastic,
    "gaussian_smooth": generate_gaussian_smooth,
    "gaussian_multiple": generate_gaussian_multiple,
    "gaussian_large": generate_gaussian_large,
    "gaussian_quadratic": generate_gaussian_quadratic,
    "poisson_linear": generate_poisson_linear,
    "poisson_smooth": generate_poisson_smooth,
    "gamma_linear": generate_gamma_linear,
    "gamma_smooth": generate_gamma_smooth,
    "studentt_linear": generate_studentt_linear,
    "studentt_smooth": generate_studentt_smooth,
    "negative_binomial_linear": generate_negative_binomial_linear,
    "negative_binomial_smooth": generate_negative_binomial_smooth,
    "beta_linear": generate_beta_linear,
}


def run_r_fit(data_path: Path, scenario: str, r_script: Path) -> FitResult:
    """Execute R fitting script and parse results."""
    output_path = data_path.parent / f"r_result_{scenario}.json"
    
    cmd = [
        "Rscript", str(r_script),
        "--data", str(data_path),
        "--scenario", scenario,
        "--output", str(output_path)
    ]
    
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (time.perf_counter() - start) * 1000
    
    if result.returncode != 0:
        return FitResult(
            implementation="R/mgcv",
            scenario=scenario,
            converged=False,
            iterations=0,
            fit_time_ms=elapsed,
            coefficients={},
            fitted_mu=[],
            fitted_sigma=[],
            edf={},
            log_likelihood=None,
            aic=None,
            error=result.stderr[:500]
        )
    
    with open(output_path) as f:
        r_result = json.load(f)
    
    return FitResult(
        implementation="R/mgcv",
        scenario=scenario,
        converged=r_result.get("converged", True),
        iterations=r_result.get("iterations", 0),
        fit_time_ms=r_result.get("fit_time_ms", elapsed),
        coefficients=r_result.get("coefficients", {}),
        fitted_mu=r_result.get("fitted_mu", []),
        fitted_sigma=r_result.get("fitted_sigma", []),
        edf=r_result.get("edf", {}),
        log_likelihood=r_result.get("log_likelihood"),
        aic=r_result.get("aic"),
        error=r_result.get("error")
    )


def run_rust_fit(data_path: Path, scenario: str, rust_binary: Path) -> FitResult:
    """Execute Rust fitting binary and parse results."""
    output_path = data_path.parent / f"rust_result_{scenario}.json"
    
    cmd = [
        str(rust_binary),
        "--data", str(data_path),
        "--scenario", scenario,
        "--output", str(output_path)
    ]
    
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (time.perf_counter() - start) * 1000
    
    if result.returncode != 0:
        return FitResult(
            implementation="Rust/gamlss_rs",
            scenario=scenario,
            converged=False,
            iterations=0,
            fit_time_ms=elapsed,
            coefficients={},
            fitted_mu=[],
            fitted_sigma=[],
            edf={},
            log_likelihood=None,
            aic=None,
            error=result.stderr[:500]
        )
    
    with open(output_path) as f:
        rust_result = json.load(f)
    
    return FitResult(
        implementation="Rust/gamlss_rs",
        scenario=scenario,
        converged=rust_result.get("converged", True),
        iterations=rust_result.get("iterations", 0),
        fit_time_ms=rust_result.get("fit_time_ms", elapsed),
        coefficients=rust_result.get("coefficients", {}),
        fitted_mu=rust_result.get("fitted_mu", []),
        fitted_sigma=rust_result.get("fitted_sigma", []),
        edf=rust_result.get("edf", {}),
        log_likelihood=rust_result.get("log_likelihood"),
        aic=rust_result.get("aic"),
        error=rust_result.get("error")
    )


def compare_results(
    r_result: FitResult,
    rust_result: FitResult,
    true_params: dict
) -> dict:
    """
    Compute comparison metrics between R and Rust fits.
    """
    comparison = {
        "scenario": r_result.scenario,
        "both_converged": r_result.converged and rust_result.converged,
        "r_time_ms": r_result.fit_time_ms,
        "rust_time_ms": rust_result.fit_time_ms,
        "speedup": r_result.fit_time_ms / max(rust_result.fit_time_ms, 0.001),
    }
    
    # Compare fitted values (correlation and max absolute difference)
    if r_result.fitted_mu and rust_result.fitted_mu:
        r_mu = np.array(r_result.fitted_mu)
        rust_mu = np.array(rust_result.fitted_mu)
        
        if len(r_mu) == len(rust_mu):
            comparison["fitted_mu_correlation"] = float(np.corrcoef(r_mu, rust_mu)[0, 1])
            comparison["fitted_mu_max_diff"] = float(np.max(np.abs(r_mu - rust_mu)))
            comparison["fitted_mu_rmse"] = float(np.sqrt(np.mean((r_mu - rust_mu) ** 2)))
    
    # Compare coefficients where names match
    coef_diffs = {}
    for param_name in r_result.coefficients:
        if param_name in rust_result.coefficients:
            # Ensure arrays, handling both scalar and list inputs
            r_val = r_result.coefficients[param_name]
            rust_val = rust_result.coefficients[param_name]
            r_coef = np.atleast_1d(np.array(r_val))
            rust_coef = np.atleast_1d(np.array(rust_val))

            if len(r_coef) == len(rust_coef):
                coef_diffs[param_name] = {
                    "max_diff": float(np.max(np.abs(r_coef - rust_coef))),
                    "r_values": r_coef.tolist(),
                    "rust_values": rust_coef.tolist()
                }
    
    comparison["coefficient_comparison"] = coef_diffs
    
    # Compare to true parameters
    true_param_recovery = {}
    for param_name, true_val in true_params.items():
        if isinstance(true_val, (int, float)):
            # Try to find matching coefficient
            for impl_name, result in [("r", r_result), ("rust", rust_result)]:
                for coef_name, coef_vals in result.coefficients.items():
                    if param_name.replace("_", " ") in coef_name.lower() or coef_name in param_name:
                        if coef_vals is not None:
                            # Handle both scalar and list coefficient values
                            if isinstance(coef_vals, (int, float)):
                                fitted_val = coef_vals
                                error_val = abs(true_val - coef_vals)
                            elif len(coef_vals) == 1:
                                fitted_val = coef_vals[0]
                                error_val = abs(true_val - coef_vals[0])
                            else:
                                fitted_val = coef_vals
                                error_val = None
                            true_param_recovery[f"{param_name}_{impl_name}"] = {
                                "true": true_val,
                                "fitted": fitted_val,
                                "error": error_val
                            }
    
    comparison["true_param_recovery"] = true_param_recovery
    
    # AIC comparison
    if r_result.aic is not None and rust_result.aic is not None:
        comparison["aic_diff"] = r_result.aic - rust_result.aic
    
    return comparison


def generate_executive_summary(summary_path: Path) -> str:
    """
    Generate an executive summary from comparison_summary.json.
    Returns formatted text suitable for printing to console.
    """
    with open(summary_path) as f:
        results = json.load(f)

    if not results:
        return "No comparison results found."

    # Compute aggregate statistics
    total_scenarios = len(results)
    converged_count = sum(1 for r in results if r.get("both_converged") is True)

    speedups = [r["speedup"] for r in results if "speedup" in r]
    avg_speedup = np.mean(speedups) if speedups else 0
    min_speedup = np.min(speedups) if speedups else 0
    max_speedup = np.max(speedups) if speedups else 0

    correlations = [r["fitted_mu_correlation"] for r in results if "fitted_mu_correlation" in r]
    avg_correlation = np.mean(correlations) if correlations else 0
    min_correlation = np.min(correlations) if correlations else 0

    rmses = [r["fitted_mu_rmse"] for r in results if "fitted_mu_rmse" in r]
    avg_rmse = np.mean(rmses) if rmses else 0
    max_rmse = np.max(rmses) if rmses else 0

    total_r_time = sum(r.get("r_time_ms", 0) for r in results)
    total_rust_time = sum(r.get("rust_time_ms", 0) for r in results)

    # Build executive summary
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("                    GAMLSS COMPARISON EXECUTIVE SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Overview section
    lines.append("OVERVIEW")
    lines.append("-" * 70)
    lines.append(f"  Scenarios tested:        {total_scenarios}")
    lines.append(f"  Both converged:          {converged_count}/{total_scenarios} ({100*converged_count/total_scenarios:.0f}%)")
    lines.append("")

    # Performance section
    lines.append("PERFORMANCE (Rust vs R)")
    lines.append("-" * 70)
    lines.append(f"  Average speedup:         {avg_speedup:.1f}x faster")
    lines.append(f"  Best speedup:            {max_speedup:.1f}x ({[r['scenario'] for r in results if r.get('speedup') == max_speedup][0] if max_speedup else 'N/A'})")
    lines.append(f"  Worst speedup:           {min_speedup:.1f}x ({[r['scenario'] for r in results if r.get('speedup') == min_speedup][0] if min_speedup else 'N/A'})")
    lines.append(f"  Total R time:            {total_r_time:.1f} ms")
    lines.append(f"  Total Rust time:         {total_rust_time:.1f} ms")
    lines.append("")

    # Accuracy section
    lines.append("ACCURACY (Fitted Values)")
    lines.append("-" * 70)
    lines.append(f"  Avg mu correlation:      {avg_correlation:.10f}")
    lines.append(f"  Min mu correlation:      {min_correlation:.10f}")
    lines.append(f"  Avg mu RMSE:             {avg_rmse:.6f}")
    lines.append(f"  Max mu RMSE:             {max_rmse:.6f}")
    lines.append("")

    # Per-scenario breakdown
    lines.append("SCENARIO BREAKDOWN")
    lines.append("-" * 70)
    lines.append(f"  {'Scenario':<28} {'Converged':<12} {'Speedup':<10} {'Correlation':<14}")
    lines.append(f"  {'-'*26:<28} {'-'*10:<12} {'-'*8:<10} {'-'*12:<14}")

    for r in results:
        scenario = r.get("scenario", "unknown")
        converged = "Yes" if r.get("both_converged") is True else "No"
        speedup = f"{r.get('speedup', 0):.1f}x"
        corr = r.get("fitted_mu_correlation")
        corr_str = f"{corr:.8f}" if corr else "N/A"
        lines.append(f"  {scenario:<28} {converged:<12} {speedup:<10} {corr_str:<14}")

    lines.append("")

    # Key findings
    lines.append("KEY FINDINGS")
    lines.append("-" * 70)

    if avg_speedup > 1:
        lines.append(f"  * Rust implementation is on average {avg_speedup:.1f}x faster than R")

    if min_correlation > 0.99:
        lines.append(f"  * Excellent agreement between implementations (min corr: {min_correlation:.6f})")
    elif min_correlation > 0.95:
        lines.append(f"  * Good agreement between implementations (min corr: {min_correlation:.6f})")
    else:
        lines.append(f"  * Some discrepancies exist (min corr: {min_correlation:.6f})")

    failed_scenarios = [r["scenario"] for r in results if r.get("both_converged") is not True]
    if failed_scenarios:
        lines.append(f"  * Convergence issues in: {', '.join(failed_scenarios)}")
    else:
        lines.append("  * All scenarios converged successfully in both implementations")

    # Best performing scenario
    if speedups:
        best_idx = np.argmax(speedups)
        lines.append(f"  * Best performance: {results[best_idx]['scenario']} ({speedups[best_idx]:.1f}x speedup)")

    lines.append("")
    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


def print_executive_summary(output_dir: Path = None, summary_path: Path = None):
    """
    Load comparison_summary.json and print executive summary to screen.
    """
    if summary_path is None:
        if output_dir is None:
            output_dir = Path("./output")
        summary_path = output_dir / "comparison_summary.json"

    if not summary_path.exists():
        print(f"Error: Summary file not found at {summary_path}")
        return

    summary = generate_executive_summary(summary_path)
    print(summary)
    return summary


def run_comparison_suite(
    output_dir: Path,
    scenarios: list[str] = None,
    n_obs: int = 500,
    seed: int = 42,
    r_script: Path = None,
    rust_binary: Path = None
):
    """
    Run the full comparison suite.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())
    
    results = []
    
    for scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario_name}")
        print(f"{'='*60}")
        
        generator = SCENARIOS[scenario_name]
        df, true_params = generator(n_obs, seed)
        
        data_path = output_dir / f"data_{scenario_name}.parquet"
        df.write_parquet(data_path)
        print(f"Generated {len(df)} observations -> {data_path}")
        
        # Save true parameters for reference
        params_path = output_dir / f"true_params_{scenario_name}.json"
        with open(params_path, "w") as f:
            json.dump(true_params, f, indent=2, default=str)
        
        # Run both implementations
        r_result = None
        rust_result = None

        print("Fitting with R/mgcv...")
        r_result = run_r_fit(data_path, scenario_name, r_script)
        print(f"  Converged: {r_result.converged}, Time: {r_result.fit_time_ms:.1f}ms")
        if r_result.error:
            print(f"  Error: {r_result.error[:200]}")

        print("Fitting with Rust/gamlss_rs...")
        rust_result = run_rust_fit(data_path, scenario_name, rust_binary)
        print(f"  Converged: {rust_result.converged}, Time: {rust_result.fit_time_ms:.1f}ms")
        if rust_result.error:
            print(f"  Error: {rust_result.error[:200]}")
        
        # Compare if both succeeded
        if r_result and rust_result:
            comparison = compare_results(r_result, rust_result, true_params)
            results.append(comparison)
            
            print(f"\nComparison:")
            if "fitted_mu_correlation" in comparison:
                print(f"  Fitted mu correlation: {comparison['fitted_mu_correlation']:.6f}")
                print(f"  Fitted mu RMSE: {comparison['fitted_mu_rmse']:.6f}")
            print(f"  Speedup (R/Rust): {comparison['speedup']:.2f}x")
    
    # Write summary
    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Summary written to {summary_path}")

    # Print executive summary
    print_executive_summary(summary_path=summary_path)

    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GAMLSS Comparison Framework")
    parser.add_argument("--output-dir", type=Path, default=Path("./comparison_output"))
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--n-obs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r-script", type=Path, default=None)
    parser.add_argument("--rust-binary", type=Path, default=None)
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate data, don't run fits")
    parser.add_argument("--summary", action="store_true",
                       help="Print executive summary from existing comparison_summary.json")

    args = parser.parse_args()
    
    if args.summary:
        # Just print executive summary from existing results
        print_executive_summary(output_dir=args.output_dir)
    elif args.generate_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        scenarios = args.scenarios or list(SCENARIOS.keys())

        for scenario_name in scenarios:
            generator = SCENARIOS[scenario_name]
            df, true_params = generator(args.n_obs, args.seed)

            data_path = args.output_dir / f"data_{scenario_name}.parquet"
            df.write_parquet(data_path)

            params_path = args.output_dir / f"true_params_{scenario_name}.json"
            with open(params_path, "w") as f:
                json.dump(true_params, f, indent=2, default=str)

            print(f"Generated: {data_path}")
    else:
        run_comparison_suite(
            output_dir=args.output_dir,
            scenarios=args.scenarios,
            n_obs=args.n_obs,
            seed=args.seed,
            r_script=args.r_script,
            rust_binary=args.rust_binary
        )
