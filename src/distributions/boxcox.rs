//! Shared Box-Cox spine for the BCCG / BCT / BCPE families.
//!
//! All three transform a positive response `y > 0` to a standardized residual `z`
//! via the same Box-Cox power transform; they differ only in the distribution `z`
//! follows (standard normal, Student-t, power-exponential) and the extra shape
//! parameter that distribution carries. This module owns the transform, its `ν`
//! derivative, and the inverse used by the quantile.

use super::MIN_POSITIVE;
use crate::math::{median, median_abs_deviation};
use ndarray::Array1;

/// Threshold below which `ν` is treated as 0 for the *inverse* transform, where
/// `(1 + νσz)^{1/ν}` would raise a near-1 base to a huge power and lose precision.
const NU_EPS: f64 = 1e-6;

/// Box-Cox z-score `((y/μ)^ν − 1)/(νσ)`, with the `ν → 0` log-normal limit
/// `log(y/μ)/σ`. Evaluated as `expm1(νL)/(νσ)` so it is accurate (no cancellation)
/// and smooth through `ν = 0` — the branch is only at *exactly* `ν = 0`. Assumes
/// `y, μ, σ > 0` (callers clamp).
#[inline]
pub(super) fn boxcox_z(y: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    let l = (y / mu).ln();
    if nu == 0.0 {
        l / sigma
    } else {
        (nu * l).exp_m1() / (nu * sigma)
    }
}

/// Returns `(z, ∂z/∂ν, L)` with `L = log(y/μ)`. `∂z/∂ν` subtracts two near-equal
/// terms for small `ν`, so it uses a Taylor limit there (accurate to `O(ν²)`).
#[inline]
pub(super) fn boxcox_z_dz_dnu(y: f64, mu: f64, sigma: f64, nu: f64) -> (f64, f64, f64) {
    let l = (y / mu).ln();
    let z = boxcox_z(y, mu, sigma, nu);
    let dz_dnu = if nu.abs() < NU_EPS {
        l * l / (2.0 * sigma) + nu * l * l * l / (3.0 * sigma)
    } else {
        let tm1 = (nu * l).exp_m1(); // T − 1 = (y/μ)^ν − 1
        (nu * (tm1 + 1.0) * l - tm1) / (nu * nu * sigma)
    };
    (z, dz_dnu, l)
}

/// Invert the transform for the quantile: `y = μ·(1 + νσz)^{1/ν}` (`ν = 0` →
/// `μ·e^{σz}`). `z` is the standardized residual at the requested probability; the
/// base is clamped off 0 (the truncated tail).
#[inline]
pub(super) fn boxcox_inv(mu: f64, sigma: f64, nu: f64, z: f64) -> f64 {
    if nu.abs() < NU_EPS {
        mu * (sigma * z).exp()
    } else {
        mu * (1.0 + nu * sigma * z).max(MIN_POSITIVE).powf(1.0 / nu)
    }
}

/// Robust `mu`/`sigma`/`nu` initial-value seeds shared by BCCG/BCT/BCPE: `mu`
/// seeds the median, `sigma` a robust coefficient of variation
/// (`1.4826·MAD(y)/median(y)`, clamped so the first RS iteration is not dragged
/// by skew/outliers), and `nu` starts symmetric (the identity of the Box-Cox
/// power). Returns `None` for any other parameter name so each family can layer
/// its own extra parameter (`tau`) seed on top.
pub(super) fn boxcox_seed(param: &str, y: &Array1<f64>) -> Option<f64> {
    match param {
        "mu" => Some(median(y)),
        "sigma" => {
            let med = median(y);
            let cv = 1.4826 * median_abs_deviation(y) / med.abs().max(MIN_POSITIVE);
            Some(cv.clamp(0.01, 10.0))
        }
        "nu" => Some(1.0),
        _ => None,
    }
}

/// Second-order Box-Cox mean approximation shared by BCCG/BCT/BCPE:
/// `E[Y] ≈ μ·(1 + ½σ²(1−ν))`, exact at `ν = 1` (symmetric, mean = μ) and at
/// `ν = 0` (log-normal, `μ·e^{σ²/2}` to `O(σ²)`). `μ` is the median, not the mean;
/// the approximation depends on `z`'s distribution only through its first two
/// moments (mean 0, variance 1), so it is identical whether `z` is normal,
/// Student-t, or power-exponential.
pub(super) fn boxcox_expected_value(
    mu: &Array1<f64>,
    sigma: &Array1<f64>,
    nu: &Array1<f64>,
) -> Array1<f64> {
    let n = mu.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        out[i] = mu[i] * (1.0 + 0.5 * sigma[i] * sigma[i] * (1.0 - nu[i]));
    }
    out
}

/// First-order coefficient-of-variation variance approximation `(σ·μ)²` shared by
/// BCCG and BCPE (and the base BCT layers its `t`-distribution inflation factor
/// on top of): `σ` is (approximately) the CV in the Box-Cox parameterization.
/// Used only for Pearson residuals; the preferred randomized-quantile residuals
/// go through `cdf`.
pub(super) fn boxcox_cv_variance(mu: &Array1<f64>, sigma: &Array1<f64>) -> Array1<f64> {
    crate::math::par_zip_map(mu, sigma, |m, s| (s * m) * (s * m))
}
