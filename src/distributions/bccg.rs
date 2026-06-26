//! Box-Cox–Cole-Green (BCCG) distribution for skew positive continuous data.
//!
//! BCCG (Cole & Green 1992) models a positive response `y > 0` by a Box-Cox power
//! transform to a standard normal. It is parameterized by the **median** `μ`, the
//! approximate coefficient of variation `σ`, and the skewness `ν`. It is the engine
//! behind LMS centile curves (growth charts) — see `model.centiles`.
//!
//! This file is the worked-example template for the Box-Cox family (DIST-1); BCT
//! (Student-t tail) and BCPE (power-exponential kurtosis) extend the same spine,
//! changing only the distribution `z` follows and the extra parameter column.

use super::boxcox::{boxcox_inv, boxcox_z, boxcox_z_dz_dnu};
use super::{
    require, DerivativesResult, Distribution, GamlssError, IdentityLink, Link, LogLink,
    MIN_POSITIVE, MIN_WEIGHT,
};
use crate::math::{std_normal_cdf, std_normal_quantile};
use ndarray::Array1;
use std::collections::HashMap;

/// Box-Cox–Cole-Green distribution for skew positive continuous data (`y > 0`).
///
/// Parameters: `μ` (median, log link), `σ` (≈ coefficient of variation, log link),
/// `ν` (skewness / Box-Cox power, identity link). `ν = 1` is symmetric about the
/// median; `ν = 0` is the log-normal limit.
///
/// The lower tail is left un-truncated (the exact gamlss density renormalizes by
/// `Φ(1/(σ|ν|))`, which is ≈ 1 in the usual regime where `1/(σ|ν|)` is large); the
/// approximation matches `dBCCG`/`pBCCG` to ~1e-6 for typical fits.
#[derive(Debug, Clone, Copy, Default)]
pub struct BCCG;

impl BCCG {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for BCCG {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma", "nu"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(LogLink)),
            "sigma" => Ok(Box::new(LogLink)),
            "nu" => Ok(Box::new(IdentityLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    /// Robust seeds: `μ₀ = median(y)` (μ is the median), `σ₀` = a robust coefficient
    /// of variation `1.4826·MAD(y)/median(y)`, and `ν₀ = 1` (start symmetric — the
    /// identity of the Box-Cox power). Mirrors `StudentT`'s median/MAD seeding so the
    /// first RS iteration is not dragged by skew/outliers.
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => median(y),
            "sigma" => {
                let med = median(y);
                let cv = 1.4826 * median_abs_deviation(y) / med.abs().max(MIN_POSITIVE);
                cv.clamp(0.01, 10.0)
            }
            "nu" => 1.0,
            other => {
                debug_assert!(
                    matches!(other, "mu" | "sigma" | "nu"),
                    "BCCG has no parameter '{other}'"
                );
                1.0
            }
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Box-Cox z-score and η-scale score/Fisher pairs. Full derivation in
        // docs/math/mathematics.md §1.9. By the definition of z, T = (y/μ)^ν = 1+νσz,
        // so the η-scale scores collapse to the clean forms below.
        //   u_μ (log)      = μ·dl/dμ = z/σ + ν(z²−1)
        //   u_σ (log)      = σ·dl/dσ = z²−1
        //   u_ν (identity) =   dl/dν = −z·∂z/∂ν + log(y/μ)
        // Expected Fisher information (matches gamlss BCCG):
        //   w_μ = μ²·I_μμ = 1/σ² + 2ν²,   w_σ = σ²·I_σσ = 2,   w_ν = I_νν = 7σ²/4.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = y.len();

        let mut u_mu = Array1::<f64>::zeros(n);
        let mut w_mu = Array1::<f64>::zeros(n);
        let mut u_sigma = Array1::<f64>::zeros(n);
        let mut w_sigma = Array1::<f64>::zeros(n);
        let mut u_nu = Array1::<f64>::zeros(n);
        let mut w_nu = Array1::<f64>::zeros(n);

        for i in 0..n {
            let m = mu[i].max(MIN_POSITIVE);
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i];
            let yi = y[i].max(MIN_POSITIVE);
            let (z, dz_dnu, l) = boxcox_z_dz_dnu(yi, m, s, nu_i); // l = log(y/μ)

            u_mu[i] = z / s + nu_i * (z * z - 1.0);
            w_mu[i] = 1.0 / (s * s) + 2.0 * nu_i * nu_i;
            u_sigma[i] = z * z - 1.0;
            w_sigma[i] = 2.0;
            u_nu[i] = -z * dz_dnu + l;
            w_nu[i] = (7.0 * s * s / 4.0).max(MIN_WEIGHT);
        }

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
            ("nu".to_string(), (u_nu, w_nu)),
        ]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = y.len();
        let half_ln_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mu[i].max(MIN_POSITIVE);
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i];
            let yi = y[i].max(MIN_POSITIVE);
            let z = boxcox_z(yi, m, s, nu_i);
            // log f(y) = −½z² − ½log(2π) + (ν−1)·log(y) − ν·log(μ) − log(σ)
            out[i] = -0.5 * z * z - half_ln_2pi + (nu_i - 1.0) * yi.ln() - nu_i * m.ln() - s.ln();
        }
        Ok(out)
    }

    /// First-order coefficient-of-variation approximation `Var(Y) ≈ (σ·μ)²` — `σ` is
    /// (approximately) the CV in the BCCG parameterization. Used only for Pearson
    /// residuals; the preferred randomized-quantile residuals go through `cdf`.
    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(crate::math::par_zip_map(mu, sigma, |m, s| (s * m) * (s * m)))
    }

    /// `μ` is the **median**, not the mean. The exact mean has no closed form; this
    /// is the second-order expansion `E[Y] ≈ μ·(1 + ½σ²(1−ν))`, exact at `ν = 1`
    /// (symmetric, mean = μ) and at `ν = 0` (log-normal, `μ·e^{σ²/2}` to O(σ²)).
    fn expected_value(
        &self,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = mu.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            out[i] = mu[i] * (1.0 + 0.5 * sigma[i] * sigma[i] * (1.0 - nu[i]));
        }
        Ok(out)
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // F(y) = Φ(z); z is monotone increasing in y on y > 0, so this is a valid CDF.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = y.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            if y[i] <= 0.0 {
                continue; // support is y > 0
            }
            let m = mu[i].max(MIN_POSITIVE);
            let s = sigma[i].max(MIN_POSITIVE);
            out[i] = std_normal_cdf(boxcox_z(y[i], m, s, nu[i]));
        }
        Ok(out)
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // Invert F = Φ(z): z_p = Φ⁻¹(p), then back out y from the Box-Cox transform.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = p.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mu[i].max(MIN_POSITIVE);
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i];
            let zp = std_normal_quantile(p[i].clamp(1e-12, 1.0 - 1e-12));
            out[i] = boxcox_inv(m, s, nu_i, zp);
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "BCCG"
    }
}

/// Median of `y` (finite entries). Returns 0.0 for an empty slice (`validate_inputs`
/// rejects empty `y` on the public path, so this is only a defensive default).
fn median(y: &Array1<f64>) -> f64 {
    let mut v: Vec<f64> = y.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len().is_multiple_of(2) {
        0.5 * (v[m - 1] + v[m])
    } else {
        v[m]
    }
}

/// Median absolute deviation about the median: `median(|yᵢ − median(y)|)`.
fn median_abs_deviation(y: &Array1<f64>) -> f64 {
    let med = median(y);
    let dev = y.mapv(|x| (x - med).abs());
    median(&dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::test_helpers::{
        check_cdf_monotone_in_unit, check_cdf_pdf_consistency, check_cdf_quantile_roundtrip,
        check_score_via_finite_diff, derivative_keys_match_parameters, params_view,
    };
    use ndarray::array;

    #[test]
    fn bccg_derivative_keys_match_parameters() {
        let y = array![0.5, 1.5, 3.0, 7.0];
        let mu = array![1.0, 2.0, 4.0, 6.0];
        let sigma = array![0.3, 0.25, 0.2, 0.35];
        let nu = array![1.0, 0.5, -0.5, 1.5];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        p.insert("nu", &nu);
        derivative_keys_match_parameters(&BCCG, p, &y);
    }

    #[test]
    fn loglik_bccg_finite_on_typical_inputs() {
        let owned = [
            ("mu", array![2.0, 2.0, 4.0]),
            ("sigma", array![0.3, 0.25, 0.2]),
            ("nu", array![1.0, 0.0, -0.5]),
        ];
        let p = params_view(&owned);
        let ll = BCCG.loglik(&array![1.0, 2.0, 5.0], &p).unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn bccg_nu_zero_is_lognormal() {
        // At ν = 0, z = log(y/μ)/σ, so log f(y) = log-normal density.
        // Standard log-normal log-density: −log(y) − log(σ) − ½log(2π) − ½(log(y/μ)/σ)².
        let owned = [
            ("mu", array![2.0]),
            ("sigma", array![0.4]),
            ("nu", array![0.0]),
        ];
        let p = params_view(&owned);
        let y = array![3.0];
        let ll = BCCG.loglik(&y, &p).unwrap();
        let z = (3.0_f64 / 2.0).ln() / 0.4;
        let expected =
            -3.0_f64.ln() - 0.4_f64.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() - 0.5 * z * z;
        assert!((ll - expected).abs() < 1e-12, "ll={ll} expected={expected}");
    }

    #[test]
    fn score_matches_finite_diff_bccg() {
        // Spans symmetric (ν=1), skew (ν=0.5, 1.5), reverse-skew (ν=-0.5), and the
        // ν≈0 log-normal limit so every derivative branch is exercised.
        let y = array![1.0, 2.5, 5.0, 0.8, 3.0];
        let owned = [
            ("mu", array![1.5, 2.0, 4.0, 1.0, 2.5]),
            ("sigma", array![0.3, 0.25, 0.2, 0.4, 0.3]),
            ("nu", array![1.0, 0.5, 1.5, -0.5, 1e-8]),
        ];
        check_score_via_finite_diff(&BCCG, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&BCCG, &y, &owned, "sigma", 1e-5);
        check_score_via_finite_diff(&BCCG, &y, &owned, "nu", 1e-5);
    }

    #[test]
    fn cdf_quantile_roundtrip_bccg() {
        let y = array![0.6, 1.5, 3.0, 7.0, 2.0];
        let owned = [
            ("mu", array![1.0, 2.0, 4.0, 6.0, 2.5]),
            ("sigma", array![0.3, 0.25, 0.2, 0.35, 0.3]),
            ("nu", array![1.0, 0.5, -0.5, 1.5, 0.0]),
        ];
        check_cdf_quantile_roundtrip(&BCCG, &y, &owned, 1e-6);
        check_cdf_pdf_consistency(&BCCG, &y, &owned, 1e-5, 1e-3);
    }

    #[test]
    fn cdf_monotone_bccg_and_zero_below_support() {
        let grid = Array1::from_iter((0..80).map(|i| 0.05 + i as f64 * 0.1));
        let owned = [
            ("mu", array![3.0]),
            ("sigma", array![0.3]),
            ("nu", array![0.8]),
        ];
        check_cdf_monotone_in_unit(&BCCG, &grid, &owned);
        // y ≤ 0 sits outside the y > 0 support.
        let boundary = [
            ("mu", array![3.0, 3.0]),
            ("sigma", array![0.3, 0.3]),
            ("nu", array![0.8, 0.8]),
        ];
        let p = params_view(&boundary);
        let at_boundary = BCCG.cdf(&array![0.0, -1.0], &p).unwrap();
        assert_eq!(at_boundary[0], 0.0);
        assert_eq!(at_boundary[1], 0.0);
    }

    #[test]
    fn median_quantile_is_mu() {
        // p = 0.5 ⇒ z_p = 0 ⇒ y = μ for every ν. (μ is the median.)
        let owned = [
            ("mu", array![2.0, 5.0]),
            ("sigma", array![0.3, 0.2]),
            ("nu", array![0.5, -1.0]),
        ];
        let p = params_view(&owned);
        let med = BCCG.quantile(&array![0.5, 0.5], &p).unwrap();
        assert!((med[0] - 2.0).abs() < 1e-9);
        assert!((med[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn expected_value_is_mu_at_nu_one() {
        // At ν = 1 the distribution is symmetric about μ, so the mean equals μ.
        let owned = [
            ("mu", array![4.0]),
            ("sigma", array![0.3]),
            ("nu", array![1.0]),
        ];
        let p = params_view(&owned);
        let ev = BCCG.expected_value(&p).unwrap();
        assert!((ev[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn variance_bccg_is_cv_squared_times_mu_squared() {
        let owned = [
            ("mu", array![4.0]),
            ("sigma", array![0.25]),
            ("nu", array![1.0]),
        ];
        let p = params_view(&owned);
        let v = BCCG.variance(&p).unwrap();
        // (σμ)² = (0.25·4)² = 1.
        assert!((v[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn initial_value_seeds_are_robust_and_sane() {
        let y = array![2.0, 2.1, 1.9, 2.2, 2.0, 1.8, 2.3, 50.0];
        let mu0 = BCCG.initial_value("mu", &y);
        assert!((mu0 - 2.05).abs() < 0.3, "median seed near the core (got {mu0})");
        let sigma0 = BCCG.initial_value("sigma", &y);
        assert!(sigma0 > 0.0 && sigma0 < 1.0, "robust CV seed (got {sigma0})");
        assert_eq!(BCCG.initial_value("nu", &y), 1.0);
    }
}
