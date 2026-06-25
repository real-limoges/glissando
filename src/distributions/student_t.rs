//! Student's t distribution for heavy-tailed continuous data.

use super::{
    require, DerivativesResult, Distribution, FlooredLogLink, GamlssError, IdentityLink, Link,
    LogLink, MIN_POSITIVE, MIN_WEIGHT,
};
use crate::math::{digamma_batch, par_zip3_map, par_zip_map, trigamma_batch};
use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;

/// Lower bound on the degrees-of-freedom `ν`, enforced via [`FlooredLogLink`].
/// Keeps `ν > 2` so the variance `σ²·ν/(ν−2)` stays finite while the optimizer
/// explores the heavy-tail region. Never binds when the true `ν` is well above 2.
const NU_FLOOR: f64 = 2.0;

/// Starting value for `ν`. A fixed moderate seed is deliberately preferred over a
/// sample-kurtosis estimate: for regression data the *marginal* kurtosis reflects the
/// spread of the mean structure, not the noise tails, so a kurtosis inversion biases
/// `ν` and (in the multi-smooth weighted case) can seed the optimizer into a degenerate
/// over-smoothed basin. 5 is a standard heavy-tail default, well clear of the `ν > 2`
/// finite-variance boundary.
const NU_INIT: f64 = 5.0;

/// Student's t distribution for heavy-tailed continuous data.
///
/// Parameters: `μ` (location, identity), `σ` (scale, log), `ν` (degrees of freedom,
/// floored log link with `ν ≥ 2`). As `ν → ∞` the distribution approaches Gaussian.
#[derive(Debug, Clone, Copy, Default)]
pub struct StudentT;

impl StudentT {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for StudentT {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma", "nu"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(IdentityLink)),
            "sigma" => Ok(Box::new(LogLink)),
            "nu" => Ok(Box::new(FlooredLogLink { floor: NU_FLOOR })),
            other => Err(self.unknown_param(other)),
        }
    }

    /// Robust IRLS seeds for heavy-tailed data. The trait default (sample mean,
    /// sample SD) is non-robust: under heavy tails the mean is pulled by outliers and
    /// the SD overestimates the scale `σ`. Instead:
    /// - `μ` = median(y),
    /// - `σ` = 1.4826·MAD(y) (the MAD-to-σ consistency factor for a normal core),
    /// - `ν` = [`NU_INIT`] (a fixed moderate seed; see its doc for why a kurtosis
    ///   estimate is avoided).
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => median(y),
            "sigma" => {
                let s = 1.4826 * median_abs_deviation(y);
                if s < 1e-4 {
                    1.0
                } else {
                    s
                }
            }
            "nu" => NU_INIT,
            other => {
                debug_assert!(
                    matches!(other, "mu" | "sigma" | "nu"),
                    "StudentT has no parameter '{other}'"
                );
                NU_INIT
            }
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Student-t log-likelihood, location-scale parameterization. Full derivation
        // in docs/math/mathematics.md.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;

        let z = (y - mu) / sigma;
        let z_sq = z.mapv(|v| v * v);

        // The "robustifying weight" w = (ν+1)/(ν+z²) downweights outliers (large |z|).
        // It → 1 as ν → ∞, recovering Gaussian behavior.
        let w_robust = par_zip_map(nu, &z_sq, |nu_i, z2_i| (nu_i + 1.0) / (nu_i + z2_i));

        // μ derivatives (identity link).
        let u_mu = (&w_robust * &z) / sigma;
        let w_mu = &w_robust / sigma.mapv(|s| s * s);

        // σ derivatives (log link). Chain rule: dl/dη = σ · dl/dσ = w·z² − 1.
        let u_sigma = &w_robust * &z_sq - 1.0;
        let w_sigma: Array1<f64> = nu.mapv(|nu_i| (2.0 * nu_i) / (nu_i + 3.0));

        // ν derivatives (log link). Score involves digamma differences.
        let nu_plus_1_half = nu.mapv(|nu_i| (nu_i + 1.0) / 2.0);
        let nu_half = nu.mapv(|nu_i| nu_i / 2.0);
        let d1 = digamma_batch(&nu_plus_1_half);
        let d2 = digamma_batch(&nu_half);

        let term3 = par_zip_map(nu, &z_sq, |nu_i, z2_i| (1.0 + z2_i / nu_i).ln());
        let term4 = par_zip3_map(nu, &w_robust, &z_sq, |nu_i, w_i, z2_i| {
            (w_i * z2_i - 1.0) / nu_i
        });

        let dl_dnu = 0.5 * (&d1 - &d2 - &term3 + &term4);
        // Chain rule for log link: u_η = ν · dl/dν.
        let u_nu = &dl_dnu * nu;

        // Fisher information for ν uses trigamma (the second derivative of log-Γ).
        let t1 = trigamma_batch(&nu_half);
        let t2 = trigamma_batch(&nu_plus_1_half);
        let t3: Array1<f64> = nu.mapv(|nu_i| (2.0 * (nu_i + 3.0)) / (nu_i * (nu_i + 1.0)));
        // The `+ t3` term subtracts from the negative Hessian — sign is correct.
        let i_nu = 0.25 * (&t1 - &t2 + &t3);
        // For log link `W_η = I_ν · ν²`, floored to keep the weight matrix positive definite.
        let w_nu = par_zip_map(&i_nu, nu, |i, nu_i| (i * nu_i * nu_i).abs().max(MIN_WEIGHT));

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
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i].max(MIN_POSITIVE);
            let z = (y[i] - mu[i]) / s;
            out[i] = ln_gamma((nu_i + 1.0) / 2.0)
                - ln_gamma(nu_i / 2.0)
                - 0.5 * (std::f64::consts::PI * nu_i).ln()
                - s.ln()
                - 0.5 * (nu_i + 1.0) * (1.0 + z * z / nu_i).ln();
        }
        Ok(out)
    }

    /// `Var(Y) = σ²·ν/(ν−2)` for `ν > 2`. For `ν ≤ 2` the variance is undefined; we
    /// clamp the denominator at `MIN_POSITIVE` so Pearson residuals stay finite.
    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        Ok(par_zip_map(sigma, nu, |s, nu_i| {
            let denom = (nu_i - 2.0).max(MIN_POSITIVE);
            s * s * nu_i / denom
        }))
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // Location-scale t: F(y) = T_ν((y−μ)/σ). ν varies per observation, so build
        // one StudentsT per row (mirrors the indexed loglik_pointwise loop above).
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = y.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i].max(MIN_POSITIVE);
            out[i] = StudentsT::new(mu[i], s, nu_i)
                .expect("valid StudentsT params")
                .cdf(y[i]);
        }
        Ok(out)
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = p.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i].max(MIN_POSITIVE);
            out[i] = StudentsT::new(mu[i], s, nu_i)
                .expect("valid StudentsT params")
                .inverse_cdf(p[i].clamp(1e-12, 1.0 - 1e-12));
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "StudentT"
    }
}

/// Median of `y`. Returns 0.0 for an empty slice (`validate_inputs` rejects empty
/// `y` on the public path, so this is only a defensive default).
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
    fn studentt_derivatives() {
        let y = array![0.0, 1.0, 2.0, -1.5];
        let mu = array![0.0, 0.5, 1.5, -1.0];
        let sigma = array![1.0, 1.0, 0.8, 1.2];
        let nu = array![5.0, 10.0, 4.0, 8.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        p.insert("nu", &nu);
        derivative_keys_match_parameters(&StudentT, p, &y);
    }

    #[test]
    fn loglik_studentt_matches_cauchy_at_zero() {
        // Student-t with ν=1, μ=0, σ=1 is standard Cauchy. Density at y=0 is 1/π.
        let owned = [
            ("mu", array![0.0]),
            ("sigma", array![1.0]),
            ("nu", array![1.0]),
        ];
        let p = params_view(&owned);
        let ll = StudentT.loglik(&array![0.0], &p).unwrap();
        let expected = -std::f64::consts::PI.ln();
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn loglik_studentt_finite_on_typical_inputs() {
        let owned = [
            ("mu", array![0.0, 1.0, 2.0]),
            ("sigma", array![1.0, 1.5, 0.5]),
            ("nu", array![5.0, 10.0, 4.0]),
        ];
        let p = params_view(&owned);
        let ll = StudentT.loglik(&array![0.5, 0.5, 1.5], &p).unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn variance_studentt_uses_sigma_sq_nu_over_nu_minus_two() {
        let owned = [
            ("mu", array![0.0]),
            ("sigma", array![1.0]),
            ("nu", array![4.0]),
        ];
        let p = params_view(&owned);
        // σ²·ν/(ν−2) = 1·4/2 = 2.
        let v = StudentT.variance(&p).unwrap();
        assert!((v[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn variance_studentt_clamps_at_low_nu() {
        // ν ≤ 2 is undefined; clamp keeps the value finite for downstream Pearson math.
        let owned = [("sigma", array![1.0]), ("nu", array![1.5])];
        let p = params_view(&owned);
        let v = StudentT.variance(&p).unwrap();
        assert!(v[0].is_finite());
        assert!(v[0] > 0.0);
    }

    #[test]
    fn initial_value_is_robust_to_outliers() {
        // A clean core around 10 with a few gross outliers. The non-robust trait
        // default (mean/SD) would be dragged toward the outliers; median/MAD resist.
        let y = array![9.8, 10.1, 9.9, 10.2, 10.0, 9.7, 10.3, 9.95, 10.05, 1000.0, -800.0];
        let mu0 = StudentT.initial_value("mu", &y);
        assert!(
            (mu0 - 10.0).abs() < 0.5,
            "median seed should sit near the core (got {mu0})"
        );
        let sigma0 = StudentT.initial_value("sigma", &y);
        assert!(
            sigma0 > 0.0 && sigma0 < 2.0,
            "MAD-based scale seed should reflect the core spread, not the outliers (got {sigma0})"
        );
        let nu0 = StudentT.initial_value("nu", &y);
        assert_eq!(
            nu0, NU_INIT,
            "ν seed is a fixed moderate default, not derived from the (outlier-sensitive) kurtosis"
        );
    }

    #[test]
    fn nu_link_floors_below_two() {
        // The floored log link must keep ν ≥ 2 regardless of how negative η drifts,
        // so the variance σ²ν/(ν−2) stays finite during iteration.
        let link = StudentT.default_link("nu").unwrap();
        assert!(link.inv_link(-50.0) >= NU_FLOOR - 1e-12);
        assert!(link.inv_link(-1.0) >= NU_FLOOR - 1e-12);
        // Above the floor it behaves like a plain log link.
        assert!((link.inv_link(2.0_f64.ln()) - 2.0).abs() < 1e-9);
        assert!((link.inv_link(10.0_f64.ln()) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn score_matches_finite_diff_studentt() {
        let y = array![-1.0, 0.5, 2.0];
        let owned = [
            ("mu", array![0.0, 0.5, 1.0]),
            ("sigma", array![1.0, 1.2, 0.8]),
            ("nu", array![5.0, 8.0, 4.0]),
        ];
        check_score_via_finite_diff(&StudentT, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&StudentT, &y, &owned, "sigma", 1e-5);
        check_score_via_finite_diff(&StudentT, &y, &owned, "nu", 1e-5);
    }

    #[test]
    fn cdf_quantile_roundtrip_studentt() {
        let y = array![-3.0, -0.5, 0.0, 1.2, 4.0];
        let owned = [
            ("mu", array![0.0, 0.5, 1.0, 2.0, 1.5]),
            ("sigma", array![1.0, 1.5, 0.8, 2.0, 1.2]),
            ("nu", array![5.0, 10.0, 4.0, 8.0, 30.0]),
        ];
        check_cdf_quantile_roundtrip(&StudentT, &y, &owned, 1e-6);
        check_cdf_pdf_consistency(&StudentT, &y, &owned, 1e-4, 1e-3);
    }

    #[test]
    fn cdf_monotone_studentt_and_median_is_mu() {
        let grid = Array1::from_iter((0..60).map(|i| -8.0 + i as f64 * 0.25));
        let owned = [
            ("mu", array![1.0]),
            ("sigma", array![1.3]),
            ("nu", array![6.0]),
        ];
        check_cdf_monotone_in_unit(&StudentT, &grid, &owned);
        let p = params_view(&owned);
        let med = StudentT.quantile(&array![0.5], &p).unwrap();
        assert!((med[0] - 1.0).abs() < 1e-7);
    }
}
