//! Box-Cox-t (BCT) distribution for skew, heavy-tailed positive continuous data.
//!
//! BCT extends [`BCCG`](super::BCCG) with a fourth parameter `τ > 0` (degrees of
//! freedom): the standardized Box-Cox residual `z` follows a Student-`t` with `τ`
//! df instead of a standard normal, adding heavy tails to the skew-positive fit.
//! As `τ → ∞` the `t` approaches the normal and BCT reduces to BCCG.
//!
//! It shares the Box-Cox spine ([`super::boxcox`]) with BCCG and BCPE; only the
//! distribution `z` follows (here Student-`t`) and the extra `τ` column differ. The
//! `τ` score/Fisher pair mirror [`StudentT`](super::StudentT)'s df parameter.

use super::boxcox::{
    boxcox_cv_variance, boxcox_expected_value, boxcox_inv, boxcox_seed, boxcox_z, boxcox_z_dz_dnu,
};
use super::{
    clamp_prob, require, DerivativesResult, Distribution, GamlssError, IdentityLink, Link, LogLink,
    MIN_POSITIVE, MIN_WEIGHT,
};
use crate::math::{digamma, trigamma};
use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Starting value for `τ`: a moderately heavy tail, well clear of the `τ = 0`
/// boundary where the `t` degenerates.
const TAU_INIT: f64 = 10.0;

/// Box-Cox-t distribution for skew, heavy-tailed positive continuous data (`y > 0`).
///
/// Parameters: `μ` (median, log link), `σ` (≈ CV, log link), `ν` (skewness,
/// identity link), `τ` (degrees of freedom, log link). `τ → ∞` recovers BCCG.
#[derive(Debug, Clone, Copy, Default)]
pub struct BCT;

impl BCT {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for BCT {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma", "nu", "tau"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(LogLink)),
            "sigma" => Ok(Box::new(LogLink)),
            "nu" => Ok(Box::new(IdentityLink)),
            "tau" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    /// Robust seeds: `μ₀ = median(y)`, `σ₀` = robust CV, `ν₀ = 1` (symmetric), and
    /// `τ₀` a fixed moderate df (see [`TAU_INIT`]). A fixed `τ` seed is preferred to
    /// a kurtosis estimate for the same reason as [`StudentT`](super::StudentT).
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        boxcox_seed(param, y).unwrap_or_else(|| {
            if param != "tau" {
                debug_assert!(false, "BCT has no parameter '{param}'");
            }
            TAU_INIT
        })
    }

    eta_derivatives_passthrough!();

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Box-Cox spine (z, ∂z/∂ν) shared with BCCG; the `t` robustifying weight
        // w_t = (τ+1)/(τ+z²) downweights outliers and → 1 as τ → ∞ (→ BCCG). Full
        // derivation in docs/math/mathematics.md [BCCG]. η-scale scores:
        //   u_μ = w_t·z·T/σ − ν   (T = (y/μ)^ν = 1+νσz)
        //   u_σ = w_t·z² − 1
        //   u_ν = −w_t·z·∂z/∂ν + log(y/μ)
        //   u_τ = τ·½[ψ((τ+1)/2) − ψ(τ/2) − ln(1+z²/τ) + (w_t·z²−1)/τ]
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let tau = require(self, params, "tau")?;
        let n = y.len();

        let mut u_mu = Array1::<f64>::zeros(n);
        let mut w_mu = Array1::<f64>::zeros(n);
        let mut u_sigma = Array1::<f64>::zeros(n);
        let mut w_sigma = Array1::<f64>::zeros(n);
        let mut u_nu = Array1::<f64>::zeros(n);
        let mut w_nu = Array1::<f64>::zeros(n);
        let mut u_tau = Array1::<f64>::zeros(n);
        let mut w_tau = Array1::<f64>::zeros(n);

        for i in 0..n {
            let m = mu[i].max(MIN_POSITIVE);
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i];
            let t = tau[i].max(MIN_POSITIVE);
            let yi = y[i].max(MIN_POSITIVE);

            let (z, dz_dnu, l) = boxcox_z_dz_dnu(yi, m, s, nu_i);
            let z2 = z * z;
            let w_t = (t + 1.0) / (t + z2); // t robustifying weight
            let big_t = 1.0 + nu_i * s * z; // T = (y/μ)^ν

            u_mu[i] = w_t * z * big_t / s - nu_i;
            u_sigma[i] = w_t * z2 - 1.0;
            u_nu[i] = -w_t * z * dz_dnu + l;

            // τ score (log link): the same digamma form as StudentT's df parameter.
            let dl_dtau = 0.5
                * (digamma((t + 1.0) / 2.0) - digamma(t / 2.0) - (1.0 + z2 / t).ln()
                    + (w_t * z2 - 1.0) / t);
            u_tau[i] = t * dl_dtau;

            // Expected Fisher information (η-scale), each reducing to BCCG at τ → ∞:
            //   w_μ = (τ+1)/((τ+3)σ²) + 2ν²,  w_σ = 2τ/(τ+3),  w_ν = (7σ²/4)(τ+1)/(τ+3).
            let shrink = (t + 1.0) / (t + 3.0);
            w_mu[i] = shrink / (s * s) + 2.0 * nu_i * nu_i;
            w_sigma[i] = 2.0 * t / (t + 3.0);
            w_nu[i] = (7.0 * s * s / 4.0) * shrink;
            // τ information mirrors StudentT: i_τ·τ², → 0 as τ → ∞ (df unidentifiable
            // for a normal), floored to keep W positive definite.
            let i_tau = 0.25
                * (trigamma(t / 2.0) - trigamma((t + 1.0) / 2.0)
                    + 2.0 * (t + 3.0) / (t * (t + 1.0)));
            w_tau[i] = (i_tau * t * t).abs().max(MIN_WEIGHT);
        }

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
            ("nu".to_string(), (u_nu, w_nu)),
            ("tau".to_string(), (u_tau, w_tau)),
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
        let tau = require(self, params, "tau")?;
        let n = y.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mu[i].max(MIN_POSITIVE);
            let s = sigma[i].max(MIN_POSITIVE);
            let t = tau[i].max(MIN_POSITIVE);
            let yi = y[i].max(MIN_POSITIVE);
            let z = boxcox_z(yi, m, s, nu[i]);
            // log h(z) = Student-t density; plus the Box-Cox Jacobian terms.
            let log_h = ln_gamma((t + 1.0) / 2.0)
                - ln_gamma(t / 2.0)
                - 0.5 * (PI * t).ln()
                - 0.5 * (t + 1.0) * (1.0 + z * z / t).ln();
            out[i] = log_h + (nu[i] - 1.0) * yi.ln() - nu[i] * m.ln() - s.ln();
        }
        Ok(out)
    }

    /// `Var(Y) ≈ (σμ)²·τ/(τ−2)` for `τ > 2` — the BCCG CV approximation inflated by
    /// the `t` variance factor. Clamped (denominator floored) so it stays finite for
    /// `τ ≤ 2`; used only for Pearson residuals.
    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let tau = require(self, params, "tau")?;
        let cv2 = boxcox_cv_variance(mu, sigma);
        let n = mu.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let infl = tau[i] / (tau[i] - 2.0).max(MIN_POSITIVE);
            out[i] = cv2[i] * infl;
        }
        Ok(out)
    }

    /// `μ` is the median; the second-order mean approximation matches BCCG (the `t`
    /// is symmetric, so the leading skew correction is unchanged).
    fn expected_value(
        &self,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        Ok(boxcox_expected_value(mu, sigma, nu))
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // F(y) = T_τ(z), the standard Student-t CDF of the Box-Cox z-score.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let tau = require(self, params, "tau")?;
        let n = y.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            if y[i] <= 0.0 {
                continue; // support is y > 0
            }
            let m = mu[i].max(MIN_POSITIVE);
            let s = sigma[i].max(MIN_POSITIVE);
            let t = tau[i].max(MIN_POSITIVE);
            let z = boxcox_z(y[i], m, s, nu[i]);
            out[i] = StudentsT::new(0.0, 1.0, t)
                .expect("valid StudentsT df")
                .cdf(z);
        }
        Ok(out)
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // z_p = T_τ⁻¹(p), then invert the Box-Cox transform.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let tau = require(self, params, "tau")?;
        let n = p.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mu[i].max(MIN_POSITIVE);
            let s = sigma[i].max(MIN_POSITIVE);
            let t = tau[i].max(MIN_POSITIVE);
            let zp = StudentsT::new(0.0, 1.0, t)
                .expect("valid StudentsT df")
                .inverse_cdf(clamp_prob(p[i]));
            out[i] = boxcox_inv(m, s, nu[i], zp);
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "BCT"
    }
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
    fn bct_derivative_keys_match_parameters() {
        let y = array![0.5, 1.5, 3.0, 7.0];
        let mu = array![1.0, 2.0, 4.0, 6.0];
        let sigma = array![0.3, 0.25, 0.2, 0.35];
        let nu = array![1.0, 0.5, -0.5, 1.5];
        let tau = array![8.0, 5.0, 12.0, 20.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        p.insert("nu", &nu);
        p.insert("tau", &tau);
        derivative_keys_match_parameters(&BCT, p, &y);
    }

    #[test]
    fn score_matches_finite_diff_bct() {
        let y = array![1.0, 2.5, 5.0, 0.8, 3.0];
        let owned = [
            ("mu", array![1.5, 2.0, 4.0, 1.0, 2.5]),
            ("sigma", array![0.3, 0.25, 0.2, 0.4, 0.3]),
            ("nu", array![1.0, 0.5, 1.5, -0.5, 1e-8]),
            ("tau", array![6.0, 8.0, 5.0, 12.0, 10.0]),
        ];
        check_score_via_finite_diff(&BCT, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&BCT, &y, &owned, "sigma", 1e-5);
        check_score_via_finite_diff(&BCT, &y, &owned, "nu", 1e-5);
        check_score_via_finite_diff(&BCT, &y, &owned, "tau", 1e-5);
    }

    #[test]
    fn cdf_quantile_roundtrip_bct() {
        let y = array![0.6, 1.5, 3.0, 7.0, 2.0];
        let owned = [
            ("mu", array![1.0, 2.0, 4.0, 6.0, 2.5]),
            ("sigma", array![0.3, 0.25, 0.2, 0.35, 0.3]),
            ("nu", array![1.0, 0.5, -0.5, 1.5, 0.0]),
            ("tau", array![8.0, 6.0, 12.0, 20.0, 7.0]),
        ];
        check_cdf_quantile_roundtrip(&BCT, &y, &owned, 1e-6);
        check_cdf_pdf_consistency(&BCT, &y, &owned, 1e-5, 1e-3);
    }

    #[test]
    fn cdf_monotone_bct() {
        let grid = Array1::from_iter((0..80).map(|i| 0.05 + i as f64 * 0.1));
        let owned = [
            ("mu", array![3.0]),
            ("sigma", array![0.3]),
            ("nu", array![0.8]),
            ("tau", array![6.0]),
        ];
        check_cdf_monotone_in_unit(&BCT, &grid, &owned);
    }

    #[test]
    fn loglik_bct_approaches_bccg_for_large_tau() {
        // As τ → ∞ the t → normal, so BCT log-density should approach BCCG's.
        use crate::distributions::BCCG;
        let owned_bct = [
            ("mu", array![2.0]),
            ("sigma", array![0.3]),
            ("nu", array![0.5]),
            ("tau", array![1e6]),
        ];
        let owned_bccg = [
            ("mu", array![2.0]),
            ("sigma", array![0.3]),
            ("nu", array![0.5]),
        ];
        let y = array![2.7];
        let ll_bct = BCT.loglik(&y, &params_view(&owned_bct)).unwrap();
        let ll_bccg = BCCG.loglik(&y, &params_view(&owned_bccg)).unwrap();
        assert!(
            (ll_bct - ll_bccg).abs() < 1e-3,
            "BCT(τ=1e6) {ll_bct} should approach BCCG {ll_bccg}"
        );
    }

    #[test]
    fn median_quantile_is_mu() {
        let owned = [
            ("mu", array![2.0, 5.0]),
            ("sigma", array![0.3, 0.2]),
            ("nu", array![0.5, -1.0]),
            ("tau", array![6.0, 10.0]),
        ];
        let p = params_view(&owned);
        let med = BCT.quantile(&array![0.5, 0.5], &p).unwrap();
        assert!((med[0] - 2.0).abs() < 1e-9);
        assert!((med[1] - 5.0).abs() < 1e-9);
    }
}
