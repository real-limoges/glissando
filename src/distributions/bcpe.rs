//! Box-Cox power-exponential (BCPE) distribution for skew, variable-kurtosis
//! positive continuous data.
//!
//! BCPE extends [`BCCG`](super::BCCG) with a fourth parameter `τ > 0` (kurtosis):
//! the standardized Box-Cox residual `z` follows a **power-exponential** (a.k.a.
//! generalized normal / Subbotin) with shape `τ` instead of a standard normal.
//! `τ = 2` is the normal (so BCPE reduces to BCCG), `τ < 2` is leptokurtic
//! (heavier-than-normal peak/tails), `τ > 2` is platykurtic.
//!
//! The power-exponential is variance-standardized so `σ` keeps its CV meaning:
//! with `c² = 2^{-2/τ}\,Γ(1/τ)/Γ(3/τ)`,
//! `log h(z) = N(τ) − ½|z/c|^τ`, `N(τ) = log τ − log 2 − \tfrac{3}{2}\logΓ(1/τ) + \tfrac{1}{2}\logΓ(3/τ)`.
//! Shares the Box-Cox spine ([`super::boxcox`]) with BCCG and BCT.

use super::boxcox::{boxcox_inv, boxcox_z, boxcox_z_dz_dnu};
use super::{
    require, DerivativesResult, Distribution, GamlssError, IdentityLink, Link, LogLink,
    MIN_POSITIVE, MIN_WEIGHT,
};
use crate::math::{digamma, trigamma};
use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, Gamma as SGamma};
use statrs::function::gamma::{gamma_lr, ln_gamma};
use std::collections::HashMap;
use std::f64::consts::LN_2;

/// Starting value for `τ`: the normal (`τ = 2`), i.e. start at the BCCG identity
/// and let the kurtosis move from there.
const TAU_INIT: f64 = 2.0;

/// Box-Cox power-exponential distribution for skew, variable-kurtosis positive data.
///
/// Parameters: `μ` (median, log link), `σ` (≈ CV, log link), `ν` (skewness,
/// identity link), `τ` (kurtosis / PE shape, log link). `τ = 2` recovers BCCG.
#[derive(Debug, Clone, Copy, Default)]
pub struct BCPE;

impl BCPE {
    pub fn new() -> Self {
        Self
    }
}

/// Normalizing-constant exponent `c` of the standardized power-exponential, where
/// `c² = 2^{-2/τ}·Γ(1/τ)/Γ(3/τ)` (makes `Var(z) = 1`).
#[inline]
fn pe_c(tau: f64) -> f64 {
    (-(LN_2) / tau + 0.5 * ln_gamma(1.0 / tau) - 0.5 * ln_gamma(3.0 / tau)).exp()
}

/// Log normalizing constant `N(τ) = log τ − log 2 − \tfrac32\logΓ(1/τ) + \tfrac12\logΓ(3/τ)`,
/// so `log h(z) = N(τ) − ½|z/c|^τ`.
#[inline]
fn pe_log_norm(tau: f64) -> f64 {
    tau.ln() - LN_2 - 1.5 * ln_gamma(1.0 / tau) + 0.5 * ln_gamma(3.0 / tau)
}

impl Distribution for BCPE {
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

    /// Robust seeds: `μ₀ = median(y)`, `σ₀` = robust CV, `ν₀ = 1` (symmetric),
    /// `τ₀ = 2` (start at the normal / BCCG).
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => median(y),
            "sigma" => {
                let med = median(y);
                let cv = 1.4826 * median_abs_deviation(y) / med.abs().max(MIN_POSITIVE);
                cv.clamp(0.01, 10.0)
            }
            "nu" => 1.0,
            "tau" => TAU_INIT,
            other => {
                debug_assert!(
                    matches!(other, "mu" | "sigma" | "nu" | "tau"),
                    "BCPE has no parameter '{other}'"
                );
                TAU_INIT
            }
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Box-Cox spine (z, ∂z/∂ν) shared with BCCG; the PE score replaces the
        // normal's −z. With a = z/c, gₜ = |a|^τ, and D = (τ/2c)|a|^{τ−1}sign(z)
        // (= z at τ=2), full derivation in docs/math/mathematics.md §1.9:
        //   u_μ = D·T/σ − ν   (T = (y/μ)^ν = 1+νσz)
        //   u_σ = (τ/2)gₜ − 1   (= z·D − 1)
        //   u_ν = −D·∂z/∂ν + log(y/μ)
        //   u_τ = τ[N'(τ) − gₜ·log gₜ /(2τ) + ½gₜ·B(τ)]
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
            let c = pe_c(t);
            let aa = z / c;
            let abs_a = aa.abs();
            let gt = abs_a.powf(t); // |z/c|^τ
            let big_t = 1.0 + nu_i * s * z; // T = (y/μ)^ν
                                            // D = −dl/dz; 0 at the mode (z=0) for τ ≥ 1.
            let d_score = if z == 0.0 {
                0.0
            } else {
                (t / (2.0 * c)) * abs_a.powf(t - 1.0) * z.signum()
            };

            u_mu[i] = d_score * big_t / s - nu_i;
            u_sigma[i] = 0.5 * t * gt - 1.0;
            u_nu[i] = -d_score * dz_dnu + l;

            // τ score: derivative of N(τ) − ½gₜ w.r.t. τ (z fixed). a = 1/τ.
            let a = 1.0 / t;
            let psi_a = digamma(a);
            let psi_3a = digamma(3.0 * a);
            let n_prime = 1.0 / t + (3.0 / (2.0 * t * t)) * (psi_a - psi_3a);
            let b_coef = LN_2 / t - psi_a / (2.0 * t) + 3.0 * psi_3a / (2.0 * t);
            let gt_ln_gt = if z == 0.0 { 0.0 } else { gt * gt.ln() };
            let dl_dtau = n_prime - gt_ln_gt / (2.0 * t) + 0.5 * gt * b_coef;
            u_tau[i] = t * dl_dtau;

            // Expected Fisher information (η-scale). I_loc(τ) = E[D²] is the location
            // info (unit scale); all reduce to BCCG (=normal) at τ = 2.
            let gamma_ratio = (ln_gamma(2.0 - a) - ln_gamma(a)).exp();
            let i_loc = (t * t / (4.0 * c * c)) * 2.0_f64.powf(2.0 - 2.0 / t) * gamma_ratio;
            w_mu[i] = i_loc / (s * s) + 2.0 * nu_i * nu_i;
            w_sigma[i] = t; // E[u_σ²] = τ
            w_nu[i] = (7.0 * s * s / 4.0) * i_loc;

            // Exact τ information via Gamma(a, 1) moments of v = ½gₜ:
            //   ∂ℓ/∂τ = N' + P·v + Q·v·log v,  P = −ψ(a)/(2τ) + 3ψ(3a)/(2τ),  Q = −1/τ.
            let p_coef = -psi_a / (2.0 * t) + 3.0 * psi_3a / (2.0 * t);
            let q_coef = -1.0 / t;
            let ev = a;
            let ev2 = a * (a + 1.0);
            let psi_a1 = digamma(a + 1.0);
            let psi_a2 = digamma(a + 2.0);
            let tri_a2 = trigamma(a + 2.0);
            let evlnv = a * psi_a1;
            let ev2lnv = a * (a + 1.0) * psi_a2;
            let ev2ln2v = a * (a + 1.0) * (psi_a2 * psi_a2 + tri_a2);
            let e_dldt2 = n_prime * n_prime
                + p_coef * p_coef * ev2
                + q_coef * q_coef * ev2ln2v
                + 2.0 * n_prime * p_coef * ev
                + 2.0 * n_prime * q_coef * evlnv
                + 2.0 * p_coef * q_coef * ev2lnv;
            w_tau[i] = (t * t * e_dldt2).max(MIN_WEIGHT);
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
            let c = pe_c(t);
            let gt = (z / c).abs().powf(t);
            // log h(z) = N(τ) − ½|z/c|^τ; plus the Box-Cox Jacobian terms.
            out[i] = pe_log_norm(t) - 0.5 * gt + (nu[i] - 1.0) * yi.ln() - nu[i] * m.ln() - s.ln();
        }
        Ok(out)
    }

    /// `Var(Y) ≈ (σμ)²` — `σ` is (approximately) the CV by the variance-1
    /// standardization of the PE. Used only for Pearson residuals.
    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(crate::math::par_zip_map(mu, sigma, |m, s| {
            (s * m) * (s * m)
        }))
    }

    /// `μ` is the median; the second-order mean approximation matches BCCG (the PE
    /// is symmetric, so the leading skew correction is unchanged).
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
        // F(y) = ½ + ½·sign(z)·P(1/τ, ½|z/c|^τ), the power-exponential CDF of z,
        // where P is the regularized lower incomplete gamma.
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
            let c = pe_c(t);
            let s_arg = 0.5 * (z / c).abs().powf(t);
            let half_p = 0.5 * gamma_lr(1.0 / t, s_arg);
            out[i] = if z >= 0.0 { 0.5 + half_p } else { 0.5 - half_p };
        }
        Ok(out)
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // Invert the PE CDF for z, then invert the Box-Cox transform. For p ≥ ½:
        // s = P⁻¹(1/τ, 2p−1) (a Gamma(1/τ, 1) quantile), z = c·(2s)^{1/τ}.
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
            let c = pe_c(t);
            let pi = p[i].clamp(1e-12, 1.0 - 1e-12);
            let zp = if pi == 0.5 {
                0.0
            } else {
                let (q, sign) = if pi > 0.5 {
                    (2.0 * pi - 1.0, 1.0)
                } else {
                    (1.0 - 2.0 * pi, -1.0)
                };
                let s_arg = SGamma::new(1.0 / t, 1.0)
                    .expect("valid Gamma shape")
                    .inverse_cdf(q);
                sign * c * (2.0 * s_arg).powf(1.0 / t)
            };
            out[i] = boxcox_inv(m, s, nu[i], zp);
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "BCPE"
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
    fn bcpe_derivative_keys_match_parameters() {
        let y = array![0.5, 1.5, 3.0, 7.0];
        let mu = array![1.0, 2.0, 4.0, 6.0];
        let sigma = array![0.3, 0.25, 0.2, 0.35];
        let nu = array![1.0, 0.5, -0.5, 1.5];
        let tau = array![2.0, 1.5, 3.0, 2.5];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        p.insert("nu", &nu);
        p.insert("tau", &tau);
        derivative_keys_match_parameters(&BCPE, p, &y);
    }

    #[test]
    fn score_matches_finite_diff_bcpe() {
        // Spans leptokurtic (τ<2), normal (τ=2), and platykurtic (τ>2), plus skew
        // and the ν≈0 limit, so every score branch is exercised.
        let y = array![1.0, 2.5, 5.0, 0.8, 3.0];
        let owned = [
            ("mu", array![1.5, 2.0, 4.0, 1.0, 2.5]),
            ("sigma", array![0.3, 0.25, 0.2, 0.4, 0.3]),
            ("nu", array![1.0, 0.5, 1.5, -0.5, 1e-8]),
            ("tau", array![2.0, 1.5, 3.0, 2.5, 1.8]),
        ];
        check_score_via_finite_diff(&BCPE, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&BCPE, &y, &owned, "sigma", 1e-5);
        check_score_via_finite_diff(&BCPE, &y, &owned, "nu", 1e-5);
        check_score_via_finite_diff(&BCPE, &y, &owned, "tau", 1e-4);
    }

    #[test]
    fn cdf_quantile_roundtrip_bcpe() {
        let y = array![0.6, 1.5, 3.0, 7.0, 2.0];
        let owned = [
            ("mu", array![1.0, 2.0, 4.0, 6.0, 2.5]),
            ("sigma", array![0.3, 0.25, 0.2, 0.35, 0.3]),
            ("nu", array![1.0, 0.5, -0.5, 1.5, 0.0]),
            ("tau", array![2.0, 1.5, 3.0, 2.5, 1.8]),
        ];
        check_cdf_quantile_roundtrip(&BCPE, &y, &owned, 1e-5);
        check_cdf_pdf_consistency(&BCPE, &y, &owned, 1e-5, 1e-3);
    }

    #[test]
    fn cdf_monotone_bcpe() {
        let grid = Array1::from_iter((0..80).map(|i| 0.05 + i as f64 * 0.1));
        let owned = [
            ("mu", array![3.0]),
            ("sigma", array![0.3]),
            ("nu", array![0.8]),
            ("tau", array![1.6]),
        ];
        check_cdf_monotone_in_unit(&BCPE, &grid, &owned);
    }

    #[test]
    fn bcpe_reduces_to_bccg_at_tau_two() {
        // τ = 2 ⇒ power-exponential is the standard normal ⇒ BCPE = BCCG.
        use crate::distributions::BCCG;
        let owned_bcpe = [
            ("mu", array![2.0, 3.0]),
            ("sigma", array![0.3, 0.25]),
            ("nu", array![0.5, -0.5]),
            ("tau", array![2.0, 2.0]),
        ];
        let owned_bccg = [
            ("mu", array![2.0, 3.0]),
            ("sigma", array![0.3, 0.25]),
            ("nu", array![0.5, -0.5]),
        ];
        let y = array![2.7, 2.4];
        let ll_bcpe = BCPE.loglik(&y, &params_view(&owned_bcpe)).unwrap();
        let ll_bccg = BCCG.loglik(&y, &params_view(&owned_bccg)).unwrap();
        assert!(
            (ll_bcpe - ll_bccg).abs() < 1e-9,
            "BCPE(τ=2) {ll_bcpe} should equal BCCG {ll_bccg}"
        );

        // CDF should match too.
        let cdf_bcpe = BCPE.cdf(&y, &params_view(&owned_bcpe)).unwrap();
        let cdf_bccg = BCCG.cdf(&y, &params_view(&owned_bccg)).unwrap();
        for i in 0..y.len() {
            assert!(
                (cdf_bcpe[i] - cdf_bccg[i]).abs() < 1e-9,
                "row {i}: BCPE cdf {} vs BCCG {}",
                cdf_bcpe[i],
                cdf_bccg[i]
            );
        }
    }

    #[test]
    fn median_quantile_is_mu() {
        let owned = [
            ("mu", array![2.0, 5.0]),
            ("sigma", array![0.3, 0.2]),
            ("nu", array![0.5, -1.0]),
            ("tau", array![1.5, 3.0]),
        ];
        let p = params_view(&owned);
        let med = BCPE.quantile(&array![0.5, 0.5], &p).unwrap();
        assert!((med[0] - 2.0).abs() < 1e-9);
        assert!((med[1] - 5.0).abs() < 1e-9);
    }
}
