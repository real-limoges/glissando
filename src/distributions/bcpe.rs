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
//! It shares the Box-Cox spine ([`super::boxcox`]) with BCCG and BCT.

use super::boxcox::{
    boxcox_cv_variance, boxcox_expected_value, boxcox_inv, boxcox_seed, boxcox_z, boxcox_z_dz_dnu,
};
use super::{
    clamp_prob, require, DerivativeMap, Distribution, GamlssError, IdentityLink, Link, LogLink,
    Natural, ScoreInfo, DENOM_FLOOR, MIN_POSITIVE,
};
use crate::math::{digamma, trigamma};
use crate::Param;
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
    fn parameters(&self) -> &[Param] {
        &[Param::Mu, Param::Sigma, Param::Nu, Param::Tau]
    }

    fn default_link(&self, param: Param) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            Param::Mu => Ok(Box::new(LogLink)),
            Param::Sigma => Ok(Box::new(LogLink)),
            Param::Nu => Ok(Box::new(IdentityLink)),
            Param::Tau => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    /// Robust seeds: `μ₀ = median(y)`, `σ₀` = robust CV, `ν₀ = 1` (symmetric),
    /// `τ₀ = 2` (start at the normal / BCCG).
    fn initial_value(&self, param: Param, y: &Array1<f64>) -> f64 {
        boxcox_seed(param, y).unwrap_or_else(|| {
            if param != Param::Tau {
                debug_assert!(false, "BCPE has no parameter '{param}'");
            }
            TAU_INIT
        })
    }

    eta_derivatives_via_chain!();

    fn theta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<DerivativeMap<Natural>, GamlssError> {
        // Box-Cox spine (z, ∂z/∂ν) shared with BCCG. The PE score swaps out the
        // normal's −z. With a = z/c, gₜ = |a|^τ, and D = (τ/2c)|a|^{τ−1}sign(z)
        // (= z at τ=2). Natural scale; chain_to_eta reapplies the default links (log, log,
        // identity, log):
        //   dl/dμ = [D·T/σ − ν] / μ   (T = (y/μ)^ν = 1+νσz)
        //   dl/dσ = [(τ/2)gₜ − 1] / σ   (numerator = z·D − 1)
        //   dl/dν = −D·∂z/∂ν + log(y/μ)
        //   dl/dτ = N'(τ) − gₜ·log gₜ /(2τ) + ½gₜ·B(τ)
        let mu = require(self, params, Param::Mu)?;
        let sigma = require(self, params, Param::Sigma)?;
        let nu = require(self, params, Param::Nu)?;
        let tau = require(self, params, Param::Tau)?;
        let n = y.len();

        let mut u_mu = Array1::<f64>::zeros(n);
        let mut i_mu = Array1::<f64>::zeros(n);
        let mut u_sigma = Array1::<f64>::zeros(n);
        let mut i_sigma = Array1::<f64>::zeros(n);
        let mut u_nu = Array1::<f64>::zeros(n);
        let mut i_nu = Array1::<f64>::zeros(n);
        let mut u_tau = Array1::<f64>::zeros(n);
        let mut i_tau = Array1::<f64>::zeros(n);

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

            // Guard each reciprocal at the power it is used at. Raising an
            // already-guarded reciprocal to a power overflows to infinity for a
            // parameter the log link can still underflow to, and inf · 0 is NaN.
            let inv_m = 1.0 / m.max(DENOM_FLOOR);
            let inv_m_sq = 1.0 / (m * m).max(DENOM_FLOOR);
            let inv_s = 1.0 / s.max(DENOM_FLOOR);
            let inv_s_sq = 1.0 / (s * s).max(DENOM_FLOOR);

            u_mu[i] = (d_score * big_t / s - nu_i) * inv_m;
            u_sigma[i] = (0.5 * t * gt - 1.0) * inv_s;
            u_nu[i] = -d_score * dz_dnu + l;

            // τ score: derivative of N(τ) − ½gₜ w.r.t. τ (z fixed). a = 1/τ.
            let a = 1.0 / t;
            let psi_a = digamma(a);
            let psi_3a = digamma(3.0 * a);
            let n_prime = 1.0 / t + (3.0 / (2.0 * t * t)) * (psi_a - psi_3a);
            let b_coef = LN_2 / t - psi_a / (2.0 * t) + 3.0 * psi_3a / (2.0 * t);
            let gt_ln_gt = if z == 0.0 { 0.0 } else { gt * gt.ln() };
            // This was already separable, so converting it meant deleting a trailing `t *`.
            u_tau[i] = n_prime - gt_ln_gt / (2.0 * t) + 0.5 * gt * b_coef;

            // Expected Fisher information. I_loc(τ) = E[D²] is the location info
            // (unit scale); all reduce to BCCG (=normal) at τ = 2.
            let gamma_ratio = (ln_gamma(2.0 - a) - ln_gamma(a)).exp();
            let i_loc = (t * t / (4.0 * c * c)) * 2.0_f64.powf(2.0 - 2.0 / t) * gamma_ratio;
            i_mu[i] = (i_loc * inv_s_sq + 2.0 * nu_i * nu_i) * inv_m_sq;
            i_sigma[i] = t * inv_s_sq; // E[(σ·∂l/∂σ)²] = τ
            i_nu[i] = (7.0 * s * s / 4.0) * i_loc;

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
            i_tau[i] = e_dldt2;
        }

        Ok(HashMap::from([
            (Param::Mu, ScoreInfo::new(u_mu, i_mu)),
            (Param::Sigma, ScoreInfo::new(u_sigma, i_sigma)),
            (Param::Nu, ScoreInfo::new(u_nu, i_nu)),
            (Param::Tau, ScoreInfo::new(u_tau, i_tau)),
        ]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, Param::Mu)?;
        let sigma = require(self, params, Param::Sigma)?;
        let nu = require(self, params, Param::Nu)?;
        let tau = require(self, params, Param::Tau)?;
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

    /// `Var(Y) ≈ (σμ)²`; `σ` is (approximately) the CV, thanks to the variance-1
    /// standardization of the PE. Used only for Pearson residuals.
    fn variance(&self, params: &HashMap<Param, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, Param::Mu)?;
        let sigma = require(self, params, Param::Sigma)?;
        Ok(boxcox_cv_variance(mu, sigma))
    }

    /// `μ` is the median; the second-order mean approximation matches BCCG (the PE
    /// is symmetric, so the leading skew correction is unchanged).
    fn expected_value(
        &self,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, Param::Mu)?;
        let sigma = require(self, params, Param::Sigma)?;
        let nu = require(self, params, Param::Nu)?;
        Ok(boxcox_expected_value(mu, sigma, nu))
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // F(y) = ½ + ½·sign(z)·P(1/τ, ½|z/c|^τ), the power-exponential CDF of z,
        // where P is the regularized lower incomplete gamma.
        let mu = require(self, params, Param::Mu)?;
        let sigma = require(self, params, Param::Sigma)?;
        let nu = require(self, params, Param::Nu)?;
        let tau = require(self, params, Param::Tau)?;
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
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // Invert the PE CDF for z, then invert the Box-Cox transform. For p ≥ ½:
        // s = P⁻¹(1/τ, 2p−1) (a Gamma(1/τ, 1) quantile), z = c·(2s)^{1/τ}.
        let mu = require(self, params, Param::Mu)?;
        let sigma = require(self, params, Param::Sigma)?;
        let nu = require(self, params, Param::Nu)?;
        let tau = require(self, params, Param::Tau)?;
        let n = p.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mu[i].max(MIN_POSITIVE);
            let s = sigma[i].max(MIN_POSITIVE);
            let t = tau[i].max(MIN_POSITIVE);
            let c = pe_c(t);
            let pi = clamp_prob(p[i]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::test_helpers::{
        check_cdf_monotone_in_unit, check_cdf_pdf_consistency, check_cdf_quantile_roundtrip,
        check_eta_score_via_finite_diff, check_score_via_finite_diff, default_link_derivatives,
        derivative_keys_match_parameters, finite_array, no_nan_array, params_view,
    };
    use crate::distributions::{LogLink, SqrtLink};
    use ndarray::array;

    #[test]
    fn bcpe_derivative_keys_match_parameters() {
        let y = array![0.5, 1.5, 3.0, 7.0];
        let mu = array![1.0, 2.0, 4.0, 6.0];
        let sigma = array![0.3, 0.25, 0.2, 0.35];
        let nu = array![1.0, 0.5, -0.5, 1.5];
        let tau = array![2.0, 1.5, 3.0, 2.5];
        let mut p = HashMap::new();
        p.insert(Param::Mu, &mu);
        p.insert(Param::Sigma, &sigma);
        p.insert(Param::Nu, &nu);
        p.insert(Param::Tau, &tau);
        derivative_keys_match_parameters(&BCPE, p, &y);
    }

    #[test]
    fn score_matches_finite_diff_bcpe() {
        // Spans leptokurtic (τ<2), normal (τ=2), and platykurtic (τ>2), plus skew
        // and the ν≈0 limit, so every score branch is exercised.
        let y = array![1.0, 2.5, 5.0, 0.8, 3.0];
        let owned = [
            (Param::Mu, array![1.5, 2.0, 4.0, 1.0, 2.5]),
            (Param::Sigma, array![0.3, 0.25, 0.2, 0.4, 0.3]),
            (Param::Nu, array![1.0, 0.5, 1.5, -0.5, 1e-8]),
            (Param::Tau, array![2.0, 1.5, 3.0, 2.5, 1.8]),
        ];
        check_score_via_finite_diff(&BCPE, &y, &owned, Param::Mu, 1e-5);
        check_score_via_finite_diff(&BCPE, &y, &owned, Param::Sigma, 1e-5);
        check_score_via_finite_diff(&BCPE, &y, &owned, Param::Nu, 1e-5);
        check_score_via_finite_diff(&BCPE, &y, &owned, Param::Tau, 1e-4);
    }

    #[test]
    fn score_matches_finite_diff_under_non_default_links() {
        // μ, σ (and τ) default to log, ν to identity, so
        // only a non-default link can tell a natural-scale score from an η-scale
        // one. ν is held positive here so a log link is well defined on it; the
        // default-link test above covers the negative and near-zero branches.
        let y = array![1.0, 2.5, 5.0, 0.8, 3.0];
        let owned = [
            (Param::Mu, array![1.5, 2.0, 4.0, 1.0, 2.5]),
            (Param::Sigma, array![0.3, 0.25, 0.2, 0.4, 0.3]),
            (Param::Nu, array![1.0, 0.5, 1.5, 0.75, 2.0]),
            (Param::Tau, array![2.0, 1.5, 3.0, 2.5, 1.8]),
        ];
        check_eta_score_via_finite_diff(&BCPE, &y, &owned, Param::Mu, &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&BCPE, &y, &owned, Param::Sigma, &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&BCPE, &y, &owned, Param::Nu, &LogLink, 1e-5);
        check_eta_score_via_finite_diff(&BCPE, &y, &owned, Param::Tau, &SqrtLink, 1e-4);
    }

    #[test]
    fn derivatives_stay_finite_at_saturated_parameters() {
        // The natural scores carry `1/μ`, `1/μ²`, `1/σ` and `1/σ²` that the η-scale
        // forms cancel.
        let y = array![1.0, 2.0, 3.0];
        let owned = [
            (Param::Mu, array![0.0, 1e-320, 1e-8]),
            (Param::Sigma, array![1e-8, 0.0, 1e-320]),
            (Param::Nu, array![1.0, 0.5, -0.5]),
            // τ stays clear of 0.5, where the pre-existing `i_loc` normalizer hits
            // `ln_gamma(2 − 1/τ) = ln_gamma(0) = ∞`. That singularity is
            // not what this test is checking.
            (Param::Tau, array![1.5, 2.0, 3.0]),
        ];
        let p = params_view(&owned);
        let natural = BCPE.theta_derivatives(&y, &p).unwrap();
        let chained = default_link_derivatives(&BCPE, &y, &p).unwrap();
        for name in [Param::Mu, Param::Sigma, Param::Nu, Param::Tau] {
            let (u_n, i_n) = (&natural[&name].score, &natural[&name].info);
            assert!(no_nan_array(u_n) && no_nan_array(i_n), "natural {name}");
            let (u, w) = (&chained[&name].score, &chained[&name].info);
            assert!(finite_array(u) && finite_array(w), "chained {name}: {u:?}");
        }
    }

    #[test]
    fn cdf_quantile_roundtrip_bcpe() {
        let y = array![0.6, 1.5, 3.0, 7.0, 2.0];
        let owned = [
            (Param::Mu, array![1.0, 2.0, 4.0, 6.0, 2.5]),
            (Param::Sigma, array![0.3, 0.25, 0.2, 0.35, 0.3]),
            (Param::Nu, array![1.0, 0.5, -0.5, 1.5, 0.0]),
            (Param::Tau, array![2.0, 1.5, 3.0, 2.5, 1.8]),
        ];
        check_cdf_quantile_roundtrip(&BCPE, &y, &owned, 1e-5);
        check_cdf_pdf_consistency(&BCPE, &y, &owned, 1e-5, 1e-3);
    }

    #[test]
    fn cdf_monotone_bcpe() {
        let grid = Array1::from_iter((0..80).map(|i| 0.05 + i as f64 * 0.1));
        let owned = [
            (Param::Mu, array![3.0]),
            (Param::Sigma, array![0.3]),
            (Param::Nu, array![0.8]),
            (Param::Tau, array![1.6]),
        ];
        check_cdf_monotone_in_unit(&BCPE, &grid, &owned);
    }

    #[test]
    fn bcpe_reduces_to_bccg_at_tau_two() {
        // τ = 2 ⇒ power-exponential is the standard normal ⇒ BCPE = BCCG.
        use crate::distributions::BCCG;
        let owned_bcpe = [
            (Param::Mu, array![2.0, 3.0]),
            (Param::Sigma, array![0.3, 0.25]),
            (Param::Nu, array![0.5, -0.5]),
            (Param::Tau, array![2.0, 2.0]),
        ];
        let owned_bccg = [
            (Param::Mu, array![2.0, 3.0]),
            (Param::Sigma, array![0.3, 0.25]),
            (Param::Nu, array![0.5, -0.5]),
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
            (Param::Mu, array![2.0, 5.0]),
            (Param::Sigma, array![0.3, 0.2]),
            (Param::Nu, array![0.5, -1.0]),
            (Param::Tau, array![1.5, 3.0]),
        ];
        let p = params_view(&owned);
        let med = BCPE.quantile(&array![0.5, 0.5], &p).unwrap();
        assert!((med[0] - 2.0).abs() < 1e-9);
        assert!((med[1] - 5.0).abs() < 1e-9);
    }
}
