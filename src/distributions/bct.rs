//! Box-Cox-t (BCT) distribution for skew, heavy-tailed positive continuous data.
//!
//! BCT extends [`BCCG`](super::BCCG) with a fourth parameter `τ > 0` (degrees of
//! freedom): the standardized Box-Cox residual `z` follows a Student-`t` with `τ`
//! df instead of a standard normal, adding heavy tails to the skew-positive fit.
//! As `τ → ∞` the `t` approaches the normal and BCT reduces to BCCG.
//!
//! It shares the Box-Cox spine ([`super::boxcox`]) with BCCG and BCPE; only the
//! distribution `z` follows (here Student-`t`) and the extra `τ` column differ. The
//! `τ` score/Fisher pair reuses [`StudentT`](super::StudentT)'s df parameter.

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

    /// Robust seeds: `μ₀ = median(y)`, `σ₀` = robust CV, `ν₀ = 1` (symmetric), and
    /// `τ₀` a fixed moderate df (see `TAU_INIT`). A fixed `τ` seed is used instead
    /// of a kurtosis estimate, for the same reason as [`StudentT`](super::StudentT).
    fn initial_value(&self, param: Param, y: &Array1<f64>) -> f64 {
        boxcox_seed(param, y).unwrap_or_else(|| {
            if param != Param::Tau {
                debug_assert!(false, "BCT has no parameter '{param}'");
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
        // Box-Cox spine (z, ∂z/∂ν) shared with BCCG. The `t` robustifying weight
        // w_t = (τ+1)/(τ+z²) downweights outliers and → 1 as τ → ∞ (→ BCCG).
        // Natural-scale scores; chain_to_eta reapplies the default links (log, log,
        // identity, log) and recovers the old η-scale values exactly:
        //   dl/dμ = [w_t·z·T/σ − ν] / μ   (T = (y/μ)^ν = 1+νσz)
        //   dl/dσ = [w_t·z² − 1] / σ
        //   dl/dν = −w_t·z·∂z/∂ν + log(y/μ)
        //   dl/dτ = ½[ψ((τ+1)/2) − ψ(τ/2) − ln(1+z²/τ) + (w_t·z²−1)/τ]
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
        let mut i_tau_out = Array1::<f64>::zeros(n);

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

            // Guard each reciprocal at the power it is used at. Raising an
            // already-guarded reciprocal to a power overflows to infinity for a
            // parameter the log link can still underflow to, and inf · 0 is NaN.
            let inv_m = 1.0 / m.max(DENOM_FLOOR);
            let inv_m_sq = 1.0 / (m * m).max(DENOM_FLOOR);
            let inv_s = 1.0 / s.max(DENOM_FLOOR);
            let inv_s_sq = 1.0 / (s * s).max(DENOM_FLOOR);

            u_mu[i] = (w_t * z * big_t / s - nu_i) * inv_m;
            u_sigma[i] = (w_t * z2 - 1.0) * inv_s;
            u_nu[i] = -w_t * z * dz_dnu + l;

            // τ score: the same digamma form as StudentT's df parameter. It was
            // already separable, so converting it meant deleting the trailing `t *`.
            u_tau[i] = 0.5
                * (digamma((t + 1.0) / 2.0) - digamma(t / 2.0) - (1.0 + z2 / t).ln()
                    + (w_t * z2 - 1.0) / t);

            // Expected Fisher information, each reducing to BCCG at τ → ∞:
            //   I_μμ = [(τ+1)/((τ+3)σ²) + 2ν²]/μ²,  I_σσ = [2τ/(τ+3)]/σ²,
            //   I_νν = (7σ²/4)(τ+1)/(τ+3).
            let shrink = (t + 1.0) / (t + 3.0);
            i_mu[i] = (shrink * inv_s_sq + 2.0 * nu_i * nu_i) * inv_m_sq;
            i_sigma[i] = (2.0 * t / (t + 3.0)) * inv_s_sq;
            i_nu[i] = (7.0 * s * s / 4.0) * shrink;
            // τ information mirrors StudentT and → 0 as τ → ∞ (df is unidentifiable
            // for a normal). The `.abs()` survives the un-fold: it used to wrap
            // `i_τ·τ²`, and τ² > 0, so `|i_τ·τ²| = |i_τ|·τ²`.
            i_tau_out[i] = (0.25
                * (trigamma(t / 2.0) - trigamma((t + 1.0) / 2.0)
                    + 2.0 * (t + 3.0) / (t * (t + 1.0))))
                .abs();
        }

        Ok(HashMap::from([
            (Param::Mu, ScoreInfo::new(u_mu, i_mu)),
            (Param::Sigma, ScoreInfo::new(u_sigma, i_sigma)),
            (Param::Nu, ScoreInfo::new(u_nu, i_nu)),
            (Param::Tau, ScoreInfo::new(u_tau, i_tau_out)),
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
            // log h(z) = Student-t density; plus the Box-Cox Jacobian terms.
            let log_h = ln_gamma((t + 1.0) / 2.0)
                - ln_gamma(t / 2.0)
                - 0.5 * (PI * t).ln()
                - 0.5 * (t + 1.0) * (1.0 + z * z / t).ln();
            out[i] = log_h + (nu[i] - 1.0) * yi.ln() - nu[i] * m.ln() - s.ln();
        }
        Ok(out)
    }

    /// `Var(Y) ≈ (σμ)²·τ/(τ−2)` for `τ > 2`: the BCCG CV approximation inflated by
    /// the `t` variance factor. The denominator is floored so it stays finite for
    /// `τ ≤ 2`. Used only for Pearson residuals.
    fn variance(&self, params: &HashMap<Param, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, Param::Mu)?;
        let sigma = require(self, params, Param::Sigma)?;
        let tau = require(self, params, Param::Tau)?;
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
        // F(y) = T_τ(z), the standard Student-t CDF of the Box-Cox z-score.
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
            out[i] = StudentsT::new(0.0, 1.0, t)
                .expect("valid StudentsT df")
                .cdf(z);
        }
        Ok(out)
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // z_p = T_τ⁻¹(p), then invert the Box-Cox transform.
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
        check_eta_score_via_finite_diff, check_score_via_finite_diff, default_link_derivatives,
        derivative_keys_match_parameters, finite_array, no_nan_array, params_view,
    };
    use crate::distributions::{LogLink, SqrtLink};
    use ndarray::array;

    #[test]
    fn bct_derivative_keys_match_parameters() {
        let y = array![0.5, 1.5, 3.0, 7.0];
        let mu = array![1.0, 2.0, 4.0, 6.0];
        let sigma = array![0.3, 0.25, 0.2, 0.35];
        let nu = array![1.0, 0.5, -0.5, 1.5];
        let tau = array![8.0, 5.0, 12.0, 20.0];
        let mut p = HashMap::new();
        p.insert(Param::Mu, &mu);
        p.insert(Param::Sigma, &sigma);
        p.insert(Param::Nu, &nu);
        p.insert(Param::Tau, &tau);
        derivative_keys_match_parameters(&BCT, p, &y);
    }

    #[test]
    fn score_matches_finite_diff_bct() {
        let y = array![1.0, 2.5, 5.0, 0.8, 3.0];
        let owned = [
            (Param::Mu, array![1.5, 2.0, 4.0, 1.0, 2.5]),
            (Param::Sigma, array![0.3, 0.25, 0.2, 0.4, 0.3]),
            (Param::Nu, array![1.0, 0.5, 1.5, -0.5, 1e-8]),
            (Param::Tau, array![6.0, 8.0, 5.0, 12.0, 10.0]),
        ];
        check_score_via_finite_diff(&BCT, &y, &owned, Param::Mu, 1e-5);
        check_score_via_finite_diff(&BCT, &y, &owned, Param::Sigma, 1e-5);
        check_score_via_finite_diff(&BCT, &y, &owned, Param::Nu, 1e-5);
        check_score_via_finite_diff(&BCT, &y, &owned, Param::Tau, 1e-5);
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
            (Param::Tau, array![6.0, 8.0, 5.0, 12.0, 10.0]),
        ];
        check_eta_score_via_finite_diff(&BCT, &y, &owned, Param::Mu, &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&BCT, &y, &owned, Param::Sigma, &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&BCT, &y, &owned, Param::Nu, &LogLink, 1e-5);
        check_eta_score_via_finite_diff(&BCT, &y, &owned, Param::Tau, &SqrtLink, 1e-4);
    }

    #[test]
    fn derivatives_stay_finite_at_saturated_parameters() {
        // Un-folding introduces `1/μ`, `1/μ²`, `1/σ` and `1/σ²` that the previous
        // η-scale forms canceled.
        let y = array![1.0, 2.0, 3.0];
        let owned = [
            (Param::Mu, array![0.0, 1e-320, 1e-8]),
            (Param::Sigma, array![1e-8, 0.0, 1e-320]),
            (Param::Nu, array![1.0, 0.5, -0.5]),
            (Param::Tau, array![1e-320, 2.0, 3.0]),
        ];
        let p = params_view(&owned);
        let natural = BCT.theta_derivatives(&y, &p).unwrap();
        let chained = default_link_derivatives(&BCT, &y, &p).unwrap();
        for name in [Param::Mu, Param::Sigma, Param::Nu, Param::Tau] {
            let (u_n, i_n) = (&natural[&name].score, &natural[&name].info);
            assert!(no_nan_array(u_n) && no_nan_array(i_n), "natural {name}");
            let (u, w) = (&chained[&name].score, &chained[&name].info);
            assert!(finite_array(u) && finite_array(w), "chained {name}: {u:?}");
        }
    }

    #[test]
    fn cdf_quantile_roundtrip_bct() {
        let y = array![0.6, 1.5, 3.0, 7.0, 2.0];
        let owned = [
            (Param::Mu, array![1.0, 2.0, 4.0, 6.0, 2.5]),
            (Param::Sigma, array![0.3, 0.25, 0.2, 0.35, 0.3]),
            (Param::Nu, array![1.0, 0.5, -0.5, 1.5, 0.0]),
            (Param::Tau, array![8.0, 6.0, 12.0, 20.0, 7.0]),
        ];
        check_cdf_quantile_roundtrip(&BCT, &y, &owned, 1e-6);
        check_cdf_pdf_consistency(&BCT, &y, &owned, 1e-5, 1e-3);
    }

    #[test]
    fn cdf_monotone_bct() {
        let grid = Array1::from_iter((0..80).map(|i| 0.05 + i as f64 * 0.1));
        let owned = [
            (Param::Mu, array![3.0]),
            (Param::Sigma, array![0.3]),
            (Param::Nu, array![0.8]),
            (Param::Tau, array![6.0]),
        ];
        check_cdf_monotone_in_unit(&BCT, &grid, &owned);
    }

    #[test]
    fn loglik_bct_approaches_bccg_for_large_tau() {
        // As τ → ∞ the t → normal, so BCT log-density should approach BCCG's.
        use crate::distributions::BCCG;
        let owned_bct = [
            (Param::Mu, array![2.0]),
            (Param::Sigma, array![0.3]),
            (Param::Nu, array![0.5]),
            (Param::Tau, array![1e6]),
        ];
        let owned_bccg = [
            (Param::Mu, array![2.0]),
            (Param::Sigma, array![0.3]),
            (Param::Nu, array![0.5]),
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
            (Param::Mu, array![2.0, 5.0]),
            (Param::Sigma, array![0.3, 0.2]),
            (Param::Nu, array![0.5, -1.0]),
            (Param::Tau, array![6.0, 10.0]),
        ];
        let p = params_view(&owned);
        let med = BCT.quantile(&array![0.5, 0.5], &p).unwrap();
        assert!((med[0] - 2.0).abs() < 1e-9);
        assert!((med[1] - 5.0).abs() < 1e-9);
    }
}
