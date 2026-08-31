//! Beta distribution for proportions on `(0, 1)`.

use super::{
    clamp_prob, require, DerivativesResult, Distribution, GamlssError, Link, LogLink, LogitLink,
    MIN_POSITIVE, TRIGAMMA_FLOOR,
};
use crate::math::{digamma_batch, par_zip3_map, par_zip_map, trigamma_batch};
use ndarray::Array1;
use statrs::distribution::{Beta as SBeta, ContinuousCDF};
use statrs::function::beta::beta_reg;
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;

/// Beta distribution for proportions on `(0, 1)`.
///
/// Parameters: `μ` (mean, logit link) and `φ` (precision, log link).
/// Shape `α = μφ`, `β = (1−μ)φ`. `Var(Y) = μ(1−μ)/(1+φ)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Beta;

impl Beta {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Beta {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "phi"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(LogitLink)),
            "phi" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    eta_derivatives_via_chain!();

    fn theta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Beta (μ, φ) parameterization: α = μφ, β = (1−μ)φ.
        // l = log Γ(φ) − log Γ(α) − log Γ(β) + (α−1)·log(y) + (β−1)·log(1−y).
        let mu = require(self, params, "mu")?;
        let phi = require(self, params, "phi")?;

        // **The floor sits on the Gamma-function arguments, not on μ or φ.** This
        // family used to clamp `μ ∈ [MIN_POSITIVE, 1−MIN_POSITIVE]`. The folded
        // η-scale form could afford that; the un-folded one can't. `chain_to_eta`
        // multiplies by a `mu_eta` computed from η independently of anything clamped
        // here, so a clamp that binds breaks the telescoping. μ now gets to be
        // probit/cloglog/cauchit, and under a probit at η = −10 the true
        // μ = Φ(−10) ≈ 7.6e-24, fourteen orders of magnitude below the old clamp,
        // which would have evaluated ψ and ψ' at the wrong α entirely. Same argument
        // spelled out at length in `binomial.rs`.
        //
        // What actually needs guarding is α = μφ and β = (1−μ)φ hitting exactly 0,
        // where ψ(0) = −∞ and ψ'(0) = +∞. `TRIGAMMA_FLOOR` is the binding one of the
        // two (ψ' ~ 1/x² overflows ~150 decades before ψ ~ −1/x does) and sits far
        // below anything a link produces inside its own η clamp.
        let y_clamped = y.mapv(|v| v.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE));

        let one_minus_mu = mu.mapv(|m| 1.0 - m);
        let alpha = par_zip_map(mu, phi, |m, p| (m * p).max(TRIGAMMA_FLOOR));
        let beta_param = par_zip_map(&one_minus_mu, phi, |om, p| (om * p).max(TRIGAMMA_FLOOR));
        let phi_floored = phi.mapv(|p| p.max(TRIGAMMA_FLOOR));

        let log_y = y_clamped.mapv(|v| v.ln());
        let log_1_minus_y = y_clamped.mapv(|v| (1.0 - v).ln());

        let psi_alpha = digamma_batch(&alpha);
        let psi_beta = digamma_batch(&beta_param);
        let psi_phi = digamma_batch(&phi_floored);
        let psi_prime_alpha = trigamma_batch(&alpha);
        let psi_prime_beta = trigamma_batch(&beta_param);
        let psi_prime_phi = trigamma_batch(&phi_floored);

        // Natural scale (Altitude #1). This family was already separable, so the
        // conversion is just deleting the two trailing chain-rule multiplies.
        // `chain_to_eta` reapplies them from the resolved link (`mu_eta = μ(1−μ)`
        // for logit, `φ` for log) and reproduces the old η-scale values under the
        // defaults. Weights come back unfloored.

        // μ. dl/dμ = φ·[log(y) − log(1−y) − ψ(α) + ψ(β)].
        let dl_dmu = phi * (&log_y - &log_1_minus_y - &psi_alpha + &psi_beta);

        // I_μ = φ²·(ψ'(α) + ψ'(β)).
        let phi_sq = phi.mapv(|p| p * p);
        let i_mu = &phi_sq * (&psi_prime_alpha + &psi_prime_beta);

        // φ. dl/dφ = ψ(φ) − μ·ψ(α) − (1−μ)·ψ(β) + μ·log(y) + (1−μ)·log(1−y).
        let dl_dphi = &psi_phi - mu * &psi_alpha - &one_minus_mu * &psi_beta
            + mu * &log_y
            + &one_minus_mu * &log_1_minus_y;

        // I_φ = μ²·ψ'(α) + (1−μ)²·ψ'(β) − ψ'(φ). ψ' is decreasing and convex, so
        // I_φ > 0. An earlier expression had the sign inverted and leaned on
        // `.abs()` to rescue it. That's gone now.
        let mu_sq = mu.mapv(|m| m * m);
        let one_minus_mu_sq = one_minus_mu.mapv(|v| v * v);
        let i_phi = &mu_sq * &psi_prime_alpha + &one_minus_mu_sq * &psi_prime_beta - &psi_prime_phi;

        Ok(HashMap::from([
            ("mu".to_string(), (dl_dmu, i_mu)),
            ("phi".to_string(), (dl_dphi, i_phi)),
        ]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let phi = require(self, params, "phi")?;
        Ok(par_zip3_map(y, mu, phi, |yi, mui, phii| {
            let alpha = mui * phii;
            let beta = (1.0 - mui) * phii;
            let yc = yi.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            ln_gamma(phii) - ln_gamma(alpha) - ln_gamma(beta)
                + (alpha - 1.0) * yc.ln()
                + (beta - 1.0) * (1.0 - yc).ln()
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let phi = require(self, params, "phi")?;
        Ok(par_zip_map(mu, phi, |m, p| {
            let m_clamped = m.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            m_clamped * (1.0 - m_clamped) / (1.0 + p.max(MIN_POSITIVE))
        }))
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // (μ, φ) parameterization: α = μφ, β = (1−μ)φ; F(y) = I_y(α, β) = beta_reg(α, β, y).
        let mu = require(self, params, "mu")?;
        let phi = require(self, params, "phi")?;
        Ok(par_zip3_map(y, mu, phi, |yi, mui, phii| {
            let yc = yi.clamp(0.0, 1.0);
            if yc <= 0.0 {
                return 0.0;
            }
            if yc >= 1.0 {
                return 1.0;
            }
            let m = mui.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            let p = phii.max(MIN_POSITIVE);
            beta_reg(m * p, (1.0 - m) * p, yc)
        }))
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let phi = require(self, params, "phi")?;
        Ok(par_zip3_map(p, mu, phi, |pi, mui, phii| {
            let m = mui.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            let ph = phii.max(MIN_POSITIVE);
            SBeta::new(m * ph, (1.0 - m) * ph)
                .expect("valid Beta params")
                .inverse_cdf(clamp_prob(pi))
        }))
    }

    fn name(&self) -> &'static str {
        "Beta"
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
    use crate::distributions::{CloglogLink, ProbitLink, SqrtLink};
    use ndarray::array;

    #[test]
    fn score_tracks_a_mu_far_below_the_old_clamp() {
        // Regression for the `μ.clamp(MIN_POSITIVE, 1−MIN_POSITIVE)` this body used to
        // apply. Under a probit link at η = −10 the true μ is Φ(−10) ≈ 7.6e-24. The
        // clamp evaluated ψ and ψ' at 1e-10 instead, fourteen orders of magnitude
        // away, so `chain_to_eta` multiplied a `mu_eta` taken from the real η into a
        // score taken from a fictional μ, and the product stopped telescoping. Two μ
        // that far apart had better not collapse onto the same derivative.
        let y = array![0.5];
        let clamped = [("mu", array![1e-10]), ("phi", array![10.0])];
        let truthful = [("mu", array![7.6e-24]), ("phi", array![10.0])];
        let a = Beta.theta_derivatives(&y, &params_view(&clamped)).unwrap();
        let b = Beta.theta_derivatives(&y, &params_view(&truthful)).unwrap();
        let (ua, ub) = (a["mu"].0[0], b["mu"].0[0]);
        assert!(
            ua.is_finite() && ub.is_finite() && (ua - ub).abs() > 1.0,
            "μ = 1e-10 and μ = 7.6e-24 gave {ua} and {ub}"
        );
    }

    #[test]
    fn beta_derivatives() {
        let y = array![0.1, 0.5, 0.9, 0.25];
        let mu = array![0.2, 0.5, 0.8, 0.3];
        let phi = array![5.0, 10.0, 15.0, 8.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("phi", &phi);
        derivative_keys_match_parameters(&Beta, p, &y);
    }

    #[test]
    fn loglik_beta_finite() {
        let owned = [
            ("mu", array![0.2, 0.5, 0.8]),
            ("phi", array![10.0, 10.0, 10.0]),
        ];
        let p = params_view(&owned);
        let ll = Beta.loglik(&array![0.1, 0.5, 0.9], &p).unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn variance_beta_uses_mu_one_minus_mu_over_one_plus_phi() {
        let owned = [("mu", array![0.5]), ("phi", array![3.0])];
        let p = params_view(&owned);
        let v = Beta.variance(&p).unwrap();
        // 0.5·0.5/(1+3) = 0.0625
        assert!((v[0] - 0.0625).abs() < 1e-12);
    }

    #[test]
    fn score_matches_finite_diff_beta() {
        let y = array![0.2, 0.5, 0.85];
        let owned = [
            ("mu", array![0.3, 0.5, 0.7]),
            ("phi", array![10.0, 12.0, 8.0]),
        ];
        check_score_via_finite_diff(&Beta, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&Beta, &y, &owned, "phi", 1e-5);
    }

    #[test]
    fn score_matches_finite_diff_under_non_default_links() {
        // The Altitude #1 gate. μ lives on (0,1), so probit and cloglog are the
        // meaningful overrides; φ is positive, so sqrt is.
        let y = array![0.2, 0.5, 0.85];
        let owned = [
            ("mu", array![0.3, 0.5, 0.7]),
            ("phi", array![10.0, 12.0, 8.0]),
        ];
        check_eta_score_via_finite_diff(&Beta, &y, &owned, "mu", &ProbitLink, 1e-5);
        check_eta_score_via_finite_diff(&Beta, &y, &owned, "mu", &CloglogLink, 1e-5);
        check_eta_score_via_finite_diff(&Beta, &y, &owned, "phi", &SqrtLink, 1e-5);
    }

    #[test]
    fn derivatives_stay_finite_at_saturated_parameters() {
        // Beta was already separable, so no new division appears here; the gate
        // still runs so the family is covered uniformly with the rest of Phase 2b.
        let y = array![0.01, 0.5, 0.99];
        let owned = [
            ("mu", array![0.0, 1.0, 1e-12]),
            ("phi", array![0.0, 1e-320, 1e8]),
        ];
        let p = params_view(&owned);
        let natural = Beta.theta_derivatives(&y, &p).unwrap();
        let chained = default_link_derivatives(&Beta, &y, &p).unwrap();
        for name in ["mu", "phi"] {
            let (u_n, i_n) = &natural[name];
            assert!(no_nan_array(u_n) && no_nan_array(i_n), "natural {name}");
            let (u, w) = &chained[name];
            assert!(finite_array(u) && finite_array(w), "chained {name}: {u:?}");
            assert!(w.iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn cdf_quantile_roundtrip_beta() {
        let y = array![0.1, 0.35, 0.5, 0.7, 0.9];
        let owned = [
            ("mu", array![0.2, 0.4, 0.5, 0.6, 0.8]),
            ("phi", array![10.0, 12.0, 8.0, 15.0, 6.0]),
        ];
        check_cdf_quantile_roundtrip(&Beta, &y, &owned, 1e-6);
        check_cdf_pdf_consistency(&Beta, &y, &owned, 1e-4, 1e-3);
    }

    #[test]
    fn cdf_monotone_beta_and_unit_endpoints() {
        let grid = Array1::from_iter((1..40).map(|i| i as f64 / 40.0));
        let owned = [("mu", array![0.45]), ("phi", array![9.0])];
        check_cdf_monotone_in_unit(&Beta, &grid, &owned);
        // F(0) = 0 and F(1) = 1 at the unit-interval endpoints.
        let endpoint_params = [("mu", array![0.45, 0.45]), ("phi", array![9.0, 9.0])];
        let p = params_view(&endpoint_params);
        let at_endpoints = Beta.cdf(&array![0.0, 1.0], &p).unwrap();
        assert_eq!(at_endpoints[0], 0.0);
        assert_eq!(at_endpoints[1], 1.0);
    }
}
