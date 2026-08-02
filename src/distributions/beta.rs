//! Beta distribution for proportions on `(0, 1)`.

use super::{
    clamp_prob, require, DerivativesResult, Distribution, GamlssError, Link, LogLink, LogitLink,
    MIN_POSITIVE, MIN_WEIGHT,
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

    eta_derivatives_passthrough!();

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Beta (μ, φ) parameterization: α = μφ, β = (1−μ)φ.
        // l = log Γ(φ) − log Γ(α) − log Γ(β) + (α−1)·log(y) + (β−1)·log(1−y).
        let mu = require(self, params, "mu")?;
        let phi = require(self, params, "phi")?;

        let mu_safe = mu.mapv(|m| m.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE));
        let phi_safe = phi.mapv(|p| p.max(MIN_POSITIVE));
        let y_clamped = y.mapv(|v| v.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE));

        let one_minus_mu = mu_safe.mapv(|m| 1.0 - m);
        let alpha = &mu_safe * &phi_safe;
        let beta_param = &one_minus_mu * &phi_safe;

        let log_y = y_clamped.mapv(|v| v.ln());
        let log_1_minus_y = y_clamped.mapv(|v| (1.0 - v).ln());

        let psi_alpha = digamma_batch(&alpha);
        let psi_beta = digamma_batch(&beta_param);
        let psi_phi = digamma_batch(&phi_safe);
        let psi_prime_alpha = trigamma_batch(&alpha);
        let psi_prime_beta = trigamma_batch(&beta_param);
        let psi_prime_phi = trigamma_batch(&phi_safe);

        // μ (logit link). dl/dμ = φ·[log(y) − log(1−y) − ψ(α) + ψ(β)].
        // Chain rule: dl/dη = μ(1−μ)·dl/dμ.
        let dl_dmu = &phi_safe * (&log_y - &log_1_minus_y - &psi_alpha + &psi_beta);
        let mu_1_minus_mu = &mu_safe * &one_minus_mu;
        let u_mu = &mu_1_minus_mu * &dl_dmu;

        // Fisher info for μ on η-scale: w = (μ(1−μ))² · φ²·(ψ'(α) + ψ'(β)).
        let phi_sq = phi_safe.mapv(|p| p * p);
        let i_mu = &phi_sq * (&psi_prime_alpha + &psi_prime_beta);
        let mu_1_minus_mu_sq = mu_1_minus_mu.mapv(|v| v * v);
        let w_mu = (&mu_1_minus_mu_sq * &i_mu).mapv(|v| v.max(MIN_WEIGHT));

        // φ (log link). dl/dφ = ψ(φ) − μ·ψ(α) − (1−μ)·ψ(β) + μ·log(y) + (1−μ)·log(1−y).
        // Chain rule: dl/dη = φ · dl/dφ.
        let dl_dphi = &psi_phi - &mu_safe * &psi_alpha - &one_minus_mu * &psi_beta
            + &mu_safe * &log_y
            + &one_minus_mu * &log_1_minus_y;
        let u_phi = &phi_safe * &dl_dphi;

        // Fisher info for φ on η-scale: I_φ = μ²·ψ'(α) + (1−μ)²·ψ'(β) − ψ'(φ),
        // so w = φ²·I_φ. (ψ' is decreasing and convex, so I_φ > 0; the previous
        // expression had the sign inverted and relied on `.abs()` to rescue it.)
        let mu_sq = mu_safe.mapv(|m| m * m);
        let one_minus_mu_sq = one_minus_mu.mapv(|v| v * v);
        let i_phi = &mu_sq * &psi_prime_alpha + &one_minus_mu_sq * &psi_prime_beta - &psi_prime_phi;
        let w_phi = (&phi_sq * &i_phi).mapv(|v| v.max(MIN_WEIGHT));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("phi".to_string(), (u_phi, w_phi)),
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
        check_score_via_finite_diff, derivative_keys_match_parameters, params_view,
    };
    use ndarray::array;

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
