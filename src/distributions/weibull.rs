//! Weibull distribution for positive continuous data

use super::{
    clamp_prob, require, DerivativesResult, Distribution, GamlssError, Link, LogLink, MIN_POSITIVE,
    MIN_WEIGHT,
};
use crate::math::{par_zip3_map, par_zip_map};
use ndarray::Array1;
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;

/// Weibull distribution (gamlss `WEI` parameterization).
///
/// Parameters: `μ` (scale, log link) and `σ` (shape, log link). Support `y > 0`.
/// With `z = (y/μ)^σ`, `Var(Y) = μ²·[Γ(1+2/σ) − Γ(1+1/σ)²]` and the mean is
/// `μ·Γ(1+1/σ)` — neither equals `μ`, so both moment methods are overridden.
#[derive(Debug, Clone, Copy, Default)]
pub struct Weibull;

impl Weibull {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Weibull {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" | "sigma" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    /// σ is the Weibull shape; the default `y.std()` seed is meaningless for it.
    /// Seed σ = 1 (Exponential), where the scale μ ≈ mean(y); RS refines both.
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => y.mean().expect("validate_inputs rejects empty y"),
            "sigma" => 1.0,
            _ => 0.1,
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // z = (y/μ)^σ ~ Exp(1) at the truth.
        // μ (η = ln μ): u = σ(z−1),               w = σ².
        // σ (η = ln σ): u = 1 + σ·ln(y/μ)·(1−z),  w = π²/6 + (1−γ)² (constant).
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        const EULER: f64 = 0.577_215_664_901_532_9;
        let w_sigma_const = std::f64::consts::PI.powi(2) / 6.0 + (1.0 - EULER).powi(2);

        let u_mu = par_zip3_map(y, mu, sigma, |yi, mui, si| {
            let m = mui.max(MIN_POSITIVE);
            let z = (yi.max(MIN_POSITIVE) / m).powf(si);
            si * (z - 1.0)
        });
        let w_mu = sigma.mapv(|s| (s * s).max(MIN_WEIGHT));

        let u_sigma = par_zip3_map(y, mu, sigma, |yi, mui, si| {
            let m = mui.max(MIN_POSITIVE);
            let r = yi.max(MIN_POSITIVE) / m;
            let z = r.powf(si);
            1.0 + si * r.ln() * (1.0 - z)
        });
        let w_sigma = Array1::from_elem(y.len(), w_sigma_const.max(MIN_WEIGHT));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
        ]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip3_map(y, mu, sigma, |yi, mui, si| {
            let yv = yi.max(MIN_POSITIVE);
            let m = mui.max(MIN_POSITIVE);
            let z = (yv / m).powf(si);
            si.ln() - si * m.ln() + (si - 1.0) * yv.ln() - z
        }))
    }

    fn expected_value(
        &self,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        // E[Y] = μ·Γ(1 + 1/σ)
        Ok(par_zip_map(mu, sigma, |m, s| {
            m * ln_gamma(1.0 + 1.0 / s.max(MIN_POSITIVE)).exp()
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        // V[Y] = μ²·[Γ(1+2/σ) − Γ(1+1/σ)²]
        Ok(par_zip_map(mu, sigma, |m, s| {
            let s = s.max(MIN_POSITIVE);
            let g1 = ln_gamma(1.0 + 1.0 / s).exp();
            let g2 = ln_gamma(1.0 + 2.0 / s).exp();
            m * m * (g2 - g1 * g1)
        }))
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // F(y) = 1 − exp(−(y/μ)^σ)
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip3_map(y, mu, sigma, |yi, mui, si| {
            if yi <= 0.0 {
                return 0.0; // support is y > 0
            }
            let z = (yi / mui.max(MIN_POSITIVE)).powf(si.max(MIN_POSITIVE));
            -(-z).exp_m1()
        }))
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // Q(p) = μ·(−ln(1 − p))^(1/σ)
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip3_map(p, mu, sigma, |pi, mui, si| {
            let pc = clamp_prob(pi);
            mui.max(MIN_POSITIVE) * (-(-pc).ln_1p()).powf(1.0 / si.max(MIN_POSITIVE))
        }))
    }

    fn name(&self) -> &'static str {
        "Weibull"
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
    fn weibull_derivatives() {
        let y = array![0.5, 1.5, 3.0, 7.0];
        let mu = array![1.0, 2.0, 4.0, 6.0];
        let sigma = array![0.8, 1.0, 1.5, 2.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        derivative_keys_match_parameters(&Weibull, p, &y);
    }

    #[test]
    fn loglik_weibull_finite_on_typical_inputs() {
        let owned = [
            ("mu", array![2.0, 2.0, 4.0]),
            ("sigma", array![0.8, 1.0, 1.5]),
        ];
        let p = params_view(&owned);
        let ll = Weibull.loglik(&array![1.0, 2.0, 5.0], &p).unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn mean_and_variance_match_gamma_function_moments() {
        // σ = 1 is Exponential(μ): E[Y] = μ, V[Y] = μ².
        let owned = [("mu", array![3.0]), ("sigma", array![1.0])];
        let p = params_view(&owned);
        let m = Weibull.expected_value(&p).unwrap();
        let v = Weibull.variance(&p).unwrap();
        assert!((m[0] - 3.0).abs() < 1e-9);
        assert!((v[0] - 9.0).abs() < 1e-9);
    }

    #[test]
    fn score_matches_finite_diff_weibull() {
        let y = array![1.0, 2.5, 5.0];
        let owned = [
            ("mu", array![1.5, 2.0, 4.0]),
            ("sigma", array![0.8, 1.0, 1.5]),
        ];
        check_score_via_finite_diff(&Weibull, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&Weibull, &y, &owned, "sigma", 1e-5);
    }

    #[test]
    fn cdf_quantile_roundtrip_weibull() {
        let y = array![0.5, 1.5, 3.0, 7.0];
        let owned = [
            ("mu", array![1.0, 2.0, 4.0, 6.0]),
            ("sigma", array![0.8, 1.0, 1.5, 2.0]),
        ];
        check_cdf_quantile_roundtrip(&Weibull, &y, &owned, 1e-6);
        check_cdf_pdf_consistency(&Weibull, &y, &owned, 1e-4, 1e-3);
    }

    #[test]
    fn cdf_monotone_weibull_and_zero_below_support() {
        let grid = Array1::from_iter((0..60).map(|i| i as f64 * 0.2));
        let owned = [("mu", array![3.0]), ("sigma", array![1.5])];
        check_cdf_monotone_in_unit(&Weibull, &grid, &owned);
        // Both boundary points (y = 0 and y < 0) sit outside the y > 0 support.
        let boundary_params = [("mu", array![3.0, 3.0]), ("sigma", array![1.5, 1.5])];
        let p = params_view(&boundary_params);
        let at_boundary = Weibull.cdf(&array![0.0, -1.0], &p).unwrap();
        assert_eq!(at_boundary[0], 0.0);
        assert_eq!(at_boundary[1], 0.0);
    }

    #[test]
    fn initial_values_are_sensible() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        assert!((Weibull.initial_value("mu", &y) - 2.5).abs() < 1e-12);
        assert_eq!(Weibull.initial_value("sigma", &y), 1.0);
        assert_eq!(Weibull.initial_value("other", &y), 0.1);
    }

    #[test]
    fn default_link_is_log_for_both_and_errs_on_unknown() {
        assert!(Weibull.default_link("mu").is_ok());
        assert!(Weibull.default_link("sigma").is_ok());
        assert!(Weibull.default_link("nu").is_err());
    }
}
