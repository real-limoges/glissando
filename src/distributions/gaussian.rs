//! Gaussian (Normal) distribution.

use super::{
    require, DerivativesResult, Distribution, GamlssError, IdentityLink, Link, LogLink,
    MIN_POSITIVE,
};
use crate::math::par_zip3_map;
use ndarray::Array1;
use std::collections::HashMap;

/// Gaussian (Normal) distribution.
///
/// Parameters: `μ` (mean, identity link) and `σ` (std dev, log link). `Var(Y) = σ²`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gaussian;

impl Gaussian {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Gaussian {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(IdentityLink)),
            "sigma" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Gaussian log-likelihood:  l = −0.5·log(2π) − log(σ) − (y−μ)²/(2σ²).
        //   μ (identity link):  u = (y−μ)/σ²,                w = 1/σ².
        //   σ (log link, η = log σ):  u = ((y−μ)² − σ²)/σ²,  w = 2.
        // Full derivation in docs/math/mathematics.md.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        let sigma_sq = sigma.mapv(|s| s * s);
        let residual = y - mu;
        let residual_sq = residual.mapv(|r| r * r);

        let u_mu = &residual / &sigma_sq;
        let w_mu = sigma_sq.mapv(|s2| 1.0 / s2);

        let u_sigma = (&residual_sq - &sigma_sq) / &sigma_sq;
        let w_sigma = Array1::from_elem(y.len(), 2.0);

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
        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        Ok(par_zip3_map(y, mu, sigma, |yi, mui, si| {
            let s = si.max(MIN_POSITIVE);
            let z = (yi - mui) / s;
            -0.5 * log_2pi - s.ln() - 0.5 * z * z
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let sigma = require(self, params, "sigma")?;
        Ok(sigma.mapv(|s| s * s))
    }

    fn name(&self) -> &'static str {
        "Gaussian"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::test_helpers::{
        check_score_via_finite_diff, derivative_keys_match_parameters, params_view,
    };
    use ndarray::array;
    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;

    #[test]
    fn gaussian_derivatives() {
        let y = array![0.0, 1.0, -1.0, 2.5];
        let mu = array![0.0, 0.5, -0.5, 2.0];
        let sigma = array![1.0, 1.0, 2.0, 0.5];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        derivative_keys_match_parameters(&Gaussian, p, &y);
    }

    #[test]
    fn gaussian_mu_score_zero_when_y_equals_mu() {
        let y = array![0.5, 1.5, -2.0];
        let mu = y.clone();
        let sigma = array![1.0, 1.0, 1.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        let derivs = Gaussian.derivatives(&y, &p).unwrap();
        let (u_mu, w_mu) = &derivs["mu"];
        assert!(u_mu.iter().all(|&v| v.abs() < 1e-12));
        // w_mu = 1/sigma^2 = 1.0
        assert!(w_mu.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn gaussian_sigma_fisher_info_constant() {
        let y = array![0.0, 1.0, 2.0];
        let mu = array![0.0, 0.0, 0.0];
        let sigma = array![1.0, 2.0, 3.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        let derivs = Gaussian.derivatives(&y, &p).unwrap();
        let (_, w_sigma) = &derivs["sigma"];
        assert!(w_sigma.iter().all(|&v| (v - 2.0).abs() < 1e-12));
    }

    #[test]
    fn loglik_gaussian_matches_manual_formula() {
        let owned = [("mu", array![0.0]), ("sigma", array![1.0])];
        let p = params_view(&owned);
        let ll = Gaussian.loglik(&array![0.0], &p).unwrap();
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn variance_gaussian_is_sigma_squared() {
        let owned = [("mu", array![0.0, 0.0]), ("sigma", array![2.0, 3.0])];
        let p = params_view(&owned);
        let v = Gaussian.variance(&p).unwrap();
        assert!((v[0] - 4.0).abs() < 1e-12);
        assert!((v[1] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn expected_value_default_is_mu() {
        let owned = [("mu", array![1.0, 2.0]), ("sigma", array![1.0, 1.0])];
        let p = params_view(&owned);
        let e = Gaussian.expected_value(&p).unwrap();
        assert_eq!(e, array![1.0, 2.0]);
    }

    #[test]
    fn score_matches_finite_diff_gaussian() {
        let y = array![-1.0, 0.0, 1.0, 2.0];
        let owned = [
            ("mu", array![-0.5, 0.5, 0.5, 1.5]),
            ("sigma", array![1.0, 1.5, 0.8, 1.2]),
        ];
        check_score_via_finite_diff(&Gaussian, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&Gaussian, &y, &owned, "sigma", 1e-5);
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn loglik_gaussian_pointwise_matches_naive(
            n in 1usize..20,
            mu_val in -5.0f64..5.0,
            sigma_val in 0.1f64..3.0,
        ) {
            let y = Array1::from_iter((0..n).map(|i| i as f64 * 0.1));
            let mu = Array1::from_elem(n, mu_val);
            let sigma = Array1::from_elem(n, sigma_val);
            let owned = [("mu", mu), ("sigma", sigma)];
            let p = params_view(&owned);
            let actual = Gaussian.loglik(&y, &p).unwrap();
            let log_2pi = (2.0 * std::f64::consts::PI).ln();
            let expected: f64 = (0..n).map(|i| {
                let z = (y[i] - mu_val) / sigma_val;
                -0.5 * log_2pi - sigma_val.ln() - 0.5 * z * z
            }).sum();
            prop_assert!((actual - expected).abs() < 1e-9);
        }
    }
}
