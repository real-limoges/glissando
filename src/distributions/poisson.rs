//! Poisson distribution for count data.

use super::{require, DerivativesResult, Distribution, GamlssError, Link, LogLink, MIN_POSITIVE};
use crate::math::par_zip_map;
use ndarray::Array1;
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;

/// Poisson distribution for count data.
///
/// Single parameter `μ` (mean / rate) with log link. `Var(Y) = μ`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Poisson;

impl Poisson {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Poisson {
    fn parameters(&self) -> &[&'static str] {
        &["mu"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Log-likelihood: l = y·log(μ) − μ.
        // Score on η = log(μ): u = y − μ.   Fisher info: w = μ.
        let mu = require(self, params, "mu")?;
        let u = y - mu;
        let w = mu.to_owned();
        Ok(HashMap::from([("mu".to_string(), (u, w))]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        Ok(par_zip_map(y, mu, |yi, mui| {
            yi * mui.max(MIN_POSITIVE).ln() - mui - ln_gamma(yi + 1.0)
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        Ok(require(self, params, "mu")?.to_owned())
    }

    fn name(&self) -> &'static str {
        "Poisson"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::test_helpers::{
        check_score_via_finite_diff, derivative_keys_match_parameters, params_view,
    };
    use ndarray::array;

    #[test]
    fn poisson_derivatives() {
        let y = array![0.0, 1.0, 5.0, 10.0];
        let mu = array![1.0, 2.0, 4.0, 9.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        derivative_keys_match_parameters(&Poisson, p, &y);
    }

    #[test]
    fn poisson_score_zero_when_y_equals_mu() {
        let y = array![1.0, 2.0, 4.0];
        let mu = y.clone();
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        let derivs = Poisson.derivatives(&y, &p).unwrap();
        let (u, _) = &derivs["mu"];
        assert!(u.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn poisson_unknown_parameter_errors() {
        let y = array![1.0];
        let p: HashMap<&str, &Array1<f64>> = HashMap::new();
        let err = Poisson.derivatives(&y, &p).unwrap_err();
        assert!(matches!(err, GamlssError::UnknownParameter { .. }));
    }

    #[test]
    fn loglik_poisson_matches_manual() {
        // l = y log(μ) − μ − log Γ(y+1). y=0, μ=1 → −1.
        let owned = [("mu", array![1.0])];
        let p = params_view(&owned);
        let ll = Poisson.loglik(&array![0.0], &p).unwrap();
        assert!((ll - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn variance_poisson_is_mu() {
        let owned = [("mu", array![1.0, 4.0, 9.0])];
        let p = params_view(&owned);
        let v = Poisson.variance(&p).unwrap();
        assert_eq!(v, array![1.0, 4.0, 9.0]);
    }

    #[test]
    fn score_matches_finite_diff_poisson() {
        let y = array![0.0, 3.0, 7.0, 12.0];
        let owned = [("mu", array![1.0, 3.5, 6.0, 10.0])];
        check_score_via_finite_diff(&Poisson, &y, &owned, "mu", 1e-5);
    }
}
