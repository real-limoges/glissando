//! Binomial distribution: counts of successes out of `n` trials.

use super::{
    require, DerivativesResult, Distribution, GamlssError, Link, LogitLink, MIN_POSITIVE,
    MIN_WEIGHT,
};
use crate::math::{par_zip3_map, par_zip_map};
use ndarray::Array1;
use statrs::function::gamma::ln_gamma;
use std::borrow::Cow;
use std::collections::HashMap;

/// Binomial distribution: response `y` is the count of successes out of `n` trials.
///
/// Single parameter `μ` ∈ `(0, 1)` (success probability) with logit link.
#[derive(Debug, Clone)]
pub struct Binomial {
    /// Trials per observation. Length 1 broadcasts; otherwise must match `y.len()`.
    n_trials: Array1<f64>,
}

impl Binomial {
    /// Construct a Binomial with a constant number of trials shared across observations.
    pub fn new(n_trials: usize) -> Self {
        Self {
            n_trials: Array1::from_elem(1, n_trials as f64),
        }
    }

    /// Construct a Binomial with per-observation trial counts.
    pub fn with_trials(n_trials: Array1<f64>) -> Self {
        Self { n_trials }
    }

    /// Returns trials broadcast to `n_obs`. Borrows when length already matches; otherwise allocates.
    fn trials(&self, n_obs: usize) -> Cow<'_, Array1<f64>> {
        if self.n_trials.len() == 1 {
            Cow::Owned(Array1::from_elem(n_obs, self.n_trials[0]))
        } else {
            Cow::Borrowed(&self.n_trials)
        }
    }
}

impl Distribution for Binomial {
    fn parameters(&self) -> &[&'static str] {
        &["mu"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(LogitLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Binomial log-likelihood: l = y·log(μ) + (n−y)·log(1−μ) + log C(n, y).
        // With logit link η = logit(μ) and dμ/dη = μ(1−μ):
        //   u_η = y − n·μ,    w_η = n·μ·(1−μ)  (floored at MIN_WEIGHT).
        let mu = require(self, params, "mu")?;
        let n = self.trials(y.len());

        let mu_safe = mu.mapv(|m| m.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE));
        let u_mu = y - &(n.as_ref() * &mu_safe);

        let mu_1_minus_mu = &mu_safe * &mu_safe.mapv(|m| 1.0 - m);
        let w_mu = (n.as_ref() * &mu_1_minus_mu).mapv(|v| v.max(MIN_WEIGHT));

        Ok(HashMap::from([("mu".to_string(), (u_mu, w_mu))]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let n = self.trials(y.len());
        Ok(par_zip3_map(y, mu, n.as_ref(), |yi, mui, ni| {
            let m = mui.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            ln_gamma(ni + 1.0) - ln_gamma(yi + 1.0) - ln_gamma(ni - yi + 1.0)
                + yi * m.ln()
                + (ni - yi) * (1.0 - m).ln()
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let n = self.trials(mu.len());
        Ok(par_zip_map(n.as_ref(), mu, |ni, mi| {
            let m = mi.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            ni * m * (1.0 - m)
        }))
    }

    fn expected_value(
        &self,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let n = self.trials(mu.len());
        Ok(n.as_ref() * mu)
    }

    fn name(&self) -> &'static str {
        "Binomial"
    }

    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => {
                // y is counts; convert to a probability via the first trial count.
                // `validate_inputs` rejects empty `y`, so the `unwrap_or` is unreachable
                // on the public path.
                let n = self.n_trials[0];
                let p = y.mean().unwrap_or(n / 2.0) / n;
                // Clamp away from {0, 1} so the IRLS loop has a well-conditioned start.
                p.clamp(0.1, 0.9)
            }
            _ => 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::test_helpers::{
        check_score_via_finite_diff, derivative_keys_match_parameters, params_view,
    };
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn binomial_derivatives_constant_trials() {
        let bin = Binomial::new(20);
        let y = array![5.0, 10.0, 15.0, 8.0];
        let mu = array![0.25, 0.5, 0.7, 0.4];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        derivative_keys_match_parameters(&bin, p, &y);
    }

    #[test]
    fn binomial_per_observation_trials() {
        let trials = array![10.0, 20.0, 5.0];
        let bin = Binomial::with_trials(trials);
        let y = array![3.0, 10.0, 2.0];
        let mu = array![0.3, 0.5, 0.4];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        let derivs = bin.derivatives(&y, &p).unwrap();
        let (u_mu, _) = &derivs["mu"];
        assert_relative_eq!(u_mu[0], 3.0 - 10.0 * 0.3, epsilon = 1e-12);
        assert_relative_eq!(u_mu[1], 10.0 - 20.0 * 0.5, epsilon = 1e-12);
    }

    #[test]
    fn binomial_score_zero_when_y_equals_n_mu() {
        let bin = Binomial::new(10);
        let y = array![3.0, 5.0, 7.0];
        let mu = array![0.3, 0.5, 0.7];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        let derivs = bin.derivatives(&y, &p).unwrap();
        let (u, _) = &derivs["mu"];
        assert!(u.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn binomial_initial_value_clamped_in_unit_interval() {
        let bin = Binomial::new(10);
        // All-zero counts → naive 0.0, must be clamped up.
        let y_zero = array![0.0, 0.0];
        let v0 = bin.initial_value("mu", &y_zero);
        assert!((0.1..=0.9).contains(&v0));
        // All-max counts → naive 1.0, must be clamped down.
        let y_full = array![10.0, 10.0];
        let v1 = bin.initial_value("mu", &y_full);
        assert!((0.1..=0.9).contains(&v1));
    }

    #[test]
    fn loglik_binomial_matches_manual() {
        // n=2, y=1, mu=0.5 → log C(2,1) + 1·log(0.5) + 1·log(0.5) = log 2 + 2 log 0.5
        let bin = Binomial::new(2);
        let owned = [("mu", array![0.5])];
        let p = params_view(&owned);
        let ll = bin.loglik(&array![1.0], &p).unwrap();
        let expected = 2.0_f64.ln() + 2.0 * 0.5_f64.ln();
        assert!((ll - expected).abs() < 1e-9);
    }

    #[test]
    fn variance_binomial_is_n_mu_one_minus_mu() {
        let bin = Binomial::new(10);
        let owned = [("mu", array![0.3, 0.5])];
        let p = params_view(&owned);
        let v = bin.variance(&p).unwrap();
        assert!((v[0] - 10.0 * 0.3 * 0.7).abs() < 1e-12);
        assert!((v[1] - 10.0 * 0.5 * 0.5).abs() < 1e-12);
    }

    #[test]
    fn expected_value_binomial_is_n_times_mu() {
        let bin = Binomial::new(10);
        let owned = [("mu", array![0.3, 0.5])];
        let p = params_view(&owned);
        let e = bin.expected_value(&p).unwrap();
        assert!((e[0] - 3.0).abs() < 1e-12);
        assert!((e[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn score_matches_finite_diff_binomial() {
        let bin = Binomial::new(10);
        let y = array![3.0, 5.0, 8.0];
        let owned = [("mu", array![0.3, 0.5, 0.7])];
        check_score_via_finite_diff(&bin, &y, &owned, "mu", 1e-5);
    }
}
