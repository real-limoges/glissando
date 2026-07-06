//! Negative Binomial (NB2) distribution for overdispersed count data.

use super::{
    discrete_quantile, require, DerivativesResult, Distribution, GamlssError, Link, LogLink,
    MIN_POSITIVE, MIN_WEIGHT,
};
use crate::math::{digamma_batch, par_zip3_map, par_zip_map};
use ndarray::Array1;
use statrs::function::beta::beta_reg;
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;

/// Negative Binomial distribution (NB2) for overdispersed count data.
///
/// Parameters: `μ` (mean, log link) and `σ` (overdispersion, log link).
/// `Var(Y) = μ + σμ²`. As `σ → 0` it approaches Poisson.
#[derive(Debug, Clone, Copy, Default)]
pub struct NegativeBinomial;

impl NegativeBinomial {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for NegativeBinomial {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" | "sigma" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    /// NB2 σ is the overdispersion coefficient (`Var = μ + σμ²`), not a standard
    /// deviation — the trait default of `sd(y)` is on the wrong scale entirely
    /// (often 10–30× too large for count data). Seed it with the method-of-moments
    /// estimate `(var(y) − mean(y))/mean(y)²`, floored at 0.1 like gamlss NBI's
    /// `sigma.initial`.
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => y.mean().expect("validate_inputs rejects empty y"),
            "sigma" => {
                let m = y.mean().expect("validate_inputs rejects empty y");
                let v = y.std(1.0).powi(2);
                let mom = (v - m) / (m * m).max(MIN_POSITIVE);
                // n = 1 gives std(ddof=1) = NaN; fall back to a moderate seed
                // rather than letting NaN poison the whole fit.
                if mom.is_finite() {
                    mom.clamp(0.1, 10.0)
                } else {
                    0.5
                }
            }
            _ => 0.1,
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // NB2 log-likelihood:
        //   l = log Γ(y + 1/σ) − log Γ(1/σ) − log y!
        //       + (1/σ)·log(1/(1+σμ)) + y·log(σμ/(1+σμ)).
        // μ (log link):  u = (y−μ)/(1+σμ),  w = μ/(1+σμ).
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        let mu_safe = mu.mapv(|m| m.max(MIN_POSITIVE));
        let sigma_safe = sigma.mapv(|s| s.max(MIN_POSITIVE));
        let one_plus_sigma_mu = par_zip_map(&sigma_safe, &mu_safe, |s, m| 1.0 + s * m);

        let u_mu = (y - &mu_safe) / &one_plus_sigma_mu;
        let w_mu = &mu_safe / &one_plus_sigma_mu;

        // σ (log link, r = 1/σ):
        //   dl/dr = ψ(y+r) − ψ(r) − log(1+σμ) + (μ−y)/(r+μ)
        //   dl/dσ = −(1/σ²)·dl/dr,   dl/dη = σ·dl/dσ = −(1/σ)·dl/dr.
        let r = sigma_safe.mapv(|s| 1.0 / s);
        let y_plus_r = y + &r;
        let psi_y_r = digamma_batch(&y_plus_r);
        let psi_r = digamma_batch(&r);
        let log_term = one_plus_sigma_mu.mapv(|v| v.ln());
        let r_plus_mu = &r + &mu_safe;
        let ratio_term = (&mu_safe - y) / &r_plus_mu;

        let u_sigma = (-1.0 / &sigma_safe) * (&psi_y_r - &psi_r - &log_term + &ratio_term);

        // Working weight for σ: gamlss NBI's squared-score convention
        // (d2ldd2 = −dldd², i.e. w_η = u_η²), floored at MIN_WEIGHT. The exact
        // expected information has no closed form (it involves E[ψ'(y+r)]); the
        // previous ψ'(r)/σ² approximation dropped same-order terms and behaved
        // like 1/σ near the Poisson boundary, over-damping σ updates and pushing
        // λ/EDF for σ smooths away from the gamlss oracle.
        let w_sigma = u_sigma.mapv(|u| (u * u).max(MIN_WEIGHT));

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
            let r = 1.0 / si.max(MIN_POSITIVE);
            let p = r / (r + mui);
            ln_gamma(yi + r) - ln_gamma(r) - ln_gamma(yi + 1.0)
                + r * p.max(MIN_POSITIVE).ln()
                + yi * (1.0 - p).max(MIN_POSITIVE).ln()
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip_map(mu, sigma, |m, s| m + s * m * m))
    }

    fn is_discrete(&self) -> bool {
        true
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // size r = 1/σ, success prob q = r/(r+μ); F(⌊y⌋) = I_q(r, ⌊y⌋+1) = beta_reg(r, ⌊y⌋+1, q).
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip3_map(y, mu, sigma, |yi, mui, si| {
            if yi < 0.0 {
                return 0.0;
            }
            let r = 1.0 / si.max(MIN_POSITIVE);
            let q = r / (r + mui.max(MIN_POSITIVE));
            beta_reg(r, yi.floor() + 1.0, q)
        }))
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip3_map(p, mu, sigma, |pi, mui, si| {
            let r = 1.0 / si.max(MIN_POSITIVE);
            let q = r / (r + mui.max(MIN_POSITIVE));
            discrete_quantile(pi.clamp(0.0, 1.0 - 1e-12), |k| {
                beta_reg(r, k as f64 + 1.0, q)
            })
        }))
    }

    fn name(&self) -> &'static str {
        "NegativeBinomial"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::test_helpers::{
        check_cdf_monotone_in_unit, check_discrete_cdf_matches_pmf, check_score_via_finite_diff,
        derivative_keys_match_parameters, params_view,
    };
    use ndarray::array;

    #[test]
    fn negative_binomial_derivatives() {
        let y = array![0.0, 3.0, 10.0, 25.0];
        let mu = array![1.0, 4.0, 8.0, 20.0];
        let sigma = array![0.5, 0.5, 0.5, 0.5];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        derivative_keys_match_parameters(&NegativeBinomial, p, &y);
    }

    #[test]
    fn loglik_negative_binomial_finite() {
        let owned = [
            ("mu", array![1.0, 4.0, 8.0]),
            ("sigma", array![0.5, 0.5, 0.5]),
        ];
        let p = params_view(&owned);
        let ll = NegativeBinomial
            .loglik(&array![0.0, 5.0, 10.0], &p)
            .unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn variance_negative_binomial_is_mu_plus_sigma_mu_squared() {
        let owned = [("mu", array![2.0]), ("sigma", array![0.5])];
        let p = params_view(&owned);
        let v = NegativeBinomial.variance(&p).unwrap();
        // 2 + 0.5·4 = 4
        assert!((v[0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn score_matches_finite_diff_negative_binomial() {
        let y = array![0.0, 4.0, 10.0];
        let owned = [
            ("mu", array![1.0, 4.0, 8.0]),
            ("sigma", array![0.5, 0.3, 0.4]),
        ];
        check_score_via_finite_diff(&NegativeBinomial, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&NegativeBinomial, &y, &owned, "sigma", 1e-5);
    }

    #[test]
    fn cdf_matches_pmf_negative_binomial() {
        let ks = array![0.0, 2.0, 5.0, 10.0, 20.0];
        let owned = [
            ("mu", array![2.0, 4.0, 6.0, 8.0, 15.0]),
            ("sigma", array![0.5, 0.5, 0.3, 0.4, 0.2]),
        ];
        check_discrete_cdf_matches_pmf(&NegativeBinomial, &ks, &owned, 1e-9);
    }

    #[test]
    fn cdf_monotone_and_quantile_inverts_negative_binomial() {
        let grid = Array1::from_iter((0..40).map(|i| i as f64));
        let owned = [("mu", array![6.0]), ("sigma", array![0.4])];
        check_cdf_monotone_in_unit(&NegativeBinomial, &grid, &owned);
        let p = params_view(&owned);
        for &prob in &[0.05, 0.5, 0.95] {
            let q = NegativeBinomial.quantile(&array![prob], &p).unwrap()[0];
            let f_q = NegativeBinomial.cdf(&array![q], &p).unwrap()[0];
            assert!(f_q >= prob - 1e-12, "F(q)={f_q} < p={prob}");
        }
    }
}
