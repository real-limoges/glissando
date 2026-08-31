//! Negative Binomial (NB2) distribution for overdispersed count data.

use super::{
    discrete_quantile, require, DerivativesResult, Distribution, GamlssError, Link, LogLink,
    DENOM_FLOOR, MIN_POSITIVE,
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
    /// deviation, so the trait default of `sd(y)` is on the wrong scale entirely
    /// (often 10–30× too large for count data). I seed it with the method-of-moments
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

    eta_derivatives_via_chain!();

    fn theta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // NB2 log-likelihood:
        //   l = log Γ(y + 1/σ) − log Γ(1/σ) − log y!
        //       + (1/σ)·log(1/(1+σμ)) + y·log(σμ/(1+σμ)).
        // Natural scale (Altitude #1):
        //   μ: ∂l/∂μ = (y−μ)/(μ(1+σμ)),   i_μ = 1/(μ(1+σμ)).
        // Under the default log link `mu_eta = μ`, so `chain_to_eta` recovers the
        // previous `u_μ = (y−μ)/(1+σμ)` and `w_μ = μ/(1+σμ)`. Weights are returned
        // unfloored.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        // **Every guard here is on a denominator, never on μ or σ themselves.** This
        // body used to clamp both up to `MIN_POSITIVE`, which the folded η-scale form
        // could afford and the un-folded one cannot: `chain_to_eta` multiplies by a
        // `mu_eta` computed from η independently of anything clamped here, so a clamp
        // that binds breaks the telescoping, and `exp(MIN_ETA) ≈ 9.4e-14` is already
        // below `MIN_POSITIVE`. See the same argument at length in `binomial.rs`.
        let one_plus_sigma_mu = par_zip_map(sigma, mu, |s, m| 1.0 + s * m);

        let i_mu = par_zip_map(mu, &one_plus_sigma_mu, |m, d| {
            1.0 / (m * d).max(DENOM_FLOOR)
        });
        let u_mu = (y - mu) * &i_mu;

        // σ (r = 1/σ):
        //   dl/dr = ψ(y+r) − ψ(r) − log(1+σμ) + (μ−y)/(r+μ)
        //   dl/dσ = −(1/σ²)·dl/dr.
        let r = sigma.mapv(|s| 1.0 / s.max(DENOM_FLOOR));
        let y_plus_r = y + &r;
        let psi_y_r = digamma_batch(&y_plus_r);
        let psi_r = digamma_batch(&r);
        let log_term = one_plus_sigma_mu.mapv(|v| v.ln());
        let ratio_term = par_zip3_map(mu, y, &r, |m, yi, ri| (m - yi) / (ri + m).max(DENOM_FLOOR));

        let inv_sigma_sq = sigma.mapv(|s| 1.0 / (s * s).max(DENOM_FLOOR));
        let u_sigma = -(&inv_sigma_sq) * (&psi_y_r - &psi_r - &log_term + &ratio_term);

        // Working "information" for σ: gamlss NBI's squared-score convention
        // (d2ldd2 = −dldd², i.e. the weight is the squared score). The exact
        // expected information has no closed form (it involves E[ψ'(y+r)]); an
        // earlier ψ'(r)/σ² approximation dropped same-order terms and behaved like
        // 1/σ near the Poisson boundary, over-damping σ updates and pushing λ/EDF
        // for σ smooths away from the gamlss oracle.
        //
        // The convention is chain-rule covariant, so it needs no special handling
        // here: taking `i_σ := (∂l/∂σ)²` gives `mu_eta²·i_σ = (mu_eta·∂l/∂σ)² =
        // u_η²`, which is the previous η-scale weight *for any link*, not just the
        // default log one. Returned unfloored, like every other weight.
        let i_sigma = u_sigma.mapv(|u| u * u);

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, i_mu)),
            ("sigma".to_string(), (u_sigma, i_sigma)),
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
        check_cdf_monotone_in_unit, check_discrete_cdf_matches_pmf,
        check_eta_score_via_finite_diff, check_score_via_finite_diff, default_link_derivatives,
        derivative_keys_match_parameters, finite_array, no_nan_array, params_view,
    };
    use crate::distributions::{InverseLink, SqrtLink};
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
    fn score_matches_finite_diff_under_non_default_links() {
        // The Altitude #1 gate. Both parameters default to log, so a non-log link
        // is the only thing that can tell a natural-scale score from an η-scale one.
        let y = array![0.0, 4.0, 10.0];
        let owned = [
            ("mu", array![1.0, 4.0, 8.0]),
            ("sigma", array![0.5, 0.3, 0.4]),
        ];
        check_eta_score_via_finite_diff(&NegativeBinomial, &y, &owned, "mu", &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&NegativeBinomial, &y, &owned, "mu", &InverseLink, 1e-5);
        check_eta_score_via_finite_diff(&NegativeBinomial, &y, &owned, "sigma", &SqrtLink, 1e-5);
    }

    #[test]
    fn derivatives_stay_finite_at_saturated_parameters() {
        // Un-folding introduces `1/(μ(1+σμ))` and `1/σ²` that the previous η-scale
        // forms canceled.
        let y = array![0.0, 3.0, 7.0];
        let owned = [
            ("mu", array![0.0, 1e-320, 1e-8]),
            ("sigma", array![1e-8, 0.0, 1e-320]),
        ];
        let p = params_view(&owned);
        let natural = NegativeBinomial.theta_derivatives(&y, &p).unwrap();
        let chained = default_link_derivatives(&NegativeBinomial, &y, &p).unwrap();
        for name in ["mu", "sigma"] {
            let (u_n, i_n) = &natural[name];
            assert!(no_nan_array(u_n) && no_nan_array(i_n), "natural {name}");
            let (u, w) = &chained[name];
            assert!(finite_array(u) && finite_array(w), "chained {name}: {u:?}");
            assert!(w.iter().all(|&v| v >= 0.0));
        }
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
