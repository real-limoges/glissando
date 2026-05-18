//! Student's t distribution for heavy-tailed continuous data.

use super::{
    require, DerivativesResult, Distribution, GamlssError, IdentityLink, Link, LogLink,
    MIN_POSITIVE, MIN_WEIGHT,
};
use crate::math::{digamma_batch, par_zip3_map, par_zip_map, trigamma_batch};
use ndarray::Array1;
use statrs::function::gamma::ln_gamma;
use std::collections::HashMap;

/// Student's t distribution for heavy-tailed continuous data.
///
/// Parameters: `μ` (location, identity), `σ` (scale, log), `ν` (degrees of freedom, log).
/// As `ν → ∞` the distribution approaches Gaussian.
#[derive(Debug, Clone, Copy, Default)]
pub struct StudentT;

impl StudentT {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for StudentT {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma", "nu"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" => Ok(Box::new(IdentityLink)),
            "sigma" | "nu" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Student-t log-likelihood, location-scale parameterization. Full derivation
        // in docs/mathematics.md.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;

        let z = (y - mu) / sigma;
        let z_sq = z.mapv(|v| v * v);

        // The "robustifying weight" w = (ν+1)/(ν+z²) downweights outliers (large |z|).
        // It → 1 as ν → ∞, recovering Gaussian behavior.
        let w_robust = par_zip_map(nu, &z_sq, |nu_i, z2_i| (nu_i + 1.0) / (nu_i + z2_i));

        // μ derivatives (identity link).
        let u_mu = (&w_robust * &z) / sigma;
        let w_mu = &w_robust / sigma.mapv(|s| s * s);

        // σ derivatives (log link). Chain rule: dl/dη = σ · dl/dσ = w·z² − 1.
        let u_sigma = &w_robust * &z_sq - 1.0;
        let w_sigma: Array1<f64> = nu.mapv(|nu_i| (2.0 * nu_i) / (nu_i + 3.0));

        // ν derivatives (log link). Score involves digamma differences.
        let nu_plus_1_half = nu.mapv(|nu_i| (nu_i + 1.0) / 2.0);
        let nu_half = nu.mapv(|nu_i| nu_i / 2.0);
        let d1 = digamma_batch(&nu_plus_1_half);
        let d2 = digamma_batch(&nu_half);

        let term3 = par_zip_map(nu, &z_sq, |nu_i, z2_i| (1.0 + z2_i / nu_i).ln());
        let term4 = par_zip3_map(nu, &w_robust, &z_sq, |nu_i, w_i, z2_i| {
            (w_i * z2_i - 1.0) / nu_i
        });

        let dl_dnu = 0.5 * (&d1 - &d2 - &term3 + &term4);
        // Chain rule for log link: u_η = ν · dl/dν.
        let u_nu = &dl_dnu * nu;

        // Fisher information for ν uses trigamma (the second derivative of log-Γ).
        let t1 = trigamma_batch(&nu_half);
        let t2 = trigamma_batch(&nu_plus_1_half);
        let t3: Array1<f64> = nu.mapv(|nu_i| (2.0 * (nu_i + 3.0)) / (nu_i * (nu_i + 1.0)));
        // The `+ t3` term subtracts from the negative Hessian — sign is correct.
        let i_nu = 0.25 * (&t1 - &t2 + &t3);
        // For log link `W_η = I_ν · ν²`, floored to keep the weight matrix positive definite.
        let w_nu = par_zip_map(&i_nu, nu, |i, nu_i| (i * nu_i * nu_i).abs().max(MIN_WEIGHT));

        Ok(HashMap::from([
            ("mu".to_string(), (u_mu, w_mu)),
            ("sigma".to_string(), (u_sigma, w_sigma)),
            ("nu".to_string(), (u_nu, w_nu)),
        ]))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        let n = y.len();
        let mut out = Array1::<f64>::zeros(n);
        for i in 0..n {
            let s = sigma[i].max(MIN_POSITIVE);
            let nu_i = nu[i].max(MIN_POSITIVE);
            let z = (y[i] - mu[i]) / s;
            out[i] = ln_gamma((nu_i + 1.0) / 2.0)
                - ln_gamma(nu_i / 2.0)
                - 0.5 * (std::f64::consts::PI * nu_i).ln()
                - s.ln()
                - 0.5 * (nu_i + 1.0) * (1.0 + z * z / nu_i).ln();
        }
        Ok(out)
    }

    /// `Var(Y) = σ²·ν/(ν−2)` for `ν > 2`. For `ν ≤ 2` the variance is undefined; we
    /// clamp the denominator at `MIN_POSITIVE` so Pearson residuals stay finite.
    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let sigma = require(self, params, "sigma")?;
        let nu = require(self, params, "nu")?;
        Ok(par_zip_map(sigma, nu, |s, nu_i| {
            let denom = (nu_i - 2.0).max(MIN_POSITIVE);
            s * s * nu_i / denom
        }))
    }

    fn name(&self) -> &'static str {
        "StudentT"
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
    fn studentt_derivatives() {
        let y = array![0.0, 1.0, 2.0, -1.5];
        let mu = array![0.0, 0.5, 1.5, -1.0];
        let sigma = array![1.0, 1.0, 0.8, 1.2];
        let nu = array![5.0, 10.0, 4.0, 8.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        p.insert("nu", &nu);
        derivative_keys_match_parameters(&StudentT, p, &y);
    }

    #[test]
    fn loglik_studentt_matches_cauchy_at_zero() {
        // Student-t with ν=1, μ=0, σ=1 is standard Cauchy. Density at y=0 is 1/π.
        let owned = [
            ("mu", array![0.0]),
            ("sigma", array![1.0]),
            ("nu", array![1.0]),
        ];
        let p = params_view(&owned);
        let ll = StudentT.loglik(&array![0.0], &p).unwrap();
        let expected = -std::f64::consts::PI.ln();
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn loglik_studentt_finite_on_typical_inputs() {
        let owned = [
            ("mu", array![0.0, 1.0, 2.0]),
            ("sigma", array![1.0, 1.5, 0.5]),
            ("nu", array![5.0, 10.0, 4.0]),
        ];
        let p = params_view(&owned);
        let ll = StudentT.loglik(&array![0.5, 0.5, 1.5], &p).unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn variance_studentt_uses_sigma_sq_nu_over_nu_minus_two() {
        let owned = [
            ("mu", array![0.0]),
            ("sigma", array![1.0]),
            ("nu", array![4.0]),
        ];
        let p = params_view(&owned);
        // σ²·ν/(ν−2) = 1·4/2 = 2.
        let v = StudentT.variance(&p).unwrap();
        assert!((v[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn variance_studentt_clamps_at_low_nu() {
        // ν ≤ 2 is undefined; clamp keeps the value finite for downstream Pearson math.
        let owned = [("sigma", array![1.0]), ("nu", array![1.5])];
        let p = params_view(&owned);
        let v = StudentT.variance(&p).unwrap();
        assert!(v[0].is_finite());
        assert!(v[0] > 0.0);
    }

    #[test]
    fn score_matches_finite_diff_studentt() {
        let y = array![-1.0, 0.5, 2.0];
        let owned = [
            ("mu", array![0.0, 0.5, 1.0]),
            ("sigma", array![1.0, 1.2, 0.8]),
            ("nu", array![5.0, 8.0, 4.0]),
        ];
        check_score_via_finite_diff(&StudentT, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&StudentT, &y, &owned, "sigma", 1e-5);
        check_score_via_finite_diff(&StudentT, &y, &owned, "nu", 1e-5);
    }
}
