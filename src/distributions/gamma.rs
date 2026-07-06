//! Gamma distribution for positive continuous data.

use super::{
    require, DerivativesResult, Distribution, GamlssError, Link, LogLink, MIN_POSITIVE, MIN_WEIGHT,
};
use crate::math::{digamma_batch, par_zip3_map, par_zip_map, trigamma_batch};
use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, Gamma as SGamma};
use statrs::function::gamma::{gamma_lr, ln_gamma};
use std::collections::HashMap;

/// Gamma distribution for positive continuous data.
///
/// Parameters: `μ` (mean, log link) and `σ` (coefficient of variation, log link).
/// Parameterization: shape `α = 1/σ²`, scale `θ = μσ²`. `Var(Y) = μ²σ²`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Gamma;

impl Gamma {
    pub fn new() -> Self {
        Self
    }
}

impl Distribution for Gamma {
    fn parameters(&self) -> &[&'static str] {
        &["mu", "sigma"]
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        match param {
            "mu" | "sigma" => Ok(Box::new(LogLink)),
            other => Err(self.unknown_param(other)),
        }
    }

    /// Gamma σ is the coefficient of variation `CV = SD(Y)/E(Y)`, not the raw SD.
    /// The default `initial_value` returns `y.std()`, which is wildly wrong for
    /// Gamma data (e.g. μ=4.5, σ=0.45 → SD≈2.0, but the init should be 0.45).
    /// A bad σ_init causes REML to over-penalize the σ smooth on the first RS
    /// iteration and warm-start into a full-collapse trap.
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => y.mean().expect("validate_inputs rejects empty y"),
            "sigma" => {
                let mu = y.mean().expect("validate_inputs rejects empty y");
                let cv = y.std(1.0) / mu.max(MIN_POSITIVE);
                cv.clamp(0.05, 10.0)
            }
            _ => 0.1,
        }
    }

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Gamma (μ, σ) parameterization: α = 1/σ², θ = μσ².
        // l = −α·log(θ) − log Γ(α) + (α−1)·log(y) − y/θ.
        // μ (log link, η = log μ):  u = (y−μ)/(μσ²),  w = 1/σ².
        // σ (log link, η = log σ):  u = (2/σ²)·[ψ(α) + 2 log σ − log(y/μ) + y/μ − 1].
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        let mu_safe = mu.mapv(|m| m.max(MIN_POSITIVE));
        let sigma_safe = sigma.mapv(|s| s.max(MIN_POSITIVE));
        let sigma_sq = sigma_safe.mapv(|s| s * s);
        let alpha = sigma_sq.mapv(|s2| 1.0 / s2);

        // Clamp y to the support, mirroring loglik_pointwise: a zero/negative row
        // would otherwise send ln(y/μ) to −∞/NaN and poison the whole PWLS solve.
        let y_safe = y.mapv(|yi| yi.max(MIN_POSITIVE));

        let u_mu = (&y_safe - &mu_safe) / (&mu_safe * &sigma_sq);
        let w_mu = sigma_sq.mapv(|s2| 1.0 / s2);

        let psi_alpha = digamma_batch(&alpha);
        let log_sigma = sigma_safe.mapv(|s| s.ln());
        let y_over_mu = &y_safe / &mu_safe;
        let log_y_over_mu = y_over_mu.mapv(|v| v.ln());
        let u_sigma =
            (2.0 / &sigma_sq) * (&psi_alpha + 2.0 * &log_sigma - &log_y_over_mu + &y_over_mu - 1.0);

        // Fisher info for σ on the log-link scale: w = σ²·I_σ with
        // I_σ = (4/σ⁶)·ψ'(α) − 4/σ⁴, so w = (4/σ⁴)·ψ'(α) − 4/σ².
        // Matches gamlss GA's d2ldd2 = (4/σ⁴) − (4/σ⁶)·ψ'(1/σ²) (η-scale via ×σ²)
        // and the Monte-Carlo check E[u_η²]. Since ψ'(1/σ²) > σ² for all σ > 0 the
        // expression is strictly positive; the MIN_WEIGHT floor guards round-off only.
        let psi_prime_alpha = trigamma_batch(&alpha);
        let sigma_sq_sq = sigma_sq.mapv(|s2| s2 * s2);
        let w_sigma =
            ((4.0 / &sigma_sq_sq) * &psi_prime_alpha - 4.0 / &sigma_sq).mapv(|v| v.max(MIN_WEIGHT));

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
            let s = si.max(MIN_POSITIVE);
            let alpha = 1.0 / (s * s);
            let theta = mui * s * s;
            (alpha - 1.0) * yi.max(MIN_POSITIVE).ln()
                - yi / theta
                - alpha * theta.ln()
                - ln_gamma(alpha)
        }))
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip_map(mu, sigma, |m, s| m * m * s * s))
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // shape α = 1/σ², scale s = μσ²; F(y) = P(α, y/s) = gamma_lr(α, y/s).
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip3_map(y, mu, sigma, |yi, mui, si| {
            if yi <= 0.0 {
                return 0.0; // support is y > 0
            }
            let s = si.max(MIN_POSITIVE);
            let shape = 1.0 / (s * s);
            let scale = mui.max(MIN_POSITIVE) * s * s;
            gamma_lr(shape, yi / scale)
        }))
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // statrs Gamma is (shape, rate); rate = 1/scale = 1/(μσ²).
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip3_map(p, mu, sigma, |pi, mui, si| {
            let s = si.max(MIN_POSITIVE);
            let shape = 1.0 / (s * s);
            let rate = 1.0 / (mui.max(MIN_POSITIVE) * s * s);
            SGamma::new(shape, rate)
                .expect("valid Gamma params")
                .inverse_cdf(pi.clamp(1e-12, 1.0 - 1e-12))
        }))
    }

    fn name(&self) -> &'static str {
        "Gamma"
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
    fn gamma_derivatives() {
        let y = array![0.5, 1.5, 3.0, 7.0];
        let mu = array![1.0, 2.0, 4.0, 6.0];
        let sigma = array![0.5, 0.4, 0.3, 0.6];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        derivative_keys_match_parameters(&Gamma, p, &y);
    }

    #[test]
    fn loglik_gamma_finite_on_typical_inputs() {
        let owned = [
            ("mu", array![2.0, 2.0, 4.0]),
            ("sigma", array![0.5, 0.4, 0.3]),
        ];
        let p = params_view(&owned);
        let ll = Gamma.loglik(&array![1.0, 2.0, 5.0], &p).unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn variance_gamma_is_mu_squared_sigma_squared() {
        let owned = [("mu", array![2.0, 3.0]), ("sigma", array![0.5, 0.5])];
        let p = params_view(&owned);
        let v = Gamma.variance(&p).unwrap();
        // μ²σ² = 4·0.25 = 1; 9·0.25 = 2.25.
        assert!((v[0] - 1.0).abs() < 1e-12);
        assert!((v[1] - 2.25).abs() < 1e-12);
    }

    #[test]
    fn score_matches_finite_diff_gamma() {
        let y = array![1.0, 2.5, 5.0];
        let owned = [
            ("mu", array![1.5, 2.0, 4.0]),
            ("sigma", array![0.5, 0.4, 0.3]),
        ];
        check_score_via_finite_diff(&Gamma, &y, &owned, "mu", 1e-5);
        check_score_via_finite_diff(&Gamma, &y, &owned, "sigma", 1e-5);
    }

    #[test]
    fn cdf_quantile_roundtrip_gamma() {
        let y = array![0.5, 1.5, 3.0, 7.0];
        let owned = [
            ("mu", array![1.0, 2.0, 4.0, 6.0]),
            ("sigma", array![0.5, 0.4, 0.3, 0.6]),
        ];
        check_cdf_quantile_roundtrip(&Gamma, &y, &owned, 1e-6);
        check_cdf_pdf_consistency(&Gamma, &y, &owned, 1e-4, 1e-3);
    }

    #[test]
    fn cdf_monotone_gamma_and_zero_below_support() {
        let grid = Array1::from_iter((0..60).map(|i| i as f64 * 0.2));
        let owned = [("mu", array![3.0]), ("sigma", array![0.5])];
        check_cdf_monotone_in_unit(&Gamma, &grid, &owned);
        // Both boundary points (y = 0 and y < 0) sit outside the y > 0 support.
        let boundary_params = [("mu", array![3.0, 3.0]), ("sigma", array![0.5, 0.5])];
        let p = params_view(&boundary_params);
        let at_boundary = Gamma.cdf(&array![0.0, -1.0], &p).unwrap();
        assert_eq!(at_boundary[0], 0.0);
        assert_eq!(at_boundary[1], 0.0);
    }
}
