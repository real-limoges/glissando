//! Gamma distribution for positive continuous data.

use super::{
    clamp_prob, require, DerivativesResult, Distribution, GamlssError, Link, LogLink, DENOM_FLOOR,
    MIN_POSITIVE,
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

    eta_derivatives_via_chain!();

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Gamma (μ, σ) parameterization: α = 1/σ², θ = μσ².
        // l = −α·log(θ) − log Γ(α) + (α−1)·log(y) − y/θ.
        // Natural scale (Altitude #1):
        //   μ: ∂l/∂μ = (y−μ)/(μ²σ²),   i_μ = 1/(μ²σ²).
        //   σ: ∂l/∂σ = (2/σ³)·[ψ(α) + 2 log σ − log(y/μ) + y/μ − 1],
        //      i_σ = (4/σ⁶)·ψ'(α) − 4/σ⁴.
        // Both default links are log, so `chain_to_eta` (mu_eta = μ, σ) recovers
        // the previous η-scale pairs exactly. Weights are returned unfloored.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        let mu_safe = mu.mapv(|m| m.max(MIN_POSITIVE));
        let sigma_safe = sigma.mapv(|s| s.max(MIN_POSITIVE));
        let sigma_sq = sigma_safe.mapv(|s| s * s);
        let alpha = sigma_sq.mapv(|s2| 1.0 / s2);

        // Guard each reciprocal at the power it is used at, rather than clamping μ
        // or σ any further: raising an already-guarded reciprocal to a power would
        // overflow to infinity for a parameter the log link can still underflow to,
        // and `inf · 0` is NaN. σ⁶ is why this matters more here than elsewhere; it
        // underflows around σ ≈ 1e-52, where the old σ⁴ form reached to σ ≈ 1e-77.
        let inv_mu_sq_sigma_sq = par_zip_map(&mu_safe, &sigma_sq, |m, s2| {
            1.0 / (m * m * s2).max(DENOM_FLOOR)
        });
        let inv_sigma_cubed = sigma_safe.mapv(|s| 1.0 / (s * s * s).max(DENOM_FLOOR));
        let inv_sigma_4 = sigma_sq.mapv(|s2| 1.0 / (s2 * s2).max(DENOM_FLOOR));
        let inv_sigma_6 = sigma_sq.mapv(|s2| 1.0 / (s2 * s2 * s2).max(DENOM_FLOOR));

        // Clamp y to the support, mirroring loglik_pointwise: a zero/negative row
        // would otherwise send ln(y/μ) to −∞/NaN and poison the whole PWLS solve.
        let y_safe = y.mapv(|yi| yi.max(MIN_POSITIVE));

        let u_mu = (&y_safe - &mu_safe) * &inv_mu_sq_sigma_sq;
        let i_mu = inv_mu_sq_sigma_sq;

        let psi_alpha = digamma_batch(&alpha);
        let log_sigma = sigma_safe.mapv(|s| s.ln());
        let y_over_mu = &y_safe / &mu_safe;
        let log_y_over_mu = y_over_mu.mapv(|v| v.ln());
        let u_sigma = 2.0
            * &inv_sigma_cubed
            * (&psi_alpha + 2.0 * &log_sigma - &log_y_over_mu + &y_over_mu - 1.0);

        // Fisher info for σ on its own scale: I_σ = (4/σ⁶)·ψ'(α) − 4/σ⁴. Matches
        // gamlss GA's d2ldd2 = (4/σ⁴) − (4/σ⁶)·ψ'(1/σ²) and the Monte-Carlo check
        // E[u_η²] once chained. Since ψ'(1/σ²) > σ² for all σ > 0 the expression is
        // strictly positive; the surviving MIN_WEIGHT floor in `scoring::step`
        // guards round-off only.
        let psi_prime_alpha = trigamma_batch(&alpha);
        let i_sigma = 4.0 * &inv_sigma_6 * &psi_prime_alpha - 4.0 * &inv_sigma_4;

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

    fn cdf_theta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> super::CdfThetaResult {
        // μ enters F = P(α, x) only through x = y/(μσ²), α = 1/σ², so ∂x/∂μ = −x/μ
        // holds the shape α fixed and the shape-derivative that blocks σ never
        // appears. Writing `mass` for the γ-density factor xᵅ·e⁻ˣ/Γ(α), the
        // natural-scale derivatives (Altitude #1) are:
        //   ∂F/∂μ  = −mass/μ
        //   ∂²F/∂μ² = mass·(1 + α − x)/μ²
        // The caller chains to η. Under the default log link mu_eta = mu_eta2 = μ,
        // which recovers the previous η-scale pair exactly:
        // μ·(−mass/μ) = −mass and mass(1+α−x) − mass = (x − α)·(−mass).
        // σ enters both α and x; its CDF derivative needs ∂P/∂α (non-elementary)
        // and is left to the wrapper's numeric fallback.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        let mut d1 = Array1::<f64>::zeros(y.len());
        let mut d2 = Array1::<f64>::zeros(y.len());
        for i in 0..y.len() {
            if !y[i].is_finite() || y[i] <= 0.0 {
                continue; // outside support or ±∞ bound: F flat, derivatives ≡ 0
            }
            let s = sigma[i].max(MIN_POSITIVE);
            let alpha = 1.0 / (s * s);
            let x = y[i] / (mu[i].max(MIN_POSITIVE) * s * s);
            // γ-density mass at x: xᵅ·e⁻ˣ / Γ(α) = exp(α·ln x − x − lnΓ(α)).
            let mass = (alpha * x.ln() - x - ln_gamma(alpha)).exp();
            // Un-folding puts μ in a denominator for the first time, so it gets a
            // `DENOM_FLOOR` guard rather than the `MIN_POSITIVE` clamp `x` uses:
            // the caller multiplies by a `mu_eta` computed from η independently of
            // anything clamped here, and `MIN_POSITIVE = 1e-10` sits *above* the
            // log link's own floor (`exp(MIN_ETA) ≈ 9.4e-14`), so a clamp that
            // binds would break the telescoping. Each power is guarded where it is
            // used — `μ²` can underflow to zero for a μ that `μ` alone survives.
            let denom1 = mu[i].max(DENOM_FLOOR);
            let denom2 = (mu[i] * mu[i]).max(DENOM_FLOOR);
            d1[i] = -mass / denom1;
            d2[i] = mass * (1.0 + alpha - x) / denom2;
        }
        Ok(HashMap::from([("mu".to_string(), (d1, d2))]))
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
                .inverse_cdf(clamp_prob(pi))
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
        check_cdf_theta_derivatives_via_finite_diff, check_eta_score_via_finite_diff,
        check_score_via_finite_diff, default_link_derivatives, derivative_keys_match_parameters,
        finite_array, params_view,
    };
    use crate::distributions::{InverseLink, SqrtLink};
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
    fn score_matches_finite_diff_under_non_default_links() {
        // The Altitude #1 gate. `inverse` on μ is the link behind the largest of the
        // four `link_mle_oracle` shortfalls (45.5 in fitted log-likelihood), so it is
        // the one this family most needs covered.
        let y = array![1.0, 2.5, 5.0];
        let owned = [
            ("mu", array![1.5, 2.0, 4.0]),
            ("sigma", array![0.5, 0.4, 0.3]),
        ];
        check_eta_score_via_finite_diff(&Gamma, &y, &owned, "mu", &InverseLink, 1e-5);
        check_eta_score_via_finite_diff(&Gamma, &y, &owned, "mu", &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&Gamma, &y, &owned, "sigma", &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&Gamma, &y, &owned, "sigma", &InverseLink, 1e-5);
    }

    #[test]
    fn derivatives_stay_finite_at_saturated_parameters() {
        // Un-folding introduces `1/(μ²σ²)`, `1/σ³`, `1/σ⁴` and `1/σ⁶`, all of which
        // the previous η-scale forms cancelled down to at most `1/σ⁴`. σ⁶ is the
        // first to underflow, so this fixture is the one that pins the guard.
        let y = array![1.0, 2.0, 3.0];
        let owned = [
            ("mu", array![0.0, 1e-320, 1e-8]),
            ("sigma", array![1e-60, 0.0, 1e-320]),
        ];
        let p = params_view(&owned);
        let natural = Gamma.derivatives(&y, &p).unwrap();
        let chained = default_link_derivatives(&Gamma, &y, &p).unwrap();
        for name in ["mu", "sigma"] {
            let (u_n, i_n) = &natural[name];
            assert!(finite_array(u_n) && finite_array(i_n), "natural {name}");
            let (u, w) = &chained[name];
            assert!(finite_array(u) && finite_array(w), "chained {name}: {u:?}");
        }
    }

    #[test]
    fn cdf_theta_derivatives_match_finite_diff_gamma() {
        // Only μ is analytic; σ is intentionally left to the numeric fallback.
        let y = array![0.5, 1.5, 3.0, 7.0];
        let owned = [
            ("mu", array![1.0, 2.0, 4.0, 6.0]),
            ("sigma", array![0.5, 0.4, 0.3, 0.6]),
        ];
        check_cdf_theta_derivatives_via_finite_diff(&Gamma, &y, &owned, "mu", 2e-4);
        let p = params_view(&owned);
        let derivs = Gamma.cdf_theta_derivatives(&y, &p).unwrap();
        assert!(!derivs.contains_key("sigma"));
    }

    #[test]
    fn cdf_theta_derivatives_stay_finite_at_a_saturated_mu() {
        // Un-folding put μ and μ² in denominators that the η-scale form (`−mass`,
        // `(x−α)·−mass`) had cancelled away entirely, which is why each power gets
        // its own `DENOM_FLOOR`: μ² underflows to exactly zero for a μ that μ alone
        // survives, and `inf · 0` is NaN.
        let y = array![1.0, 2.0, 0.5, 3.0];
        let owned = [
            ("mu", array![0.0, 1e-320, 1e-8, 1e13]),
            ("sigma", array![0.5, 0.5, 1e-8, 1e13]),
        ];
        let p = params_view(&owned);
        let d = Gamma.cdf_theta_derivatives(&y, &p).unwrap();
        let (d1, d2) = &d["mu"];
        assert!(finite_array(d1) && finite_array(d2), "{d1:?} {d2:?}");
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
