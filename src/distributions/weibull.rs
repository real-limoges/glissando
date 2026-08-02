//! Weibull distribution for positive continuous data

use super::{
    clamp_prob, require, DerivativesResult, Distribution, GamlssError, Link, LogLink, DENOM_FLOOR,
    MIN_POSITIVE,
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

    eta_derivatives_via_chain!();

    fn theta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // z = (y/μ)^σ ~ Exp(1) at the truth. Natural scale (Altitude #1):
        //   μ: ∂l/∂μ = σ(z−1)/μ,                    i_μ = σ²/μ².
        //   σ: ∂l/∂σ = [1 + σ·ln(y/μ)·(1−z)]/σ,     i_σ = (π²/6 + (1−γ)²)/σ².
        // Both default links are log, so `chain_to_eta` (mu_eta = μ, σ) recovers
        // the previous `u_μ = σ(z−1)`, `w_μ = σ²`, `u_σ = 1 + σ·ln(y/μ)(1−z)` and
        // the constant `w_σ`. Weights are returned unfloored.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        const EULER: f64 = 0.577_215_664_901_532_9;
        let i_sigma_numer = std::f64::consts::PI.powi(2) / 6.0 + (1.0 - EULER).powi(2);

        // Guard each reciprocal rather than clamping μ or σ, so the value the chain
        // rule multiplies back in stays exactly the caller's parameter. Guard each
        // denominator *at the power it is used at*: squaring an already-guarded
        // reciprocal would overflow to infinity for a parameter the log link can
        // still underflow to, and `inf · 0` is NaN.
        //
        // `z = (y/μ)^σ` must be built from the **same** guarded μ as the reciprocal
        // it multiplies. Evaluating the numerator at a μ clamped up to
        // `MIN_POSITIVE` while dividing by a μ floored at `DENOM_FLOOR` left the two
        // 290 orders of magnitude apart, and the mismatch was not a rounding
        // difference: at μ = 0 with σ = 5 the numerator stayed a finite ~1e50 while
        // `1/μ` reached 1e300, so their product overflowed to +∞ where the honest
        // value is ~1e350 (beyond f64 either way), but `chain_to_eta` then met that
        // ∞ with a `mu_eta` of exactly 0 (`sqrt` at η = 0, `log` at η ≤ −745) and
        // produced NaN. Sharing one guard leaves the overflow to `chain_to_eta`,
        // which annihilates it against a zero `mu_eta` and saturates it otherwise.
        let mu_guarded = mu.mapv(|m| m.max(DENOM_FLOOR));
        let sigma_guarded = sigma.mapv(|s| s.max(DENOM_FLOOR));
        let inv_mu = mu_guarded.mapv(|m| 1.0 / m);
        let inv_mu_sq = mu.mapv(|m| 1.0 / (m * m).max(DENOM_FLOOR));
        let inv_sigma = sigma_guarded.mapv(|s| 1.0 / s);
        let inv_sigma_sq = sigma.mapv(|s| 1.0 / (s * s).max(DENOM_FLOOR));

        let u_mu = par_zip3_map(y, &mu_guarded, sigma, |yi, m, si| {
            let z = (yi.max(MIN_POSITIVE) / m).powf(si);
            si * (z - 1.0)
        }) * &inv_mu;
        let i_mu = par_zip_map(sigma, &inv_mu_sq, |s, ims| (s * s) * ims);

        let u_sigma = par_zip3_map(y, &mu_guarded, sigma, |yi, m, si| {
            let r = yi.max(MIN_POSITIVE) / m;
            let z = r.powf(si);
            1.0 + si * r.ln() * (1.0 - z)
        }) * &inv_sigma;
        let i_sigma = inv_sigma_sq.mapv(|iss| i_sigma_numer * iss);

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
        check_eta_score_via_finite_diff, check_score_via_finite_diff, default_link_derivatives,
        derivative_keys_match_parameters, finite_array, no_nan_array, params_view, ParamLinks,
    };
    use crate::distributions::{InverseLink, SqrtLink};
    use ndarray::array;

    #[test]
    fn chained_derivatives_survive_a_zero_mu_eta() {
        // At η = 0 the sqrt link gives μ = 0 *and* `mu_eta` = 0, and σ = 5 is enough
        // for `z = (y/μ)^σ` to overflow rather than merely grow, so the natural-scale
        // score is +∞ and the product is `∞ · 0`. Nothing downstream would catch the
        // resulting NaN: `scoring::step`'s `w < MIN_WEIGHT` and `step > MAX_STEP`
        // tests are both false for NaN, so it reaches the PWLS solve and poisons it.
        let y = array![1.0, 2.0, 3.0];
        let owned = [
            ("mu", array![0.0, 0.0, 1.0]),
            ("sigma", array![5.0, 5.0, 5.0]),
        ];
        let p = params_view(&owned);
        let links = ParamLinks::overriding(&Weibull, &p, "mu", &SqrtLink);
        let d = Weibull.eta_derivatives(&y, &p, &links.context()).unwrap();
        for name in ["mu", "sigma"] {
            let (u, w) = &d[name];
            assert!(
                finite_array(u) && finite_array(w),
                "{name}: u={u:?} w={w:?}"
            );
        }
        // A frozen row contributes nothing, not a saturated something.
        assert_eq!((d["mu"].0[0], d["mu"].1[0]), (0.0, 0.0));
    }

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
    fn score_matches_finite_diff_under_non_default_links() {
        // The Altitude #1 gate: both parameters default to a log link, under which
        // `∂l/∂η` and the folded form agree by construction. `sqrt` and `inverse`
        // want a different `dμ/dη`, which this family no longer hardcodes.
        let y = array![1.0, 2.5, 5.0];
        let owned = [
            ("mu", array![1.5, 2.0, 4.0]),
            ("sigma", array![0.8, 1.0, 1.5]),
        ];
        check_eta_score_via_finite_diff(&Weibull, &y, &owned, "mu", &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&Weibull, &y, &owned, "mu", &InverseLink, 1e-5);
        check_eta_score_via_finite_diff(&Weibull, &y, &owned, "sigma", &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&Weibull, &y, &owned, "sigma", &InverseLink, 1e-5);
    }

    #[test]
    fn derivatives_stay_finite_at_saturated_parameters() {
        // Un-folding introduces `1/μ` and `1/σ` the old η-scale forms cancelled.
        let y = array![1.0, 2.0, 3.0];
        let owned = [
            ("mu", array![0.0, 1e-320, 1e-8]),
            ("sigma", array![1e-8, 0.0, 1e-320]),
        ];
        let p = params_view(&owned);
        let natural = Weibull.theta_derivatives(&y, &p).unwrap();
        let chained = default_link_derivatives(&Weibull, &y, &p).unwrap();
        for name in ["mu", "sigma"] {
            let (u_n, i_n) = &natural[name];
            assert!(no_nan_array(u_n) && no_nan_array(i_n), "natural {name}");
            let (u, w) = &chained[name];
            assert!(finite_array(u) && finite_array(w), "chained {name}: {u:?}");
            assert!(w.iter().all(|&v| v >= 0.0));
        }
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
