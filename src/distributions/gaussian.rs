//! Gaussian (Normal) distribution.

use super::{
    require, DerivativesResult, Distribution, GamlssError, IdentityLink, Link, LogLink,
    DENOM_FLOOR, MIN_POSITIVE,
};
use crate::math::{par_zip3_map, std_normal_cdf, std_normal_pdf, std_normal_quantile};
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

    eta_derivatives_via_chain!();

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Gaussian log-likelihood:  l = −0.5·log(2π) − log(σ) − (y−μ)²/(2σ²).
        // Natural scale (Altitude #1; no link folded in):
        //   μ:  ∂l/∂μ = (y−μ)/σ²,               i_μ = 1/σ².
        //   σ:  ∂l/∂σ = ((y−μ)² − σ²)/σ³,       i_σ = 2/σ².
        // Full derivation in docs/math/mathematics.md.
        //
        // `chain_to_eta` recovers the classic η-scale pairs under the default
        // links: μ is identity (`mu_eta = 1`, so its entries pass through
        // untouched), and σ is log (`mu_eta = σ`), giving `u_η = ((y−μ)²−σ²)/σ²`
        // and `w_η = σ²·(2/σ²) = 2`. The weights are returned unfloored.
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        let sigma_sq = sigma.mapv(|s| s * s);
        // Guard each reciprocal at its own power of σ rather than clamping σ, so
        // the value `chain_to_eta` multiplies back in stays exactly the caller's σ.
        // Guarding σ instead and then cubing would overflow to infinity for a σ the
        // log link can still underflow to, and `inf · 0` is NaN.
        let inv_sigma_sq = sigma_sq.mapv(|s2| 1.0 / s2.max(DENOM_FLOOR));
        let inv_sigma_cubed = sigma.mapv(|s| 1.0 / (s * s * s).max(DENOM_FLOOR));
        let residual = y - mu;
        let residual_sq = residual.mapv(|r| r * r);

        let u_mu = &residual * &inv_sigma_sq;
        let i_mu = inv_sigma_sq.clone();

        let u_sigma = (&residual_sq - &sigma_sq) * &inv_sigma_cubed;
        let i_sigma = 2.0 * &inv_sigma_sq;

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

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        Ok(par_zip3_map(y, mu, sigma, |yi, mui, si| {
            let z = (yi - mui) / si.max(MIN_POSITIVE);
            std_normal_cdf(z)
        }))
    }

    fn cdf_theta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> super::CdfThetaResult {
        // Natural-scale (Altitude #1) location-scale derivatives of F = Φ(z),
        // z = (y−μ)/σ, std-normal pdf φ, φ'(z) = −z·φ(z). ∂z/∂μ = −1/σ and
        // ∂z/∂σ = −z/σ, so:
        //   μ:  ∂F/∂μ = −φ/σ,    ∂²F/∂μ² = φ'/σ² = −zφ/σ².
        //   σ:  ∂F/∂σ = −zφ/σ,   ∂²F/∂σ² = zφ(2 − z²)/σ².
        // The caller chains to η. Under the default links (identity, log) that
        // recovers the previous η-scale forms exactly: μ has mu_eta = 1 and
        // mu_eta2 = 0, so it is unchanged; σ has mu_eta = mu_eta2 = σ, giving
        // σ·(−zφ/σ) = −zφ and σ²·zφ(2−z²)/σ² + σ·(−zφ/σ) = zφ(1 − z²).
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;

        let mut d1_mu = Array1::<f64>::zeros(y.len());
        let mut d2_mu = Array1::<f64>::zeros(y.len());
        let mut d1_sigma = Array1::<f64>::zeros(y.len());
        let mut d2_sigma = Array1::<f64>::zeros(y.len());
        for i in 0..y.len() {
            if !y[i].is_finite() {
                continue; // ±∞ bound: F saturates, all derivatives vanish
            }
            let s = sigma[i].max(MIN_POSITIVE);
            let z = (y[i] - mu[i]) / s;
            let phi = std_normal_pdf(z);
            let z_sq = z * z;
            d1_mu[i] = -phi / s;
            d2_mu[i] = -z * phi / (s * s);
            d1_sigma[i] = -z * phi / s;
            // φ decays like e^{−z²/2}, so φ·z^k → 0 for every k and the limit here is
            // 0. Take it explicitly: z² overflows to infinity around |z| ≈ 1e154,
            // long after φ has underflowed to exactly 0, and `0 · ∞` is NaN. The
            // guard fires only where the unguarded expression is NaN, so every
            // in-range value is bit-identical. Reachable from a wrapper evaluating
            // `F` at a far-out censoring or truncation bound.
            d2_sigma[i] = if z_sq.is_finite() {
                z * phi * (2.0 - z_sq) / (s * s)
            } else {
                0.0
            };
        }
        Ok(HashMap::from([
            ("mu".to_string(), (d1_mu, d2_mu)),
            ("sigma".to_string(), (d1_sigma, d2_sigma)),
        ]))
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let sigma = require(self, params, "sigma")?;
        // Q(p) = μ + σ·Φ⁻¹(p); Φ⁻¹ is shared with the quantile residuals (INFER-1).
        Ok(par_zip3_map(p, mu, sigma, |pi, mui, si| {
            mui + si * std_normal_quantile(pi)
        }))
    }

    fn name(&self) -> &'static str {
        "Gaussian"
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
        let derivs = default_link_derivatives(&Gaussian, &y, &p).unwrap();
        let (u_mu, w_mu) = &derivs["mu"];
        assert!(u_mu.iter().all(|&v| v.abs() < 1e-12));
        // w_mu = 1/sigma^2 = 1.0
        assert!(w_mu.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    #[test]
    fn score_matches_finite_diff_under_non_default_links() {
        // The Altitude #1 gate. Gaussian μ is identity-linked, so a default-link
        // finite difference cannot distinguish `∂l/∂μ` from `∂l/∂η` at all; a log
        // link on μ is what makes the check bite. This replaces
        // `identity_link_parameters_are_only_checked_vacuously_today`, the Phase 0
        // characterization test that asserted the opposite.
        //
        // μ is positive throughout the fixture so the log and sqrt links are
        // well defined on it.
        let y = array![0.5, 1.0, 2.0, 3.5, 5.0];
        let owned = [
            ("mu", array![1.0, 1.5, 2.5, 3.0, 4.0]),
            ("sigma", array![1.0, 1.2, 0.8, 1.5, 1.0]),
        ];
        check_eta_score_via_finite_diff(&Gaussian, &y, &owned, "mu", &LogLink, 1e-5);
        check_eta_score_via_finite_diff(&Gaussian, &y, &owned, "mu", &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&Gaussian, &y, &owned, "sigma", &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&Gaussian, &y, &owned, "sigma", &InverseLink, 1e-5);
    }

    #[test]
    fn derivatives_stay_finite_at_a_saturated_sigma() {
        // Un-folding introduces `1/σ²` and `1/σ³` that the old `w_σ = 2` cancelled.
        // Both have to stay finite where the log link can still land, including
        // where σ has underflowed to exactly zero.
        let y = array![0.0, 1.0, 2.0];
        let owned = [
            ("mu", array![0.0, 0.0, 0.0]),
            ("sigma", array![0.0, 1e-320, 1e-8]),
        ];
        let p = params_view(&owned);
        let natural = Gaussian.derivatives(&y, &p).unwrap();
        let chained = default_link_derivatives(&Gaussian, &y, &p).unwrap();
        for name in ["mu", "sigma"] {
            let (u_n, i_n) = &natural[name];
            assert!(finite_array(u_n) && finite_array(i_n), "natural {name}");
            let (u, w) = &chained[name];
            assert!(finite_array(u) && finite_array(w), "chained {name}: {u:?}");
            assert!(w.iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn gaussian_sigma_fisher_info_constant() {
        let y = array![0.0, 1.0, 2.0];
        let mu = array![0.0, 0.0, 0.0];
        let sigma = array![1.0, 2.0, 3.0];
        let mut p = HashMap::new();
        p.insert("mu", &mu);
        p.insert("sigma", &sigma);
        let derivs = default_link_derivatives(&Gaussian, &y, &p).unwrap();
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

    #[test]
    fn cdf_theta_derivatives_match_finite_diff_gaussian() {
        let y = array![-1.5, 0.0, 0.7, 2.3];
        let owned = [
            ("mu", array![-0.5, 0.2, 1.0, 1.5]),
            ("sigma", array![1.0, 1.3, 0.8, 1.1]),
        ];
        check_cdf_theta_derivatives_via_finite_diff(&Gaussian, &y, &owned, "mu", 1e-4);
        check_cdf_theta_derivatives_via_finite_diff(&Gaussian, &y, &owned, "sigma", 1e-4);
    }

    #[test]
    fn cdf_theta_derivatives_stay_finite_at_a_saturated_sigma() {
        // Un-folding σ put a `1/σ` in `∂F/∂σ` and a `1/σ²` in `∂²F/∂σ²` where the
        // η-scale forms had none, so the saturated tail became reachable
        // arithmetic. Span both ends of what the log link can produce inside its
        // own η clamp, plus σ underflowed to exactly zero.
        let y = array![0.0, 1.0, 2.0, -1.0];
        let owned = [
            ("mu", array![0.0, 0.0, 0.0, 0.0]),
            ("sigma", array![0.0, 1e-320, 1e-8, 1e13]),
        ];
        let p = params_view(&owned);
        let d = Gaussian.cdf_theta_derivatives(&y, &p).unwrap();
        for name in ["mu", "sigma"] {
            let (d1, d2) = &d[name];
            assert!(
                finite_array(d1) && finite_array(d2),
                "{name}: {d1:?} {d2:?}"
            );
        }
    }

    #[test]
    fn cdf_quantile_roundtrip_gaussian() {
        let y = array![-2.0, -0.3, 0.0, 1.7, 4.0];
        let owned = [
            ("mu", array![0.0, 0.5, -1.0, 2.0, 3.0]),
            ("sigma", array![1.0, 1.5, 0.8, 2.0, 1.2]),
        ];
        check_cdf_quantile_roundtrip(&Gaussian, &y, &owned, 1e-7);
        check_cdf_pdf_consistency(&Gaussian, &y, &owned, 1e-4, 1e-4);
    }

    #[test]
    fn cdf_monotone_and_median_is_mu_gaussian() {
        let grid = Array1::from_iter((0..50).map(|i| -5.0 + i as f64 * 0.2));
        let owned = [("mu", array![0.7]), ("sigma", array![1.3])];
        check_cdf_monotone_in_unit(&Gaussian, &grid, &owned);
        // 50th percentile of a symmetric family is the mean.
        let p = params_view(&owned);
        let med = Gaussian.quantile(&array![0.5], &p).unwrap();
        assert!((med[0] - 0.7).abs() < 1e-9);
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn cdf_roundtrip_proptest_gaussian(
            mu_val in -5.0f64..5.0,
            sigma_val in 0.2f64..3.0,
            // Draw the point as a standardized z within ±6 so F(y) stays clear of
            // the tail where the CDF saturates to 0/1 and the round-trip loses y.
            z in -6.0f64..6.0,
        ) {
            let y_val = mu_val + z * sigma_val;
            let owned = [("mu", array![mu_val]), ("sigma", array![sigma_val])];
            let p = params_view(&owned);
            let u = Gaussian.cdf(&array![y_val], &p).unwrap();
            prop_assert!(u[0] >= 0.0 && u[0] <= 1.0);
            let back = Gaussian.quantile(&u, &p).unwrap();
            prop_assert!((back[0] - y_val).abs() < 1e-5);
        }

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
