//! Poisson distribution for count data.

use super::{
    discrete_quantile, require, DerivativesResult, Distribution, GamlssError, Link, LogLink,
    DENOM_FLOOR, MIN_POSITIVE,
};
use crate::math::par_zip_map;
use ndarray::Array1;
use statrs::function::gamma::{gamma_ur, ln_gamma};
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

    eta_derivatives_via_chain!();

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Log-likelihood: l = y·log(μ) − μ.
        // Natural scale:  ∂l/∂μ = (y−μ)/μ,   i_μ = 1/μ.
        //
        // Under the default log link `mu_eta = μ`, so `chain_to_eta` recovers the
        // classic `u_η = y − μ`, `w_η = μ` exactly. The weight is returned
        // unfloored: `MIN_WEIGHT` is applied once, in `scoring::step`, after the
        // chain rule.
        let mu = require(self, params, "mu")?;
        // `1/μ` is both the reciprocal in the score and the information itself.
        let i_mu = mu.mapv(|m| 1.0 / m.max(DENOM_FLOOR));
        let u_mu = (y - mu) * &i_mu;
        Ok(HashMap::from([("mu".to_string(), (u_mu, i_mu))]))
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

    fn is_discrete(&self) -> bool {
        true
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // F(⌊y⌋ | μ) = Q(⌊y⌋+1, μ) = gamma_ur(⌊y⌋+1, μ) — the upper-incomplete-gamma
        // identity for the Poisson CDF (no summation loop).
        let mu = require(self, params, "mu")?;
        Ok(par_zip_map(y, mu, |yi, mui| {
            if yi < 0.0 {
                return 0.0;
            }
            gamma_ur(yi.floor() + 1.0, mui.max(MIN_POSITIVE))
        }))
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        Ok(par_zip_map(p, mu, |pi, mui| {
            let m = mui.max(MIN_POSITIVE);
            discrete_quantile(pi.clamp(0.0, 1.0 - 1e-12), |k| gamma_ur(k as f64 + 1.0, m))
        }))
    }

    fn name(&self) -> &'static str {
        "Poisson"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::test_helpers::{
        check_cdf_monotone_in_unit, check_discrete_cdf_matches_pmf,
        check_eta_score_via_finite_diff, check_score_via_finite_diff, default_link_derivatives,
        derivative_keys_match_parameters, finite_array, params_view,
    };
    use crate::distributions::{InverseLink, SqrtLink};
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
        let derivs = default_link_derivatives(&Poisson, &y, &p).unwrap();
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

    #[test]
    fn score_matches_finite_diff_under_a_sqrt_link() {
        // The Altitude #1 gate. Under the default log link `∂l/∂η` and the folded
        // `y − μ` agree by construction, so the default-link check above cannot
        // tell a natural-scale score from an η-scale one. `sqrt` can: it wants
        // `dμ/dη = 2√μ`, which this family no longer hardcodes.
        //
        // This replaces `poisson_score_is_wrong_under_a_sqrt_link_today`, the
        // Phase 0 characterization test that asserted the opposite.
        let y = array![0.0, 1.0, 4.0, 9.0, 6.0];
        let owned = [("mu", array![0.5, 1.5, 3.0, 8.0, 5.0])];
        check_eta_score_via_finite_diff(&Poisson, &y, &owned, "mu", &SqrtLink, 1e-5);
        check_eta_score_via_finite_diff(&Poisson, &y, &owned, "mu", &InverseLink, 1e-5);
    }

    #[test]
    fn derivatives_stay_finite_at_a_saturated_mu() {
        // Un-folding introduces a `1/μ` the old `u = y − μ` cancelled, so the
        // saturated tail is newly reachable arithmetic. `DENOM_FLOOR` has to keep
        // both the natural score and the chained η-score finite there, including
        // where μ has underflowed to exactly zero.
        let y = array![0.0, 3.0, 10.0];
        let owned = [("mu", array![0.0, 1e-320, 1e-8])];
        let p = params_view(&owned);
        let natural = Poisson.derivatives(&y, &p).unwrap();
        let (u_nat, i_nat) = &natural["mu"];
        assert!(finite_array(u_nat) && finite_array(i_nat));

        let chained = default_link_derivatives(&Poisson, &y, &p).unwrap();
        let (u, w) = &chained["mu"];
        assert!(finite_array(u) && finite_array(w), "u={u:?} w={w:?}");
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn cdf_matches_pmf_poisson() {
        let ks = array![0.0, 1.0, 3.0, 7.0, 12.0];
        let owned = [("mu", array![2.0, 2.0, 4.0, 6.0, 10.0])];
        check_discrete_cdf_matches_pmf(&Poisson, &ks, &owned, 1e-9);
    }

    #[test]
    fn cdf_monotone_and_quantile_inverts_poisson() {
        let grid = Array1::from_iter((0..30).map(|i| i as f64));
        let owned = [("mu", array![5.0])];
        check_cdf_monotone_in_unit(&Poisson, &grid, &owned);
        // Quantile is the smallest k with F(k) ≥ p; check it brackets the CDF.
        let p = params_view(&owned);
        for &prob in &[0.05, 0.5, 0.95] {
            let q = Poisson.quantile(&array![prob], &p).unwrap()[0];
            let f_q = Poisson.cdf(&array![q], &p).unwrap()[0];
            assert!(f_q >= prob - 1e-12, "F(q)={f_q} < p={prob}");
            if q > 0.0 {
                let f_qm1 = Poisson.cdf(&array![q - 1.0], &p).unwrap()[0];
                assert!(f_qm1 < prob, "F(q-1)={f_qm1} should be < p={prob}");
            }
        }
    }
}
