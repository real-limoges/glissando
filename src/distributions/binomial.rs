//! Binomial distribution: counts of successes out of `n` trials.

use super::{
    discrete_quantile, require, DerivativesResult, Distribution, GamlssError, Link, LogitLink,
    DENOM_FLOOR, MIN_POSITIVE,
};
use crate::math::{par_zip3_map, par_zip_map};
use ndarray::Array1;
use statrs::function::beta::beta_reg;
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

    eta_derivatives_via_chain!();

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        // Binomial log-likelihood: l = y·log(μ) + (n−y)·log(1−μ) + log C(n, y).
        // Natural scale (Altitude #1):
        //   ∂l/∂μ = (y − n·μ) / (μ(1−μ)),   i_μ = n / (μ(1−μ)).
        // Under the default logit link `mu_eta = μ(1−μ)`, so `chain_to_eta` recovers
        // the classic `u_η = y − n·μ` and `w_η = n·μ(1−μ)`. Returned unfloored.
        //
        // **The guard is on the denominator, not on μ.** This family used to clamp
        // `μ ∈ [MIN_POSITIVE, 1−MIN_POSITIVE]` with `MIN_POSITIVE = 1e-10`, which the
        // folded form could afford because the division cancelled. It cannot survive
        // the un-fold: `chain_to_eta` multiplies by a `mu_eta` computed from η
        // independently of anything clamped here, so a clamp that binds breaks the
        // telescoping. Under a probit link at η = −30 (the link's own clamp)
        // μ = Φ(η) ≈ 5e-198, and clamping the denominator up to 1e-10 would collapse
        // the score by ~190 orders of magnitude, which is exactly the regime the
        // probit and cloglog acceptance gates exercise. `DENOM_FLOOR` sits far below
        // anything any link produces, so it only prevents a division by exactly zero.
        //
        // Note also that `1.0 - MIN_POSITIVE` was never the upper clamp its name
        // suggests once μ approached 1.
        let mu = require(self, params, "mu")?;
        let n = self.trials(y.len());

        let var_unit = mu.mapv(|m| (m * (1.0 - m)).max(DENOM_FLOOR));
        let u_mu = par_zip3_map(y, n.as_ref(), mu, |yi, ni, mi| yi - ni * mi) / &var_unit;
        let i_mu = n.as_ref() / &var_unit;

        Ok(HashMap::from([("mu".to_string(), (u_mu, i_mu))]))
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

    fn is_discrete(&self) -> bool {
        true
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // F(⌊y⌋ | n, μ) = I_{1−μ}(n−⌊y⌋, ⌊y⌋+1) = beta_reg(n−⌊y⌋, ⌊y⌋+1, 1−μ).
        let mu = require(self, params, "mu")?;
        let n = self.trials(y.len());
        Ok(par_zip3_map(y, mu, n.as_ref(), |yi, mui, ni| {
            if yi < 0.0 {
                return 0.0;
            }
            let k = yi.floor();
            if k >= ni {
                return 1.0;
            }
            let m = mui.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            beta_reg(ni - k, k + 1.0, 1.0 - m)
        }))
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let mu = require(self, params, "mu")?;
        let n = self.trials(p.len());
        Ok(par_zip3_map(p, mu, n.as_ref(), |pi, mui, ni| {
            let m = mui.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
            let nmax = ni;
            discrete_quantile(pi.clamp(0.0, 1.0 - 1e-12), |k| {
                let kf = k as f64;
                if kf >= nmax {
                    1.0
                } else {
                    beta_reg(nmax - kf, kf + 1.0, 1.0 - m)
                }
            })
        }))
    }

    fn name(&self) -> &'static str {
        "Binomial"
    }

    fn descriptor(&self) -> super::FamilyDescriptor {
        super::FamilyDescriptor::Binomial {
            n_trials: self.n_trials.to_vec(),
        }
    }

    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        match param {
            "mu" => {
                // y is counts; pool across observations as Σy/Σn so heterogeneous
                // per-observation trial counts don't bias the seed (`n_trials[0]`
                // alone is wrong when trials vary by row).
                let total_y: f64 = y.sum();
                let total_n: f64 = if self.n_trials.len() == 1 {
                    self.n_trials[0] * y.len() as f64
                } else {
                    self.n_trials.sum()
                };
                let p = total_y / total_n.max(MIN_POSITIVE);
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
        check_discrete_cdf_matches_pmf, check_eta_score_via_finite_diff,
        check_score_via_finite_diff, default_link_derivatives, derivative_keys_match_parameters,
        finite_array, params_view, ParamLinks,
    };
    use crate::distributions::{CauchitLink, CloglogLink, ProbitLink};
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
        let derivs = default_link_derivatives(&bin, &y, &p).unwrap();
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
        let derivs = default_link_derivatives(&bin, &y, &p).unwrap();
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

    #[test]
    fn score_matches_finite_diff_under_non_default_links() {
        // The Altitude #1 gate. μ folded the *logit* chain rule into `u = y − n·μ`,
        // so under probit the score was wrong by a factor of `φ(η)/(μ(1−μ))`. These
        // are the links behind two of the four `link_mle_oracle` shortfalls.
        //
        // This replaces `binomial_score_is_wrong_under_a_probit_link_today`, the
        // Phase 0 characterization test that asserted the opposite.
        let bin = Binomial::new(10);
        let y = array![0.0, 3.0, 5.0, 8.0, 10.0];
        let owned = [("mu", array![0.15, 0.35, 0.5, 0.8, 0.93])];
        check_eta_score_via_finite_diff(&bin, &y, &owned, "mu", &ProbitLink, 1e-5);
        check_eta_score_via_finite_diff(&bin, &y, &owned, "mu", &CloglogLink, 1e-5);
        check_eta_score_via_finite_diff(&bin, &y, &owned, "mu", &CauchitLink, 1e-5);
    }

    #[test]
    fn derivatives_stay_finite_at_a_saturated_mu() {
        // Un-folding introduces the `1/(μ(1−μ))` that `u = y − n·μ` cancelled. The
        // fixture spans both boundaries, including μ exactly 0 and exactly 1, which
        // the old `MIN_POSITIVE` clamp used to mask.
        let bin = Binomial::new(10);
        let y = array![0.0, 10.0, 5.0, 3.0];
        let owned = [("mu", array![0.0, 1.0, 1e-200, 1.0 - 1e-16])];
        let p = params_view(&owned);
        let natural = bin.derivatives(&y, &p).unwrap();
        let (u_n, i_n) = &natural["mu"];
        assert!(finite_array(u_n) && finite_array(i_n), "natural: {u_n:?}");

        let chained = default_link_derivatives(&bin, &y, &p).unwrap();
        let (u, w) = &chained["mu"];
        assert!(finite_array(u) && finite_array(w), "chained: {u:?}");
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn probit_score_telescopes_in_the_tail() {
        // The reason the guard is on the denominator rather than on μ. Deep in the
        // probit tail the natural score `(y − nμ)/(μ(1−μ))` is enormous and
        // `mu_eta = φ(η)` is minuscule; the product has to telescope back to
        // `y·φ(η)/Φ(η)`, the inverse Mills ratio times y. The old `MIN_POSITIVE`
        // clamp on μ would floor the denominator at 1e-10 and collapse the result by
        // orders of magnitude, which is what kept the probit acceptance gate red.
        //
        // η stops at −6 because of the *test helper*, not the fitter:
        // `ParamLinks` reconstructs η as `link(μ)`, and `ProbitLink::link` clamps μ
        // at `MIN_POSITIVE = 1e-10` (so η saturates around −6.36). In production η is
        // the primary quantity and μ is `inv_link(η)`, so no such round trip happens
        // and the tail extends to `MIN_ETA`. Φ(−6) ≈ 9.9e-10 is just clear of the
        // clamp, which is far enough to make the point: the folded value would be ≈ 4
        // and the correct one is ≈ 24.
        let bin = Binomial::new(10);
        let y = array![4.0];
        for &eta in &[-4.0_f64, -6.0] {
            let mu = ProbitLink.inv_link(eta);
            let owned = [("mu", array![mu])];
            let p = params_view(&owned);
            let links = ParamLinks::overriding(&bin, &p, "mu", &ProbitLink);
            let u = bin.eta_derivatives(&y, &p, &links.context()).unwrap()["mu"]
                .0
                .clone();

            let n_mu = 10.0 * mu;
            let expected = crate::math::std_normal_pdf(eta) * (4.0 - n_mu) / (mu * (1.0 - mu));
            assert!(u[0].is_finite(), "η={eta}: u is not finite");
            assert!(
                (u[0] - expected).abs() / expected < 1e-9,
                "η={eta}: u={:.6e} expected={:.6e}",
                u[0],
                expected
            );
            // The folded (buggy) value was `y − n·μ ≈ 4`. The correct one is the
            // inverse Mills ratio times y, which grows without bound down the tail.
            assert!(
                u[0] > 4.0 * 1.5,
                "η={eta}: score {} is suspiciously close to the folded `y − nμ`",
                u[0]
            );
        }
    }

    #[test]
    fn cdf_matches_pmf_binomial() {
        let bin = Binomial::new(20);
        let ks = array![0.0, 5.0, 10.0, 15.0, 20.0];
        let owned = [("mu", array![0.25, 0.4, 0.5, 0.6, 0.75])];
        check_discrete_cdf_matches_pmf(&bin, &ks, &owned, 1e-9);
    }

    #[test]
    fn cdf_endpoints_and_quantile_inverts_binomial() {
        let bin = Binomial::new(15);
        // F(n) = 1 and F(y < 0) = 0 at the support boundaries.
        let boundary_params = [("mu", array![0.4, 0.4])];
        let p = params_view(&boundary_params);
        let at_boundary = bin.cdf(&array![15.0, -1.0], &p).unwrap();
        assert_eq!(at_boundary[0], 1.0);
        assert_eq!(at_boundary[1], 0.0);

        let owned = [("mu", array![0.4])];
        let p = params_view(&owned);
        for &prob in &[0.05, 0.5, 0.95] {
            let q = bin.quantile(&array![prob], &p).unwrap()[0];
            assert!((0.0..=15.0).contains(&q), "quantile {q} outside [0,n]");
            let f_q = bin.cdf(&array![q], &p).unwrap()[0];
            assert!(f_q >= prob - 1e-12, "F(q)={f_q} < p={prob}");
        }
    }

    #[test]
    fn cdf_per_observation_trials_binomial() {
        // Distinct n per row exercises the broadcast path.
        let bin = Binomial::with_trials(array![10.0, 20.0, 5.0]);
        let ks = array![3.0, 10.0, 2.0];
        let owned = [("mu", array![0.3, 0.5, 0.4])];
        check_discrete_cdf_matches_pmf(&bin, &ks, &owned, 1e-9);
    }
}
