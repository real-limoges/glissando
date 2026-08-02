//! Censored responses (STRUCT-1): a wrapper distribution that rewrites a base
//! family's likelihood for observations known only to lie in an interval.
//!
//! Each row is observed exactly ([`CensorStatus::Event`]), or known only to be
//! below ([`CensorStatus::Left`]), above ([`CensorStatus::Right`]), or within an
//! interval ([`CensorStatus::Interval`]) of the recorded value. The pointwise
//! log-likelihood swaps the density for a survival / interval probability built
//! from the base family's `cdf`:
//!
//! ```text
//! event (exact y):  log f(y)
//! right-censored:   log S(y) = log(1 − F(y))
//! left-censored:    log F(y)
//! interval [y, hi]: log(F(hi) − F(y))
//! ```
//!
//! Like [`Binomial`](super::Binomial) and [`Ocat`](super::Ocat), a `Censored`
//! carries per-observation state (`status`, interval upper bounds) that a name
//! string cannot, so it is excluded from [`from_name`](super::from_name); build
//! it through this typed API and serialize it via the family descriptor (SER-1).

use super::structural::{
    cdf_eta_grads, check_state_len, delegate_to_base, rewrite_base_derivatives,
};
use super::{clamp_prob, DerivativesResult, Distribution, GamlssError, Link, LinkContext};
use ndarray::Array1;
use std::collections::HashMap;

/// Per-observation censoring code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum CensorStatus {
    /// Observed exactly at `y`. Likelihood contribution is the base density.
    Event,
    /// Left-censored: the true value is `≤ y`. Contribution is `F(y)`.
    Left,
    /// Right-censored: the true value is `> y` (survival). Contribution is `1 − F(y)`.
    Right,
    /// Interval-censored: the true value lies in `[y, upper]`. Contribution is
    /// `F(upper) − F(y)`.
    Interval,
}

/// A base family wrapped with per-observation censoring information.
///
/// The response `y` passed to fit / predict carries the observed time (or, for
/// interval rows, the interval's lower bound); `upper` carries the interval upper
/// bound for [`CensorStatus::Interval`] rows and is ignored for the rest. The
/// wrapper fits the base family's parameters; censoring only reshapes the
/// likelihood, so `cdf` / `quantile` / `variance` / `expected_value` delegate to
/// the base.
#[derive(Debug)]
pub struct Censored {
    base: Box<dyn Distribution>,
    status: Array1<CensorStatus>,
    /// Interval upper bounds, aligned with `status`. Only read on
    /// [`CensorStatus::Interval`] rows.
    upper: Array1<f64>,
    /// Cached `status.iter().any(|s| *s == CensorStatus::Interval)`. `status` is
    /// immutable after construction, so this is computed once here instead of
    /// rescanned on every `loglik_pointwise` / `theta_derivatives` call (i.e. every
    /// IRLS iteration).
    has_interval: bool,
}

impl Censored {
    /// Wrap `base` with per-row censoring `status`. Use for event / left / right
    /// censoring (no interval rows); `upper` is set to zeros.
    pub fn new(base: Box<dyn Distribution>, status: Array1<CensorStatus>) -> Self {
        let n = status.len();
        let has_interval = status.iter().any(|s| *s == CensorStatus::Interval);
        Self {
            base,
            status,
            upper: Array1::zeros(n),
            has_interval,
        }
    }

    /// Wrap `base` with censoring `status` and interval `upper` bounds (read only
    /// on [`CensorStatus::Interval`] rows). `upper` must match `status` in length.
    pub fn with_upper(
        base: Box<dyn Distribution>,
        status: Array1<CensorStatus>,
        upper: Array1<f64>,
    ) -> Self {
        let has_interval = status.iter().any(|s| *s == CensorStatus::Interval);
        Self {
            base,
            status,
            upper,
            has_interval,
        }
    }

    /// The wrapped base family.
    pub fn base(&self) -> &dyn Distribution {
        self.base.as_ref()
    }

    /// Per-observation censoring codes.
    pub fn status(&self) -> &Array1<CensorStatus> {
        &self.status
    }

    /// Interval upper bounds (only meaningful on interval rows).
    pub fn upper(&self) -> &Array1<f64> {
        &self.upper
    }

    /// Reject a response whose length does not match the stored `status` vector.
    fn check_len(&self, n: usize) -> Result<(), GamlssError> {
        check_state_len("Censored: status", self.status.len(), n)
    }
}

impl Distribution for Censored {
    delegate_to_base!(
        parameters,
        default_link,
        allows_link_override,
        initial_value,
        is_discrete,
        variance,
        expected_value,
        cdf,
        quantile,
    );

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        self.check_len(y.len())?;
        let base_ll = self.base.loglik_pointwise(y, params)?;
        let f_y = self.base.cdf(y, params)?;
        let f_up = if self.has_interval {
            Some(self.base.cdf(&self.upper, params)?)
        } else {
            None
        };

        let mut out = base_ll;
        for i in 0..y.len() {
            out[i] = match self.status[i] {
                CensorStatus::Event => out[i],
                CensorStatus::Right => clamp_prob(1.0 - f_y[i]).ln(),
                CensorStatus::Left => clamp_prob(f_y[i]).ln(),
                CensorStatus::Interval => {
                    let d = f_up.as_ref().expect("interval needs upper")[i] - f_y[i];
                    clamp_prob(d).ln()
                }
            };
        }
        Ok(out)
    }

    /// The CDF chain rule below reads `mu_eta2`, so the scoring loop must build a
    /// full second-order [`LinkContext`] rather than a first-order one.
    fn needs_second_order_links(&self) -> bool {
        true
    }

    /// Overrides the η-scale adapter directly: the censored rows carry *observed*
    /// information, which is not link-invariant, so there is no natural-scale
    /// `(∂l/∂θ, i_θ)` for the generic chain rule to lift. See
    /// [`Link::mu_eta2`].
    fn eta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
        ctx: &LinkContext,
    ) -> DerivativesResult {
        self.check_len(y.len())?;
        // Event rows keep the base score / Fisher weight; censored rows are
        // overwritten with the survival / interval score and observed-information
        // weight built from F, F', F''.
        let base_derivs = self.base.eta_derivatives(y, params, ctx)?;
        let f_y = self.base.cdf(y, params)?;
        let grad_y = cdf_eta_grads(self.base.as_ref(), y, &f_y, params, ctx)?;

        let (f_up, grad_up) = if self.has_interval {
            let f_up = self.base.cdf(&self.upper, params)?;
            let grad_up = cdf_eta_grads(self.base.as_ref(), &self.upper, &f_up, params, ctx)?;
            (Some(f_up), Some(grad_up))
        } else {
            (None, None)
        };

        rewrite_base_derivatives(self.base.as_ref(), base_derivs, |param, u, w| {
            let (d1y, d2y) = &grad_y[param];
            for i in 0..y.len() {
                match self.status[i] {
                    CensorStatus::Event => {}
                    CensorStatus::Right => {
                        let s = clamp_prob(1.0 - f_y[i]);
                        u[i] = -d1y[i] / s;
                        w[i] = d2y[i] / s + (d1y[i] / s).powi(2);
                    }
                    CensorStatus::Left => {
                        let fv = clamp_prob(f_y[i]);
                        u[i] = d1y[i] / fv;
                        w[i] = -d2y[i] / fv + (d1y[i] / fv).powi(2);
                    }
                    CensorStatus::Interval => {
                        let (d1u, d2u) = &grad_up.as_ref().expect("interval grads")[param];
                        let f_upper = f_up.as_ref().expect("interval upper")[i];
                        let dd = clamp_prob(f_upper - f_y[i]);
                        let d1 = d1u[i] - d1y[i];
                        let d2 = d2u[i] - d2y[i];
                        u[i] = d1 / dd;
                        w[i] = -d2 / dd + (d1 / dd).powi(2);
                    }
                }
            }
        })
    }

    fn name(&self) -> &'static str {
        "Censored"
    }

    fn descriptor(&self) -> super::FamilyDescriptor {
        super::FamilyDescriptor::Censored {
            base: Box::new(self.base.descriptor()),
            status: self.status.to_vec(),
            upper: super::descriptor::encode_bounds(&self.upper),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::test_helpers::{
        check_eta_score_via_finite_diff, check_score_via_finite_diff, default_link_derivatives,
        derivative_keys_match_parameters_observed_info, finite_array, params_view,
    };
    use crate::distributions::{Gaussian, SqrtLink};
    use ndarray::array;

    fn gaussian_owned() -> Vec<(&'static str, Array1<f64>)> {
        vec![
            ("mu", array![0.0, 0.5, 1.0, -0.5]),
            ("sigma", array![1.0, 1.2, 0.8, 1.5]),
        ]
    }

    #[test]
    fn all_event_reduces_to_base_loglik() {
        let y = array![0.3, 0.7, 1.1, -0.2];
        let owned = gaussian_owned();
        let p = params_view(&owned);
        let status = Array1::from_elem(4, CensorStatus::Event);
        let cens = Censored::new(Box::new(Gaussian::new()), status);

        let base_ll = Gaussian.loglik_pointwise(&y, &p).unwrap();
        let cens_ll = cens.loglik_pointwise(&y, &p).unwrap();
        for i in 0..4 {
            assert!((base_ll[i] - cens_ll[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn right_censored_is_log_survival() {
        let y = array![0.3, 0.7, 1.1, -0.2];
        let owned = gaussian_owned();
        let p = params_view(&owned);
        let status = Array1::from_elem(4, CensorStatus::Right);
        let cens = Censored::new(Box::new(Gaussian::new()), status);

        let f = Gaussian.cdf(&y, &p).unwrap();
        let ll = cens.loglik_pointwise(&y, &p).unwrap();
        for i in 0..4 {
            assert!((ll[i] - (1.0 - f[i]).ln()).abs() < 1e-10);
        }
    }

    #[test]
    fn interval_is_log_probability_mass() {
        let y = array![0.0, -0.5];
        let upper = array![1.0, 0.5];
        let owned = [("mu", array![0.0, 0.0]), ("sigma", array![1.0, 1.0])];
        let p = params_view(&owned);
        let status = array![CensorStatus::Interval, CensorStatus::Interval];
        let cens = Censored::with_upper(Box::new(Gaussian::new()), status, upper.clone());

        let f_lo = Gaussian.cdf(&y, &p).unwrap();
        let f_hi = Gaussian.cdf(&upper, &p).unwrap();
        let ll = cens.loglik_pointwise(&y, &p).unwrap();
        for i in 0..2 {
            assert!((ll[i] - (f_hi[i] - f_lo[i]).ln()).abs() < 1e-10);
        }
    }

    #[test]
    fn event_rows_keep_base_score() {
        // With all-event status the analytic score must equal the base family's.
        let y = array![0.3, 0.7, 1.1, -0.2];
        let owned = gaussian_owned();
        let p = params_view(&owned);
        let status = Array1::from_elem(4, CensorStatus::Event);
        let cens = Censored::new(Box::new(Gaussian::new()), status);

        let base = default_link_derivatives(&Gaussian, &y, &p).unwrap();
        let got = default_link_derivatives(&cens, &y, &p).unwrap();
        for param in ["mu", "sigma"] {
            let (ub, _) = &base[param];
            let (ug, _) = &got[param];
            for i in 0..4 {
                assert!((ub[i] - ug[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn censored_score_matches_finite_diff() {
        // The whole wrapper (mixed status) must satisfy u == d(loglik)/dη.
        let y = array![0.3, 0.7, 1.1, -0.2];
        let owned = gaussian_owned();
        let status = array![
            CensorStatus::Event,
            CensorStatus::Right,
            CensorStatus::Left,
            CensorStatus::Right,
        ];
        let cens = Censored::new(Box::new(Gaussian::new()), status);
        check_score_via_finite_diff(&cens, &y, &owned, "mu", 1e-4);
        check_score_via_finite_diff(&cens, &y, &owned, "sigma", 1e-4);
    }

    #[test]
    fn derivative_keys_and_weights_are_well_formed() {
        // The structural wrappers had no `derivative_keys_match_parameters_observed_info`
        // coverage, so nothing checked that they emit an entry per parameter
        // with finite, non-negative weights. All four status arms are present so
        // each overwrite branch is exercised.
        let y = array![0.3, 0.7, 1.1, -0.2];
        let owned = gaussian_owned();
        let upper = array![1.0, 1.5, 2.0, 0.5];
        let status = array![
            CensorStatus::Event,
            CensorStatus::Right,
            CensorStatus::Left,
            CensorStatus::Interval,
        ];
        let cens = Censored::with_upper(Box::new(Gaussian::new()), status, upper);
        derivative_keys_match_parameters_observed_info(&cens, params_view(&owned), &y);
    }

    #[test]
    fn interval_score_matches_finite_diff() {
        // Interval censoring is the one branch `censored_score_matches_finite_diff`
        // doesn't exercise; check it in isolation (mixed with Event/Right so the
        // other branches' code paths stay live in the same call).
        let y = array![0.0, -0.5, 0.3, 1.0];
        let upper = array![1.0, 0.5, 1.2, 2.0];
        let owned = gaussian_owned();
        let status = array![
            CensorStatus::Interval,
            CensorStatus::Interval,
            CensorStatus::Event,
            CensorStatus::Right,
        ];
        let cens = Censored::with_upper(Box::new(Gaussian::new()), status, upper);
        check_score_via_finite_diff(&cens, &y, &owned, "mu", 1e-4);
        check_score_via_finite_diff(&cens, &y, &owned, "sigma", 1e-4);
    }

    #[test]
    fn score_matches_finite_diff_under_a_non_default_link() {
        // Altitude #1 Phase 3 acceptance gate. Before the CDF derivatives moved to
        // the natural scale, `cdf_eta_grads` chained through Gaussian's *default*
        // links no matter what the fit resolved, so this score was wrong by a
        // factor of `mu_eta_default / mu_eta_actual` on every censored row.
        //
        // σ goes on `sqrt` rather than the usual suspects because η = √σ keeps the
        // fixture's positive σ in the link's domain. μ deliberately stays on
        // identity: this fixture has μ < 0, which `sqrt` and `inverse_square`
        // cannot represent.
        let y = array![0.0, -0.5, 0.3, 1.0];
        let upper = array![1.0, 0.5, 1.2, 2.0];
        let owned = gaussian_owned();
        let status = array![
            CensorStatus::Interval,
            CensorStatus::Left,
            CensorStatus::Event,
            CensorStatus::Right,
        ];
        let cens = Censored::with_upper(Box::new(Gaussian::new()), status, upper);
        check_eta_score_via_finite_diff(&cens, &y, &owned, "sigma", &SqrtLink, 1e-4);
    }

    #[test]
    fn derivatives_stay_finite_at_a_saturated_fixture() {
        // Altitude #1 Phase 3, gate (d). Two things changed under this test: the
        // base's CDF derivatives are now un-folded (new divisions by σ and σ²), and
        // the family-level `.max(MIN_WEIGHT)` that used to launder every censored
        // row's weight is gone. A saturating F drives `clamp_prob` to both of its
        // rails, so `d1/F` and `d2/F` are evaluated at `PROB_EPS`.
        //
        // Weights are checked for finiteness only, not for sign: these rows carry
        // observed information, which is legitimately negative in places (see
        // `derivative_keys_match_parameters_observed_info`).
        let y = array![-40.0, 40.0, 0.0, 0.0];
        let upper = array![0.0, 0.0, 0.0, 1e300];
        let owned = vec![
            ("mu", array![0.0, 0.0, 0.0, 0.0]),
            ("sigma", array![1e-320, 1e13, 1e-8, 1.0]),
        ];
        let status = array![
            CensorStatus::Right,
            CensorStatus::Left,
            CensorStatus::Event,
            CensorStatus::Interval,
        ];
        let cens = Censored::with_upper(Box::new(Gaussian::new()), status, upper);
        let p = params_view(&owned);
        let d = default_link_derivatives(&cens, &y, &p).unwrap();
        for name in ["mu", "sigma"] {
            let (u, w) = &d[name];
            assert!(finite_array(u) && finite_array(w), "{name}: {u:?} {w:?}");
        }
    }

    #[test]
    fn length_mismatch_errors() {
        let y = array![0.3, 0.7];
        let owned = [("mu", array![0.0, 0.0]), ("sigma", array![1.0, 1.0])];
        let p = params_view(&owned);
        let status = Array1::from_elem(3, CensorStatus::Event); // wrong length
        let cens = Censored::new(Box::new(Gaussian::new()), status);
        assert!(cens.loglik_pointwise(&y, &p).is_err());
    }
}
