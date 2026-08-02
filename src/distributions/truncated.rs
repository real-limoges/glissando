//! Truncated responses (STRUCT-2): a wrapper distribution observed only within
//! per-observation bounds `(lo, hi)`.
//!
//! Unlike censoring, out-of-range values are *absent*, not recorded, so the
//! density renormalizes by the in-support mass:
//!
//! ```text
//! f_T(y) = f(y) / (F(hi) − F(lo))           for lo < y < hi
//! log f_T(y) = base.loglik_pointwise(y) − log(F(hi) − F(lo))
//! ```
//!
//! Bounds may be `±∞` (an open side reduces that term to `0` or `1`). With
//! `(−∞, ∞)` the wrapper reduces exactly to the base family. `cdf` / `quantile`
//! are renormalized onto the truncated support; `variance` / `expected_value`
//! delegate to the base family (they report the *untruncated* parameter moments;
//! truncated moments would need numerical integration and are out of scope).
//!
//! Like the other structural wrappers it carries per-row state and is excluded
//! from [`from_name`](super::from_name).

use super::structural::{
    cdf_eta_grads, check_state_len, delegate_to_base, rewrite_base_derivatives,
};
use super::{
    clamp_prob, DerivativesResult, Distribution, GamlssError, Link, LinkContext, PROB_EPS,
};
use ndarray::Array1;
use std::collections::HashMap;

/// A base family restricted to per-observation support `(lower, upper)`.
#[derive(Debug)]
pub struct Truncated {
    base: Box<dyn Distribution>,
    lower: Array1<f64>,
    upper: Array1<f64>,
}

impl Truncated {
    /// Restrict `base` to the per-row open interval `(lower, upper)`. Use `±∞` for
    /// an unbounded side. `lower` and `upper` must match in length.
    pub fn new(base: Box<dyn Distribution>, lower: Array1<f64>, upper: Array1<f64>) -> Self {
        Self { base, lower, upper }
    }

    /// The wrapped base family.
    pub fn base(&self) -> &dyn Distribution {
        self.base.as_ref()
    }

    /// Per-observation lower truncation bounds.
    pub fn lower(&self) -> &Array1<f64> {
        &self.lower
    }

    /// Per-observation upper truncation bounds.
    pub fn upper(&self) -> &Array1<f64> {
        &self.upper
    }

    /// Reject a response whose length does not match the stored bound vectors.
    fn check_len(&self, n: usize) -> Result<(), GamlssError> {
        check_state_len("Truncated: lower bound", self.lower.len(), n)?;
        check_state_len("Truncated: upper bound", self.upper.len(), n)
    }

    /// `F` at a bound array that may contain `±∞`, without the per-parameter
    /// gradients `cdf_and_grads_at` also computes; cheaper, for the call sites
    /// that only need `F` (`loglik_pointwise`, `cdf`, `quantile`; only
    /// `theta_derivatives` needs the gradients). Infinite rows saturate to the limit
    /// (`+∞ → F=1`, `−∞ → F=0`) without ever passing `±∞` into the base family.
    fn cdf_at(
        &self,
        bound: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let sanitized = bound.mapv(|v| if v.is_finite() { v } else { 0.0 });
        let mut f = self.base.cdf(&sanitized, params)?;
        for i in 0..bound.len() {
            if bound[i].is_finite() {
                continue;
            }
            f[i] = if bound[i] > 0.0 { 1.0 } else { 0.0 };
        }
        Ok(f)
    }

    /// `F` and `(∂F/∂η, ∂²F/∂η²)` per parameter, evaluated at a bound array that
    /// may contain `±∞`. Infinite rows are computed as the saturating limit
    /// (`+∞ → F=1`, `−∞ → F=0`, derivatives `0`) without ever passing `±∞` into
    /// the base family.
    fn cdf_and_grads_at(
        &self,
        bound: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
        ctx: &LinkContext,
    ) -> Result<(Array1<f64>, super::CdfEtaMap), GamlssError> {
        let f = self.cdf_at(bound, params)?;
        let sanitized = bound.mapv(|v| if v.is_finite() { v } else { 0.0 });
        let mut grads = cdf_eta_grads(self.base.as_ref(), &sanitized, &f, params, ctx)?;
        for i in 0..bound.len() {
            if bound[i].is_finite() {
                continue;
            }
            for (d1, d2) in grads.values_mut() {
                d1[i] = 0.0;
                d2[i] = 0.0;
            }
        }
        Ok((f, grads))
    }
}

impl Distribution for Truncated {
    // `cdf` / `quantile` are *not* delegated: they renormalize onto the truncated
    // support (see below). `variance` / `expected_value` report the untruncated
    // base moments by design (see the module docs).
    delegate_to_base!(
        parameters,
        default_link,
        allows_link_override,
        initial_value,
        is_discrete,
        variance,
        expected_value,
    );

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        self.check_len(y.len())?;
        let base_ll = self.base.loglik_pointwise(y, params)?;
        let f_lo = self.cdf_at(&self.lower, params)?;
        let f_hi = self.cdf_at(&self.upper, params)?;
        let mut out = base_ll;
        for i in 0..y.len() {
            let mass = (f_hi[i] - f_lo[i]).max(PROB_EPS);
            out[i] -= mass.ln();
        }
        Ok(out)
    }

    /// The CDF chain rule below reads `mu_eta2`, so the scoring loop must build a
    /// full second-order [`LinkContext`] rather than a first-order one.
    fn needs_second_order_links(&self) -> bool {
        true
    }

    /// Overrides the η-scale adapter directly: the normalizer contributes *observed*
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
        // Score / weight = base contribution minus the normalizer's. For the
        // normalizer D = F(hi) − F(lo):
        //   u = u_base − D'/D,   w = w_base + D''/D − (D'/D)²   (observed info).
        let base_derivs = self.base.eta_derivatives(y, params, ctx)?;
        let (f_lo, grad_lo) = self.cdf_and_grads_at(&self.lower, params, ctx)?;
        let (f_hi, grad_hi) = self.cdf_and_grads_at(&self.upper, params, ctx)?;

        rewrite_base_derivatives(self.base.as_ref(), base_derivs, |param, u, w| {
            let (d1_lo, d2_lo) = &grad_lo[param];
            let (d1_hi, d2_hi) = &grad_hi[param];
            for i in 0..y.len() {
                let dmass = (f_hi[i] - f_lo[i]).max(PROB_EPS);
                let d1 = d1_hi[i] - d1_lo[i];
                let d2 = d2_hi[i] - d2_lo[i];
                u[i] -= d1 / dmass;
                w[i] = w[i] + d2 / dmass - (d1 / dmass).powi(2);
            }
        })
    }

    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // Renormalized onto the truncated support: F_T(y) = (F(y)−F(lo))/(F(hi)−F(lo)),
        // clamped to [0, 1] outside (lo, hi). On new data whose length does not match
        // the stored per-row bounds (prediction), delegate to the untruncated base.
        if self.lower.len() != y.len() {
            return self.base.cdf(y, params);
        }
        let f_y = self.base.cdf(y, params)?;
        let f_lo = self.cdf_at(&self.lower, params)?;
        let f_hi = self.cdf_at(&self.upper, params)?;
        let mut out = f_y;
        for i in 0..out.len() {
            let mass = (f_hi[i] - f_lo[i]).max(PROB_EPS);
            out[i] = ((out[i] - f_lo[i]) / mass).clamp(0.0, 1.0);
        }
        Ok(out)
    }

    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        // Invert the renormalized CDF: map p into the base scale via
        // p_base = F(lo) + p·(F(hi)−F(lo)), then call the base quantile. On new data
        // whose length does not match the stored bounds, delegate to the base.
        if self.lower.len() != p.len() {
            return self.base.quantile(p, params);
        }
        let f_lo = self.cdf_at(&self.lower, params)?;
        let f_hi = self.cdf_at(&self.upper, params)?;
        let mut p_base = Array1::<f64>::zeros(p.len());
        for i in 0..p.len() {
            let mass = (f_hi[i] - f_lo[i]).max(PROB_EPS);
            p_base[i] = clamp_prob(f_lo[i] + p[i].clamp(0.0, 1.0) * mass);
        }
        self.base.quantile(&p_base, params)
    }

    fn name(&self) -> &'static str {
        "Truncated"
    }

    fn descriptor(&self) -> super::FamilyDescriptor {
        super::FamilyDescriptor::Truncated {
            base: Box::new(self.base.descriptor()),
            lower: super::descriptor::encode_bounds(&self.lower),
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
    use crate::distributions::{Gaussian, InverseLink};
    use ndarray::array;

    fn full_range(n: usize) -> (Array1<f64>, Array1<f64>) {
        (
            Array1::from_elem(n, f64::NEG_INFINITY),
            Array1::from_elem(n, f64::INFINITY),
        )
    }

    #[test]
    fn full_range_reduces_to_base_loglik() {
        let y = array![0.3, 0.7, 1.1, -0.2];
        let owned = [
            ("mu", array![0.0, 0.5, 1.0, -0.5]),
            ("sigma", array![1.0, 1.2, 0.8, 1.5]),
        ];
        let p = params_view(&owned);
        let (lo, hi) = full_range(4);
        let trunc = Truncated::new(Box::new(Gaussian::new()), lo, hi);

        let base_ll = Gaussian.loglik_pointwise(&y, &p).unwrap();
        let trunc_ll = trunc.loglik_pointwise(&y, &p).unwrap();
        for i in 0..4 {
            assert!((base_ll[i] - trunc_ll[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn left_truncation_renormalizes() {
        // Left-truncated at 0: log f_T(y) = log f(y) − log(1 − F(0)).
        let y = array![1.0, 2.0, 0.5];
        let owned = [
            ("mu", array![1.0, 1.0, 1.0]),
            ("sigma", array![1.0, 1.0, 1.0]),
        ];
        let p = params_view(&owned);
        let lo = Array1::from_elem(3, 0.0);
        let hi = Array1::from_elem(3, f64::INFINITY);
        let trunc = Truncated::new(Box::new(Gaussian::new()), lo, hi);

        let base_ll = Gaussian.loglik_pointwise(&y, &p).unwrap();
        let f0 = Gaussian.cdf(&array![0.0, 0.0, 0.0], &p).unwrap();
        let ll = trunc.loglik_pointwise(&y, &p).unwrap();
        for i in 0..3 {
            let expected = base_ll[i] - (1.0 - f0[i]).ln();
            assert!((ll[i] - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn derivative_keys_and_weights_are_well_formed() {
        // See the matching test in `censored.rs`: the wrappers had no
        // `derivative_keys_match_parameters_observed_info` coverage at all. A two-sided
        // truncation is used so the normalizer `D = F(hi) − F(lo)` is a genuine
        // difference rather than collapsing to `1 − F(lo)`.
        let y = array![1.0, 2.0, 1.5, 3.0];
        let owned = [
            ("mu", array![1.0, 1.5, 1.0, 2.0]),
            ("sigma", array![1.0, 1.2, 0.9, 1.1]),
        ];
        let lo = Array1::from_elem(4, 0.0);
        let hi = Array1::from_elem(4, 5.0);
        let trunc = Truncated::new(Box::new(Gaussian::new()), lo, hi);
        derivative_keys_match_parameters_observed_info(&trunc, params_view(&owned), &y);
    }

    #[test]
    fn truncated_score_matches_finite_diff() {
        let y = array![1.0, 2.0, 1.5, 3.0];
        let owned = [
            ("mu", array![1.0, 1.5, 1.0, 2.0]),
            ("sigma", array![1.0, 1.2, 0.9, 1.1]),
        ];
        let lo = Array1::from_elem(4, 0.0);
        let hi = Array1::from_elem(4, f64::INFINITY);
        let trunc = Truncated::new(Box::new(Gaussian::new()), lo, hi);
        check_score_via_finite_diff(&trunc, &y, &owned, "mu", 1e-4);
        check_score_via_finite_diff(&trunc, &y, &owned, "sigma", 1e-4);
    }

    #[test]
    fn score_matches_finite_diff_under_a_non_default_link() {
        // Altitude #1 Phase 3 acceptance gate: the normalizer's `D'/D` term has to
        // be built from the link the fit resolved, not from Gaussian's default.
        // `inverse` puts η = 1/σ, so a positive σ stays in the link's domain; μ
        // stays on identity because this fixture allows μ ≤ 0 under perturbation.
        //
        // Finite bounds on both sides so `cdf_and_grads_at` is exercised twice
        // rather than short-circuiting on ±∞.
        let y = array![1.0, 2.0, 1.5, 3.0];
        let owned = [
            ("mu", array![1.0, 1.5, 1.0, 2.0]),
            ("sigma", array![1.0, 1.2, 0.9, 1.1]),
        ];
        let lo = Array1::from_elem(4, 0.0);
        let hi = Array1::from_elem(4, 5.0);
        let trunc = Truncated::new(Box::new(Gaussian::new()), lo, hi);
        check_eta_score_via_finite_diff(&trunc, &y, &owned, "sigma", &InverseLink, 1e-4);
    }

    #[test]
    fn derivatives_stay_finite_at_a_saturated_fixture() {
        // Altitude #1 Phase 3, gate (d). Covers a degenerate normalizer (bounds so
        // far out that `F(hi) − F(lo)` saturates and `PROB_EPS` binds), the ±∞
        // short-circuit in `cdf_and_grads_at`, and a σ at both ends of the log
        // link's reach. Finiteness only: these weights are observed information and
        // may be negative.
        let y = array![0.0, 1.0, 0.5, -1.0];
        let owned = [
            ("mu", array![0.0, 0.0, 0.0, 0.0]),
            ("sigma", array![1e-320, 1e13, 1e-8, 1.0]),
        ];
        let lo = array![f64::NEG_INFINITY, -1e300, 1e-3, 1e-3];
        let hi = array![f64::INFINITY, 1e300, 2e-3, 1e300];
        let trunc = Truncated::new(Box::new(Gaussian::new()), lo, hi);
        let p = params_view(&owned);
        let d = default_link_derivatives(&trunc, &y, &p).unwrap();
        for name in ["mu", "sigma"] {
            let (u, w) = &d[name];
            assert!(finite_array(u) && finite_array(w), "{name}: {u:?} {w:?}");
        }
    }

    #[test]
    fn cdf_is_renormalized_and_quantile_inverts() {
        // Left-truncated Gaussian at 0; F_T(lo)=0, and Q(F_T(y))≈y inside support.
        let owned = [("mu", array![1.0]), ("sigma", array![1.0])];
        let p = params_view(&owned);
        let lo = array![0.0];
        let hi = array![f64::INFINITY];
        let trunc = Truncated::new(Box::new(Gaussian::new()), lo, hi);

        let y = array![1.5];
        let u = trunc.cdf(&y, &p).unwrap();
        assert!((0.0..=1.0).contains(&u[0]));
        let back = trunc.quantile(&u, &p).unwrap();
        assert!((back[0] - 1.5).abs() < 1e-6);
        // mass below the truncation point is excluded.
        let at_lo = trunc.cdf(&array![0.0], &p).unwrap();
        assert!(at_lo[0].abs() < 1e-9);
    }
}
