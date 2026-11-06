//! Hurdle / two-part models: a point mass at zero combined with a
//! zero-truncated base for the positive part.
//!
//! ```text
//! P(Y = 0) = ξ                            the zero atom (logit-linked)
//! P(Y = y) = (1 − ξ) · g_T(y)   (y > 0)   base, zero-TRUNCATED
//! ```
//!
//! Under zero-*inflation* the base can still emit zero
//! (`P(Y=0) = π + (1−π)·g(0)`); in a hurdle the positive process is structurally
//! separate from the zero process. A hurdle suits data whose zero-generating
//! mechanism is distinct (a true two-part model); zero-inflation suits zeros that
//! contaminate a single process.
//!
//! The wrapper adds one fitted parameter `xi` (the zero probability, logit link)
//! on top of the base family's parameters. The moments assume the base has
//! non-negative support. Like the other structural wrappers it is excluded from
//! [`from_name`](super::from_name).

use super::structural::{cdf_eta_grads, delegate_to_base, rewrite_base_derivatives};
use super::{
    chain_to_eta, clamp_prob, DerivativeMap, Distribution, Eta, GamlssError, Link, LinkContext,
    LogitLink, ScoreInfo, DENOM_FLOOR, MIN_WEIGHT, PROB_EPS,
};
use crate::Param;
use ndarray::Array1;
use std::collections::HashMap;

/// A base family augmented with a zero atom: zeros come from a logit-linked
/// probability `xi`, positive values from the zero-truncated base.
#[derive(Debug)]
pub struct Hurdle {
    base: Box<dyn Distribution>,
    /// `base.parameters()` followed by `Param::Xi`; backs [`Distribution::parameters`].
    params: Vec<Param>,
}

impl Hurdle {
    /// Wrap `base` with a logit-linked zero atom `xi = P(Y = 0)`.
    pub fn new(base: Box<dyn Distribution>) -> Self {
        let mut params = base.parameters().to_vec();
        params.push(Param::Xi);
        Self { base, params }
    }

    /// The wrapped (positive-part) base family.
    pub fn base(&self) -> &dyn Distribution {
        self.base.as_ref()
    }

    /// True where the row is the structural zero (the atom), i.e. `y ≤ 0`.
    fn is_zero(y: f64) -> bool {
        y <= 0.0
    }

    /// The zero probability `xi` from a parameter map.
    fn xi<'a>(
        &self,
        params: &HashMap<Param, &'a Array1<f64>>,
    ) -> Result<&'a Array1<f64>, GamlssError> {
        params
            .get(&Param::Xi)
            .copied()
            .ok_or_else(|| self.unknown_param(Param::Xi))
    }

    /// The base CDF at zero, `F(0)`, per row.
    fn base_cdf_at_zero(
        &self,
        n: usize,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        self.base.cdf(&Array1::zeros(n), params)
    }
}

impl Distribution for Hurdle {
    fn parameters(&self) -> &[Param] {
        &self.params
    }

    fn default_link(&self, param: Param) -> Result<Box<dyn Link>, GamlssError> {
        if param == Param::Xi {
            Ok(Box::new(LogitLink))
        } else {
            self.base.default_link(param)
        }
    }

    /// Hand-written rather than delegated. `xi` is the wrapper's own parameter and
    /// is not in `base.parameters()`, so asking the base about it would let a base
    /// that refuses every name (Ocat) veto a parameter it has never heard of. The
    /// `xi` atom goes through `chain_to_eta` like any plain family, so it accepts
    /// any link.
    fn allows_link_override(&self, param: Param) -> bool {
        param == Param::Xi || self.base.allows_link_override(param)
    }

    fn initial_value(&self, param: Param, y: &Array1<f64>) -> f64 {
        if param == Param::Xi {
            // Empirical zero fraction, clamped away from {0, 1}.
            let zeros = y.iter().filter(|&&v| Self::is_zero(v)).count() as f64;
            (zeros / y.len().max(1) as f64).clamp(0.05, 0.95)
        } else {
            self.base.initial_value(param, y)
        }
    }

    delegate_to_base!(is_discrete);

    /// `ξ + (1−ξ)·(F(y) − F(0)) / (1 − F(0))` for `y ≥ 0`, else 0.
    fn cdf(
        &self,
        y: &Array1<f64>,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let xi = self.xi(params)?;
        let f_y = self.base.cdf(y, params)?;
        let f0 = self.base_cdf_at_zero(y.len(), params)?;
        Ok(Array1::from_shape_fn(y.len(), |i| {
            if y[i] < 0.0 {
                return 0.0;
            }
            let xi_i = clamp_prob(xi[i]);
            let mass = (1.0 - f0[i]).max(PROB_EPS);
            let positive = ((f_y[i] - f0[i]) / mass).clamp(0.0, 1.0);
            xi_i + (1.0 - xi_i) * positive
        }))
    }

    /// `P(Y < y)`: 0 at the atom, so a zero's residual spreads over `[0, ξ]`.
    fn cdf_left_limit(
        &self,
        y: &Array1<f64>,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Option<Array1<f64>>, GamlssError> {
        let below = if self.base.is_discrete() {
            self.cdf(&(y - 1.0), params)?
        } else {
            self.cdf(y, params)?
        };
        Ok(Some(Array1::from_shape_fn(y.len(), |i| {
            if Self::is_zero(y[i]) {
                0.0
            } else {
                below[i]
            }
        })))
    }

    /// 0 for `p ≤ ξ`, else `F⁻¹(F(0) + (p−ξ)/(1−ξ)·(1 − F(0)))`.
    fn quantile(
        &self,
        p: &Array1<f64>,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let xi = self.xi(params)?;
        let f0 = self.base_cdf_at_zero(p.len(), params)?;
        let p_base = Array1::from_shape_fn(p.len(), |i| {
            let xi_i = clamp_prob(xi[i]);
            let within = ((p[i] - xi_i) / (1.0 - xi_i)).clamp(0.0, 1.0);
            let mass = (1.0 - f0[i]).max(PROB_EPS);
            clamp_prob(f0[i] + within * mass)
        });
        let q_base = self.base.quantile(&p_base, params)?;
        Ok(Array1::from_shape_fn(p.len(), |i| {
            if p[i] <= clamp_prob(xi[i]) {
                0.0
            } else {
                q_base[i]
            }
        }))
    }

    /// `(1−ξ)·E_b / (1 − F(0))`.
    fn expected_value(
        &self,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let xi = self.xi(params)?;
        let mean_b = self.base.expected_value(params)?;
        let f0 = self.base_cdf_at_zero(mean_b.len(), params)?;
        Ok(Array1::from_shape_fn(mean_b.len(), |i| {
            let mass = (1.0 - f0[i]).max(PROB_EPS);
            (1.0 - clamp_prob(xi[i])) * mean_b[i] / mass
        }))
    }

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<Param, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let xi = self.xi(params)?;
        // Positive-part density is the base left-truncated at zero:
        // log g_T(y) = base.loglik(y) − log(1 − F(0)).
        let base_ll = self.base.loglik_pointwise(y, params)?;
        let zeros = Array1::<f64>::zeros(y.len());
        let f0 = self.base.cdf(&zeros, params)?;

        let mut out = base_ll;
        for i in 0..y.len() {
            let xi_i = clamp_prob(xi[i]);
            if Self::is_zero(y[i]) {
                out[i] = xi_i.ln();
            } else {
                let mass = (1.0 - f0[i]).max(PROB_EPS);
                out[i] = (1.0 - xi_i).ln() + out[i] - mass.ln();
            }
        }
        Ok(out)
    }

    /// The CDF chain rule below reads `mu_eta2`, so the scoring loop must build a
    /// full second-order [`LinkContext`] rather than a first-order one.
    fn needs_second_order_links(&self) -> bool {
        true
    }

    /// Overrides the η-scale adapter directly: the zero-truncation normalizer
    /// contributes *observed* information, which is not link-invariant, so there is
    /// no natural-scale `(∂l/∂θ, i_θ)` for the generic chain rule to lift. See
    /// [`Link::mu_eta2`].
    fn eta_derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<Param, &Array1<f64>>,
        ctx: &LinkContext,
    ) -> Result<DerivativeMap<Eta>, GamlssError> {
        let xi = params
            .get(&Param::Xi)
            .copied()
            .ok_or_else(|| self.unknown_param(Param::Xi))?;
        // Base parameters: zero-truncated score on positive rows, nothing on zeros.
        let base_derivs = self.base.eta_derivatives(y, params, ctx)?;
        let zeros = Array1::<f64>::zeros(y.len());
        let f0 = self.base.cdf(&zeros, params)?;
        let grad0 = cdf_eta_grads(self.base.as_ref(), &zeros, &f0, params, ctx)?;

        let mut out = rewrite_base_derivatives(self.base.as_ref(), base_derivs, |param, u, w| {
            let (d1_0, d2_0) = (&grad0[&param].d1, &grad0[&param].d2);
            for i in 0..y.len() {
                if Self::is_zero(y[i]) {
                    // Zero rows carry no information about the positive-part params.
                    //
                    // This `MIN_WEIGHT` is a *sentinel*, not a floor, and is the one
                    // family-level use of the constant.
                    // Writing 0.0 here and letting `scoring::step` floor it would be
                    // numerically identical (`u/w = 0` either way), but it would tally
                    // one `weight_floor_hits` per structural zero on every iteration,
                    // reporting every well-behaved hurdle fit as degenerate and
                    // defeating the purpose of that diagnostic.
                    u[i] = 0.0;
                    w[i] = MIN_WEIGHT;
                } else {
                    // Zero-truncation at 0: D = 1 − F(0), D' = −F'(0), D'' = −F''(0).
                    let dmass = (1.0 - f0[i]).max(PROB_EPS);
                    u[i] += d1_0[i] / dmass; // u_base − D'/D = u_base + F'(0)/D
                    w[i] = w[i] - d2_0[i] / dmass - (d1_0[i] / dmass).powi(2);
                }
            }
        })?;

        // xi atom: a Bernoulli on the zero indicator. Unlike the base parameters
        // above, this one *is* expected Fisher information, so it has a separable
        // natural scale and goes through the generic chain rule:
        //   ∂l/∂ξ = (I(y=0) − ξ) / (ξ(1−ξ)),   i_ξ = 1 / (ξ(1−ξ)).
        // Under the default logit link `mu_eta = ξ(1−ξ)`, so `chain_to_eta`
        // recovers the classic `u_η = I(y=0) − ξ` and `w_η = ξ(1−ξ)`.
        //
        // **The guard is on the denominator, not on ξ**: the same decision, for
        // the same reason, as `binomial.rs`. `clamp_prob` still applies in
        // `loglik_pointwise`, which takes a logarithm, but a clamp here would break
        // the telescoping: the caller multiplies by a `mu_eta` computed from η
        // independently of anything clamped, and under a probit link at the link's
        // own η clamp ξ = Φ(−30) ≈ 5e-198, far below `PROB_EPS`.
        let mut u_xi = Array1::<f64>::zeros(y.len());
        let mut i_xi = Array1::<f64>::zeros(y.len());
        for i in 0..y.len() {
            let xi_i = xi[i];
            let denom = (xi_i * (1.0 - xi_i)).max(DENOM_FLOOR);
            let z = if Self::is_zero(y[i]) { 1.0 } else { 0.0 };
            u_xi[i] = (z - xi_i) / denom;
            i_xi[i] = 1.0 / denom;
        }
        let xi_eta = chain_to_eta(
            HashMap::from([(Param::Xi, ScoreInfo::new(u_xi, i_xi))]),
            ctx,
        )?;
        out.extend(xi_eta);
        Ok(out)
    }

    /// `(1−ξ)·(Var_b + E_b²) / (1 − F(0)) − E[Y]²`.
    fn variance(&self, params: &HashMap<Param, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        let xi = self.xi(params)?;
        let mean_b = self.base.expected_value(params)?;
        let var_b = self.base.variance(params)?;
        let f0 = self.base_cdf_at_zero(mean_b.len(), params)?;
        Ok(Array1::from_shape_fn(mean_b.len(), |i| {
            let keep = 1.0 - clamp_prob(xi[i]);
            let mass = (1.0 - f0[i]).max(PROB_EPS);
            let second_moment = keep * (var_b[i] + mean_b[i] * mean_b[i]) / mass;
            let mean = keep * mean_b[i] / mass;
            (second_moment - mean * mean).max(0.0)
        }))
    }

    fn name(&self) -> &'static str {
        "Hurdle"
    }

    fn descriptor(&self) -> super::FamilyDescriptor {
        super::FamilyDescriptor::Hurdle {
            base: Box::new(self.base.descriptor()),
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
    use crate::distributions::{Gamma, ProbitLink, SqrtLink};
    use ndarray::array;

    fn gamma_hurdle_owned() -> Vec<(Param, Array1<f64>)> {
        vec![
            (Param::Mu, array![2.0, 3.0, 1.5, 4.0]),
            (Param::Sigma, array![0.5, 0.4, 0.6, 0.3]),
            (Param::Xi, array![0.3, 0.3, 0.3, 0.3]),
        ]
    }

    #[test]
    fn derivative_keys_and_weights_are_well_formed() {
        // See the matching test in `censored.rs`. Mixed zero / positive rows so
        // both the structural-zero mask and the zero-truncated base path are
        // exercised in one call.
        let y = array![0.0, 2.0, 0.0, 4.0];
        let owned = gamma_hurdle_owned();
        let h = Hurdle::new(Box::new(Gamma::new()));
        derivative_keys_match_parameters_observed_info(&h, params_view(&owned), &y);
    }

    #[test]
    fn cdf_has_the_zero_atom_and_the_truncated_base() {
        let owned = gamma_hurdle_owned();
        let params = params_view(&owned);
        let h = Hurdle::new(Box::new(Gamma::new()));

        let below = h.cdf(&array![-1.0, -0.5, -2.0, -0.1], &params).unwrap();
        assert!(below.iter().all(|&v| v == 0.0), "below zero: {below}");

        let at_zero = h.cdf(&Array1::zeros(4), &params).unwrap();
        for v in &at_zero {
            assert!((v - 0.3).abs() < 1e-12, "F(0) should be xi = 0.3, got {v}");
        }

        // A Gamma base puts no mass at zero, so above zero F_H = ξ + (1−ξ)·F.
        let y = array![1.0, 2.5, 0.7, 6.0];
        let base_cdf = Gamma::new().cdf(&y, &params).unwrap();
        let hurdle_cdf = h.cdf(&y, &params).unwrap();
        for i in 0..4 {
            let want = 0.3 + 0.7 * base_cdf[i];
            assert!((hurdle_cdf[i] - want).abs() < 1e-12);
        }
    }

    #[test]
    fn quantile_inverts_cdf_and_returns_zero_inside_the_atom() {
        let owned = gamma_hurdle_owned();
        let params = params_view(&owned);
        let h = Hurdle::new(Box::new(Gamma::new()));

        let inside_atom = h.quantile(&array![0.1, 0.2, 0.29, 0.05], &params).unwrap();
        assert!(inside_atom.iter().all(|&v| v == 0.0), "{inside_atom}");

        let y = array![1.0, 2.5, 0.7, 6.0];
        let back = h.quantile(&h.cdf(&y, &params).unwrap(), &params).unwrap();
        for i in 0..4 {
            assert!(
                (back[i] - y[i]).abs() < 1e-6,
                "q(F({})) = {}",
                y[i],
                back[i]
            );
        }
    }

    #[test]
    fn moments_fold_in_the_atom() {
        let owned = gamma_hurdle_owned();
        let params = params_view(&owned);
        let h = Hurdle::new(Box::new(Gamma::new()));
        let mean_b = Gamma::new().expected_value(&params).unwrap();
        let var_b = Gamma::new().variance(&params).unwrap();

        let mean = h.expected_value(&params).unwrap();
        let var = h.variance(&params).unwrap();
        for i in 0..4 {
            // F(0) = 0 for Gamma, so E = (1−ξ)·E_b and E[Y²] = (1−ξ)·(V_b + E_b²).
            let want_mean = 0.7 * mean_b[i];
            let want_var = 0.7 * (var_b[i] + mean_b[i] * mean_b[i]) - want_mean * want_mean;
            assert!((mean[i] - want_mean).abs() < 1e-12);
            assert!((var[i] - want_var).abs() < 1e-10);
        }
    }

    #[test]
    fn poisson_hurdle_truncates_the_base_zero() {
        use crate::distributions::Poisson;
        let mu = array![2.0];
        let xi = array![0.4];
        let params = HashMap::from([(Param::Mu, &mu), (Param::Xi, &xi)]);
        let h = Hurdle::new(Box::new(Poisson::new()));
        let p0 = (-2.0_f64).exp();

        let cdf = h.cdf(&array![0.0], &params).unwrap()[0];
        assert!((cdf - 0.4).abs() < 1e-12);
        let cdf1 = h.cdf(&array![1.0], &params).unwrap()[0];
        let want1 = 0.4 + 0.6 * (2.0 * p0) / (1.0 - p0);
        assert!((cdf1 - want1).abs() < 1e-12);

        // The left limit at 1 is F(0) = ξ, and at the atom it is 0.
        let (mu2, xi2) = (array![2.0, 2.0], array![0.4, 0.4]);
        let params2 = HashMap::from([(Param::Mu, &mu2), (Param::Xi, &xi2)]);
        let left = h
            .cdf_left_limit(&array![0.0, 1.0], &params2)
            .unwrap()
            .unwrap();
        assert_eq!(left[0], 0.0);
        assert!((left[1] - 0.4).abs() < 1e-12);

        let mean = h.expected_value(&params).unwrap()[0];
        assert!((mean - 0.6 * 2.0 / (1.0 - p0)).abs() < 1e-12);
    }

    #[test]
    fn left_limit_spreads_the_atom_over_a_continuous_base() {
        let owned = gamma_hurdle_owned();
        let params = params_view(&owned);
        let h = Hurdle::new(Box::new(Gamma::new()));
        let y = array![0.0, 2.0, 0.0, 4.0];
        let left = h.cdf_left_limit(&y, &params).unwrap().unwrap();
        let cdf = h.cdf(&y, &params).unwrap();
        assert_eq!(left[0], 0.0);
        assert_eq!(left[2], 0.0);
        // Continuous positive part: no jump above zero.
        assert_eq!(left[1], cdf[1]);
        assert_eq!(left[3], cdf[3]);
    }

    #[test]
    fn parameters_append_xi() {
        let h = Hurdle::new(Box::new(Gamma::new()));
        assert_eq!(h.parameters(), &[Param::Mu, Param::Sigma, Param::Xi]);
        assert_eq!(h.default_link(Param::Xi).unwrap().link(0.5), 0.0); // logit(0.5)=0
    }

    #[test]
    fn zero_rows_are_log_xi() {
        // Gamma base: F(0)=0 so the positive normalizer is 1; the zero atom is log ξ.
        let y = array![0.0, 2.0];
        let owned = [
            (Param::Mu, array![2.0, 2.0]),
            (Param::Sigma, array![0.5, 0.5]),
            (Param::Xi, array![0.25, 0.25]),
        ];
        let p = params_view(&owned);
        let h = Hurdle::new(Box::new(Gamma::new()));
        let ll = h.loglik_pointwise(&y, &p).unwrap();
        assert!((ll[0] - 0.25_f64.ln()).abs() < 1e-12);
        // positive row: log(1−ξ) + base loglik (F(0)=0 for Gamma ⇒ no truncation term).
        let base_ll = Gamma.loglik_pointwise(&y, &p).unwrap();
        assert!((ll[1] - ((1.0 - 0.25_f64).ln() + base_ll[1])).abs() < 1e-9);
    }

    #[test]
    fn xi_at_zero_matches_zero_truncated_base() {
        // ξ → 0 (no zeros) ⇒ positive rows reduce to the zero-truncated base.
        let y = array![1.0, 2.0, 3.0];
        let owned = [
            (Param::Mu, array![2.0, 2.0, 2.0]),
            (Param::Sigma, array![0.5, 0.5, 0.5]),
            (Param::Xi, array![1e-12, 1e-12, 1e-12]),
        ];
        let p = params_view(&owned);
        let h = Hurdle::new(Box::new(Gamma::new()));
        let ll = h.loglik_pointwise(&y, &p).unwrap();
        let base_ll = Gamma.loglik_pointwise(&y, &p).unwrap();
        for i in 0..3 {
            // F(0)=0 for Gamma so the zero-truncated density equals the base density.
            assert!((ll[i] - base_ll[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn hurdle_score_matches_finite_diff() {
        // Mixed zeros and positives; check every parameter including xi.
        let y = array![0.0, 2.0, 3.0, 0.0];
        let owned = gamma_hurdle_owned();
        let h = Hurdle::new(Box::new(Gamma::new()));
        check_score_via_finite_diff(&h, &y, &owned, Param::Mu, 1e-4);
        check_score_via_finite_diff(&h, &y, &owned, Param::Sigma, 1e-4);
        check_score_via_finite_diff(&h, &y, &owned, Param::Xi, 1e-4);
    }

    #[test]
    fn score_matches_finite_diff_under_a_non_default_link() {
        // Covers both of this wrapper's link-dependent sites in one fixture:
        //   μ on `sqrt`:   the zero-truncation normalizer's `F'(0)/D` term, built
        //                  from Gamma's analytic CDF derivative;
        //   ξ on `probit`: the zero atom, which must chain through the resolved
        //                  link rather than logit.
        // Gamma's μ is strictly positive, so η = √μ stays in the link's domain.
        let y = array![0.0, 2.0, 3.0, 0.0];
        let owned = gamma_hurdle_owned();
        let h = Hurdle::new(Box::new(Gamma::new()));
        check_eta_score_via_finite_diff(&h, &y, &owned, Param::Mu, &SqrtLink, 1e-4);
        check_eta_score_via_finite_diff(&h, &y, &owned, Param::Xi, &ProbitLink, 1e-4);
    }

    #[test]
    fn derivatives_stay_finite_at_a_saturated_fixture() {
        // ξ at both rails: the natural-scale atom divides by ξ(1−ξ), and
        // `DENOM_FLOOR` keeps it finite there.
        // μ and σ sweep the log link's reach at the same time so the zero-truncation
        // normalizer's `F'(0)/D` is evaluated in the saturated tail too.
        let y = array![0.0, 2.0, 0.0, 3.0];
        let owned = [
            (Param::Mu, array![1e-320, 1e13, 2.0, 1e-8]),
            (Param::Sigma, array![1e-8, 1e13, 0.5, 1e-320]),
            (Param::Xi, array![0.0, 1.0, 1e-320, 0.5]),
        ];
        let h = Hurdle::new(Box::new(Gamma::new()));
        let p = params_view(&owned);
        let d = default_link_derivatives(&h, &y, &p).unwrap();
        for name in [Param::Mu, Param::Sigma, Param::Xi] {
            let (u, w) = (&d[&name].score, &d[&name].info);
            assert!(finite_array(u) && finite_array(w), "{name}: {u:?} {w:?}");
        }
    }
}
