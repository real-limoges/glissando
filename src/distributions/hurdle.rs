//! Hurdle / two-part models (STRUCT-3): a point mass at zero combined with a
//! zero-truncated base for the positive part.
//!
//! ```text
//! P(Y = 0) = ξ                          — the zero atom (logit-linked)
//! P(Y = y) = (1 − ξ) · g_T(y)   (y > 0) — base, zero-TRUNCATED
//! ```
//!
//! This is a clean generalization of zero-inflation. Contrast with
//! zero-*inflation* (DIST-5), where the base can still emit zero
//! (`P(Y=0) = π + (1−π)·g(0)`): a hurdle's positive process is structurally
//! separate from the zero process — reach for it when the zero-generating
//! mechanism is distinct (a true two-part model), and for zero-inflation when
//! zeros are a contamination of one process.
//!
//! The wrapper adds one fitted parameter `xi` (the zero probability, logit link)
//! on top of the base family's parameters; the positive part reuses the
//! zero-truncation machinery (STRUCT-2). Like the other structural wrappers it is
//! excluded from [`from_name`](super::from_name).

use super::structural::{cdf_eta_grads, delegate_to_base, rewrite_base_derivatives};
use super::{
    clamp_prob, DerivativesResult, Distribution, GamlssError, Link, LogitLink, MIN_WEIGHT, PROB_EPS,
};
use ndarray::Array1;
use std::collections::HashMap;

/// A base family augmented with a zero atom: zeros come from a logit-linked
/// probability `xi`, positive values from the zero-truncated base.
#[derive(Debug)]
pub struct Hurdle {
    base: Box<dyn Distribution>,
    /// `base.parameters()` followed by `"xi"`; backs [`Distribution::parameters`].
    params: Vec<&'static str>,
}

impl Hurdle {
    /// Wrap `base` with a logit-linked zero atom `xi = P(Y = 0)`.
    pub fn new(base: Box<dyn Distribution>) -> Self {
        let mut params = base.parameters().to_vec();
        params.push("xi");
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
}

impl Distribution for Hurdle {
    fn parameters(&self) -> &[&'static str] {
        &self.params
    }

    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
        if param == "xi" {
            Ok(Box::new(LogitLink))
        } else {
            self.base.default_link(param)
        }
    }

    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        if param == "xi" {
            // Empirical zero fraction, clamped away from {0, 1}.
            let zeros = y.iter().filter(|&&v| Self::is_zero(v)).count() as f64;
            (zeros / y.len().max(1) as f64).clamp(0.05, 0.95)
        } else {
            self.base.initial_value(param, y)
        }
    }

    delegate_to_base!(is_discrete, expected_value, cdf, quantile);

    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        let xi = params
            .get("xi")
            .copied()
            .ok_or_else(|| self.unknown_param("xi"))?;
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

    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult {
        let xi = params
            .get("xi")
            .copied()
            .ok_or_else(|| self.unknown_param("xi"))?;
        // Base parameters: zero-truncated score on positive rows, nothing on zeros.
        let base_derivs = self.base.derivatives(y, params)?;
        let zeros = Array1::<f64>::zeros(y.len());
        let f0 = self.base.cdf(&zeros, params)?;
        let grad0 = cdf_eta_grads(self.base.as_ref(), &zeros, &f0, params)?;

        let mut out = rewrite_base_derivatives(self.base.as_ref(), base_derivs, |param, u, w| {
            let (d1_0, d2_0) = &grad0[param];
            for i in 0..y.len() {
                if Self::is_zero(y[i]) {
                    // Zero rows carry no information about the positive-part params.
                    u[i] = 0.0;
                    w[i] = MIN_WEIGHT;
                } else {
                    // Zero-truncation at 0: D = 1 − F(0), D' = −F'(0), D'' = −F''(0).
                    let dmass = (1.0 - f0[i]).max(PROB_EPS);
                    u[i] += d1_0[i] / dmass; // u_base − D'/D = u_base + F'(0)/D
                    w[i] = (w[i] - d2_0[i] / dmass - (d1_0[i] / dmass).powi(2)).max(MIN_WEIGHT);
                }
            }
        })?;

        // xi atom (logit link): a Bernoulli on the zero indicator.
        //   u = I(y=0) − xi,   w = xi(1 − xi).
        let mut u_xi = Array1::<f64>::zeros(y.len());
        let mut w_xi = Array1::<f64>::zeros(y.len());
        for i in 0..y.len() {
            let xi_i = clamp_prob(xi[i]);
            let z = if Self::is_zero(y[i]) { 1.0 } else { 0.0 };
            u_xi[i] = z - xi_i;
            w_xi[i] = (xi_i * (1.0 - xi_i)).max(MIN_WEIGHT);
        }
        out.insert("xi".to_string(), (u_xi, w_xi));
        Ok(out)
    }

    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError> {
        // Reports the untruncated base variance (the zero atom and truncation are
        // not folded in — a known diagnostic approximation, as for Truncated).
        self.base.variance(params)
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
    use crate::distributions::test_helpers::{check_score_via_finite_diff, params_view};
    use crate::distributions::Gamma;
    use ndarray::array;

    fn gamma_hurdle_owned() -> Vec<(&'static str, Array1<f64>)> {
        vec![
            ("mu", array![2.0, 3.0, 1.5, 4.0]),
            ("sigma", array![0.5, 0.4, 0.6, 0.3]),
            ("xi", array![0.3, 0.3, 0.3, 0.3]),
        ]
    }

    #[test]
    fn parameters_append_xi() {
        let h = Hurdle::new(Box::new(Gamma::new()));
        assert_eq!(h.parameters(), &["mu", "sigma", "xi"]);
        assert_eq!(h.default_link("xi").unwrap().link(0.5), 0.0); // logit(0.5)=0
    }

    #[test]
    fn zero_rows_are_log_xi() {
        // Gamma base: F(0)=0 so the positive normalizer is 1; the zero atom is log ξ.
        let y = array![0.0, 2.0];
        let owned = [
            ("mu", array![2.0, 2.0]),
            ("sigma", array![0.5, 0.5]),
            ("xi", array![0.25, 0.25]),
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
            ("mu", array![2.0, 2.0, 2.0]),
            ("sigma", array![0.5, 0.5, 0.5]),
            ("xi", array![1e-12, 1e-12, 1e-12]),
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
        check_score_via_finite_diff(&h, &y, &owned, "mu", 1e-4);
        check_score_via_finite_diff(&h, &y, &owned, "sigma", 1e-4);
        check_score_via_finite_diff(&h, &y, &owned, "xi", 1e-4);
    }
}
