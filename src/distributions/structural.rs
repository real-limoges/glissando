//! Shared infrastructure for the structural-likelihood wrappers
//! ([`Censored`](super::Censored) / [`Truncated`](super::Truncated) /
//! [`Hurdle`](super::Hurdle)).
//!
//! The wrappers all rewrite a base family's likelihood around its CDF, so they
//! share one routine: evaluate `(∂F/∂η, ∂²F/∂η²)` per base parameter, taking the
//! family's analytic value where it supplies one
//! ([`Distribution::cdf_eta_derivatives`]) and otherwise central-differencing
//! `base.cdf` on that parameter's η. Location/scale parameters land on the
//! analytic path; non-elementary shape parameters fall back to the numeric one.

use super::{CdfEtaResult, Distribution, GamlssError, Link};
use ndarray::Array1;
use std::collections::HashMap;

/// Step size for the central-difference fallback, on the η (link) scale. Small
/// enough that the first derivative is accurate to ~`eps²`, large enough that the
/// second-difference (which divides by `eps²`) keeps the CDF round-off from
/// dominating.
const FD_EPS: f64 = 1e-5;

/// Reject a response whose length does not match a wrapper's stored per-row state.
///
/// `name` identifies the wrapper and the field (e.g. `"Censored: status"`), so the
/// message reads as `"<name> length {stored} does not match response length {n}"`.
pub(crate) fn check_state_len(name: &str, stored: usize, n: usize) -> Result<(), GamlssError> {
    if stored != n {
        return Err(GamlssError::Input(format!(
            "{name} length {stored} does not match response length {n}"
        )));
    }
    Ok(())
}

/// Emit `Distribution` methods that are pure one-line passthroughs to `self.base`.
///
/// Each wrapper lists only the methods it genuinely forwards unchanged; anything
/// the wrapper reshapes (e.g. `Truncated::cdf`, which renormalizes) stays
/// hand-written. Requires `Array1`, `HashMap`, `Link` and `GamlssError` in scope
/// at the call site.
macro_rules! delegate_to_base {
    (@method parameters) => {
        fn parameters(&self) -> &[&'static str] {
            self.base.parameters()
        }
    };
    (@method default_link) => {
        fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError> {
            self.base.default_link(param)
        }
    };
    (@method initial_value) => {
        fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
            self.base.initial_value(param, y)
        }
    };
    (@method is_discrete) => {
        fn is_discrete(&self) -> bool {
            self.base.is_discrete()
        }
    };
    (@method variance) => {
        fn variance(
            &self,
            params: &HashMap<&str, &Array1<f64>>,
        ) -> Result<Array1<f64>, GamlssError> {
            self.base.variance(params)
        }
    };
    (@method expected_value) => {
        fn expected_value(
            &self,
            params: &HashMap<&str, &Array1<f64>>,
        ) -> Result<Array1<f64>, GamlssError> {
            self.base.expected_value(params)
        }
    };
    (@method cdf) => {
        fn cdf(
            &self,
            y: &Array1<f64>,
            params: &HashMap<&str, &Array1<f64>>,
        ) -> Result<Array1<f64>, GamlssError> {
            self.base.cdf(y, params)
        }
    };
    (@method quantile) => {
        fn quantile(
            &self,
            p: &Array1<f64>,
            params: &HashMap<&str, &Array1<f64>>,
        ) -> Result<Array1<f64>, GamlssError> {
            self.base.quantile(p, params)
        }
    };
    ($($m:ident),* $(,)?) => {
        $(delegate_to_base!(@method $m);)*
    };
}

pub(crate) use delegate_to_base;

/// `(∂F/∂η, ∂²F/∂η²)` at the points `at`, for every parameter of `base`.
///
/// Analytic for the parameters `base.cdf_eta_derivatives` supplies; a central
/// difference of `base.cdf` (perturbing the parameter on its default-link η) for
/// the rest. Derivatives are taken w.r.t. the family's *default* link, matching
/// the convention the families' own [`Distribution::derivatives`] use.
pub(crate) fn cdf_eta_grads(
    base: &dyn Distribution,
    at: &Array1<f64>,
    params: &HashMap<&str, &Array1<f64>>,
) -> CdfEtaResult {
    let analytic = base.cdf_eta_derivatives(at, params)?;
    let mut out = HashMap::new();
    for &p in base.parameters() {
        if let Some(v) = analytic.get(p) {
            out.insert(p.to_string(), v.clone());
        } else {
            let link = base.default_link(p)?;
            let grads = numeric_cdf_grad(base, at, params, p, link.as_ref())?;
            out.insert(p.to_string(), grads);
        }
    }
    Ok(out)
}

/// Central-difference `(∂F/∂η, ∂²F/∂η²)` for a single parameter, perturbing it on
/// the η-scale through `link` and re-evaluating `base.cdf` at `at`.
fn numeric_cdf_grad(
    base: &dyn Distribution,
    at: &Array1<f64>,
    params: &HashMap<&str, &Array1<f64>>,
    param: &str,
    link: &dyn Link,
) -> Result<(Array1<f64>, Array1<f64>), GamlssError> {
    let orig = params
        .get(param)
        .copied()
        .ok_or_else(|| base.unknown_param(param))?;
    let n = at.len();

    let mut plus = orig.clone();
    let mut minus = orig.clone();
    for i in 0..n {
        let eta = link.link(orig[i]);
        plus[i] = link.inv_link(eta + FD_EPS);
        minus[i] = link.inv_link(eta - FD_EPS);
    }

    let f0 = base.cdf(at, params)?;
    let mut p_plus = params.clone();
    p_plus.insert(param, &plus);
    let f_plus = base.cdf(at, &p_plus)?;
    let mut p_minus = params.clone();
    p_minus.insert(param, &minus);
    let f_minus = base.cdf(at, &p_minus)?;

    let d1 = (&f_plus - &f_minus) / (2.0 * FD_EPS);
    let d2 = (&f_plus - 2.0 * &f0 + &f_minus) / (FD_EPS * FD_EPS);
    Ok((d1, d2))
}
