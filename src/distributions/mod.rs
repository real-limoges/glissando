//! Probability distributions for GAMLSS models.
//!
//! Each distribution defines its parameter names (μ, σ, ν, …), default link functions,
//! and the score / Fisher-information pairs that drive the Rigby–Stasinopoulos
//! IRLS update. Derivatives are batched (vectorized over observations).

use crate::error::GamlssError;
use ndarray::Array1;
use std::collections::HashMap;
use std::fmt::Debug;

/// Floor for positive parameters (μ, σ, …) to avoid log(0) or division by zero.
pub(crate) const MIN_POSITIVE: f64 = 1e-10;
/// Linear-predictor ceiling for log/logit links (prevents `exp` overflow).
pub(crate) const MAX_ETA: f64 = 30.0;
/// Linear-predictor floor for log/logit links (prevents `exp` underflow).
pub(crate) const MIN_ETA: f64 = -30.0;
/// Lower bound on Fisher-information weights to keep `W` positive definite.
pub(crate) const MIN_WEIGHT: f64 = 1e-6;

// ============================================================================
// Link functions
// ============================================================================

/// A link function `g` mapping the response-scale parameter `μ` to the linear predictor `η = g(μ)`.
pub trait Link: Debug + Send + Sync {
    /// Apply the link: `η = g(μ)`.
    fn link(&self, mu: f64) -> f64;
    /// Apply the inverse link: `μ = g⁻¹(η)`.
    fn inv_link(&self, eta: f64) -> f64;
}

/// Identity link: `η = μ`. Used for unbounded continuous parameters (e.g. Gaussian mean).
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityLink;

impl Link for IdentityLink {
    fn link(&self, mu: f64) -> f64 {
        mu
    }
    fn inv_link(&self, eta: f64) -> f64 {
        eta
    }
}

/// Log link: `η = log(μ)`. Used for positive parameters (Poisson rate, Gamma mean).
///
/// `link` clamps `log(μ)` to [`MIN_ETA`]; `inv_link` clamps `η` to [`MAX_ETA`].
#[derive(Debug, Clone, Copy, Default)]
pub struct LogLink;

impl Link for LogLink {
    fn link(&self, mu: f64) -> f64 {
        mu.ln().max(MIN_ETA)
    }
    fn inv_link(&self, eta: f64) -> f64 {
        eta.min(MAX_ETA).exp()
    }
}

/// Logit link: `η = log(μ / (1 - μ))`. Used for `(0, 1)` probability parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogitLink;

impl Link for LogitLink {
    fn link(&self, mu: f64) -> f64 {
        let m = mu.clamp(MIN_POSITIVE, 1.0 - MIN_POSITIVE);
        (m / (1.0 - m)).ln()
    }
    fn inv_link(&self, eta: f64) -> f64 {
        let e = eta.clamp(MIN_ETA, MAX_ETA);
        1.0 / (1.0 + (-e).exp())
    }
}

// ============================================================================
// Distribution trait
// ============================================================================

/// Score / Fisher-information pairs keyed by distribution-parameter name.
pub type DerivativesResult = Result<HashMap<String, (Array1<f64>, Array1<f64>)>, GamlssError>;

/// A statistical distribution for GAMLSS, defining parameters, link functions, and
/// score / Fisher-information pairs that drive the IRLS algorithm.
pub trait Distribution: Debug + Send + Sync {
    /// Distribution-parameter names (e.g. `["mu", "sigma"]`).
    fn parameters(&self) -> &[&'static str];

    /// Default link function for the named parameter.
    ///
    /// # Errors
    ///
    /// Returns `GamlssError::UnknownParameter` if the name is not one of [`Self::parameters`].
    fn default_link(&self, param: &str) -> Result<Box<dyn Link>, GamlssError>;

    /// Score (`u`) and Fisher information (`w`) on the linear-predictor scale, for each
    /// parameter, evaluated against the current `params` snapshot.
    fn derivatives(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> DerivativesResult;

    /// Per-observation log-density `log f(y_i | params_i)`, used to assemble the model
    /// log-likelihood and observation-level diagnostics (WAIC, leave-one-out, etc.).
    fn loglik_pointwise(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError>;

    /// Marginal `Var(Y_i | params_i)` on the response scale, used for Pearson residuals.
    ///
    /// Distinct from the Fisher-information weight returned by [`Self::derivatives`],
    /// which is on the linear-predictor scale.
    fn variance(&self, params: &HashMap<&str, &Array1<f64>>) -> Result<Array1<f64>, GamlssError>;

    /// Marginal `E[Y_i | params_i]` on the response scale.
    ///
    /// Default returns `params["mu"]` cloned. Distributions where `mu` is not the
    /// expected value of `Y` (e.g. [`Binomial`] where `E[Y] = n·μ`) override.
    fn expected_value(
        &self,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<Array1<f64>, GamlssError> {
        Ok(require(self, params, "mu")?.to_owned())
    }

    /// Total model log-likelihood: `Σ log f(y_i | params_i)`.
    fn loglik(
        &self,
        y: &Array1<f64>,
        params: &HashMap<&str, &Array1<f64>>,
    ) -> Result<f64, GamlssError> {
        Ok(self.loglik_pointwise(y, params)?.sum())
    }

    /// Stable distribution name (e.g. `"Gaussian"`); used in error messages and
    /// for the WASM `from_name` lookup.
    fn name(&self) -> &'static str;

    /// Initial response-scale value for a parameter, used to seed the IRLS loop.
    /// Override for distributions where `y` is not directly a sample of the parameter.
    fn initial_value(&self, param: &str, y: &Array1<f64>) -> f64 {
        // `validate_inputs` rejects empty `y` before fitting, so `mean` returning `None`
        // is unreachable on the public path. The `unwrap_or` below keeps the fn pure.
        match param {
            "mu" => y.mean().unwrap_or(0.5),
            "sigma" => {
                let s = y.std(1.0);
                if s < 1e-4 {
                    1.0
                } else {
                    s
                }
            }
            "nu" => 5.0,
            "phi" => 1.0,
            _ => 0.1,
        }
    }

    /// Build a fresh `UnknownParameter` error tagged with this distribution's name.
    fn unknown_param(&self, param: &str) -> GamlssError {
        GamlssError::UnknownParameter {
            distribution: self.name().to_string(),
            param: param.to_string(),
        }
    }
}

/// Look up `param` in a derivatives-input map or yield an `UnknownParameter` error.
pub(crate) fn require<'a, D: Distribution + ?Sized>(
    dist: &D,
    params: &HashMap<&str, &'a Array1<f64>>,
    name: &str,
) -> Result<&'a Array1<f64>, GamlssError> {
    params
        .get(name)
        .copied()
        .ok_or_else(|| dist.unknown_param(name))
}

// ============================================================================
// Distribution implementations
// ============================================================================

mod beta;
mod binomial;
mod gamma;
mod gaussian;
mod negative_binomial;
mod poisson;
mod student_t;

pub use beta::Beta;
pub use binomial::Binomial;
pub use gamma::Gamma;
pub use gaussian::Gaussian;
pub use negative_binomial::NegativeBinomial;
pub use poisson::Poisson;
pub use student_t::StudentT;

/// Construct a stateless distribution from its name (e.g. for WASM JSON I/O).
///
/// Excludes [`Binomial`] because it requires `n_trials` state that cannot be recovered
/// from the name alone.
///
/// # Errors
///
/// Returns `GamlssError::Input` if `name` does not match a supported stateless distribution.
///
/// # Examples
///
/// ```
/// use glissando::distributions::{from_name, Distribution};
///
/// let d = from_name("Gaussian").unwrap();
/// assert_eq!(d.name(), "Gaussian");
/// assert_eq!(d.parameters(), &["mu", "sigma"]);
///
/// assert!(from_name("Wishart").is_err());
/// ```
pub fn from_name(name: &str) -> Result<Box<dyn Distribution>, GamlssError> {
    match name {
        "Gaussian" => Ok(Box::new(Gaussian)),
        "Poisson" => Ok(Box::new(Poisson)),
        "StudentT" => Ok(Box::new(StudentT)),
        "Gamma" => Ok(Box::new(Gamma)),
        "NegativeBinomial" => Ok(Box::new(NegativeBinomial)),
        "Beta" => Ok(Box::new(Beta)),
        other => Err(GamlssError::Input(format!(
            "Unknown distribution: '{}'. Supported: Gaussian, Poisson, StudentT, Gamma, NegativeBinomial, Beta",
            other
        ))),
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;

    pub fn finite_array(a: &Array1<f64>) -> bool {
        a.iter().all(|v| v.is_finite())
    }

    pub fn derivative_keys_match_parameters<D: Distribution>(
        d: &D,
        params: HashMap<&str, &Array1<f64>>,
        y: &Array1<f64>,
    ) {
        let derivs = d.derivatives(y, &params).unwrap();
        let mut keys: Vec<&str> = derivs.keys().map(String::as_str).collect();
        keys.sort();
        let mut expected: Vec<&str> = d.parameters().to_vec();
        expected.sort();
        assert_eq!(keys, expected);
        for (u, w) in derivs.values() {
            assert_eq!(u.len(), y.len());
            assert_eq!(w.len(), y.len());
            assert!(finite_array(u));
            assert!(finite_array(w));
            // Fisher info should be non-negative.
            assert!(w.iter().all(|&v| v >= 0.0));
        }
    }

    /// Build a `params` view from owned arrays for test ergonomics.
    pub fn params_view<'a>(
        owned: &'a [(&'static str, Array1<f64>)],
    ) -> HashMap<&'a str, &'a Array1<f64>> {
        owned.iter().map(|(k, v)| (*k, v)).collect()
    }

    /// Check that the analytic score `u` returned by `derivatives()` matches the
    /// central difference of `loglik_pointwise` for `target` on the η-scale.
    pub fn check_score_via_finite_diff<D: Distribution + ?Sized>(
        family: &D,
        y: &Array1<f64>,
        owned: &[(&'static str, Array1<f64>)],
        target: &str,
        tol: f64,
    ) {
        let p: HashMap<&str, &Array1<f64>> = owned.iter().map(|(k, v)| (*k, v)).collect();
        let derivs = family.derivatives(y, &p).unwrap();
        let analytic_u = derivs.get(target).unwrap().0.clone();

        let link = family.default_link(target).unwrap();
        let eps: f64 = 1e-6;
        let idx = owned.iter().position(|(k, _)| *k == target).unwrap();

        let mut perturbed: Vec<(&'static str, Array1<f64>)> =
            owned.iter().map(|(k, v)| (*k, v.clone())).collect();

        for i in 0..y.len() {
            let mu_orig = owned[idx].1[i];
            let eta = link.link(mu_orig);

            perturbed[idx].1[i] = link.inv_link(eta + eps);
            let p_plus: HashMap<&str, &Array1<f64>> =
                perturbed.iter().map(|(k, v)| (*k, v)).collect();
            let l_plus = family.loglik_pointwise(y, &p_plus).unwrap()[i];

            perturbed[idx].1[i] = link.inv_link(eta - eps);
            let p_minus: HashMap<&str, &Array1<f64>> =
                perturbed.iter().map(|(k, v)| (*k, v)).collect();
            let l_minus = family.loglik_pointwise(y, &p_minus).unwrap()[i];

            perturbed[idx].1[i] = mu_orig;

            let numeric_u = (l_plus - l_minus) / (2.0 * eps);
            let scale = analytic_u[i].abs().max(1.0);
            assert!(
                (analytic_u[i] - numeric_u).abs() / scale < tol,
                "{}::{} obs {}: analytic u={:.6e}, numeric u={:.6e}",
                family.name(),
                target,
                i,
                analytic_u[i],
                numeric_u
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;

    // --- Links ---

    #[test]
    fn identity_link_is_noop() {
        let l = IdentityLink;
        assert_eq!(l.link(0.0), 0.0);
        assert_eq!(l.link(3.5), 3.5);
        assert_eq!(l.inv_link(-2.0), -2.0);
    }

    #[test]
    fn log_link_clamps_underflow_and_overflow() {
        let l = LogLink;
        assert!(l.link(0.0).is_finite());
        assert!(l.link(0.0) <= MIN_ETA + 1e-9);
        assert!(l.inv_link(1e6).is_finite());
        assert!(l.inv_link(1e6) <= MAX_ETA.exp() + 1.0);
    }

    #[test]
    fn logit_link_handles_boundary_probabilities() {
        let l = LogitLink;
        assert!(l.link(0.0).is_finite());
        assert!(l.link(1.0).is_finite());
        assert!((0.0..=1.0).contains(&l.inv_link(-1e6)));
        assert!((0.0..=1.0).contains(&l.inv_link(1e6)));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn identity_link_round_trip(x in -1e6f64..1e6) {
            let l = IdentityLink;
            prop_assert!(close(l.link(l.inv_link(x)), x, 1e-9));
        }

        #[test]
        fn log_link_round_trip_in_safe_range(eta in (MIN_ETA + 1.0) .. (MAX_ETA - 1.0)) {
            let l = LogLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-9));
        }

        #[test]
        fn logit_link_round_trip_in_safe_range(eta in -20.0f64..20.0) {
            let l = LogitLink;
            prop_assert!(close(l.link(l.inv_link(eta)), eta, 1e-6));
        }

        #[test]
        fn log_link_inv_link_strictly_positive(eta in -50.0f64..50.0) {
            let l = LogLink;
            prop_assert!(l.inv_link(eta) > 0.0);
        }

        #[test]
        fn logit_link_inv_link_in_unit_interval(eta in -1e6f64..1e6) {
            let l = LogitLink;
            let p = l.inv_link(eta);
            prop_assert!((0.0..=1.0).contains(&p));
        }
    }

    // --- from_name ---

    #[test]
    fn from_name_returns_expected_variants() {
        for name in &[
            "Gaussian",
            "Poisson",
            "StudentT",
            "Gamma",
            "NegativeBinomial",
            "Beta",
        ] {
            let d = from_name(name).unwrap();
            assert_eq!(d.name(), *name);
        }
    }

    #[test]
    fn from_name_unknown_returns_input_error() {
        let err = from_name("Wishart").unwrap_err();
        assert!(matches!(err, GamlssError::Input(_)));
    }

    // --- initial_value (cross-distribution) ---

    #[test]
    fn initial_value_finite_for_typical_parameters() {
        let y = ndarray::array![1.0, 2.0, 3.0, 4.0];
        for d in [
            from_name("Gaussian").unwrap(),
            from_name("Poisson").unwrap(),
            from_name("StudentT").unwrap(),
            from_name("Gamma").unwrap(),
            from_name("Beta").unwrap(),
            from_name("NegativeBinomial").unwrap(),
        ] {
            for p in d.parameters() {
                let v = d.initial_value(p, &y);
                assert!(
                    v.is_finite(),
                    "{}::{} initial_value not finite: {}",
                    d.name(),
                    p,
                    v
                );
            }
        }
    }

    // --- unknown_param helper ---

    #[test]
    fn unknown_param_carries_distribution_name() {
        let err = Gaussian.unknown_param("zeta");
        match err {
            GamlssError::UnknownParameter {
                distribution,
                param,
            } => {
                assert_eq!(distribution, "Gaussian");
                assert_eq!(param, "zeta");
            }
            other => panic!("expected UnknownParameter, got {:?}", other),
        }
    }
}
