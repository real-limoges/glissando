//! Aggregate diagnostics for a fitted [`GamlssModel`](crate::GamlssModel): residuals,
//! log-likelihood, EDF, AIC, BIC.
//!
//! Per-family math (log-density, marginal variance, expected value) lives on the
//! [`Distribution`] trait. This module composes those primitives — it never
//! hand-dispatches on the family name.

mod residuals;

pub use residuals::*;

use super::FittedParameter;
use crate::distributions::Distribution;
use crate::GamlssError;
use indexmap::IndexMap;
use ndarray::Array1;
use std::collections::HashMap;

/// Aggregated model diagnostics including residuals, EDF, and information criteria.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelDiagnostics {
    /// `(y − E[Y]) / √Var(Y)`
    pub pearson_residuals: Array1<f64>,
    /// `y − E[Y]`
    pub response_residuals: Array1<f64>,
    /// Summed across all distribution parameters.
    pub total_edf: f64,
    pub aic: f64,
    pub bic: f64,
    pub log_likelihood: f64,
    pub n_obs: usize,
}

/// Generalized Akaike Information Criterion: `−2·loglik + k·EDF`.
///
/// The penalty `k` is the selection knob: `k = 2` recovers AIC (predictive
/// accuracy, permissive), `k = log(n)` recovers BIC/SBC (consistency,
/// parsimonious). `k = 3.84 ≈ χ²₁,₀.₉₅` is another common choice. Every
/// information-criterion score in the crate flows through this one definition,
/// so AIC, BIC, and the stepwise / ANOVA scores stay mutually consistent.
///
/// `EDF` is the *effective* degrees of freedom (the summed smoother traces,
/// generally fractional), not the raw coefficient count.
///
/// # Examples
///
/// ```
/// use glissando::diagnostics::compute_gaic;
///
/// // k = 2 is AIC; k = log(n) is BIC.
/// assert!((compute_gaic(-100.0, 5.0, 2.0) - 210.0).abs() < 1e-12);
/// // Larger k penalizes complexity harder for the same fit.
/// assert!(compute_gaic(-100.0, 5.0, 4.0) > compute_gaic(-100.0, 5.0, 2.0));
/// ```
pub fn compute_gaic(log_likelihood: f64, total_edf: f64, k: f64) -> f64 {
    -2.0 * log_likelihood + k * total_edf
}

/// Computes Akaike Information Criterion: `−2·loglik + 2·EDF`.
///
/// Thin wrapper over [`compute_gaic`] at `k = 2`.
///
/// # Examples
///
/// ```
/// use glissando::diagnostics::compute_aic;
///
/// // Two models with the same log-likelihood: lower-EDF model has lower AIC.
/// let aic_simple = compute_aic(-100.0, 3.0);
/// let aic_complex = compute_aic(-100.0, 5.0);
/// assert!(aic_simple < aic_complex);
/// ```
pub fn compute_aic(log_likelihood: f64, total_edf: f64) -> f64 {
    compute_gaic(log_likelihood, total_edf, 2.0)
}

/// Computes Bayesian Information Criterion: `−2·loglik + log(n)·EDF`.
///
/// Thin wrapper over [`compute_gaic`] at `k = log(n)`.
pub fn compute_bic(log_likelihood: f64, total_edf: f64, n_obs: usize) -> f64 {
    compute_gaic(log_likelihood, total_edf, (n_obs as f64).ln())
}

pub fn total_edf(fitted_params: &IndexMap<String, FittedParameter>) -> f64 {
    fitted_params.values().map(|p| p.edf).sum()
}

/// Snapshot of fitted parameters on the response scale, in the shape the
/// [`Distribution`] trait expects (parameter name → fitted values). The single
/// source for this view; `GamlssModel`'s scoring methods and `selection` build
/// on it rather than re-collecting the map.
pub(crate) fn fitted_params_view(
    models: &IndexMap<String, FittedParameter>,
) -> HashMap<&str, &Array1<f64>> {
    models
        .iter()
        .map(|(k, v)| (k.as_str(), &v.fitted_values))
        .collect()
}

pub(crate) fn compute<D: Distribution + ?Sized>(
    models: &IndexMap<String, FittedParameter>,
    family: &D,
    y: &Array1<f64>,
) -> Result<ModelDiagnostics, GamlssError> {
    let params = fitted_params_view(models);
    let expected = family.expected_value(&params)?;
    let pearson = pearson_residuals(family, y, &params)?;
    let response = response_residuals(y, &expected);
    let log_likelihood = family.loglik(y, &params)?;
    let edf = total_edf(models);
    let n_obs = y.len();
    let aic = compute_aic(log_likelihood, edf);
    let bic = compute_bic(log_likelihood, edf, n_obs);

    Ok(ModelDiagnostics {
        pearson_residuals: pearson,
        response_residuals: response,
        total_edf: edf,
        aic,
        bic,
        log_likelihood,
        n_obs,
    })
}

/// Randomized (normalized) quantile residuals — gamlss's default residual
/// (Dunn & Smyth, 1996).
///
/// By the probability integral transform, `U = F(Y | θ̂) ∼ Uniform(0,1)` for a
/// correct continuous fit, so `r = Φ⁻¹(U) ∼ N(0,1)` regardless of the family —
/// one Q-Q/worm-plot yardstick across every distribution. For a discrete
/// response `F` jumps, so the **randomized PIT** spreads each atom across its
/// jump interval: `u_i = F(y_i−1) + v_i·(F(y_i) − F(y_i−1))` with `v_i ∼ U(0,1)`.
///
/// `seed` makes the discrete randomization reproducible (via `StdRng`); it is
/// ignored for continuous families, where `u = F(y)` is deterministic.
///
/// # Errors
/// Propagates [`GamlssError`] from the family's `cdf` evaluation.
pub fn quantile_residuals<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    params: &HashMap<&str, &Array1<f64>>,
    seed: Option<u64>,
) -> Result<Array1<f64>, GamlssError> {
    use crate::math::std_normal_quantile;
    use rand::rngs::StdRng;
    use rand::{rng, Rng, SeedableRng};
    use rand_distr::{Distribution as _, StandardUniform};

    let u = if family.is_discrete() {
        let upper = family.cdf(y, params)?; // F(y)
        let lower = family.cdf(&(y - 1.0), params)?; // F(y−1)
                                                     // One generic filler, two RNG concretizations — mirrors `sample_posterior_seeded`.
        fn randomize<R: Rng>(rng: &mut R, lower: &Array1<f64>, upper: &Array1<f64>) -> Array1<f64> {
            ndarray::Zip::from(lower).and(upper).map_collect(|&a, &b| {
                let v: f64 = StandardUniform.sample(rng); // v ∼ U[0,1)
                a + (b - a) * v
            })
        }
        match seed {
            Some(s) => randomize(&mut StdRng::seed_from_u64(s), &lower, &upper),
            None => randomize(&mut rng(), &lower, &upper),
        }
    } else {
        family.cdf(y, params)? // continuous: u = F(y)
    };

    Ok(u.mapv(std_normal_quantile))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::Term;
    use crate::types::{Coefficients, CovarianceMatrix};
    use ndarray::{array, Array2};

    // --- compute_gaic / compute_aic / compute_bic ---

    #[test]
    fn compute_gaic_recovers_aic_and_bic() {
        let (ll, edf, n) = (-100.0, 5.0, 100usize);
        assert!((compute_gaic(ll, edf, 2.0) - compute_aic(ll, edf)).abs() < 1e-12);
        assert!((compute_gaic(ll, edf, (n as f64).ln()) - compute_bic(ll, edf, n)).abs() < 1e-12);
    }

    #[test]
    fn compute_gaic_monotone_in_k() {
        // Larger penalty ⇒ larger GAIC for the same fit (edf > 0).
        let (ll, edf) = (-50.0, 4.0);
        assert!(compute_gaic(ll, edf, 4.0) > compute_gaic(ll, edf, 2.0));
        assert!(compute_gaic(ll, edf, 2.0) > compute_gaic(ll, edf, 1.0));
    }

    #[test]
    fn compute_gaic_exact_formula_and_edge_cases() {
        // Pin the exact arithmetic −2·ll + k·edf across ordinary and edge inputs.
        assert!((compute_gaic(-10.0, 3.0, 2.0) - (20.0 + 6.0)).abs() < 1e-12);
        // k = 0 ⇒ pure global deviance −2·ll (no complexity penalty).
        assert!((compute_gaic(-10.0, 3.0, 0.0) - 20.0).abs() < 1e-12);
        // edf = 0 ⇒ penalty term vanishes for any k.
        assert!((compute_gaic(-10.0, 0.0, 7.5) - 20.0).abs() < 1e-12);
        // Negative k is still evaluated literally (no clamping).
        assert!((compute_gaic(-10.0, 3.0, -2.0) - (20.0 - 6.0)).abs() < 1e-12);
        // Positive log-likelihood (tight continuous fit) ⇒ negative deviance.
        assert!((compute_gaic(5.0, 1.0, 2.0) - (-10.0 + 2.0)).abs() < 1e-12);
    }

    #[test]
    fn compute_gaic_prefers_lower_edf_at_equal_loglik() {
        // Two fits with equal log-likelihood: the lower-EDF one wins for all k > 0.
        let ll = -75.0;
        for &k in &[0.5, 2.0, (1000f64).ln()] {
            assert!(compute_gaic(ll, 3.0, k) < compute_gaic(ll, 6.0, k));
        }
    }

    #[test]
    fn compute_aic_formula() {
        assert!((compute_aic(-100.0, 5.0) - 210.0).abs() < 1e-12);
        assert!((compute_aic(0.0, 0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn compute_bic_formula() {
        let n = 100usize;
        let bic = compute_bic(-100.0, 5.0, n);
        let expected = 200.0 + (n as f64).ln() * 5.0;
        assert!((bic - expected).abs() < 1e-12);
    }

    #[test]
    fn bic_exceeds_aic_for_n_above_e_squared() {
        let ll = -50.0;
        let edf = 4.0;
        let aic = compute_aic(ll, edf);
        let bic = compute_bic(ll, edf, 100);
        assert!(bic > aic);
    }

    // --- total_edf ---

    fn dummy_fitted_param(edf: f64) -> FittedParameter {
        FittedParameter {
            coefficients: Coefficients(array![0.0]),
            covariance: CovarianceMatrix(Array2::<f64>::zeros((1, 1))),
            terms: vec![Term::Intercept],
            lambdas: array![],
            eta: array![0.0],
            fitted_values: array![0.0],
            edf,
            term_edf: vec![edf],
            term_blocks: vec![("(intercept)".to_string(), 0, 1)],
        }
    }

    #[test]
    fn total_edf_sums_per_parameter_edf() {
        let mut params = IndexMap::new();
        params.insert("mu".to_string(), dummy_fitted_param(3.5));
        params.insert("sigma".to_string(), dummy_fitted_param(1.2));
        assert!((total_edf(&params) - 4.7).abs() < 1e-12);
    }

    #[test]
    fn total_edf_empty_returns_zero() {
        let params: IndexMap<String, FittedParameter> = IndexMap::new();
        assert_eq!(total_edf(&params), 0.0);
    }
}
