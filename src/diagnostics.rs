//! Aggregate diagnostics for a fitted [`GamlssModel`](crate::GamlssModel): residuals,
//! log-likelihood, EDF, AIC, BIC.
//!
//! Per-family math (log-density, marginal variance, expected value) lives on the
//! [`Distribution`] trait. This module composes those primitives — it never
//! hand-dispatches on the family name.

use crate::distributions::Distribution;
use crate::fitting::FittedParameter;
use crate::GamlssError;
use ndarray::Array1;
use std::collections::HashMap;

const MIN_POSITIVE: f64 = 1e-10;

/// Aggregated model diagnostics including residuals, EDF, and information criteria.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelDiagnostics {
    /// Pearson residuals: `(y − E[Y]) / √Var(Y)`.
    pub pearson_residuals: Array1<f64>,
    /// Raw response residuals: `y − E[Y]`.
    pub response_residuals: Array1<f64>,
    /// Total effective degrees of freedom across all parameters.
    pub total_edf: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Model log-likelihood evaluated at fitted parameters.
    pub log_likelihood: f64,
    /// Number of observations.
    pub n_obs: usize,
}

/// Computes Pearson residuals via the family's marginal moments:
/// `r_i = (y_i − E[Y_i]) / √Var(Y_i)`.
///
/// Variance is floored at [`MIN_POSITIVE`] before the square-root to keep
/// residuals finite when the fitted variance is degenerate.
pub fn pearson_residuals<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    params: &HashMap<&str, &Array1<f64>>,
) -> Result<Array1<f64>, GamlssError> {
    let e = family.expected_value(params)?;
    let v = family.variance(params)?;
    let sd = v.mapv(|vi| vi.max(MIN_POSITIVE).sqrt());
    Ok((y - &e) / &sd)
}

/// Computes Akaike Information Criterion: `−2·loglik + 2·EDF`.
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
    -2.0 * log_likelihood + 2.0 * total_edf
}

/// Computes Bayesian Information Criterion: `−2·loglik + log(n)·EDF`.
pub fn compute_bic(log_likelihood: f64, total_edf: f64, n_obs: usize) -> f64 {
    -2.0 * log_likelihood + (n_obs as f64).ln() * total_edf
}

/// Sums effective degrees of freedom across all fitted parameters.
pub fn total_edf(fitted_params: &HashMap<String, FittedParameter>) -> f64 {
    fitted_params.values().map(|p| p.edf).sum()
}

/// Computes raw response residuals: `y − E[Y]`.
pub fn response_residuals(y: &Array1<f64>, expected: &Array1<f64>) -> Array1<f64> {
    y - expected
}

/// Snapshot of fitted parameters on the response scale, in the shape the
/// [`Distribution`] trait expects.
fn fitted_params_view(models: &HashMap<String, FittedParameter>) -> HashMap<&str, &Array1<f64>> {
    models
        .iter()
        .map(|(k, v)| (k.as_str(), &v.fitted_values))
        .collect()
}

pub(crate) fn compute<D: Distribution + ?Sized>(
    models: &HashMap<String, FittedParameter>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::Gaussian;
    use crate::terms::Term;
    use crate::types::{Coefficients, CovarianceMatrix};
    use ndarray::{array, Array2};

    // --- compute_aic / compute_bic ---

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

    // --- response_residuals ---

    #[test]
    fn response_residuals_subtracts() {
        let y = array![3.0, 5.0, 7.0];
        let e = array![1.0, 2.0, 4.0];
        let r = response_residuals(&y, &e);
        assert_eq!(r, array![2.0, 3.0, 3.0]);
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
        }
    }

    #[test]
    fn total_edf_sums_per_parameter_edf() {
        let mut params = HashMap::new();
        params.insert("mu".to_string(), dummy_fitted_param(3.5));
        params.insert("sigma".to_string(), dummy_fitted_param(1.2));
        assert!((total_edf(&params) - 4.7).abs() < 1e-12);
    }

    #[test]
    fn total_edf_empty_returns_zero() {
        let params: HashMap<String, FittedParameter> = HashMap::new();
        assert_eq!(total_edf(&params), 0.0);
    }

    // --- pearson_residuals composes E[Y] and Var(Y) ---

    #[test]
    fn pearson_residuals_gaussian_via_trait() {
        // For Gaussian, E[Y]=mu, Var(Y)=sigma^2, so Pearson = (y-mu)/sigma.
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.0, 2.5];
        let sigma = array![0.5, 0.5, 0.5];
        let params: HashMap<&str, &Array1<f64>> = HashMap::from([("mu", &mu), ("sigma", &sigma)]);
        let r = pearson_residuals(&Gaussian, &y, &params).unwrap();
        assert!((r[0] - (-1.0)).abs() < 1e-10);
        assert!((r[1] - 0.0).abs() < 1e-10);
        assert!((r[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_residuals_handles_zero_variance_gracefully() {
        // sigma -> 0 would divide by zero; MIN_POSITIVE clamp must keep residuals finite.
        let y = array![0.0];
        let mu = array![0.0];
        let sigma = array![0.0];
        let params: HashMap<&str, &Array1<f64>> = HashMap::from([("mu", &mu), ("sigma", &sigma)]);
        let r = pearson_residuals(&Gaussian, &y, &params).unwrap();
        assert!(r.iter().all(|v| v.is_finite()));
    }
}
