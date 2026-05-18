//! GAMLSS fitting implementation via the Rigby-Stasinopoulos (RS) algorithm.
//!
//! The RS algorithm iteratively cycles through distribution parameters, fitting each as a
//! penalized additive model while holding others fixed. For each parameter:
//!
//! 1. Compute score (u) and Fisher information (w) from the distribution
//! 2. Form working response: z = η + u/w
//! 3. Optimize smoothing parameters (λ) via GCV using L-BFGS
//! 4. Solve penalized weighted least squares: (X'WX + Σλ·S)·β = X'W·z
//! 5. Update linear predictor: η = X·β
//!
//! The module also handles posterior inference (sampling from the approximate posterior of coefficients).

pub mod assembler;
pub mod diagnostics;
mod scoring;
mod solver;

use self::assembler::assemble_model_matrices;

use super::distributions::{Distribution, Link};
use super::error::GamlssError;
use super::terms::{Smooth, Term};
use super::types::*;
use crate::linalg;
use ndarray::{Array1, Array2};
use rand::{rng, Rng};
use rand_distr::{Distribution as _, StandardNormal};
use std::collections::HashMap;

const DEFAULT_MAX_ITER: usize = 200;
const DEFAULT_TOLERANCE: f64 = 1e-3;

/// Configuration options for the GAMLSS fitting algorithm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FitConfig {
    /// Maximum number of RS algorithm iterations (default: 200).
    pub max_iterations: usize,
    /// Convergence tolerance for coefficient changes (default: 1e-3).
    pub tolerance: f64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITER,
            tolerance: DEFAULT_TOLERANCE,
        }
    }
}

/// Diagnostic information from the model fitting process.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FitDiagnostics {
    /// Whether the algorithm converged within the maximum iterations.
    pub converged: bool,
    /// Number of RS algorithm iterations performed.
    pub iterations: usize,
    /// Maximum coefficient change in the final iteration.
    pub final_change: f64,
    /// Maximum gradient at convergence (if computed).
    pub max_gradient: Option<f64>,
    /// Per-parameter diagnostic information.
    pub param_diagnostics: HashMap<String, ParamDiagnostic>,
}

/// Diagnostic information for a single distribution parameter.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParamDiagnostic {
    /// Sum of absolute changes in linear predictor (eta) in final iteration.
    pub final_eta_change: f64,
    /// Sum of absolute changes in smoothing parameters (lambda) in final iteration.
    pub final_lambda_change: f64,
    /// Effective degrees of freedom for this parameter's model.
    pub edf: f64,
}

/// Fitted results for a single distribution parameter (e.g., mu, sigma).
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FittedParameter {
    /// Estimated regression coefficients (beta).
    pub coefficients: Coefficients,
    /// Covariance matrix of the coefficient estimates.
    pub covariance: CovarianceMatrix,
    /// Terms included in this parameter's model formula.
    pub terms: Vec<Term>,
    /// Optimized smoothing parameters for each penalty matrix.
    pub lambdas: Array1<f64>,
    /// Linear predictor values (X * beta).
    pub eta: Array1<f64>,
    /// Fitted values on the response scale (link^-1(eta)).
    pub fitted_values: Array1<f64>,
    /// Effective degrees of freedom.
    pub edf: f64,
}

pub(super) struct FittingParameter {
    pub(super) terms: Vec<Term>,
    pub(super) link: Box<dyn Link>,
    pub(super) x_matrix: ModelMatrix,
    pub(super) penalty_matrices: Vec<PenaltyMatrix>,
    pub(super) beta: Coefficients,
    pub(super) eta: Array1<f64>,
    pub(super) lambdas: Array1<f64>,
    pub(super) covariance: Option<CovarianceMatrix>,
    pub(super) edf: f64,
}

/// Fits a GAMLSS model using the RS (Rigby-Stasinopoulos) algorithm.
///
/// The RS algorithm cycles through distribution parameters (μ, σ, ν, ...) fitting each
/// as a penalized additive model while holding others fixed. Each parameter update uses
/// penalized iteratively reweighted least squares (P-IRLS) with a working response z
/// and working weights w derived from the distribution's score and Fisher information.
pub(crate) fn fit_gamlss<D: Distribution + ?Sized>(
    data: &DataSet,
    y: &Array1<f64>,
    formula: &Formula,
    family: &D,
    config: &FitConfig,
) -> Result<(HashMap<String, FittedParameter>, FitDiagnostics), GamlssError> {
    let n_obs = y.len();
    let mut models: HashMap<String, FittingParameter> = HashMap::new();

    for param_name in family.parameters() {
        let param_name_str = param_name.to_string();
        let terms = formula.get(&param_name_str).ok_or_else(|| {
            GamlssError::Input(format!("Formula missing for parameter {}", param_name))
        })?;
        let link = family.default_link(param_name)?;

        let (x_model, penalty_matrices, total_coeffs) =
            assemble_model_matrices(data, n_obs, terms)?;

        // Initialize on response scale using distribution-specific logic
        let response_scale_start = family.initial_value(param_name, y);
        let eta_start = link.link(response_scale_start);

        let mut beta = Coefficients(Array1::zeros(total_coeffs));
        if total_coeffs > 0 {
            beta.0[0] = eta_start;
        }

        let eta = Array1::from_elem(n_obs, eta_start);
        let lambdas = Array1::<f64>::ones(penalty_matrices.len());

        models.insert(
            param_name_str,
            FittingParameter {
                terms: terms.clone(),
                link,
                x_matrix: x_model,
                penalty_matrices,
                beta,
                eta,
                lambdas,
                covariance: None,
                edf: 0.0,
            },
        );
    }

    let mut converged = false;
    let mut final_iteration = 0;
    let mut final_change = f64::MAX;
    let mut param_diagnostics = HashMap::new();

    for cycle in 0..config.max_iterations {
        param_diagnostics.clear();
        let mut max_diff = 0.0;

        for param_name in family.parameters() {
            let update = scoring::step(family, y, &models, param_name)?;
            if update.max_diff > max_diff {
                max_diff = update.max_diff;
            }
            param_diagnostics.insert(
                param_name.to_string(),
                ParamDiagnostic {
                    final_eta_change: update.eta_change,
                    final_lambda_change: update.lambda_change,
                    edf: update.edf,
                },
            );

            let model = models.get_mut(*param_name).ok_or_else(|| {
                GamlssError::Internal(format!("Model for parameter '{}' not found", param_name))
            })?;
            model.beta = update.beta;
            model.eta = update.eta;
            model.lambdas = update.lambdas;
            model.covariance = Some(update.covariance);
            model.edf = update.edf;
        }

        final_iteration = cycle + 1;
        final_change = max_diff;

        if max_diff < config.tolerance {
            converged = true;
            break;
        }
    }

    let mut final_results = HashMap::new();

    for (name, model) in models {
        let fitted_values = model.eta.mapv(|e| model.link.inv_link(e));

        let covariance = model.covariance.ok_or_else(|| {
            GamlssError::Internal(format!(
                "Covariance matrix not computed for parameter '{}'",
                name
            ))
        })?;

        let fitted_param = FittedParameter {
            coefficients: model.beta,
            covariance,
            terms: model.terms,
            lambdas: model.lambdas,
            eta: model.eta,
            fitted_values,
            edf: model.edf,
        };
        final_results.insert(name, fitted_param);
    }

    let diagnostics = FitDiagnostics {
        converged,
        iterations: final_iteration,
        final_change,
        max_gradient: None,
        param_diagnostics,
    };

    Ok((final_results, diagnostics))
}

// ============================================================================
// Posterior sampling
// ============================================================================

/// Draws samples from the approximate posterior N(beta_hat, V_beta) via Cholesky decomposition.
///
/// Returns an empty vec if the covariance matrix is not positive definite.
pub fn sample_posterior(
    beta_hat: &Coefficients,
    v_beta: &CovarianceMatrix,
    n_samples: usize,
) -> Vec<Array1<f64>> {
    let Ok(l_factor) = linalg::cholesky_lower(&v_beta.0) else {
        return vec![];
    };

    let mut rng_rs = rng();
    sample_from_cholesky(&beta_hat.0, &l_factor, n_samples, &mut rng_rs)
}

pub(crate) fn sample_from_cholesky(
    mean: &Array1<f64>,
    l_factor: &Array2<f64>,
    n_samples: usize,
    rng: &mut impl Rng,
) -> Vec<Array1<f64>> {
    let dim = mean.len();

    (0..n_samples)
        .map(|_| {
            let z = Array1::<f64>::from_shape_fn(dim, |_| StandardNormal.sample(rng));
            mean + l_factor.dot(&z)
        })
        .collect()
}
