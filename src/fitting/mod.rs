pub mod assembler;
pub mod inference;
mod solver;

use self::assembler::assemble_model_matrices;
pub use self::inference::sample_posterior;
use self::solver::{fit_pwls, run_optimization};

use super::distributions::{Distribution, Link};
use super::error::GamlssError;
use super::terms::{Smooth, Term};
use super::types::*;
use ndarray::Array1;
use polars::prelude::DataFrame;
use std::collections::HashMap;

const DEFAULT_MAX_ITER: usize = 200;
const DEFAULT_TOLERANCE: f64 = 1e-3;

/// Minimum weight for IRLS to prevent division by near-zero
const MIN_WEIGHT: f64 = 1e-6;

/// Configuration options for the GAMLSS fitting algorithm.
#[derive(Debug, Clone)]
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

struct FittingParameter {
    terms: Vec<Term>,
    link: Box<dyn Link>,
    x_matrix: ModelMatrix,
    penalty_matrices: Vec<PenaltyMatrix>,
    beta: Coefficients,
    eta: Array1<f64>,
    lambdas: Array1<f64>,
    covariance: Option<CovarianceMatrix>,
    edf: f64,
}

/// Fits a GAMLSS model using the RS (Rigby-Stasinopoulos) algorithm.
///
/// The RS algorithm cycles through distribution parameters (μ, σ, ν, ...) fitting each
/// as a penalized additive model while holding others fixed. Each parameter update uses
/// penalized iteratively reweighted least squares (P-IRLS) with a working response z
/// and working weights w derived from the distribution's score and Fisher information.
pub(crate) fn fit_gamlss<D: Distribution>(
    data: &DataFrame,
    y: &Array1<f64>,
    formula: &HashMap<String, Vec<Term>>,
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

    let param_names: Vec<String> = models.keys().cloned().collect();

    for cycle in 0..config.max_iterations {
        param_diagnostics.clear();
        let mut max_diff = 0.0;

        for param_name in family.parameters() {
            let param_key = param_name.to_string();

            // Build current parameter values on response scale (batched)
            let current_params: HashMap<&str, Array1<f64>> = param_names
                .iter()
                .map(|name| {
                    let model = &models[name];
                    let fitted_values = model.eta.mapv(|e| model.link.inv_link(e));
                    (name.as_str(), fitted_values)
                })
                .collect();

            // Create reference map for batched derivatives call
            let params_ref: HashMap<&str, &Array1<f64>> =
                current_params.iter().map(|(k, v)| (*k, v)).collect();

            // Single batched call replaces n_obs individual calls
            let all_derivs = family.derivatives(y, &params_ref)?;
            let (deriv_u, deriv_w) = all_derivs.get(*param_name).ok_or_else(|| {
                GamlssError::Input(format!("No derivation for {} found", param_name))
            })?;

            let model = models.get_mut(&param_key).ok_or_else(|| {
                GamlssError::Internal(format!("Model for parameter '{}' not found", param_key))
            })?;

            // Fisher scoring: z = η + u/w forms working response for weighted least squares
            let safe_w = deriv_w.mapv(|w: f64| w.max(MIN_WEIGHT));
            let adjustment = deriv_u / &safe_w;
            let safe_adjustment = adjustment.mapv(|v| v.clamp(-20.0, 20.0));

            let z = &model.eta + &safe_adjustment;
            let w = safe_w;

            // Optimize lambdas (warm-start from previous values for faster convergence)
            // Skip optimization if no penalty matrices (purely parametric model)
            let best_lambdas = if model.penalty_matrices.is_empty() {
                Array1::zeros(0)
            } else {
                run_optimization::<D>(
                    &model.x_matrix,
                    &z,
                    &w,
                    &model.penalty_matrices,
                    Some(&model.lambdas),
                )?
            };

            let (new_beta, cov_matrix, edf) = fit_pwls(
                &model.x_matrix,
                &z,
                &w,
                &model.penalty_matrices,
                &best_lambdas,
            )?;

            // Use max absolute change for convergence
            let diff = (&new_beta.0 - &model.beta.0)
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f64, |a, b| a.max(b));
            if diff > max_diff {
                max_diff = diff;
            }

            let new_eta = model.x_matrix.dot(&new_beta.0);
            let eta_change = (&new_eta - &model.eta).mapv(f64::abs).sum();
            let lambda_change = (&best_lambdas - &model.lambdas).mapv(f64::abs).sum();

            model.beta = new_beta;
            model.eta = new_eta;
            model.lambdas = best_lambdas;
            model.covariance = Some(cov_matrix);
            model.edf = edf;

            param_diagnostics.insert(
                param_key.clone(),
                ParamDiagnostic {
                    final_eta_change: eta_change,
                    final_lambda_change: lambda_change,
                    edf,
                },
            );
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
