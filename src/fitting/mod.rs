mod assembler;
mod solver;
mod inference;

use self::assembler::assemble_model_matrices;
use self::solver::{run_optimization, fit_pwls};
pub use self::inference::sample_posterior;

use super::error::GamlssError;
use super::distributions::{Distribution, Link};
use super::terms::{Term, Smooth};
use super::types::*;
use ndarray::{Array1, Array2};
use polars::prelude::DataFrame;
use std::collections::HashMap;

const MAX_GAMLSS_ITER: usize = 20;

#[derive(Debug)]
pub struct FittedParameter {
    pub coefficients: Coefficients,
    pub covariance: CovarianceMatrix,
    pub terms: Vec<Term>,
    pub lambdas: Array1<f64>,
    pub eta: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub edf: f64,
}

pub struct FittingParameter {
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

pub(crate) fn fit_gamlss<D: Distribution>(
    data: &DataFrame,
    y: &Array1<f64>,
    formula: &HashMap<String, Vec<Term>>,
    family: &D
) -> Result<HashMap<String, FittedParameter>, GamlssError> {
    let n_obs = y.len();
    let mut models: HashMap<String, FittingParameter> = HashMap::new();

    // =========================================================
    // 1. INITIALIZATION PHASE
    // =========================================================
    for param_name in family.parameters() {
        let param_name_str = param_name.to_string();
        let terms = formula.get(&param_name_str)
            .ok_or_else(|| GamlssError::Input(format!("Formula missing for parameter {}", param_name)))?;
        let link = family.default_link(param_name);

        let (x_model, penalty_matrices, total_coeffs) = assemble_model_matrices(data, n_obs, terms)?;

        // --- SMART INITIALIZATION (Fixing the Bugs) ---
        // 1. Determine Start Value on RESPONSE Scale (Physical units)
        let response_scale_start = if param_name_str == "mu" {
            y.mean().unwrap_or(0.0)
        } else if param_name_str == "sigma" {
            // Initialize sigma to the standard deviation of the data
            let s = y.std(1.0);
            if s < 1e-4 { 1.0 } else { s }
        } else if param_name_str == "nu" {
            10.0 // Start with high degrees of freedom (Gaussian-like)
        } else {
            0.1
        };

        // 2. Convert to LINEAR PREDICTOR Scale (Eta)
        // This prevents the "Double Link" bug.
        let eta_start = link.link(response_scale_start);

        // 3. Initialize Beta
        // If the first term is an intercept, set it to eta_start.
        let mut beta = Coefficients(Array1::zeros(total_coeffs));
        if total_coeffs > 0 {
            beta.0[0] = eta_start;
        }

        // 4. Initialize Eta
        let eta = Array1::from_elem(n_obs, eta_start);
        let lambdas = Array1::<f64>::ones(penalty_matrices.len());

        models.insert(param_name_str, FittingParameter {
            terms: terms.clone(),
            link,
            x_matrix: x_model,
            penalty_matrices,
            beta,
            eta,
            lambdas,
            covariance: None,
            edf: 0.0,
        });
    } // <--- Initialization Loop Ends Here

    // =========================================================
    // 2. GAMLSS CYCLE PHASE
    // =========================================================
    for _cycle in 0..MAX_GAMLSS_ITER {
        let mut max_diff = 0.0;

        for param_name in family.parameters() {
            let param_key = param_name.to_string();
            let mut current_params = HashMap::new();

            // Snapshot current state
            for (name, model) in &models {
                let fitted_values = model.eta.mapv(|e| model.link.inv_link(e));
                current_params.insert(name.clone(), fitted_values);
            }

            // Calculate derivatives
            let mut deriv_u = Array1::zeros(n_obs);
            let mut deriv_w = Array1::zeros(n_obs);

            for i in 0..n_obs {
                let mut obs_params = HashMap::new();
                for (name, value) in &current_params {
                    obs_params.insert(name.clone(), value[i]);
                }

                let all_derivs = family.derivatives(y[i], &obs_params);
                let (u, w) = all_derivs.get(*param_name)
                    .ok_or_else(|| GamlssError::Input(format!("No derivation for {} found", param_name)))?;
                deriv_u[i] = *u;
                deriv_w[i] = *w;
            }

            let model = models.get_mut(&param_key).unwrap();

            // FIX: Enforce minimum weight to prevent collapse/NaNs
            // If weight is too small, the solver sees "no data" and returns prior (0).
            let safe_w = deriv_w.mapv(|w| w.max(1e-6));

            // Calculate Working Response (z)
            // Clamp the adjustment to prevent explosion in early iterations
            let adjustment = &deriv_u / &safe_w;
            let safe_adjustment = adjustment.mapv(|v| v.max(-20.0).min(20.0));

            let z = &model.eta + &safe_adjustment;
            let w = safe_w;

            // Fit Sub-model
            let best_lambdas = run_optimization::<D>(
                &model.x_matrix,
                &z,
                &w,
                &model.penalty_matrices
            )?;

            let (new_beta, cov_matrix, edf) = fit_pwls(
                &model.x_matrix,
                &z,
                &w,
                &model.penalty_matrices,
                &best_lambdas
            )?;

            let diff = (&new_beta.0 - &model.beta.0).mapv(f64::abs).sum();
            if diff > max_diff {
                max_diff = diff;
            }

            model.beta = new_beta;
            model.eta = model.x_matrix.dot(&model.beta.0);
            model.lambdas = best_lambdas;
            model.covariance = Some(cov_matrix);
            model.edf = edf;
        }

        if max_diff < 1e-6 {
            break;
        }
    }

    // =========================================================
    // 3. FINALIZE
    // =========================================================
    let mut final_results = HashMap::new();

    for (name, model) in models {
        let fitted_values = model.eta.mapv(|e| model.link.inv_link(e));

        let fitted_param = FittedParameter {
            coefficients: model.beta,
            covariance: model.covariance.expect("Covariance matrix not computed"),
            terms: model.terms,
            lambdas: model.lambdas,
            eta: model.eta,
            fitted_values,
            edf: model.edf,
        };
        final_results.insert(name, fitted_param);
    }

    Ok(final_results)
}