mod assembler;
mod solver;
mod inference;

use self::assembler::assemble_model_matrices;
use self::solver::{run_optimization, fit_pwls};
pub use self::inference::sample_posterior;

use super::error::GamlssError;
use super::families::{Distribution, Link};
use super::terms::{Term, Smooth};
use super::types::*;
use ndarray::{Array1, Array2};
use polars::prelude::DataFrame;
use std::collections::HashMap;


const MAX_GAMLSS_ITER: usize = 10;

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
    // mutable stuff
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

    for param_name in family.parameters() {
        let param_name_str = param_name.to_string();
        let terms = formula.get(&param_name_str)
            .ok_or_else(|| GamlssError::Input(format!("Formula missing for parameter {}", param_name)))?;
        let link = family.default_link(param_name);

        let (x_model, penalty_matrices, total_coeffs) = assemble_model_matrices(data, n_obs, terms)?;

        let eta = Array1::<f64>::from_elem(n_obs, 0.1);
        let beta = Coefficients(Array1::zeros(total_coeffs));

        let start_val = if models.is_empty() {
            y.mean().unwrap_or(0.1)
        } else {
            0.1
        };

        let eta = Array1::from_elem(n_obs, start_val);
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

        for _cycle in 0..MAX_GAMLSS_ITER {
            let mut max_diff = 0.0;

            for param_name in family.parameters() {
                let param_key = param_name.to_string();
                let mut current_params = HashMap::new();

                for (name, model) in &models {
                    let fitted_values = model.eta.mapv(|e| model.link.inv_link(e));
                    current_params.insert(name.clone(), fitted_values);
                }

                let mut deriv_u = Array1::zeros(n_obs);
                let mut deriv_w = Array1::zeros(n_obs);

                for i in 0..n_obs {
                    let mut obs_params = HashMap::new();
                    for (name, value) in &current_params {
                        obs_params.insert(name.clone(), value[i]);
                    }

                    let all_derivs = family.derivatives(y[i], &obs_params);
                    let (u, w) = all_derivs.get(*param_name)
                        .ok_or_else(|| GamlssError::Input(format!("No derivation for {} found in model", param_name)))?;
                    deriv_u[i] = *u;
                    deriv_w[i] = *w;
                }

                let model = models.get_mut(&param_key).unwrap();

                // I'm forcing the matrix to be positive definite here for the solver
                let safe_w = deriv_w.mapv(|w| w.max(1e-8));

                let z = &model.eta + (&deriv_u / &safe_w);
                let w = safe_w;

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
    }
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
