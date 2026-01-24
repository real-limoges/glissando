#![recursion_limit = "1024"]
pub mod diagnostics;
pub mod distributions;
mod error;
pub mod fitting;
mod math;
pub mod preprocessing;
mod splines;
mod terms;
mod types;

pub use diagnostics::ModelDiagnostics;
pub use error::GamlssError;
pub use fitting::{FitConfig, FitDiagnostics};
pub use terms::{Smooth, Term};
pub use types::*;

use distributions::Distribution;
use fitting::assembler::assemble_model_matrices;
use ndarray::Array1;
use polars::prelude::DataFrame;
use preprocessing::validate_inputs;
use std::collections::HashMap;

#[derive(Debug)]
pub struct GamlssModel {
    pub models: HashMap<String, fitting::FittedParameter>,
    pub diagnostics: FitDiagnostics,
}

impl GamlssModel {
    pub fn fit<D: Distribution>(
        data: &DataFrame,
        y_name: &str,
        formula: &HashMap<String, Vec<Term>>,
        family: &D,
    ) -> Result<Self, GamlssError> {
        Self::fit_with_config(data, y_name, formula, family, FitConfig::default())
    }

    pub fn fit_with_config<D: Distribution>(
        data: &DataFrame,
        y_name: &str,
        formula: &HashMap<String, Vec<Term>>,
        family: &D,
        config: FitConfig,
    ) -> Result<Self, GamlssError> {
        validate_inputs(data, y_name, formula, family)?;

        let y_series = data.column(y_name).map_err(|e| {
            GamlssError::Input(format!("Target Column '{}' not found: {}", y_name, e))
        })?;

        let binding = y_series.f64()?.to_ndarray()?;
        let y_vec = binding
            .to_shape(y_series.len())
            .map_err(|e| GamlssError::Shape(e.to_string()))?;

        let y_vector = y_vec.to_owned();

        let (fitted_models, diagnostics) =
            fitting::fit_gamlss(data, &y_vector, formula, family, &config)?;

        Ok(Self {
            models: fitted_models,
            diagnostics,
        })
    }

    pub fn converged(&self) -> bool {
        self.diagnostics.converged
    }

    /// Predict fitted values for new data.
    ///
    /// Returns a HashMap with parameter names as keys and fitted values (on response scale)
    /// as values. The distribution is needed to obtain the appropriate link functions.
    pub fn predict<D: Distribution>(
        &self,
        new_data: &DataFrame,
        family: &D,
    ) -> Result<HashMap<String, Array1<f64>>, GamlssError> {
        let n_obs = new_data.height();
        let mut predictions = HashMap::new();

        for (param_name, fitted_param) in &self.models {
            let (x_matrix, _, _) = assemble_model_matrices(new_data, n_obs, &fitted_param.terms)?;
            let eta = x_matrix.0.dot(&fitted_param.coefficients.0);
            let link = family.default_link(param_name)?;
            let fitted = eta.mapv(|e| link.inv_link(e));

            predictions.insert(param_name.clone(), fitted);
        }

        Ok(predictions)
    }

    /// Predict fitted values with standard errors for new data.
    ///
    /// Returns predictions on the linear predictor (eta) scale along with standard errors.
    /// Standard errors are computed via: se = sqrt(diag(X * V * X'))
    /// where V is the covariance matrix of the coefficients.
    pub fn predict_with_se<D: Distribution>(
        &self,
        new_data: &DataFrame,
        family: &D,
    ) -> Result<HashMap<String, PredictionResult>, GamlssError> {
        let n_obs = new_data.height();
        let mut results = HashMap::new();

        for (param_name, fitted_param) in &self.models {
            let (x_matrix, _, _) = assemble_model_matrices(new_data, n_obs, &fitted_param.terms)?;
            let eta = x_matrix.0.dot(&fitted_param.coefficients.0);

            let v = &fitted_param.covariance.0;
            let mut se_eta = Array1::zeros(n_obs);
            for i in 0..n_obs {
                let x_i = x_matrix.0.row(i);
                let v_x_i = v.dot(&x_i);
                let var_eta_i = x_i.dot(&v_x_i);
                se_eta[i] = var_eta_i.max(0.0).sqrt();
            }

            let link = family.default_link(param_name)?;
            let fitted = eta.mapv(|e| link.inv_link(e));

            results.insert(
                param_name.clone(),
                PredictionResult {
                    fitted,
                    eta: eta.clone(),
                    se_eta,
                },
            );
        }

        Ok(results)
    }

    /// Sample from the posterior distribution of coefficients for a given parameter.
    ///
    /// Uses Cholesky decomposition of the covariance matrix to generate samples
    /// from the approximate posterior N(beta_hat, V_beta).
    pub fn posterior_samples(&self, param_name: &str, n_samples: usize) -> Vec<Coefficients> {
        let fitted_param = match self.models.get(param_name) {
            Some(p) => p,
            None => return vec![],
        };

        fitting::sample_posterior(
            &fitted_param.coefficients,
            &fitted_param.covariance,
            n_samples,
        )
        .into_iter()
        .map(Coefficients)
        .collect()
    }

    /// Generate prediction samples by sampling from posterior and propagating through predictions.
    ///
    /// For each posterior sample of coefficients, computes predictions on new data.
    /// Returns samples of fitted values on the response scale.
    pub fn predict_samples<D: Distribution>(
        &self,
        new_data: &DataFrame,
        family: &D,
        n_samples: usize,
    ) -> Result<HashMap<String, Vec<Array1<f64>>>, GamlssError> {
        let n_obs = new_data.height();
        let mut results = HashMap::new();

        for (param_name, fitted_param) in &self.models {
            let (x_matrix, _, _) = assemble_model_matrices(new_data, n_obs, &fitted_param.terms)?;

            let beta_samples = fitting::sample_posterior(
                &fitted_param.coefficients,
                &fitted_param.covariance,
                n_samples,
            );

            let link = family.default_link(param_name)?;

            let prediction_samples: Vec<Array1<f64>> = beta_samples
                .iter()
                .map(|beta| {
                    let eta = x_matrix.0.dot(beta);
                    eta.mapv(|e| link.inv_link(e))
                })
                .collect();

            results.insert(param_name.clone(), prediction_samples);
        }

        Ok(results)
    }
}

/// Result of prediction with standard errors
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Fitted values on the response scale
    pub fitted: Array1<f64>,
    /// Linear predictor values
    pub eta: Array1<f64>,
    /// Standard errors on the linear predictor scale
    pub se_eta: Array1<f64>,
}
