#![recursion_limit = "1024"]
pub mod diagnostics;
pub mod distributions;
mod error;
pub mod fitting;
mod linalg;
mod math;
pub mod preprocessing;
mod splines;
mod terms;
mod types;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use diagnostics::ModelDiagnostics;
pub use error::GamlssError;
pub use fitting::{FitConfig, FitDiagnostics, ParamDiagnostic};
pub use terms::{Smooth, Term};
pub use types::*;

use distributions::Distribution;
use fitting::assembler::assemble_model_matrices;
use ndarray::Array1;
use preprocessing::validate_inputs;
use std::collections::HashMap;

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GamlssModel {
    pub models: HashMap<String, fitting::FittedParameter>,
    pub diagnostics: FitDiagnostics,
}

impl GamlssModel {
    pub fn fit<D: Distribution>(
        y: &Array1<f64>,
        data: &DataSet,
        formula: &Formula,
        family: &D,
    ) -> Result<Self, GamlssError> {
        Self::fit_with_config(y, data, formula, family, FitConfig::default())
    }

    pub fn fit_with_config<D: Distribution>(
        y: &Array1<f64>,
        data: &DataSet,
        formula: &Formula,
        family: &D,
        config: FitConfig,
    ) -> Result<Self, GamlssError> {
        validate_inputs(y, data, formula, family)?;

        let (fitted_models, diagnostics) = fitting::fit_gamlss(data, y, formula, family, &config)?;

        Ok(Self {
            models: fitted_models,
            diagnostics,
        })
    }

    pub fn converged(&self) -> bool {
        self.diagnostics.converged
    }

    /// Includes the distribution name so it can be deserialized without knowing the type upfront.
    #[cfg(feature = "serde")]
    pub fn to_json<D: Distribution + ?Sized>(&self, family: &D) -> Result<String, GamlssError> {
        let wrapper = SerializedModel {
            distribution: family.name().to_string(),
            model: self,
        };
        serde_json::to_string(&wrapper).map_err(|e| GamlssError::Input(e.to_string()))
    }

    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<(Self, String), GamlssError> {
        let wrapper: OwnedSerializedModel =
            serde_json::from_str(json).map_err(|e| GamlssError::Input(e.to_string()))?;
        Ok((wrapper.model, wrapper.distribution))
    }

    /// Predict fitted values for new data.
    ///
    /// Returns a HashMap with parameter names as keys and fitted values (on response scale)
    /// as values. The distribution is needed to obtain the appropriate link functions.
    pub fn predict<D: Distribution + ?Sized>(
        &self,
        new_data: &DataSet,
        family: &D,
    ) -> Result<HashMap<String, Array1<f64>>, GamlssError> {
        let n_obs = new_data.n_obs().unwrap_or(0);
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
    pub fn predict_with_se<D: Distribution + ?Sized>(
        &self,
        new_data: &DataSet,
        family: &D,
    ) -> Result<HashMap<String, PredictionResult>, GamlssError> {
        let n_obs = new_data.n_obs().unwrap_or(0);
        let mut results = HashMap::new();

        for (param_name, fitted_param) in &self.models {
            let (x_matrix, _, _) = assemble_model_matrices(new_data, n_obs, &fitted_param.terms)?;
            let eta = x_matrix.0.dot(&fitted_param.coefficients.0);

            let v = &fitted_param.covariance.0;
            let se_eta: Array1<f64> = x_matrix
                .0
                .rows()
                .into_iter()
                .map(|x_i| {
                    let v_x_i = v.dot(&x_i);
                    x_i.dot(&v_x_i).max(0.0).sqrt()
                })
                .collect();

            let link = family.default_link(param_name)?;
            let fitted = eta.view().mapv(|e| link.inv_link(e));

            results.insert(
                param_name.clone(),
                PredictionResult {
                    fitted,
                    eta,
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
    pub fn predict_samples<D: Distribution + ?Sized>(
        &self,
        new_data: &DataSet,
        family: &D,
        n_samples: usize,
    ) -> Result<HashMap<String, Vec<Array1<f64>>>, GamlssError> {
        let n_obs = new_data.n_obs().unwrap_or(0);
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

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PredictionResult {
    pub fitted: Array1<f64>,
    pub eta: Array1<f64>,
    pub se_eta: Array1<f64>,
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize)]
struct SerializedModel<'a> {
    distribution: String,
    model: &'a GamlssModel,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct OwnedSerializedModel {
    distribution: String,
    model: GamlssModel,
}
