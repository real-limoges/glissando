#![recursion_limit = "1024"]
//! Generalized Additive Models for Location, Scale, and Shape (GAMLSS) in Rust.
//!
//! GAMLSS extends traditional regression by modeling multiple distribution parameters
//! (mean, variance, shape) as functions of predictors using the Rigby-Stasinopoulos
//! algorithm with penalized B-splines for nonlinear effects.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use gamlss_rs::{GamlssModel, DataSet, Formula, Term};
//! use gamlss_rs::distributions::Gaussian;
//! use ndarray::Array1;
//!
//! let y = Array1::from_vec(vec![2.1, 4.0, 5.9, 8.1, 10.0]);
//! let mut data = DataSet::new();
//! data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
//!
//! let formula = Formula::new()
//!     .with_terms("mu", vec![Term::Intercept, Term::Linear { col_name: "x".to_string() }])
//!     .with_terms("sigma", vec![Term::Intercept]);
//!
//! let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
//! ```

pub mod diagnostics;
pub mod distributions;
mod error;
pub mod fitting;
mod linalg;
mod math;
pub mod preprocessing;
#[cfg(feature = "python")]
mod python;
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

/// A fitted GAMLSS model containing per-parameter results and convergence diagnostics.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GamlssModel {
    /// Fitted results keyed by parameter name (e.g., "mu", "sigma").
    pub models: HashMap<String, fitting::FittedParameter>,
    /// Convergence diagnostics from the RS algorithm.
    pub diagnostics: FitDiagnostics,
}

impl GamlssModel {
    /// Fits a GAMLSS model with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `GamlssError` if inputs are invalid, the algorithm fails to converge,
    /// or a linear algebra operation fails.
    pub fn fit<D: Distribution + ?Sized>(
        data: &DataSet,
        y: &Array1<f64>,
        formula: &Formula,
        family: &D,
    ) -> Result<Self, GamlssError> {
        Self::fit_with_config(data, y, formula, family, FitConfig::default())
    }

    /// Fits a GAMLSS model with custom iteration limits and tolerance.
    ///
    /// # Errors
    ///
    /// Returns `GamlssError` if inputs are invalid, the algorithm fails to converge,
    /// or a linear algebra operation fails.
    pub fn fit_with_config<D: Distribution + ?Sized>(
        data: &DataSet,
        y: &Array1<f64>,
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

    /// Serializes the model to JSON, including the distribution name for later deserialization.
    ///
    /// # Errors
    ///
    /// Returns `GamlssError::Input` if serialization fails.
    #[cfg(feature = "serde")]
    pub fn to_json<D: Distribution + ?Sized>(&self, family: &D) -> Result<String, GamlssError> {
        let wrapper = SerializedModel {
            distribution: family.name().to_string(),
            model: self,
        };
        serde_json::to_string(&wrapper).map_err(|e| GamlssError::Input(e.to_string()))
    }

    /// Deserializes a model from JSON, returning the model and distribution name.
    ///
    /// # Errors
    ///
    /// Returns `GamlssError::Input` if deserialization fails.
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

/// Prediction output containing fitted values, linear predictor, and standard errors.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PredictionResult {
    /// Fitted values on the response scale (link⁻¹(eta)).
    pub fitted: Array1<f64>,
    /// Linear predictor values (X * beta).
    pub eta: Array1<f64>,
    /// Standard errors on the linear predictor scale.
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
