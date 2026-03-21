//! WebAssembly bindings via wasm-bindgen for browser-based GAMLSS fitting and prediction.
//!
//! Provides [`WasmGamlssModel`] with JSON-based I/O for use from JavaScript.
//! Feature-gated behind the `wasm` feature flag.

use std::collections::HashMap;
use wasm_bindgen::prelude::*;

use crate::distributions::{
    Beta, Distribution, Gamma, Gaussian, NegativeBinomial, Poisson, StudentT,
};
use crate::error::GamlssError;
use crate::fitting::FitConfig;
use crate::types::{DataSet, Formula};
use crate::GamlssModel;
use ndarray::Array1;

fn to_js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Binomial is excluded because it requires state (n_trials) that
/// cannot be recovered from the distribution name alone.
fn get_distribution(name: &str) -> Result<Box<dyn Distribution>, GamlssError> {
    match name {
        "Gaussian" => Ok(Box::new(Gaussian)),
        "Poisson" => Ok(Box::new(Poisson)),
        "StudentT" => Ok(Box::new(StudentT)),
        "Gamma" => Ok(Box::new(Gamma)),
        "NegativeBinomial" => Ok(Box::new(NegativeBinomial)),
        "Beta" => Ok(Box::new(Beta)),
        _ => Err(GamlssError::Input(format!(
            "Unknown distribution: '{}'. Supported: Gaussian, Poisson, StudentT, Gamma, NegativeBinomial, Beta",
            name
        ))),
    }
}

/// Expects a JSON array of numbers, e.g. `[1.0, 2.0, 3.0]`.
fn parse_y_json(json: &str) -> Result<Array1<f64>, JsError> {
    let values: Vec<f64> = serde_json::from_str(json).map_err(to_js_err)?;
    Ok(Array1::from_vec(values))
}

/// Expects `{"col_name": [1.0, 2.0, ...], ...}`.
fn parse_data_json(json: &str) -> Result<DataSet, JsError> {
    let raw: HashMap<String, Vec<f64>> = serde_json::from_str(json).map_err(to_js_err)?;
    Ok(DataSet::from_vecs(raw))
}

/// Expects `{"mu": [{"Intercept": null}, ...], "sigma": [...]}`.
fn parse_formula_json(json: &str) -> Result<Formula, JsError> {
    serde_json::from_str(json).map_err(to_js_err)
}

/// WASM wrapper for GAMLSS models.
///
/// Supports both fitting models in the browser and loading pre-fitted models
/// serialized via `GamlssModel::to_json()`.
#[wasm_bindgen]
pub struct WasmGamlssModel {
    model: GamlssModel,
    distribution_name: String,
}

#[wasm_bindgen]
impl WasmGamlssModel {
    /// Fit a GAMLSS model in the browser.
    ///
    /// - `y_json`: Response variable as a JSON array, e.g. `[1.0, 2.0, 3.0]`
    /// - `data_json`: Predictor data as JSON object, e.g. `{"x": [1.0, 2.0], "z": [3.0, 4.0]}`
    /// - `formula_json`: Formula mapping parameter names to terms, e.g.
    ///   `{"mu": [{"Intercept": null}, {"Linear": {"col_name": "x"}}]}`
    /// - `distribution`: Distribution name (Gaussian, Poisson, StudentT, Gamma, NegativeBinomial, Beta)
    pub fn fit(
        y_json: &str,
        data_json: &str,
        formula_json: &str,
        distribution: &str,
    ) -> Result<WasmGamlssModel, JsError> {
        let y = parse_y_json(y_json)?;
        let data = parse_data_json(data_json)?;
        let formula = parse_formula_json(formula_json)?;
        let family = get_distribution(distribution).map_err(to_js_err)?;

        let model = GamlssModel::fit(&data, &y, &formula, family.as_ref()).map_err(to_js_err)?;

        Ok(WasmGamlssModel {
            model,
            distribution_name: distribution.to_string(),
        })
    }

    /// Fit a GAMLSS model with custom configuration.
    ///
    /// `config_json` is a JSON object with optional fields:
    /// `{"max_iterations": 200, "tolerance": 0.001}`
    #[wasm_bindgen(js_name = "fitWithConfig")]
    pub fn fit_with_config(
        y_json: &str,
        data_json: &str,
        formula_json: &str,
        distribution: &str,
        config_json: &str,
    ) -> Result<WasmGamlssModel, JsError> {
        let y = parse_y_json(y_json)?;
        let data = parse_data_json(data_json)?;
        let formula = parse_formula_json(formula_json)?;
        let family = get_distribution(distribution).map_err(to_js_err)?;
        let config: FitConfig = serde_json::from_str(config_json).map_err(to_js_err)?;

        let model = GamlssModel::fit_with_config(&data, &y, &formula, family.as_ref(), config)
            .map_err(to_js_err)?;

        Ok(WasmGamlssModel {
            model,
            distribution_name: distribution.to_string(),
        })
    }

    #[wasm_bindgen(js_name = "fromJson")]
    pub fn from_json(json: &str) -> Result<WasmGamlssModel, JsError> {
        let (model, distribution_name) = GamlssModel::from_json(json).map_err(to_js_err)?;
        Ok(WasmGamlssModel {
            model,
            distribution_name,
        })
    }

    #[wasm_bindgen(js_name = "toJson")]
    pub fn to_json(&self) -> Result<String, JsError> {
        let family = get_distribution(&self.distribution_name).map_err(to_js_err)?;
        self.model.to_json(family.as_ref()).map_err(to_js_err)
    }

    /// Input/output are JSON: `{"col": [values]}` → `{"param": [predictions]}`.
    pub fn predict(&self, data_json: &str) -> Result<String, JsError> {
        let family = get_distribution(&self.distribution_name).map_err(to_js_err)?;
        let new_data = parse_data_json(data_json)?;
        let predictions = self
            .model
            .predict(&new_data, family.as_ref())
            .map_err(to_js_err)?;

        let result: HashMap<String, Vec<f64>> = predictions
            .into_iter()
            .map(|(k, v)| (k, v.to_vec()))
            .collect();
        serde_json::to_string(&result).map_err(to_js_err)
    }

    #[wasm_bindgen(js_name = "predictWithSe")]
    pub fn predict_with_se(&self, data_json: &str) -> Result<String, JsError> {
        let family = get_distribution(&self.distribution_name).map_err(to_js_err)?;
        let new_data = parse_data_json(data_json)?;
        let results = self
            .model
            .predict_with_se(&new_data, family.as_ref())
            .map_err(to_js_err)?;

        let output: HashMap<String, PredictionWithSe> = results
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    PredictionWithSe {
                        fitted: v.fitted.to_vec(),
                        eta: v.eta.to_vec(),
                        se_eta: v.se_eta.to_vec(),
                    },
                )
            })
            .collect();
        serde_json::to_string(&output).map_err(to_js_err)
    }

    pub fn converged(&self) -> bool {
        self.model.converged()
    }

    #[wasm_bindgen(js_name = "fittedValues")]
    pub fn fitted_values(&self, param: &str) -> Result<Vec<f64>, JsError> {
        let fitted_param = self.model.models.get(param).ok_or_else(|| {
            JsError::new(&format!(
                "Parameter '{}' not found. Available: {:?}",
                param,
                self.model.models.keys().collect::<Vec<_>>()
            ))
        })?;
        Ok(fitted_param.fitted_values.to_vec())
    }

    pub fn coefficients(&self, param: &str) -> Result<Vec<f64>, JsError> {
        let fitted_param = self.model.models.get(param).ok_or_else(|| {
            JsError::new(&format!(
                "Parameter '{}' not found. Available: {:?}",
                param,
                self.model.models.keys().collect::<Vec<_>>()
            ))
        })?;
        Ok(fitted_param.coefficients.to_vec())
    }

    /// Generate prediction samples by sampling from the posterior distribution of coefficients.
    ///
    /// Returns JSON: `{"mu": [[s1_v1, s1_v2, ...], [s2_v1, ...], ...], "sigma": [...]}`
    /// where each inner array is one sample's predictions across all observations.
    #[wasm_bindgen(js_name = "predictSamples")]
    pub fn predict_samples(&self, data_json: &str, n_samples: usize) -> Result<String, JsError> {
        let family = get_distribution(&self.distribution_name).map_err(to_js_err)?;
        let new_data = parse_data_json(data_json)?;
        let results = self
            .model
            .predict_samples(&new_data, family.as_ref(), n_samples)
            .map_err(to_js_err)?;

        let output: HashMap<String, Vec<Vec<f64>>> = results
            .into_iter()
            .map(|(k, samples)| (k, samples.into_iter().map(|s| s.to_vec()).collect()))
            .collect();
        serde_json::to_string(&output).map_err(to_js_err)
    }

    #[wasm_bindgen(js_name = "diagnosticsJson")]
    pub fn diagnostics_json(&self) -> Result<String, JsError> {
        serde_json::to_string(&self.model.diagnostics).map_err(to_js_err)
    }
}

#[derive(serde::Serialize)]
struct PredictionWithSe {
    fitted: Vec<f64>,
    eta: Vec<f64>,
    se_eta: Vec<f64>,
}
