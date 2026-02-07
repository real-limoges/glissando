use std::collections::HashMap;
use wasm_bindgen::prelude::*;

use crate::distributions::{
    Beta, Distribution, Gamma, Gaussian, NegativeBinomial, Poisson, StudentT,
};
use crate::error::GamlssError;
use crate::types::DataSet;
use crate::GamlssModel;

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

/// Expects `{"col_name": [1.0, 2.0, ...], ...}`.
fn parse_data_json(json: &str) -> Result<DataSet, GamlssError> {
    let raw: HashMap<String, Vec<f64>> =
        serde_json::from_str(json).map_err(|e| GamlssError::Input(e.to_string()))?;
    Ok(DataSet::from_vecs(raw))
}

/// Models are fitted natively and serialized to JSON via `GamlssModel::to_json()`.
/// This wrapper deserializes them for browser-based prediction.
#[wasm_bindgen]
pub struct WasmGamlssModel {
    model: GamlssModel,
    distribution_name: String,
}

#[wasm_bindgen]
impl WasmGamlssModel {
    #[wasm_bindgen(js_name = "fromJson")]
    pub fn from_json(json: &str) -> Result<WasmGamlssModel, JsError> {
        let (model, distribution_name) =
            GamlssModel::from_json(json).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmGamlssModel {
            model,
            distribution_name,
        })
    }

    #[wasm_bindgen(js_name = "toJson")]
    pub fn to_json(&self) -> Result<String, JsError> {
        let family =
            get_distribution(&self.distribution_name).map_err(|e| JsError::new(&e.to_string()))?;
        self.model
            .to_json(family.as_ref())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Input/output are JSON: `{"col": [values]}` → `{"param": [predictions]}`.
    pub fn predict(&self, data_json: &str) -> Result<String, JsError> {
        let family =
            get_distribution(&self.distribution_name).map_err(|e| JsError::new(&e.to_string()))?;
        let new_data = parse_data_json(data_json)?;
        let predictions = self
            .model
            .predict(&new_data, family.as_ref())
            .map_err(|e| JsError::new(&e.to_string()))?;

        let result: HashMap<String, Vec<f64>> = predictions
            .into_iter()
            .map(|(k, v)| (k, v.to_vec()))
            .collect();
        serde_json::to_string(&result).map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(js_name = "predictWithSe")]
    pub fn predict_with_se(&self, data_json: &str) -> Result<String, JsError> {
        let family =
            get_distribution(&self.distribution_name).map_err(|e| JsError::new(&e.to_string()))?;
        let new_data = parse_data_json(data_json)?;
        let results = self
            .model
            .predict_with_se(&new_data, family.as_ref())
            .map_err(|e| JsError::new(&e.to_string()))?;

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
        serde_json::to_string(&output).map_err(|e| JsError::new(&e.to_string()))
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

    #[wasm_bindgen(js_name = "diagnosticsJson")]
    pub fn diagnostics_json(&self) -> Result<String, JsError> {
        serde_json::to_string(&self.model.diagnostics).map_err(|e| JsError::new(&e.to_string()))
    }
}

#[derive(serde::Serialize)]
struct PredictionWithSe {
    fitted: Vec<f64>,
    eta: Vec<f64>,
    se_eta: Vec<f64>,
}
