//! WebAssembly bindings via wasm-bindgen for browser-based GAMLSS fitting and prediction.
//!
//! Provides [`WasmGamlssModel`] with JSON-based I/O for use from JavaScript.
//! Feature-gated behind the `wasm` feature flag. This is a thin marshalling
//! shim over [`crate::json`]; the wire formats and distribution dispatch are
//! documented there.

use wasm_bindgen::prelude::*;

use crate::distributions::Distribution;
use crate::json;
use crate::GamlssModel;

fn to_js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// WASM wrapper for GAMLSS models.
///
/// Supports both fitting models in the browser and loading pre-fitted models
/// serialized via `GamlssModel::to_json()`.
#[wasm_bindgen]
pub struct WasmGamlssModel {
    model: GamlssModel,
    family: Box<dyn Distribution>,
}

#[wasm_bindgen]
impl WasmGamlssModel {
    /// Fit a GAMLSS model in the browser.
    ///
    /// - `y_json`: Response variable as a JSON array, e.g. `[1.0, 2.0, 3.0]`
    /// - `data_json`: Predictor data as JSON object, e.g. `{"x": [1.0, 2.0], "z": [3.0, 4.0]}`
    /// - `formula_json`: Formula mapping parameter names to terms, e.g.
    ///   `{"mu": [{"Intercept": null}, {"Linear": {"col_name": "x"}}]}`.
    ///   For a natural cubic regression spline (`bs="cr"`):
    ///   `{"mu": [{"Smooth": {"CrSpline1D": {"col_name": "x", "k": 6, "pc": null, "knots": []}}}]}`
    ///   (leave `knots` empty — resolved from training data at fit time).
    /// - `distribution`: Distribution name (Gaussian, Poisson, StudentT, Gamma, NegativeBinomial, Beta)
    pub fn fit(
        y_json: &str,
        data_json: &str,
        formula_json: &str,
        distribution: &str,
    ) -> Result<WasmGamlssModel, JsError> {
        let (model, family) =
            json::fit(y_json, data_json, formula_json, distribution, None).map_err(to_js_err)?;
        Ok(WasmGamlssModel { model, family })
    }

    /// Fit a GAMLSS model with custom configuration.
    ///
    /// `config_json` is a JSON object with optional fields:
    /// `{"max_iterations": 200, "tolerance": 0.001, "criterion": "reml"}`.
    /// `criterion` accepts `"reml"` (default), `"gcv"`, or `"fellner_schall"`.
    #[wasm_bindgen(js_name = "fitWithConfig")]
    pub fn fit_with_config(
        y_json: &str,
        data_json: &str,
        formula_json: &str,
        distribution: &str,
        config_json: &str,
    ) -> Result<WasmGamlssModel, JsError> {
        let (model, family) = json::fit(
            y_json,
            data_json,
            formula_json,
            distribution,
            Some(config_json),
        )
        .map_err(to_js_err)?;
        Ok(WasmGamlssModel { model, family })
    }

    #[wasm_bindgen(js_name = "fromJson")]
    pub fn from_json(json: &str) -> Result<WasmGamlssModel, JsError> {
        let (model, family) = json::load(json).map_err(to_js_err)?;
        Ok(WasmGamlssModel { model, family })
    }

    #[wasm_bindgen(js_name = "toJson")]
    pub fn to_json(&self) -> Result<String, JsError> {
        self.model.to_json(self.family.as_ref()).map_err(to_js_err)
    }

    /// Input/output are JSON: `{"col": [values]}` → `{"param": [predictions]}`.
    pub fn predict(&self, data_json: &str) -> Result<String, JsError> {
        json::predict(&self.model, self.family.as_ref(), data_json).map_err(to_js_err)
    }

    #[wasm_bindgen(js_name = "predictWithSe")]
    pub fn predict_with_se(&self, data_json: &str) -> Result<String, JsError> {
        json::predict_with_se(&self.model, self.family.as_ref(), data_json).map_err(to_js_err)
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
        json::predict_samples(&self.model, self.family.as_ref(), data_json, n_samples)
            .map_err(to_js_err)
    }

    #[wasm_bindgen(js_name = "diagnosticsJson")]
    pub fn diagnostics_json(&self) -> Result<String, JsError> {
        json::diagnostics(&self.model).map_err(to_js_err)
    }
}
