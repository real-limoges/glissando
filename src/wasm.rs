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
    /// Fit a GAMLSS model. Wire formats are documented on [`crate::json`].
    ///
    /// `weights_json` is an optional JSON array of per-observation prior weights,
    /// e.g. `"[0.5, 1.0, 1.5]"`.  Pass `null` / `undefined` for unweighted fitting.
    pub fn fit(
        y_json: &str,
        data_json: &str,
        formula_json: &str,
        distribution: &str,
        weights_json: Option<String>,
    ) -> Result<WasmGamlssModel, JsError> {
        let (model, family) = json::fit(
            y_json,
            data_json,
            formula_json,
            distribution,
            None,
            weights_json.as_deref(),
        )
        .map_err(to_js_err)?;
        Ok(WasmGamlssModel { model, family })
    }

    /// Fit with an explicit config JSON. Beyond the algorithm knobs, `config_json`
    /// accepts `"links"` to override a parameter's link by name, e.g.
    /// `{"links": {"mu": "probit"}}` (see [`crate::json`] for the accepted names).
    #[wasm_bindgen(js_name = "fitWithConfig")]
    pub fn fit_with_config(
        y_json: &str,
        data_json: &str,
        formula_json: &str,
        distribution: &str,
        config_json: &str,
        weights_json: Option<String>,
    ) -> Result<WasmGamlssModel, JsError> {
        let (model, family) = json::fit(
            y_json,
            data_json,
            formula_json,
            distribution,
            Some(config_json),
            weights_json.as_deref(),
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
    ///
    /// Pass an integer `seed` for reproducible output; omit or pass `null`/`undefined`
    /// for non-deterministic sampling.
    #[wasm_bindgen(js_name = "predictSamples")]
    pub fn predict_samples(
        &self,
        data_json: &str,
        n_samples: usize,
        seed: Option<u64>,
    ) -> Result<String, JsError> {
        json::predict_samples(
            &self.model,
            self.family.as_ref(),
            data_json,
            n_samples,
            seed,
        )
        .map_err(to_js_err)
    }

    /// Returns the linear-predictor design matrix X for `data_json` and `param`
    /// as a JSON array of rows (`[[col0, col1, ...], ...]`).
    ///
    /// Equivalent to mgcv's `predict(type="lpmatrix")`.
    #[wasm_bindgen(js_name = "designMatrix")]
    pub fn design_matrix(&self, data_json: &str, param: &str) -> Result<String, JsError> {
        json::design_matrix(&self.model, data_json, param).map_err(to_js_err)
    }

    /// Returns the `p × p` posterior covariance matrix for `param`
    /// as a JSON array of rows.
    #[wasm_bindgen(js_name = "covarianceMatrix")]
    pub fn covariance_matrix(&self, param: &str) -> Result<String, JsError> {
        json::covariance_matrix(&self.model, param).map_err(to_js_err)
    }

    /// Returns the term → coefficient column block map for `param` as a JSON
    /// object `{"term_name": [first, last], ...}`.
    #[wasm_bindgen(js_name = "termIndexMap")]
    pub fn term_index_map(&self, param: &str) -> Result<String, JsError> {
        json::term_index_map(&self.model, param).map_err(to_js_err)
    }

    #[wasm_bindgen(js_name = "diagnosticsJson")]
    pub fn diagnostics_json(&self) -> Result<String, JsError> {
        json::diagnostics(&self.model).map_err(to_js_err)
    }

    /// Randomized normalized quantile residuals against the response `y` the
    /// model was fit on, as a JSON array. `seed` makes the discrete-family
    /// randomization reproducible; pass `null`/`undefined` for an unseeded draw.
    #[wasm_bindgen(js_name = "quantileResiduals")]
    pub fn quantile_residuals(&self, y_json: &str, seed: Option<u64>) -> Result<String, JsError> {
        json::quantile_residuals(&self.model, self.family.as_ref(), y_json, seed).map_err(to_js_err)
    }

    /// Response-scale centile curves for `data_json`, as `{"C<pct>": [..], ...}`.
    /// `percentiles_json` is a JSON array of centile levels in percent.
    pub fn centiles(&self, data_json: &str, percentiles_json: &str) -> Result<String, JsError> {
        json::centiles(
            &self.model,
            self.family.as_ref(),
            data_json,
            percentiles_json,
        )
        .map_err(to_js_err)
    }

    /// Per-observation quantile prediction; `p_json` is a JSON array of per-row
    /// centile levels in `(0,1)`. Returns a JSON array of predicted responses.
    #[wasm_bindgen(js_name = "quantilePrediction")]
    pub fn quantile_prediction(&self, data_json: &str, p_json: &str) -> Result<String, JsError> {
        json::quantile_prediction(&self.model, self.family.as_ref(), data_json, p_json)
            .map_err(to_js_err)
    }

    /// Generalized AIC at penalty `k` against the response `y` this model was fit
    /// on. Returns `{"gaic": value}`. `k = 2` ≡ AIC, `k = ln(n)` ≡ BIC.
    pub fn gaic(&self, y_json: &str, k: f64) -> Result<String, JsError> {
        json::gaic(&self.model, self.family.as_ref(), y_json, k).map_err(to_js_err)
    }

    /// Likelihood-ratio test treating `self` as the nested (small) model and the
    /// model serialized in `bigger_model_json` as the alternative (big) model.
    /// Returns `{lr_stat, df, p_value}`. Errors if the pair is mis-ordered.
    #[wasm_bindgen(js_name = "lrTest")]
    pub fn lr_test(&self, bigger_model_json: &str, y_json: &str) -> Result<String, JsError> {
        let (big, _) = json::load(bigger_model_json).map_err(to_js_err)?;
        json::lr_test(&self.model, &big, self.family.as_ref(), y_json).map_err(to_js_err)
    }

    /// Stepwise term selection by GAIC(`k`). Returns `{trace, formula, model}`
    /// where `model` is in the same wire form `fromJson` accepts. `scope_json` is
    /// a list of `{"param": "...", "candidates": [<term>, ...]}`; `direction` is
    /// `"forward"`, `"backward"`, or `"both"`.
    #[wasm_bindgen(js_name = "stepGaic")]
    #[allow(clippy::too_many_arguments)]
    pub fn step_gaic(
        y_json: &str,
        data_json: &str,
        distribution: &str,
        start_formula_json: &str,
        scope_json: &str,
        k: f64,
        direction: &str,
        config_json: Option<String>,
    ) -> Result<String, JsError> {
        json::step_gaic(
            y_json,
            data_json,
            distribution,
            start_formula_json,
            scope_json,
            k,
            direction,
            config_json.as_deref(),
        )
        .map_err(to_js_err)
    }
}
