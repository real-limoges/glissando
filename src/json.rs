//! JSON marshalling facade for embedding glissando behind your own FFI.
//!
//! glissando's typed Rust API ([`crate::GamlssModel`], [`crate::DataSet`],
//! [`crate::Formula`]) is the right surface for in-process Rust callers. But an
//! embedder behind a *different* boundary — a Rustler NIF, a C ABI, a JSON
//! microservice — usually wants to hand glissando strings and get strings back,
//! without depending on `ndarray` types or re-deriving the wire format. This
//! module is that contract: the same tested JSON marshalling the WASM bindings
//! use, exposed for any embedder.
//!
//! Gated behind the `serialization` feature.
//!
//! # Wire formats
//!
//! - **response** (`parse_response`): a JSON array of numbers — `[1.0, 2.0, 3.0]`.
//! - **data** (`parse_data`): an object of equal-length columns —
//!   `{"x": [1.0, 2.0], "z": [3.0, 4.0]}`.
//! - **formula** (`parse_formula`): parameter name → term list —
//!   `{"mu": [{"Intercept": null}, {"Linear": {"col_name": "x"}}], "sigma": [{"Intercept": null}]}`.
//! - **config** (`parse_config`): `{"max_iterations": 200, "tolerance": 0.001, "criterion": "reml"}`
//!   (`criterion` is `"reml"`, `"gcv"`, or `"fellner_schall"`; all fields optional).
//! - **predictions** (`serialize_predictions`): `{"mu": [..], "sigma": [..]}`.
//! - **predictions with SE** (`serialize_predictions_with_se`):
//!   `{"mu": {"fitted": [..], "eta": [..], "se_eta": [..]}, ...}`.
//! - **samples** (`serialize_samples`): `{"mu": [[s1..], [s2..], ...], ...}`.
//!
//! # Distribution dispatch
//!
//! [`fit`] and [`load`] resolve a distribution by name via
//! [`crate::distributions::from_name`], which covers `Gaussian`, `Poisson`,
//! `StudentT`, `Gamma`, `NegativeBinomial`, and `Beta`. `Binomial` is excluded
//! because it needs `n_trials` state that a name cannot carry — construct it
//! through the typed API instead.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "serialization")] {
//! use glissando::json;
//!
//! let y = "[1.2, 2.1, 2.9, 4.2, 4.8]";
//! let data = r#"{"x": [1.0, 2.0, 3.0, 4.0, 5.0]}"#;
//! let formula = r#"{
//!     "mu":    [{"Intercept": null}, {"Linear": {"col_name": "x"}}],
//!     "sigma": [{"Intercept": null}]
//! }"#;
//!
//! let (model, family) = json::fit(y, data, formula, "Gaussian", None).unwrap();
//! assert!(model.converged());
//!
//! // Keep the model in memory and predict interactively.
//! let preds = json::predict(&model, family.as_ref(), r#"{"x": [6.0, 7.0]}"#).unwrap();
//! assert!(preds.contains("mu"));
//! # }
//! ```

use std::collections::HashMap;

use ndarray::Array1;

use crate::distributions::{from_name, Distribution};
use crate::fitting::FitConfig;
use crate::types::{DataSet, Formula};
use crate::{GamlssError, GamlssModel, PredictionResult};

/// Per-parameter prediction with standard errors, in wire form.
#[derive(serde::Serialize)]
struct PredictionWithSe {
    fitted: Vec<f64>,
    eta: Vec<f64>,
    se_eta: Vec<f64>,
}

fn json_err(e: impl std::fmt::Display) -> GamlssError {
    GamlssError::Input(e.to_string())
}

// ---------------------------------------------------------------------------
// Parsing (JSON → typed inputs)
// ---------------------------------------------------------------------------

/// Parse the response vector from a JSON array of numbers, e.g. `[1.0, 2.0, 3.0]`.
///
/// # Errors
/// Returns [`GamlssError::Input`] if the JSON is malformed or not an array of numbers.
pub fn parse_response(json: &str) -> Result<Array1<f64>, GamlssError> {
    let values: Vec<f64> = serde_json::from_str(json).map_err(json_err)?;
    Ok(Array1::from_vec(values))
}

/// Parse a [`DataSet`] from a JSON object of equal-length numeric columns,
/// e.g. `{"x": [1.0, 2.0], "z": [3.0, 4.0]}`.
///
/// # Errors
/// Returns [`GamlssError::Input`] for malformed JSON, or the validation error
/// from [`DataSet::from_vecs`] if columns differ in length.
pub fn parse_data(json: &str) -> Result<DataSet, GamlssError> {
    let raw: HashMap<String, Vec<f64>> = serde_json::from_str(json).map_err(json_err)?;
    DataSet::from_vecs(raw)
}

/// Parse a [`Formula`] mapping parameter names to term lists, e.g.
/// `{"mu": [{"Intercept": null}, {"Linear": {"col_name": "x"}}]}`.
///
/// # Errors
/// Returns [`GamlssError::Input`] if the JSON does not match the term schema.
pub fn parse_formula(json: &str) -> Result<Formula, GamlssError> {
    serde_json::from_str(json).map_err(json_err)
}

/// Parse a [`FitConfig`] from JSON; all fields are optional and default to
/// [`FitConfig::default`].
///
/// # Errors
/// Returns [`GamlssError::Input`] if the JSON is malformed.
pub fn parse_config(json: &str) -> Result<FitConfig, GamlssError> {
    serde_json::from_str(json).map_err(json_err)
}

// ---------------------------------------------------------------------------
// Serialization (typed outputs → JSON)
// ---------------------------------------------------------------------------

/// Serialize fitted-value predictions (`{"param": [values]}`) as produced by
/// [`GamlssModel::predict`].
///
/// # Errors
/// Returns [`GamlssError::Input`] if serialization fails.
pub fn serialize_predictions(
    predictions: &HashMap<String, Array1<f64>>,
) -> Result<String, GamlssError> {
    let result: HashMap<&str, Vec<f64>> = predictions
        .iter()
        .map(|(k, v)| (k.as_str(), v.to_vec()))
        .collect();
    serde_json::to_string(&result).map_err(json_err)
}

/// Serialize predictions with standard errors (`{"param": {"fitted", "eta",
/// "se_eta"}}`) as produced by [`GamlssModel::predict_with_se`].
///
/// # Errors
/// Returns [`GamlssError::Input`] if serialization fails.
pub fn serialize_predictions_with_se(
    predictions: &HashMap<String, PredictionResult>,
) -> Result<String, GamlssError> {
    let output: HashMap<&str, PredictionWithSe> = predictions
        .iter()
        .map(|(k, v)| {
            (
                k.as_str(),
                PredictionWithSe {
                    fitted: v.fitted.to_vec(),
                    eta: v.eta.to_vec(),
                    se_eta: v.se_eta.to_vec(),
                },
            )
        })
        .collect();
    serde_json::to_string(&output).map_err(json_err)
}

/// Serialize posterior prediction samples (`{"param": [[sample], ...]}`) as
/// produced by [`GamlssModel::predict_samples`].
///
/// # Errors
/// Returns [`GamlssError::Input`] if serialization fails.
pub fn serialize_samples(
    samples: &HashMap<String, Vec<Array1<f64>>>,
) -> Result<String, GamlssError> {
    let output: HashMap<&str, Vec<Vec<f64>>> = samples
        .iter()
        .map(|(k, runs)| (k.as_str(), runs.iter().map(|s| s.to_vec()).collect()))
        .collect();
    serde_json::to_string(&output).map_err(json_err)
}

// ---------------------------------------------------------------------------
// End-to-end convenience (string in, model / string out)
// ---------------------------------------------------------------------------

/// Fit a model from JSON inputs, returning the fitted model and the resolved
/// distribution (boxed so callers can keep predicting interactively).
///
/// `config_json` is optional; pass `None` for [`FitConfig::default`].
///
/// # Errors
/// Returns [`GamlssError`] for malformed JSON, an unknown distribution name, or
/// any fitting failure.
pub fn fit(
    y_json: &str,
    data_json: &str,
    formula_json: &str,
    distribution: &str,
    config_json: Option<&str>,
) -> Result<(GamlssModel, Box<dyn Distribution>), GamlssError> {
    let y = parse_response(y_json)?;
    let data = parse_data(data_json)?;
    let formula = parse_formula(formula_json)?;
    let family = from_name(distribution)?;
    let config = match config_json {
        Some(c) => parse_config(c)?,
        None => FitConfig::default(),
    };
    let model = GamlssModel::fit_with_config(&data, &y, &formula, family.as_ref(), config)?;
    Ok((model, family))
}

/// Deserialize a model previously written with [`GamlssModel::to_json`],
/// returning it alongside its resolved distribution.
///
/// # Errors
/// Returns [`GamlssError`] if the JSON is malformed or names a distribution that
/// cannot be reconstructed from its name (e.g. `Binomial`).
pub fn load(json: &str) -> Result<(GamlssModel, Box<dyn Distribution>), GamlssError> {
    let (model, distribution_name) = GamlssModel::from_json(json)?;
    let family = from_name(&distribution_name)?;
    Ok((model, family))
}

/// Predict fitted values for new JSON data, returning `{"param": [values]}`.
///
/// # Errors
/// Returns [`GamlssError`] for malformed data JSON or any prediction failure.
pub fn predict(
    model: &GamlssModel,
    family: &dyn Distribution,
    data_json: &str,
) -> Result<String, GamlssError> {
    let new_data = parse_data(data_json)?;
    let predictions = model.predict(&new_data, family)?;
    serialize_predictions(&predictions)
}

/// Predict with standard errors, returning
/// `{"param": {"fitted", "eta", "se_eta"}}`.
///
/// # Errors
/// Returns [`GamlssError`] for malformed data JSON or any prediction failure.
pub fn predict_with_se(
    model: &GamlssModel,
    family: &dyn Distribution,
    data_json: &str,
) -> Result<String, GamlssError> {
    let new_data = parse_data(data_json)?;
    let results = model.predict_with_se(&new_data, family)?;
    serialize_predictions_with_se(&results)
}

/// Generate posterior prediction samples, returning
/// `{"param": [[sample], ...]}`.
///
/// # Errors
/// Returns [`GamlssError`] for malformed data JSON or any sampling failure.
pub fn predict_samples(
    model: &GamlssModel,
    family: &dyn Distribution,
    data_json: &str,
    n_samples: usize,
) -> Result<String, GamlssError> {
    let new_data = parse_data(data_json)?;
    let results = model.predict_samples(&new_data, family, n_samples)?;
    serialize_samples(&results)
}

/// Serialize the model's [`crate::FitDiagnostics`] (convergence, per-parameter
/// EDF, clamp counters, and collapse warnings) as JSON.
///
/// # Errors
/// Returns [`GamlssError::Input`] if serialization fails.
pub fn diagnostics(model: &GamlssModel) -> Result<String, GamlssError> {
    serde_json::to_string(&model.diagnostics).map_err(json_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    const Y: &str = "[1.2, 2.1, 2.9, 4.2, 4.8, 5.9, 7.1, 7.9, 9.2, 9.8]";
    const DATA: &str = r#"{"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}"#;
    const FORMULA: &str = r#"{
        "mu":    [{"Intercept": null}, {"Linear": {"col_name": "x"}}],
        "sigma": [{"Intercept": null}]
    }"#;

    #[test]
    fn fit_predict_round_trip() {
        let (model, family) = fit(Y, DATA, FORMULA, "Gaussian", None).unwrap();
        assert!(model.converged());

        let preds = predict(&model, family.as_ref(), r#"{"x": [11.0, 12.0]}"#).unwrap();
        let parsed: HashMap<String, Vec<f64>> = serde_json::from_str(&preds).unwrap();
        assert_eq!(parsed["mu"].len(), 2);
        assert_eq!(parsed["sigma"].len(), 2);
    }

    #[test]
    fn unknown_distribution_errors() {
        assert!(fit(Y, DATA, FORMULA, "Wishart", None).is_err());
    }

    #[test]
    fn malformed_json_errors() {
        assert!(parse_response("not json").is_err());
        assert!(parse_data("not json").is_err());
        assert!(parse_formula("not json").is_err());
    }

    #[test]
    fn save_then_load_preserves_coefficients() {
        let (model, family) = fit(Y, DATA, FORMULA, "Gaussian", None).unwrap();
        let serialized = model.to_json(family.as_ref()).unwrap();
        let (restored, restored_family) = load(&serialized).unwrap();
        assert_eq!(restored_family.name(), "Gaussian");
        for (name, fp) in &model.models {
            let other = &restored.models[name];
            for (a, b) in fp.coefficients.0.iter().zip(other.coefficients.0.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }
}
