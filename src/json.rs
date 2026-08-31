//! JSON marshaling facade for embedding glissando behind your own FFI.
//!
//! If you're calling glissando in-process from Rust, the typed API
//! ([`crate::GamlssModel`], [`crate::DataSet`], [`crate::Formula`]) is the surface
//! you want. But an embedder sitting behind a *different* boundary (a Rustler NIF,
//! a C ABI, a JSON microservice) usually just wants to throw glissando strings and
//! get strings back, without taking a dependency on `ndarray` types or reverse-
//! engineering the wire format. That's what this module is: the exact same tested
//! JSON marshaling the WASM bindings run on, opened up for any embedder to use.
//!
//! Gated behind the `serialization` feature.
//!
//! # Wire formats
//!
//! - **response** (`parse_response`): a JSON array of numbers: `[1.0, 2.0, 3.0]`.
//! - **data** (`parse_data`): an object of equal-length columns:
//!   `{"x": [1.0, 2.0], "z": [3.0, 4.0]}`.
//! - **formula** (`parse_formula`): parameter name → term list:
//!   `{"mu": [{"Intercept": null}, {"Linear": {"col_name": "x"}}], "sigma": [{"Intercept": null}]}`.
//! - **config** (`parse_config`): `{"max_iterations": 200, "tolerance": 0.001, "criterion": "reml"}`
//!   (`criterion` is `"reml"`, `"gcv"`, or `"fellner_schall"`; all fields optional).
//!   Add `"links"` to override a parameter's link by name, e.g.
//!   `{"links": {"mu": "probit"}}`: accepted names: `"identity"`, `"log"`, `"logit"`,
//!   `"probit"`, `"cloglog"`, `"inverse"`, `"inverse_square"`, `"sqrt"`, `"cauchit"`.
//! - **predictions** (`serialize_predictions`): `{"mu": [..], "sigma": [..]}`.
//! - **predictions with SE** (`serialize_predictions_with_se`):
//!   `{"mu": {"fitted": [..], "eta": [..], "se_eta": [..]}, ...}`.
//! - **samples** (`serialize_samples`): `{"mu": [[s1..], [s2..], ...], ...}`.
//!
//! # Distribution dispatch
//!
//! [`fit`] and [`load`] resolve a distribution by name via
//! [`crate::distributions::from_name`], which covers `Gaussian`, `Poisson`,
//! `StudentT`, `Gamma`, `NegativeBinomial`, `Beta`, `BCCG`, `BCT`, and `BCPE`.
//! `Binomial` sits this one out: it needs `n_trials` state, and a bare name has
//! nowhere to carry that, so build it through the typed API instead.
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
//! let (model, family) = json::fit(y, data, formula, "Gaussian", None, None).unwrap();
//! assert!(model.converged());
//!
//! // Keep the model in memory and predict interactively.
//! let preds = json::predict(&model, family.as_ref(), r#"{"x": [6.0, 7.0]}"#).unwrap();
//! assert!(preds.contains("mu"));
//! # }
//! ```

use std::collections::{BTreeMap, HashMap};

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

/// # Errors
/// Returns [`GamlssError::Input`] if the JSON is malformed.
pub fn parse_response(json: &str) -> Result<Array1<f64>, GamlssError> {
    let values: Vec<f64> = serde_json::from_str(json).map_err(json_err)?;
    Ok(Array1::from_vec(values))
}

/// # Errors
/// Returns [`GamlssError::Input`] for malformed JSON or ragged columns.
pub fn parse_data(json: &str) -> Result<DataSet, GamlssError> {
    let raw: HashMap<String, Vec<f64>> = serde_json::from_str(json).map_err(json_err)?;
    DataSet::from_vecs(raw)
}

/// Parse a formula from JSON, accepting either spelling (DATA-5):
///
/// - **term-list**: the structured `{ "mu": [ {"Linear": {"col_name": "x"}}, … ] }`
///   form matching the [`Term`](crate::Term) schema; or
/// - **string formula**: the ergonomic `{ "mu": "y ~ s(x) + z", "sigma": "~ 1" }`
///   form, where each value is an R/mgcv-style formula string.
///
/// The two are disambiguated by value type (string vs array), so existing
/// term-list payloads keep working unchanged.
///
/// # Errors
/// Returns [`GamlssError::Input`] if the JSON matches neither schema or a string
/// formula is malformed.
pub fn parse_formula(json: &str) -> Result<Formula, GamlssError> {
    // Try the ergonomic string-map form first: `{param: "y ~ ..."}`. It only
    // parses when every value is a JSON string, so a term-list payload (whose
    // values are arrays) misses cleanly and drops through to the structured
    // deserializer below. No ambiguity, the value type decides.
    if let Ok(string_map) = serde_json::from_str::<HashMap<String, String>>(json) {
        return Formula::from_strings(string_map.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    serde_json::from_str(json).map_err(json_err)
}

/// All fields are optional and default to [`FitConfig::default`].
///
/// # Errors
/// Returns [`GamlssError::Input`] if the JSON is malformed.
pub fn parse_config(json: &str) -> Result<FitConfig, GamlssError> {
    serde_json::from_str(json).map_err(json_err)
}

// ---------------------------------------------------------------------------
// Serialization (typed outputs → JSON)
// ---------------------------------------------------------------------------

/// # Errors
/// Returns [`GamlssError::Input`] if serialization fails.
pub fn serialize_predictions(
    predictions: &HashMap<String, Array1<f64>>,
) -> Result<String, GamlssError> {
    // BTreeMap so the keys come out sorted and deterministic, not in HashMap's
    // roll-the-dice order. Embedders get stable output, and two predict calls
    // compare equal byte for byte instead of only sometimes.
    let result: BTreeMap<&str, Vec<f64>> = predictions
        .iter()
        .map(|(k, v)| (k.as_str(), v.to_vec()))
        .collect();
    serde_json::to_string(&result).map_err(json_err)
}

/// # Errors
/// Returns [`GamlssError::Input`] if serialization fails.
pub fn serialize_predictions_with_se(
    predictions: &HashMap<String, PredictionResult>,
) -> Result<String, GamlssError> {
    let output: BTreeMap<&str, PredictionWithSe> = predictions
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

/// # Errors
/// Returns [`GamlssError::Input`] if serialization fails.
pub fn serialize_samples(
    samples: &HashMap<String, Vec<Array1<f64>>>,
) -> Result<String, GamlssError> {
    let output: BTreeMap<&str, Vec<Vec<f64>>> = samples
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
    weights_json: Option<&str>,
) -> Result<(GamlssModel, Box<dyn Distribution>), GamlssError> {
    let y = parse_response(y_json)?;
    let data = parse_data(data_json)?;
    let formula = parse_formula(formula_json)?;
    let family = from_name(distribution)?;
    let config = match config_json {
        Some(c) => parse_config(c)?,
        None => FitConfig::default(),
    };
    let weights = weights_json.map(parse_response).transpose()?;
    let model = GamlssModel::fit_with_config(
        &data,
        &y,
        weights.as_ref(),
        &formula,
        family.as_ref(),
        config,
    )?;
    Ok((model, family))
}

/// Deserialize a model previously written with [`GamlssModel::to_json`],
/// returning it alongside its resolved distribution.
///
/// # Errors
/// Returns [`GamlssError`] if the JSON is malformed or describes a distribution
/// that cannot be reconstructed (e.g. a `Named` descriptor for an unknown family).
pub fn load(json: &str) -> Result<(GamlssModel, Box<dyn Distribution>), GamlssError> {
    let (model, descriptor) = GamlssModel::from_json(json)?;
    let family = descriptor.build()?;
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
/// Pass `seed = Some(s)` for reproducible samples; `seed = None` uses an unseeded RNG.
///
/// # Errors
/// Returns [`GamlssError`] for malformed data JSON or any sampling failure.
pub fn predict_samples(
    model: &GamlssModel,
    family: &dyn Distribution,
    data_json: &str,
    n_samples: usize,
    seed: Option<u64>,
) -> Result<String, GamlssError> {
    let new_data = parse_data(data_json)?;
    let results = model.predict_samples(&new_data, family, n_samples, seed)?;
    serialize_samples(&results)
}

/// Returns the linear-predictor design matrix X for `new_data` and the named
/// distribution parameter, serialized as a JSON array of rows (list of lists).
///
/// This is the `predict(type="lpmatrix")` equivalent from mgcv.
///
/// # Errors
/// Returns [`GamlssError`] for malformed data JSON or any prediction failure.
pub fn design_matrix(
    model: &GamlssModel,
    data_json: &str,
    param: &str,
) -> Result<String, GamlssError> {
    let new_data = parse_data(data_json)?;
    let x = model.design_matrix(&new_data, param)?;
    let rows: Vec<Vec<f64>> = x.rows().into_iter().map(|r| r.to_vec()).collect();
    serde_json::to_string(&rows).map_err(json_err)
}

/// Returns the `p × p` posterior covariance matrix `V = (X'WX + Σλ·S)⁻¹`
/// for the named distribution parameter, serialized as a JSON array of rows.
///
/// # Errors
/// Returns [`GamlssError::UnknownParameter`] if `param` is not in the model,
/// or [`GamlssError::Input`] if serialization fails.
pub fn covariance_matrix(model: &GamlssModel, param: &str) -> Result<String, GamlssError> {
    let v = model.covariance_matrix(param)?;
    let rows: Vec<Vec<f64>> = v.0.rows().into_iter().map(|r| r.to_vec()).collect();
    serde_json::to_string(&rows).map_err(json_err)
}

/// Returns the term → coefficient column block map for the named distribution
/// parameter, serialized as a JSON object `{"term_name": [first, last], ...}`.
///
/// Key order is alphabetical (BTreeMap) for deterministic output.
///
/// # Errors
/// Returns [`GamlssError::UnknownParameter`] if `param` is not in the model,
/// or [`GamlssError::Input`] if serialization fails.
pub fn term_index_map(model: &GamlssModel, param: &str) -> Result<String, GamlssError> {
    let blocks = model.term_index_map(param)?;
    let map: BTreeMap<&str, [usize; 2]> = blocks
        .iter()
        .map(|(name, first, last)| (name.as_str(), [*first, *last]))
        .collect();
    serde_json::to_string(&map).map_err(json_err)
}

/// Serialize the model's [`crate::FitDiagnostics`] (convergence, per-parameter
/// EDF, clamp counters, and collapse warnings) as JSON.
///
/// # Errors
/// Returns [`GamlssError::Input`] if serialization fails.
pub fn diagnostics(model: &GamlssModel) -> Result<String, GamlssError> {
    serde_json::to_string(&model.diagnostics).map_err(json_err)
}

// --- Distributional inference (INFER-1 / INFER-2) ---

/// Randomized normalized quantile residuals against the response `y` the model
/// was fit on, serialized as a JSON array. `seed` makes the discrete-family
/// randomization reproducible (ignored for continuous families).
///
/// # Errors
/// Returns [`GamlssError`] for malformed `y` JSON or a `cdf` failure.
pub fn quantile_residuals(
    model: &GamlssModel,
    family: &dyn Distribution,
    y_json: &str,
    seed: Option<u64>,
) -> Result<String, GamlssError> {
    let y = parse_response(y_json)?;
    let residuals = model.quantile_residuals(family, &y, seed)?;
    serde_json::to_string(&residuals.to_vec()).map_err(json_err)
}

/// Response-scale centile curves for `new_data`, serialized as
/// `{"C<pct>": [values], ...}` (deterministic key order). `percentiles_json` is
/// a JSON array of centile levels in percent, e.g. `[2, 10, 50, 90, 98]`.
///
/// # Errors
/// Returns [`GamlssError`] for malformed JSON or any prediction/quantile failure.
pub fn centiles(
    model: &GamlssModel,
    family: &dyn Distribution,
    data_json: &str,
    percentiles_json: &str,
) -> Result<String, GamlssError> {
    let new_data = parse_data(data_json)?;
    let percentiles: Vec<f64> = serde_json::from_str(percentiles_json).map_err(json_err)?;
    let curves = model.centiles(&new_data, family, &percentiles)?;
    let output: BTreeMap<&str, Vec<f64>> = curves
        .iter()
        .map(|(key, values)| (key.as_str(), values.to_vec()))
        .collect();
    serde_json::to_string(&output).map_err(json_err)
}

/// Per-observation quantile prediction: `p_json` is a JSON array of per-row
/// centile levels in `(0,1)`. Returns a JSON array of predicted responses.
///
/// # Errors
/// Returns [`GamlssError`] for malformed JSON or any prediction/quantile failure.
pub fn quantile_prediction(
    model: &GamlssModel,
    family: &dyn Distribution,
    data_json: &str,
    p_json: &str,
) -> Result<String, GamlssError> {
    let new_data = parse_data(data_json)?;
    let p = parse_response(p_json)?;
    let predicted = model.quantile_prediction(&new_data, family, &p)?;
    serde_json::to_string(&predicted.to_vec()).map_err(json_err)
}

// --- Model selection & comparison (INFER-3 / INFER-7 / INFER-4) ---

/// Generalized AIC at penalty `k`, serialized as `{"gaic": value}`. Evaluate
/// against the same response `y` the model was fit on.
///
/// # Errors
/// Returns [`GamlssError`] for malformed `y` JSON or log-likelihood failure.
pub fn gaic(
    model: &GamlssModel,
    family: &dyn Distribution,
    y_json: &str,
    k: f64,
) -> Result<String, GamlssError> {
    let y = parse_response(y_json)?;
    let value = model.gaic(family, &y, k)?;
    let out: BTreeMap<&str, f64> = BTreeMap::from([("gaic", value)]);
    serde_json::to_string(&out).map_err(json_err)
}

/// Information-criterion comparison table over labeled models, serialized as a
/// JSON array of `{label, edf, global_deviance, gaic}` rows in input order.
///
/// # Errors
/// Returns [`GamlssError`] for malformed `y` JSON or log-likelihood failure.
pub fn ic_table(
    models: &[(&str, &GamlssModel)],
    family: &dyn Distribution,
    y_json: &str,
    k: f64,
) -> Result<String, GamlssError> {
    let y = parse_response(y_json)?;
    let rows = crate::fitting::selection::ic_table(models, family, &y, k)?;
    serde_json::to_string(&rows).map_err(json_err)
}

/// Likelihood-ratio test of `small` nested in `big`, serialized as
/// `{lr_stat, df, p_value}`.
///
/// # Errors
/// Returns [`GamlssError::Input`] for a mis-ordered/non-nested pair, or for
/// malformed `y` JSON.
pub fn lr_test(
    small: &GamlssModel,
    big: &GamlssModel,
    family: &dyn Distribution,
    y_json: &str,
) -> Result<String, GamlssError> {
    let y = parse_response(y_json)?;
    let result = crate::fitting::selection::lr_test(small, big, family, &y)?;
    serde_json::to_string(&result).map_err(json_err)
}

/// Run stepwise term selection from `start_formula_json` over `scope_json`, and
/// return a single JSON object `{trace, formula, model}` where `model` is the
/// selected model in the same wire form [`load`] accepts (distribution + model).
///
/// `direction` is `"forward"`, `"backward"`, or `"both"`; `scope_json` is a list
/// of `{"param": "...", "candidates": [<term>, ...]}`.
///
/// # Errors
/// Returns [`GamlssError`] for malformed inputs, an unknown distribution, or a
/// fitting failure at the starting formula.
#[allow(clippy::too_many_arguments)]
pub fn step_gaic(
    y_json: &str,
    data_json: &str,
    distribution: &str,
    start_formula_json: &str,
    scope_json: &str,
    k: f64,
    direction: &str,
    config_json: Option<&str>,
) -> Result<String, GamlssError> {
    use crate::fitting::selection::{step_gaic as run_step_gaic, Direction, StepScope};

    let y = parse_response(y_json)?;
    let data = parse_data(data_json)?;
    let start = parse_formula(start_formula_json)?;
    let family = from_name(distribution)?;
    let scope: Vec<StepScope> = serde_json::from_str(scope_json).map_err(json_err)?;
    let dir = Direction::from_name(direction)?;
    let config = match config_json {
        Some(c) => parse_config(c)?,
        None => FitConfig::default(),
    };

    let result = run_step_gaic(&data, &y, family.as_ref(), start, &scope, k, dir, config)?;

    // Nest the selected model as an object in the exact shape `load` reads back,
    // so the caller can round-trip it straight through without reshaping anything.
    let model_wire: serde_json::Value =
        serde_json::from_str(&result.model.to_json(family.as_ref())?).map_err(json_err)?;
    let out = serde_json::json!({
        "trace": result.trace,
        "formula": result.formula,
        "model": model_wire,
    });
    serde_json::to_string(&out).map_err(json_err)
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
        let (model, family) = fit(Y, DATA, FORMULA, "Gaussian", None, None).unwrap();
        assert!(model.converged());

        let preds = predict(&model, family.as_ref(), r#"{"x": [11.0, 12.0]}"#).unwrap();
        let parsed: HashMap<String, Vec<f64>> = serde_json::from_str(&preds).unwrap();
        assert_eq!(parsed["mu"].len(), 2);
        assert_eq!(parsed["sigma"].len(), 2);
    }

    #[test]
    fn unknown_distribution_errors() {
        assert!(fit(Y, DATA, FORMULA, "Wishart", None, None).is_err());
    }

    /// DATA-5: the string-formula spelling is accepted by `parse_formula` and
    /// fits identically to the equivalent term-list payload.
    #[test]
    fn string_formula_fits_like_term_list() {
        const STR_FORMULA: &str = r#"{"mu": "y ~ x", "sigma": "~ 1"}"#;
        let (m_str, fam) = fit(Y, DATA, STR_FORMULA, "Gaussian", None, None).unwrap();
        let (m_list, _) = fit(Y, DATA, FORMULA, "Gaussian", None, None).unwrap();
        let _ = fam;
        for param in ["mu", "sigma"] {
            let a = &m_str.models[param].coefficients.0;
            let b = &m_list.models[param].coefficients.0;
            for (x, z) in a.iter().zip(b.iter()) {
                assert!((x - z).abs() < 1e-12, "{param}: {x} vs {z}");
            }
        }
    }

    /// The string-map and term-list forms are disambiguated by value type, so a
    /// term-list payload still parses as before.
    #[test]
    fn term_list_formula_still_parses() {
        let f = parse_formula(FORMULA).unwrap();
        assert_eq!(f["mu"].len(), 2);
    }

    #[test]
    fn malformed_json_errors() {
        assert!(parse_response("not json").is_err());
        assert!(parse_data("not json").is_err());
        assert!(parse_formula("not json").is_err());
    }

    #[test]
    fn save_then_load_preserves_coefficients() {
        let (model, family) = fit(Y, DATA, FORMULA, "Gaussian", None, None).unwrap();
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
