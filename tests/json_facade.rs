// The JSON facade lives behind the `serialization` feature; exclude `python`
// for the usual PyO3 extension-module linking reason.
#![cfg(all(feature = "serialization", not(feature = "python")))]

//! Native round-trip of the embedding contract (`glissando::json`), exercising
//! the public facade the way a non-WASM/non-Python embedder would — without a
//! wasm build. Mirrors the coverage in `tests/wasm.rs`.

use glissando::json;
use std::collections::HashMap;

const Y: &str = "[1.2, 2.1, 2.9, 4.2, 4.8, 5.9, 7.1, 7.9, 9.2, 9.8]";
const DATA: &str = r#"{"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}"#;
const FORMULA: &str = r#"{
    "mu":    [{"Intercept": null}, {"Linear": {"col_name": "x"}}],
    "sigma": [{"Intercept": null}]
}"#;

#[test]
fn fit_then_predict_round_trip() {
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None).expect("fit");
    assert!(model.converged());

    let preds_json = json::predict(&model, family.as_ref(), r#"{"x": [11.0, 12.0, 13.0]}"#)
        .expect("predict");
    let preds: HashMap<String, Vec<f64>> = serde_json::from_str(&preds_json).unwrap();
    assert_eq!(preds["mu"].len(), 3);
    assert_eq!(preds["sigma"].len(), 3);
    // Linear mean: predictions should be increasing in x.
    assert!(preds["mu"][0] < preds["mu"][1] && preds["mu"][1] < preds["mu"][2]);
}

#[test]
fn fit_with_config_json() {
    let cfg = r#"{"max_iterations": 50, "tolerance": 0.001, "criterion": "gcv"}"#;
    let (model, _family) = json::fit(Y, DATA, FORMULA, "Gaussian", Some(cfg)).expect("fit");
    assert!(model.converged());
}

#[test]
fn predict_with_se_shape() {
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None).expect("fit");
    let se_json =
        json::predict_with_se(&model, family.as_ref(), r#"{"x": [2.0, 4.0]}"#).expect("se");
    let parsed: serde_json::Value = serde_json::from_str(&se_json).unwrap();
    let mu = &parsed["mu"];
    assert_eq!(mu["fitted"].as_array().unwrap().len(), 2);
    assert_eq!(mu["eta"].as_array().unwrap().len(), 2);
    assert_eq!(mu["se_eta"].as_array().unwrap().len(), 2);
}

#[test]
fn diagnostics_expose_per_term_edf_and_warnings_field() {
    let (model, _family) = json::fit(Y, DATA, FORMULA, "Gaussian", None).expect("fit");
    let diag_json = json::diagnostics(&model).expect("diagnostics");
    let diag: serde_json::Value = serde_json::from_str(&diag_json).unwrap();
    assert!(diag["converged"].as_bool().unwrap());
    // The warnings field is always present (empty on a clean fit).
    assert!(diag["warnings"].is_array());
    assert!(diag["param_diagnostics"].is_object());
}

#[test]
fn save_then_load_preserves_predictions() {
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None).expect("fit");
    let before = json::predict(&model, family.as_ref(), r#"{"x": [6.0]}"#).unwrap();

    let blob = model.to_json(family.as_ref()).expect("to_json");
    let (restored, restored_family) = json::load(&blob).expect("load");

    let after = json::predict(&restored, restored_family.as_ref(), r#"{"x": [6.0]}"#).unwrap();
    assert_eq!(before, after);
}

#[test]
fn errors_surface_as_gamlss_error() {
    assert!(json::fit(Y, DATA, FORMULA, "Wishart", None).is_err());
    assert!(json::fit("not json", DATA, FORMULA, "Gaussian", None).is_err());
    assert!(json::parse_data(r#"{"x": [1.0], "z": [1.0, 2.0]}"#).is_err()); // ragged columns
}
