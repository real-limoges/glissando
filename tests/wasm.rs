//! WASM integration tests for `glissando::wasm::WasmGamlssModel`.
//!
//! Runs under `wasm-pack test --node --no-default-features --features wasm`.
//! On non-wasm targets this file is empty — the dev-dep `wasm-bindgen-test` is
//! target-scoped (see Cargo.toml) and the surface under test is wasm-only.

#![cfg(target_arch = "wasm32")]

use glissando::wasm::WasmGamlssModel;
use wasm_bindgen_test::*;

// Noisy Gaussian sample so the fit doesn't collapse to σ → 0 (which trips the IRLS guard).
const Y: &str = "[1.2, 2.1, 2.9, 4.2, 4.8, 5.9, 7.1, 7.9, 9.2, 9.8]";
const DATA: &str = r#"{"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}"#;
const FORMULA: &str = r#"{
    "mu":    [{"Intercept": null}, {"Linear": {"col_name": "x"}}],
    "sigma": [{"Intercept": null}]
}"#;

#[wasm_bindgen_test]
fn fit_linear_gaussian() {
    let model = WasmGamlssModel::fit(Y, DATA, FORMULA, "Gaussian").unwrap();
    assert!(model.converged());
    let coefs = model.coefficients("mu").unwrap();
    assert_eq!(coefs.len(), 2);
}

#[wasm_bindgen_test]
fn predict_returns_one_value_per_row() {
    let model = WasmGamlssModel::fit(Y, DATA, FORMULA, "Gaussian").unwrap();

    let new_data = r#"{"x": [6.0, 7.0, 8.0]}"#;
    let pred_json = model.predict(new_data).unwrap();
    let parsed: std::collections::HashMap<String, Vec<f64>> =
        serde_json::from_str(&pred_json).unwrap();
    assert_eq!(parsed["mu"].len(), 3);
    assert_eq!(parsed["sigma"].len(), 3);
}

#[wasm_bindgen_test]
fn json_round_trip_preserves_coefficients() {
    let model = WasmGamlssModel::fit(Y, DATA, FORMULA, "Gaussian").unwrap();
    let original_coefs = model.coefficients("mu").unwrap();

    let json = model.to_json().unwrap();
    let restored = WasmGamlssModel::from_json(&json).unwrap();
    let restored_coefs = restored.coefficients("mu").unwrap();

    assert_eq!(original_coefs.len(), restored_coefs.len());
    for (a, b) in original_coefs.iter().zip(restored_coefs.iter()) {
        assert!((a - b).abs() < 1e-12, "coef mismatch after JSON round-trip");
    }
}

#[wasm_bindgen_test]
fn malformed_json_returns_error() {
    let res = WasmGamlssModel::fit("not json", "{}", "{}", "Gaussian");
    assert!(res.is_err());
}

#[wasm_bindgen_test]
fn unknown_distribution_returns_error() {
    let y = "[1.0, 2.0]";
    let data = r#"{"x": [1.0, 2.0]}"#;
    let formula = r#"{"mu": [{"Intercept": null}], "sigma": [{"Intercept": null}]}"#;
    let res = WasmGamlssModel::fit(y, data, formula, "Wishart");
    assert!(res.is_err());
}
