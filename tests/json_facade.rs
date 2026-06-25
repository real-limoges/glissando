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
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
    assert!(model.converged());

    let preds_json =
        json::predict(&model, family.as_ref(), r#"{"x": [11.0, 12.0, 13.0]}"#).expect("predict");
    let preds: HashMap<String, Vec<f64>> = serde_json::from_str(&preds_json).unwrap();
    assert_eq!(preds["mu"].len(), 3);
    assert_eq!(preds["sigma"].len(), 3);
    // Linear mean: predictions should be increasing in x.
    assert!(preds["mu"][0] < preds["mu"][1] && preds["mu"][1] < preds["mu"][2]);
}

#[test]
fn fit_with_config_json() {
    let cfg = r#"{"max_iterations": 50, "tolerance": 0.001, "criterion": "gcv"}"#;
    let (model, _family) = json::fit(Y, DATA, FORMULA, "Gaussian", Some(cfg), None).expect("fit");
    assert!(model.converged());
}

#[test]
fn config_json_defaults_step_halving_and_gd_tolerance() {
    // Omitted FIT-1/FIT-2 keys fall back to the documented defaults via serde.
    let parsed = json::parse_config("{}").expect("parse empty config");
    assert!(parsed.step_halving, "step_halving should default to true");
    assert_eq!(parsed.gd_tolerance, 1e-3);
    assert_eq!(parsed.max_iterations, 200);
}

#[test]
fn config_json_round_trips_step_halving_and_gd_tolerance() {
    let cfg = r#"{"step_halving": false, "gd_tolerance": 5e-4}"#;
    let parsed = json::parse_config(cfg).expect("parse config");
    assert!(!parsed.step_halving);
    assert_eq!(parsed.gd_tolerance, 5e-4);
    // Untouched fields keep their defaults.
    assert_eq!(parsed.tolerance, 1e-3);
}

#[test]
fn diagnostics_json_exposes_final_deviance() {
    let (model, _family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
    let diag_json = json::diagnostics(&model).expect("diagnostics");
    let diag: serde_json::Value = serde_json::from_str(&diag_json).unwrap();
    // FIT-2 surfaces the converged global deviance through the JSON facade.
    assert!(diag["final_deviance"].is_number());
}

#[test]
fn predict_with_se_shape() {
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
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
    let (model, _family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
    let diag_json = json::diagnostics(&model).expect("diagnostics");
    let diag: serde_json::Value = serde_json::from_str(&diag_json).unwrap();
    assert!(diag["converged"].as_bool().unwrap());
    // The warnings field is always present (empty on a clean fit).
    assert!(diag["warnings"].is_array());
    assert!(diag["param_diagnostics"].is_object());
}

#[test]
fn save_then_load_preserves_predictions() {
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
    let before = json::predict(&model, family.as_ref(), r#"{"x": [6.0]}"#).unwrap();

    let blob = model.to_json(family.as_ref()).expect("to_json");
    let (restored, restored_family) = json::load(&blob).expect("load");

    let after = json::predict(&restored, restored_family.as_ref(), r#"{"x": [6.0]}"#).unwrap();
    assert_eq!(before, after);
}

#[test]
fn errors_surface_as_gamlss_error() {
    assert!(json::fit(Y, DATA, FORMULA, "Wishart", None, None).is_err());
    assert!(json::fit("not json", DATA, FORMULA, "Gaussian", None, None).is_err());
    assert!(json::parse_data(r#"{"x": [1.0], "z": [1.0, 2.0]}"#).is_err()); // ragged columns
}

// ============================================================================
// Guide 1 — design_matrix / covariance_matrix / term_index_map / seed
// ============================================================================

#[test]
fn design_matrix_json_shape_matches_data_and_coefficients() {
    let (model, _family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");

    let dm_json = json::design_matrix(&model, DATA, "mu").expect("design_matrix");
    let dm: Vec<Vec<f64>> = serde_json::from_str(&dm_json).unwrap();

    // n_rows = number of observations in DATA (10)
    assert_eq!(dm.len(), 10, "one row per observation");
    // n_cols = number of mu coefficients (intercept + linear = 2)
    let n_coeffs = model.models["mu"].coefficients.0.len();
    for (i, row) in dm.iter().enumerate() {
        assert_eq!(
            row.len(),
            n_coeffs,
            "row {i} width {actual} != n_coeffs {n_coeffs}",
            actual = row.len()
        );
    }

    // Unknown param errors
    assert!(json::design_matrix(&model, DATA, "nonexistent").is_err());
}

#[test]
fn covariance_matrix_json_is_square_and_symmetric() {
    let (model, _family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
    let cv_json = json::covariance_matrix(&model, "mu").expect("covariance_matrix");
    let cv: Vec<Vec<f64>> = serde_json::from_str(&cv_json).unwrap();

    let p = cv.len();
    for row in &cv {
        assert_eq!(row.len(), p, "covariance must be square");
    }
    // Symmetry: V[i][j] ≈ V[j][i]
    for (i, row) in cv.iter().enumerate() {
        for (j, &v_ij) in row.iter().enumerate() {
            let diff = (v_ij - cv[j][i]).abs();
            assert!(diff < 1e-10, "V[{i}][{j}] != V[{j}][{i}], diff={diff}");
        }
    }
}

#[test]
fn term_index_map_json_is_contiguous_and_sums_to_n_coeffs() {
    let (model, _family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
    let map_json = json::term_index_map(&model, "mu").expect("term_index_map");
    let map: std::collections::BTreeMap<String, [usize; 2]> =
        serde_json::from_str(&map_json).unwrap();

    let n_coeffs = model.models["mu"].coefficients.0.len();
    let total: usize = map.values().map(|[f, l]| l - f).sum();
    assert_eq!(total, n_coeffs, "total block width must equal n_coeffs");

    // Standard formula: (intercept) + x => 2 blocks
    assert!(map.contains_key("(intercept)"), "must have intercept block");
    assert!(map.contains_key("x"), "must have linear x block");
}

#[test]
fn predict_samples_seeded_json_is_reproducible() {
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
    let new_x = r#"{"x": [5.0, 6.0, 7.0]}"#;

    let run1 = json::predict_samples(&model, family.as_ref(), new_x, 10, Some(42)).unwrap();
    let run2 = json::predict_samples(&model, family.as_ref(), new_x, 10, Some(42)).unwrap();
    assert_eq!(run1, run2, "seeded runs must be byte-identical");

    let run_unseeded = json::predict_samples(&model, family.as_ref(), new_x, 10, None).unwrap();
    // Unseeded output almost certainly differs (would need 10 × 3 = 30 exact
    // float matches to coincide by chance).
    assert_ne!(run1, run_unseeded, "unseeded run should differ from seeded");
}

// --- Model selection & comparison facade (INFER-3 / INFER-7 / INFER-4) ---

const INTERCEPT_ONLY: &str = r#"{
    "mu":    [{"Intercept": null}],
    "sigma": [{"Intercept": null}]
}"#;

#[test]
fn gaic_json_is_finite_and_monotone_in_k() {
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");

    let read = |k: f64| -> f64 {
        let s = json::gaic(&model, family.as_ref(), Y, k).expect("gaic");
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        v["gaic"].as_f64().unwrap()
    };
    let g2 = read(2.0);
    let g_bic = read((10.0_f64).ln());
    // -2·ll can be negative for a tight Gaussian fit, so only require finiteness…
    assert!(g2.is_finite() && g_bic.is_finite());
    // …and that a larger penalty raises GAIC (edf > 0).
    assert!(
        g_bic > g2,
        "BIC-penalty GAIC {} should exceed AIC-penalty {}",
        g_bic,
        g2
    );
}

#[test]
fn ic_table_json_ranks_models() {
    let (m_null, family) = json::fit(Y, DATA, INTERCEPT_ONLY, "Gaussian", None, None).expect("fit");
    let (m_x, _) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");

    let table_json = json::ic_table(
        &[("null", &m_null), ("with_x", &m_x)],
        family.as_ref(),
        Y,
        2.0,
    )
    .expect("ic_table");
    let rows: serde_json::Value = serde_json::from_str(&table_json).unwrap();
    assert_eq!(rows.as_array().unwrap().len(), 2);
    assert_eq!(rows[0]["label"], "null");
    // The model with x fits y (linear) better → lower deviance.
    let dev_null = rows[0]["global_deviance"].as_f64().unwrap();
    let dev_x = rows[1]["global_deviance"].as_f64().unwrap();
    assert!(dev_x < dev_null);
}

#[test]
fn lr_test_json_detects_genuine_term() {
    let (small, family) = json::fit(Y, DATA, INTERCEPT_ONLY, "Gaussian", None, None).expect("fit");
    let (big, _) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");

    let test_json = json::lr_test(&small, &big, family.as_ref(), Y).expect("lr_test");
    let test: serde_json::Value = serde_json::from_str(&test_json).unwrap();
    assert!(test["lr_stat"].as_f64().unwrap() > 0.0);
    assert!(test["df"].as_f64().unwrap() > 0.0);
    // y is strongly linear in x → tiny p-value.
    assert!(test["p_value"].as_f64().unwrap() < 0.05);

    // Mis-ordered pair surfaces an error.
    assert!(json::lr_test(&big, &small, family.as_ref(), Y).is_err());
}

#[test]
fn step_gaic_json_returns_trace_and_loadable_model() {
    let scope = r#"[{"param": "mu", "candidates": [{"Linear": {"col_name": "x"}}]}]"#;
    let out_json = json::step_gaic(
        Y,
        DATA,
        "Gaussian",
        INTERCEPT_ONLY,
        scope,
        2.0,
        "forward",
        None,
    )
    .expect("step_gaic");
    let out: serde_json::Value = serde_json::from_str(&out_json).unwrap();

    // The genuine linear term should have been added on mu.
    let trace = out["trace"].as_array().unwrap();
    assert!(!trace.is_empty(), "expected at least one accepted move");
    assert!(out["formula"]["mu"].as_array().unwrap().len() >= 2);

    // The embedded model round-trips through `load`.
    let model_blob = serde_json::to_string(&out["model"]).unwrap();
    let (restored, restored_family) = json::load(&model_blob).expect("load selected model");
    assert_eq!(restored_family.name(), "Gaussian");
    let preds = json::predict(&restored, restored_family.as_ref(), r#"{"x": [11.0]}"#).unwrap();
    let parsed: HashMap<String, Vec<f64>> = serde_json::from_str(&preds).unwrap();
    assert_eq!(parsed["mu"].len(), 1);
}

// --- INFER-1 / INFER-2 facade (quantile residuals, centiles) ---

#[test]
fn quantile_residuals_json_round_trip() {
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
    let json_out = json::quantile_residuals(&model, family.as_ref(), Y, None).expect("residuals");
    let residuals: Vec<f64> = serde_json::from_str(&json_out).unwrap();
    assert_eq!(residuals.len(), 10);
    assert!(residuals.iter().all(|r| r.is_finite()));
}

#[test]
fn centiles_json_are_ordered_and_keyed() {
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
    let json_out = json::centiles(&model, family.as_ref(), DATA, "[10, 50, 90]").expect("centiles");
    let curves: std::collections::HashMap<String, Vec<f64>> =
        serde_json::from_str(&json_out).unwrap();
    assert_eq!(curves["C10"].len(), 10);
    // Monotone in level at the first observation.
    assert!(curves["C10"][0] < curves["C50"][0]);
    assert!(curves["C50"][0] < curves["C90"][0]);
}

#[test]
fn quantile_prediction_json_constant_level() {
    let (model, family) = json::fit(Y, DATA, FORMULA, "Gaussian", None, None).expect("fit");
    let p = "[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]";
    let json_out = json::quantile_prediction(&model, family.as_ref(), DATA, p).expect("qpred");
    let predicted: Vec<f64> = serde_json::from_str(&json_out).unwrap();
    assert_eq!(predicted.len(), 10);
    // 50th percentile equals fitted mu for a Gaussian.
    let preds = json::predict(&model, family.as_ref(), DATA).unwrap();
    let parsed: std::collections::HashMap<String, Vec<f64>> = serde_json::from_str(&preds).unwrap();
    for i in 0..10 {
        assert!((predicted[i] - parsed["mu"][i]).abs() < 1e-6);
    }
}
