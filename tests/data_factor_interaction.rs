//! DATA-1 factors and DATA-2 interactions — public-API integration tests.
//!
//! Contrast coding is locked at the unit level against R's `contr.treatment` /
//! `contr.sum` (see `assembler::tests`). Here we check the end-to-end story:
//! a factor's per-level effects are recovered from a fit, an interaction term
//! recovers a level-specific slope, and resolved factor levels replay verbatim
//! through a JSON round-trip and at predict time.

// Integration tests cannot run with the `python` feature (PyO3 linking).
#![cfg(not(feature = "python"))]

use glissando::distributions::Gaussian;
use glissando::{Contrast, DataSet, Formula, GamlssModel, Term};
use ndarray::Array1;

/// Build a 3-level factor column cycling 0,1,2 and a Gaussian response whose mean
/// is `base + effect[level]` (treatment-coded: effect[0] is folded into `base`).
fn factor_dataset(n: usize, base: f64, effects: [f64; 3]) -> (Array1<f64>, DataSet) {
    let g: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let y: Vec<f64> = g
        .iter()
        .enumerate()
        .map(|(i, &gi)| {
            // Deterministic tiny wiggle keeps the fit well-posed without RNG.
            base + effects[gi as usize] + 0.02 * ((i % 5) as f64 - 2.0)
        })
        .collect();
    let mut data = DataSet::new();
    data.insert_column("g", Array1::from_vec(g));
    (Array1::from_vec(y), data)
}

/// Treatment-coded factor: intercept ≈ base + effect[0]; the two dummy
/// coefficients ≈ effect[1] − effect[0] and effect[2] − effect[0].
#[test]
fn factor_recovers_treatment_level_effects() {
    let (base, effects) = (5.0, [0.0, 2.0, -1.5]);
    let (y, data) = factor_dataset(300, base, effects);
    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::factor("g")])
        .with_terms("sigma", vec![Term::Intercept]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian).unwrap();

    let beta = &model.models["mu"].coefficients.0;
    assert_eq!(
        beta.len(),
        3,
        "intercept + 2 dummy columns for a 3-level factor"
    );
    assert!(
        (beta[0] - (base + effects[0])).abs() < 0.1,
        "intercept {}",
        beta[0]
    );
    assert!(
        (beta[1] - (effects[1] - effects[0])).abs() < 0.1,
        "g1 {}",
        beta[1]
    );
    assert!(
        (beta[2] - (effects[2] - effects[0])).abs() < 0.1,
        "g2 {}",
        beta[2]
    );
}

/// Sum-to-zero coding produces the same fitted values as treatment coding — the
/// contrast is a reparameterization, the fit is identical up to it.
#[test]
fn factor_sum_to_zero_fits_same_values_as_treatment() {
    let (y, data) = factor_dataset(300, 5.0, [0.0, 2.0, -1.5]);
    let treat = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::factor("g")])
        .with_terms("sigma", vec![Term::Intercept]);
    let sum = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Intercept, Term::factor_with("g", Contrast::SumToZero)],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let m_treat = GamlssModel::fit(&data, &y, &treat, &Gaussian).unwrap();
    let m_sum = GamlssModel::fit(&data, &y, &sum, &Gaussian).unwrap();

    let p_treat = m_treat.predict(&data, &Gaussian).unwrap();
    let p_sum = m_sum.predict(&data, &Gaussian).unwrap();
    for (a, b) in p_treat["mu"].iter().zip(p_sum["mu"].iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "contrast reparameterization changed fitted μ: {a} vs {b}"
        );
    }
}

/// A factor×continuous interaction recovers a level-specific slope: with a slope
/// that differs by factor level, the interaction column carries the difference.
#[test]
fn interaction_recovers_level_specific_slope() {
    // Two groups; slope is 1.0 in group 0 and 1.0 + 0.8 in group 1.
    let n = 400;
    let g: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let x: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 4.0).collect();
    let y: Vec<f64> = g
        .iter()
        .zip(&x)
        .map(|(&gi, &xi)| 1.0 + 1.0 * xi + 0.8 * gi * xi)
        .collect();

    let mut data = DataSet::new();
    data.insert_column("g", Array1::from_vec(g));
    data.insert_column("x", Array1::from_vec(x));
    let y = Array1::from_vec(y);

    // μ = intercept + g + x + g:x  (the interaction is the level-1 slope offset).
    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::factor("g"),
                Term::linear("x"),
                Term::interaction(Term::factor("g"), Term::linear("x")),
            ],
        )
        .with_terms("sigma", vec![Term::Intercept]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian).unwrap();

    let pred = model.predict(&data, &Gaussian).unwrap();
    for (p, yi) in pred["mu"].iter().zip(y.iter()) {
        assert!((p - yi).abs() < 0.05, "interaction fit off: {p} vs {yi}");
    }
    // The interaction coefficient (last column) recovers the slope difference 0.8.
    let beta = &model.models["mu"].coefficients.0;
    let interaction_coef = beta[beta.len() - 1];
    assert!(
        (interaction_coef - 0.8).abs() < 0.1,
        "interaction slope-difference: {interaction_coef} (expected 0.8)"
    );
}

/// A factor's levels resolve at fit time and replay through a JSON round-trip,
/// so a reloaded model predicts identically even on data missing a level.
#[cfg(feature = "serialization")]
#[test]
fn factor_levels_survive_json_roundtrip() {
    let (y, data) = factor_dataset(300, 5.0, [0.0, 2.0, -1.5]);
    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::factor("g")])
        .with_terms("sigma", vec![Term::Intercept]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian).unwrap();

    let json = model.to_json(&Gaussian).unwrap();
    let (reloaded, desc) = GamlssModel::from_json(&json).unwrap();
    assert_eq!(desc.build().unwrap().name(), "Gaussian");

    // Predict on data containing only levels {0, 2} — the stored levels keep the
    // column mapping stable (level 2 still lands in the second dummy column).
    let mut subset = DataSet::new();
    subset.insert_column("g", Array1::from_vec(vec![0.0, 2.0, 2.0, 0.0]));
    let p1 = model.predict(&subset, &Gaussian).unwrap();
    let p2 = reloaded.predict(&subset, &Gaussian).unwrap();
    for (a, b) in p1["mu"].iter().zip(p2["mu"].iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "round-trip prediction mismatch: {a} vs {b}"
        );
    }
}
