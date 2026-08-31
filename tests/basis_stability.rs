// Regression tests for fit-time basis resolution. The P-spline knot grid, the
// tensor-product marginal ranges, and random-effect level maps all get resolved
// once from TRAINING data, stored on the term, and replayed at predict time.
//
// Before the fix, `assemble_model_matrices` quietly re-derived the knot grid /
// level map from whatever data got passed to `predict`. So a prediction on a
// grid, a subset, or reordered groups was evaluated on a DIFFERENT basis than
// the one the coefficients were fitted on. Nasty.
#![cfg(not(feature = "python"))]

mod common;

use common::Generator;
use glissando::{distributions::Gaussian, DataSet, Formula, GamlssModel, Smooth, Term};
use ndarray::Array1;
use rand::RngExt;
use rand_distr::StandardNormal;

trait Draw {
    fn normal(&mut self) -> f64;
    fn uniform(&mut self) -> f64;
}
impl Draw for Generator {
    fn normal(&mut self) -> f64 {
        self.rng.sample(StandardNormal)
    }
    fn uniform(&mut self) -> f64 {
        self.rng.random::<f64>()
    }
}

/// Sine data on [0, 2π] fitted with a P-spline; predicting a SUBSET of the
/// training rows must reproduce the same fitted values as the full-data
/// prediction at those rows (the subset spans a narrower x-range, which used
/// to shift the knot grid).
#[test]
fn pspline_prediction_is_invariant_to_prediction_range() {
    let mut rng = Generator::new(4242);
    let n = 400;
    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / (n - 1) as f64 * std::f64::consts::TAU)
        .collect();
    let y_vec: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1 * rng.normal()).collect();
    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x.clone()));

    let formula = Formula::new()
        .with_terms("mu", vec![Term::smooth(Smooth::ps("x").n_splines(12))])
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    let model = GamlssModel::fit(&data, &y, &formula, &family).unwrap();

    // Training-range prediction (baseline).
    let full_pred = model.predict(&data, &family).unwrap();

    // Interior subset: x in the middle third only. Same rows, so the fitted
    // values have to agree exactly with the matching full-data entries.
    let idx: Vec<usize> = (n / 3..2 * n / 3).collect();
    let mut sub = DataSet::new();
    sub.insert_column("x", Array1::from_vec(idx.iter().map(|&i| x[i]).collect()));
    let sub_pred = model.predict(&sub, &family).unwrap();

    for (j, &i) in idx.iter().enumerate() {
        let a = full_pred["mu"][i];
        let b = sub_pred["mu"][j];
        assert!(
            (a - b).abs() < 1e-10,
            "subset prediction diverged at x={}: full={a}, subset={b}",
            x[i]
        );
    }
}

/// Random-effect predictions have to map groups by the level list resolved at
/// fit time, not by first-occurrence order in the prediction data. So I present
/// the groups in reverse order at predict time and check per-row equality.
#[test]
fn random_effect_levels_are_stable_under_reordering() {
    let mut rng = Generator::new(99);
    let n = 300;
    let n_groups = 6;
    let effects: Vec<f64> = (0..n_groups).map(|_| rng.normal()).collect();
    let g: Vec<f64> = (0..n).map(|i| (i % n_groups) as f64).collect();
    let y_vec: Vec<f64> = g
        .iter()
        .map(|&gi| 2.0 + effects[gi as usize] + 0.2 * rng.normal())
        .collect();
    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("g", Array1::from_vec(g.clone()));

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::smooth(Smooth::re("g"))])
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    let model = GamlssModel::fit(&data, &y, &formula, &family).unwrap();
    let base = model.predict(&data, &family).unwrap();

    // Reversed rows → groups first appear in the opposite order.
    let rev_rows: Vec<usize> = (0..n).rev().collect();
    let mut rev = DataSet::new();
    rev.insert_column(
        "g",
        Array1::from_vec(rev_rows.iter().map(|&i| g[i]).collect()),
    );
    let rev_pred = model.predict(&rev, &family).unwrap();

    for (j, &i) in rev_rows.iter().enumerate() {
        assert!(
            (base["mu"][i] - rev_pred["mu"][j]).abs() < 1e-10,
            "group prediction changed under row reordering (row {i})"
        );
    }
}

/// An unseen group level at predict time is an error (mgcv factor semantics),
/// not a silent column reshuffle.
#[test]
fn random_effect_unseen_level_errors() {
    let mut rng = Generator::new(5);
    let n = 60;
    let g: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let y = Array1::from_vec((0..n).map(|_| rng.normal()).collect::<Vec<f64>>());
    let mut data = DataSet::new();
    data.insert_column("g", Array1::from_vec(g));

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::smooth(Smooth::re("g"))])
        .with_terms("sigma", vec![Term::Intercept]);
    let family = Gaussian::new();
    let model = GamlssModel::fit(&data, &y, &formula, &family).unwrap();

    let mut new_data = DataSet::new();
    new_data.insert_column("g", Array1::from_vec(vec![0.0, 7.0])); // 7 unseen
    let err = model.predict(&new_data, &family);
    assert!(err.is_err(), "unseen level must be an error");
}

/// te() with an intercept must be able to represent ADDITIVE main effects
/// (mgcv semantics: one sum-to-zero constraint on the full tensor basis).
/// The old marginal-centering construction spanned only pure interactions and
/// could not fit f(x1) + g(x2) at all.
#[test]
fn tensor_with_intercept_recovers_main_effects() {
    let mut rng = Generator::new(2718);
    let n = 300;
    let x1: Vec<f64> = (0..n)
        .map(|_| rng.uniform() * std::f64::consts::TAU)
        .collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
    // Purely additive truth: zero interaction.
    let f = |a: f64, b: f64| a.sin() + 2.0 * (b - 0.5) * (b - 0.5);
    let y_vec: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| f(a, b) + 0.1 * rng.normal())
        .collect();
    let y = Array1::from_vec(y_vec.clone());
    let mut data = DataSet::new();
    data.insert_column("x1", Array1::from_vec(x1.clone()));
    data.insert_column("x2", Array1::from_vec(x2.clone()));

    let formula = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Intercept, common::tensor("x1", "x2", 5, 5)],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let family = Gaussian::new();
    let model = GamlssModel::fit(&data, &y, &formula, &family).unwrap();
    let pred = model.predict(&data, &family).unwrap();

    // R² against the noiseless truth has to be high; the interaction-only
    // construction managed essentially zero.
    let truth: Vec<f64> = x1.iter().zip(x2.iter()).map(|(&a, &b)| f(a, b)).collect();
    let mean_t = truth.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = truth.iter().map(|t| (t - mean_t) * (t - mean_t)).sum();
    let ss_res: f64 = truth
        .iter()
        .zip(pred["mu"].iter())
        .map(|(t, p)| (t - p) * (t - p))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot;
    assert!(
        r2 > 0.95,
        "te() + intercept should recover additive main effects (R² = {r2:.4})"
    );
}
