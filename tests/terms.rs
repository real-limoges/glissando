// Integration tests cannot run with the `python` feature due to PyO3's extension-module linking.
#![cfg(not(feature = "python"))]

mod common;

use common::{random, Generator};
use glissando::{distributions::Gaussian, DataSet, Formula, GamlssModel, Term};
use ndarray::Array1;
use rand::RngExt;
use rand_distr::{Distribution, Normal};

#[test]
fn random_effect_recovers_group_means() {
    // y = grand_mean + group_offset[g_i] + ε where group_offset has 4 levels.
    // Fit `Intercept + RandomEffect(group)` on Gaussian; the fitted η for each
    // observation should track the true per-group mean within noise tolerance.
    let mut gen = Generator::new(2024);

    let grand_mean = 5.0;
    let group_offsets = [-1.5, 0.0, 1.0, 2.5];
    let n_per_group = 40;
    let sigma = 0.3;
    let normal = Normal::new(0.0, sigma).unwrap();

    let mut y_vals: Vec<f64> = Vec::with_capacity(group_offsets.len() * n_per_group);
    let mut group_ids: Vec<f64> = Vec::with_capacity(group_offsets.len() * n_per_group);
    for (g_idx, off) in group_offsets.iter().enumerate() {
        for _ in 0..n_per_group {
            let noise = normal.sample(&mut gen.rng);
            y_vals.push(grand_mean + off + noise);
            group_ids.push(g_idx as f64);
        }
    }
    let y = Array1::from_vec(y_vals);
    let mut data = DataSet::new();
    data.insert_column("group", Array1::from_vec(group_ids.clone()));

    let mut formula = Formula::new();
    formula.add_terms("mu", vec![Term::Intercept, random("group")]);
    formula.add_terms("sigma", vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    assert!(model.converged());

    // The fitted η_i for an obs in group g should be near grand_mean + offset[g].
    let mu_fitted = &model.models["mu"].fitted_values;
    for (g_idx, off) in group_offsets.iter().enumerate() {
        let expected = grand_mean + off;
        // pick the first observation in this group
        let i = g_idx * n_per_group;
        let diff = (mu_fitted[i] - expected).abs();
        assert!(
            diff < 0.4,
            "group {} fitted = {:.3}, expected ≈ {:.3} (diff {:.3})",
            g_idx,
            mu_fitted[i],
            expected,
            diff
        );
    }

    // EDF should reflect the random effect's contribution above an intercept-only fit.
    // With strong group separation and 4 groups (3 free degrees of freedom after the
    // sum-to-zero constraint), we expect EDF clearly above 1.
    let edf = model.models["mu"].edf;
    assert!(
        edf > 1.5,
        "RandomEffect should add EDF beyond the intercept; got {:.3}",
        edf
    );
}

#[test]
fn random_effect_with_few_groups_runs() {
    // Smoke test: 2-group case (the minimum where sum-to-zero kicks in).
    let mut rng = rand::rng();
    let n_per_group = 20;
    let mut y_vals: Vec<f64> = Vec::new();
    let mut group_ids: Vec<f64> = Vec::new();
    for g in 0..2 {
        for _ in 0..n_per_group {
            let centre = 3.0 + g as f64 * 2.0;
            y_vals.push(centre + rng.random_range(-0.1..0.1));
            group_ids.push(g as f64);
        }
    }
    let y = Array1::from_vec(y_vals);
    let mut data = DataSet::new();
    data.insert_column("g", Array1::from_vec(group_ids));

    let mut formula = Formula::new();
    formula.add_terms("mu", vec![Term::Intercept, random("g")]);
    formula.add_terms("sigma", vec![Term::Intercept]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    assert!(model.converged());
    assert!(model.models["mu"]
        .coefficients
        .iter()
        .all(|c: &f64| c.is_finite()));
}
