//! SER-1 integration tests: structural wrappers (and a finite mixture) come
//! through a `to_json → from_json → build()` round-trip and predict identically.

#![cfg(all(feature = "serialization", not(feature = "python")))]

use glissando::distributions::{
    CensorStatus, Censored, Distribution, FamilyDescriptor, Gaussian, Hurdle, Ocat, Truncated,
};
use glissando::fitting::mixture::fit_mixture;
use glissando::{DataSet, FitConfig, Formula, GamlssModel, MixtureModel, Term};
use ndarray::Array1;

fn dummy_data(n: usize) -> DataSet {
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_iter((0..n).map(|i| i as f64)));
    data
}

fn intercept_only() -> Formula {
    Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms("sigma", vec![Term::Intercept])
}

fn latent_gaussian(mu: f64, sigma: f64, n: usize) -> Array1<f64> {
    let p = Array1::from_iter((0..n).map(|i| (i as f64 + 0.5) / n as f64));
    let owned = [
        ("mu", Array1::from_elem(n, mu)),
        ("sigma", Array1::from_elem(n, sigma)),
    ];
    let view = owned.iter().map(|(k, v)| (*k, v)).collect();
    Gaussian.quantile(&p, &view).unwrap()
}

#[test]
fn censored_descriptor_round_trips() {
    let n = 60;
    let y = latent_gaussian(5.0, 2.0, n);
    let mut status = Array1::from_elem(n, CensorStatus::Event);
    for i in 0..n {
        if y[i] > 6.0 {
            status[i] = CensorStatus::Right;
        }
    }
    let family = Censored::new(Box::new(Gaussian::new()), status.clone());
    let model = GamlssModel::fit(&dummy_data(n), &y, &intercept_only(), &family).unwrap();

    let json = model.to_json(&family).unwrap();
    let (reloaded, desc) = GamlssModel::from_json(&json).unwrap();
    assert_eq!(desc.build().unwrap().name(), "Censored");
    // The descriptor holds onto the per-row status, so the base stays Gaussian.
    match &desc {
        FamilyDescriptor::Censored {
            base, status: s, ..
        } => {
            assert!(matches!(**base, FamilyDescriptor::Named(ref n) if n == "Gaussian"));
            assert_eq!(s.len(), n);
        }
        other => panic!("expected Censored descriptor, got {other:?}"),
    }

    // Predictions come through the round-trip (using the rebuilt family).
    let rebuilt = desc.build().unwrap();
    let p1 = model.predict(&dummy_data(n), &family).unwrap();
    let p2 = reloaded.predict(&dummy_data(n), rebuilt.as_ref()).unwrap();
    for k in ["mu", "sigma"] {
        for (a, b) in p1[k].iter().zip(p2[k].iter()) {
            assert!((a - b).abs() < 1e-12, "{k}: {a} vs {b}");
        }
    }
}

#[test]
fn truncated_descriptor_round_trips_with_infinite_bounds() {
    let n = 40;
    let y = latent_gaussian(3.0, 1.0, n).mapv(|v| v.max(0.5));
    let lower = Array1::from_elem(n, 0.0);
    let upper = Array1::from_elem(n, f64::INFINITY);
    let family = Truncated::new(Box::new(Gaussian::new()), lower, upper);
    let model = GamlssModel::fit(&dummy_data(n), &y, &intercept_only(), &family).unwrap();

    let json = model.to_json(&family).unwrap();
    let (_, desc) = GamlssModel::from_json(&json).unwrap();
    // The +∞ upper bound makes it through via the sentinel encode/decode.
    let rebuilt = desc.build().unwrap();
    assert_eq!(rebuilt.name(), "Truncated");
    // Rebuilt loglik equals the original on the data, which means the bounds match.
    let owned = [
        ("mu", model.models["mu"].fitted_values.clone()),
        ("sigma", model.models["sigma"].fitted_values.clone()),
    ];
    let view = owned.iter().map(|(k, v)| (*k, v)).collect();
    let ll_orig = family.loglik_pointwise(&y, &view).unwrap();
    let ll_new = rebuilt.loglik_pointwise(&y, &view).unwrap();
    for i in 0..n {
        assert!((ll_orig[i] - ll_new[i]).abs() < 1e-12);
    }
}

#[test]
fn hurdle_descriptor_round_trips() {
    let family = Hurdle::new(Box::new(Gaussian::new()));
    let desc = family.descriptor();
    let json = serde_json::to_string(&desc).unwrap();
    let back: FamilyDescriptor = serde_json::from_str(&json).unwrap();
    let rebuilt = back.build().unwrap();
    assert_eq!(rebuilt.name(), "Hurdle");
    assert_eq!(rebuilt.parameters(), &["mu", "sigma", "xi"]);
}

#[test]
fn ocat_descriptor_round_trips() {
    // Ocat carries n_categories, the same class of state as Binomial's n_trials.
    // The `Named` fallback would rebuild it with the wrong (or no) category count,
    // so this asserts the descriptor keeps it and the fit survives save -> load.
    let n = 240;
    let n_categories = 4;

    // Deterministic LCG in [0, 1); the >> 32 / (MAX + 1) keeps the full unit
    // range (a >> 33 would cap u below 0.5 and never draw the top categories).
    let mut state = 42u64;
    let mut lcg = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as f64) / (u32::MAX as f64 + 1.0)
    };
    // eta is constant for this intercept-only case, so the category
    // probabilities are fixed; compute them once, then just draw per row.
    let thresholds = [-1.0_f64, 0.0, 1.2];
    let cum: Vec<f64> = thresholds
        .iter()
        .map(|&t| 1.0 / (1.0 + (-t).exp()))
        .collect();
    let p = [cum[0], cum[1] - cum[0], cum[2] - cum[1], 1.0 - cum[2]];
    let y: Array1<f64> = (0..n)
        .map(|_| {
            let u = lcg();
            let mut acc = 0.0;
            for (k, &pk) in p.iter().enumerate() {
                acc += pk;
                if u < acc {
                    return (k + 1) as f64;
                }
            }
            n_categories as f64
        })
        .collect();

    // Intercept-only: mu plus one threshold offset per extra category.
    let mut formula = Formula::new().with_terms("mu", vec![Term::Intercept]);
    for name in ["delta_1", "delta_2", "delta_3"] {
        formula = formula.with_terms(name, vec![Term::Intercept]);
    }

    // Guard the data itself: all four categories must be present, or this would
    // silently fit a 4-category Ocat on degenerate data and prove nothing.
    let present: std::collections::BTreeSet<i64> = y.iter().map(|&v| v as i64).collect();
    assert_eq!(
        present.len(),
        n_categories,
        "all {n_categories} categories must be sampled"
    );

    let family = Ocat::new(n_categories);
    let model = GamlssModel::fit(&dummy_data(n), &y, &formula, &family).unwrap();
    let preds = model.predict(&dummy_data(n), &family).unwrap();

    let json = model.to_json(&family).unwrap();
    let (reloaded, desc) = GamlssModel::from_json(&json).unwrap();

    // The descriptor carries n_categories, so the rebuilt family matches.
    match &desc {
        FamilyDescriptor::Ocat { n_categories: k } => assert_eq!(*k, n_categories),
        other => panic!("expected Ocat descriptor, got {other:?}"),
    }
    let rebuilt = desc.build().unwrap();
    assert_eq!(rebuilt.name(), "Ocat");
    assert_eq!(rebuilt.parameters(), family.parameters());

    // Every parameter (mu and the thresholds) predicts identically after the
    // round-trip; a wrong category count would reshape the threshold set and
    // diverge here.
    let preds2 = reloaded.predict(&dummy_data(n), rebuilt.as_ref()).unwrap();
    for key in family.parameters() {
        for (a, b) in preds[*key].iter().zip(preds2[*key].iter()) {
            assert!((a - b).abs() < 1e-12, "{key}: {a} vs {b}");
        }
    }
}

#[test]
fn mixture_round_trips() {
    // Two clusters, at 0 and 6.
    let mut vals: Vec<f64> = (0..40).map(|i| -1.0 + 2.0 * i as f64 / 39.0).collect();
    vals.extend((0..40).map(|i| 5.0 + 2.0 * i as f64 / 39.0));
    let y = Array1::from_vec(vals);
    let data = dummy_data(y.len());

    let mix = fit_mixture(
        &data,
        &y,
        &intercept_only(),
        &Gaussian::new(),
        2,
        &FitConfig::default(),
        Some(99),
    )
    .unwrap();

    let json = mix.to_json().unwrap();
    let reloaded = MixtureModel::from_json(&json).unwrap();
    assert_eq!(reloaded.components.len(), 2);
    assert_eq!(reloaded.family.build().unwrap().name(), "Gaussian");
    assert!((reloaded.log_likelihood - mix.log_likelihood).abs() < 1e-9);

    // The reloaded mixture predicts the same mean.
    let rebuilt = reloaded.family.build().unwrap();
    let m1 = mix.predict_expected_value(&data, rebuilt.as_ref()).unwrap();
    let m2 = reloaded
        .predict_expected_value(&data, rebuilt.as_ref())
        .unwrap();
    for (a, b) in m1.iter().zip(m2.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}
