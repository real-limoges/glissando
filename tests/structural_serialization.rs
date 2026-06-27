//! SER-1 integration tests: structural wrappers (and a finite mixture) survive a
//! `to_json → from_json → build()` round-trip and predict identically.

#![cfg(all(feature = "serialization", not(feature = "python")))]

use glissando::distributions::{
    CensorStatus, Censored, Distribution, FamilyDescriptor, Gaussian, Hurdle, Truncated,
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
    // The descriptor preserves the per-row status (so the base is Gaussian).
    match &desc {
        FamilyDescriptor::Censored { base, status: s, .. } => {
            assert!(matches!(**base, FamilyDescriptor::Named(ref n) if n == "Gaussian"));
            assert_eq!(s.len(), n);
        }
        other => panic!("expected Censored descriptor, got {other:?}"),
    }

    // Predictions survive the round-trip (use the rebuilt family).
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
    // The +∞ upper bound survives via the sentinel encode/decode.
    let rebuilt = desc.build().unwrap();
    assert_eq!(rebuilt.name(), "Truncated");
    // Confirm the rebuilt loglik equals the original on the data (so bounds match).
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
fn mixture_round_trips() {
    // Two clusters at 0 and 6.
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
