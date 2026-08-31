//! STRUCT-4 integration tests. Finite mixtures fit by EM should recover a known
//! two-component structure and beat a single-component fit. If they don't, EM isn't earning its keep.

#![cfg(not(feature = "python"))]

use glissando::distributions::Gaussian;
use glissando::fitting::mixture::fit_mixture;
use glissando::{DataSet, FitConfig, Formula, GamlssModel, Term};
use ndarray::Array1;

/// Two well-separated Gaussian clusters (means 0 and 6) with mild within-cluster
/// spread. I build them deterministically so the test reproduces every run.
fn two_cluster_data() -> (DataSet, Array1<f64>) {
    let mut vals = Vec::new();
    for i in 0..60 {
        // cluster A sitting around 0, spread across [−1.5, 1.5]
        vals.push(-1.5 + 3.0 * (i as f64) / 59.0);
    }
    for i in 0..60 {
        // cluster B sitting around 6, spread across [4.5, 7.5]
        vals.push(4.5 + 3.0 * (i as f64) / 59.0);
    }
    let y = Array1::from_vec(vals);
    let mut data = DataSet::new();
    // Dummy column just so the dataset reports n_obs. The formula is intercept-only.
    data.insert_column("x", Array1::from_iter((0..y.len()).map(|i| i as f64)));
    (data, y)
}

fn intercept_only() -> Formula {
    Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms("sigma", vec![Term::Intercept])
}

#[test]
fn mixture_recovers_two_components() {
    let (data, y) = two_cluster_data();
    let formula = intercept_only();
    let config = FitConfig::default();

    let mix = fit_mixture(&data, &y, &formula, &Gaussian::new(), 2, &config, Some(42)).unwrap();

    assert_eq!(mix.components.len(), 2);
    assert!(
        mix.converged,
        "EM should converge on well-separated clusters"
    );
    assert!(mix.log_likelihood.is_finite());

    // Recover the two intercepts. Identity link, so coefficient[0] just is the mean.
    let mut means: Vec<f64> = mix
        .components
        .iter()
        .map(|c| c.models["mu"].coefficients.0[0])
        .collect();
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        (means[0] - 0.0).abs() < 1.0,
        "low mean ≈ 0, got {}",
        means[0]
    );
    assert!(
        (means[1] - 6.0).abs() < 1.0,
        "high mean ≈ 6, got {}",
        means[1]
    );

    // Balanced clusters, so the weights should sit near 0.5 each and sum to 1.
    let wsum: f64 = mix.weights.iter().sum();
    assert!((wsum - 1.0).abs() < 1e-9);
    for w in &mix.weights {
        assert!((*w - 0.5).abs() < 0.2, "weight ≈ 0.5, got {w}");
    }
}

#[test]
fn mixture_beats_single_component() {
    let (data, y) = two_cluster_data();
    let formula = intercept_only();
    let config = FitConfig::default();

    let single = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    let single_ll = single
        .diagnostics(&Gaussian::new(), &y)
        .unwrap()
        .log_likelihood;

    let mix = fit_mixture(&data, &y, &formula, &Gaussian::new(), 2, &config, Some(7)).unwrap();

    assert!(
        mix.log_likelihood > single_ll,
        "two-component mixture loglik {} should exceed single-component {}",
        mix.log_likelihood,
        single_ll
    );
    // AIC should land on the mixture too, for data this genuinely bimodal.
    let single_aic = single.diagnostics(&Gaussian::new(), &y).unwrap().aic;
    assert!(
        mix.aic() < single_aic,
        "mixture AIC should beat single-component"
    );
}

#[test]
fn fit_mixture_rejects_single_component() {
    let (data, y) = two_cluster_data();
    let formula = intercept_only();
    let config = FitConfig::default();
    let err = fit_mixture(&data, &y, &formula, &Gaussian::new(), 1, &config, Some(1));
    assert!(err.is_err());
}
