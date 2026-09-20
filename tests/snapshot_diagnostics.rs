// Characterization snapshots (TEST-4) for the diagnostics surface: `ModelDiagnostics`
// (AIC/BIC/EDF/loglik) and the deterministic residual kinds, plus centiles. Like
// `tests/regression.rs`, these are frozen behavior, not expectations: a diff means
// the fit or a diagnostic moved, so investigate before re-accepting. Floats are
// rounded so the snapshot survives openblas/pure-rust trailing-digit drift.
//
// First-time creation / deliberate refresh:
//   INSTA_UPDATE=auto cargo test --features serialization --test snapshot_diagnostics
// then promote the `.snap.new` files by hand (cargo-insta is not installed).
#![cfg(not(feature = "python"))]
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::{linear_intercepts, pspline, Generator};
use glissando::distributions::{Distribution, Gaussian, Poisson};
use glissando::{Formula, GamlssModel, Term};
use ndarray::Array1;
use serde::Serialize;

fn fmt(x: f64) -> String {
    format!("{:.4e}", x)
}

/// 3-significant-figure formatter for quantities that carry openblas/pure-rust drift
/// in their 4th–5th digit (effective df, residual extremes): coarse enough to be
/// backend-stable, fine enough to still catch a real move.
fn fmt3(x: f64) -> String {
    format!("{:.2e}", x)
}

/// Compact, backend-stable summary of a residual vector: full vectors would bloat
/// the snapshot and add nothing over these numbers for a regression guard. The mean
/// is deliberately omitted; it is ~0 by construction here (machine-epsilon noise
/// that differs between backends and carries no signal).
#[derive(Debug, Serialize)]
struct ResidualSummary {
    n: usize,
    std: String,
    min: String,
    max: String,
}

impl ResidualSummary {
    fn of(r: &Array1<f64>) -> Self {
        let n = r.len();
        let mean = r.sum() / n as f64;
        let var = r.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        Self {
            n,
            std: fmt3(var.sqrt()),
            min: fmt3(r.iter().cloned().fold(f64::INFINITY, f64::min)),
            max: fmt3(r.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
        }
    }
}

#[derive(Debug, Serialize)]
struct DiagnosticsSnapshot {
    n_obs: usize,
    total_edf: String,
    log_likelihood: String,
    aic: String,
    bic: String,
    gaic_k2: String,
    gaic_bic: String,
    pearson_residuals: ResidualSummary,
    response_residuals: ResidualSummary,
}

impl DiagnosticsSnapshot {
    fn from_fit<D: Distribution + ?Sized>(
        model: &GamlssModel,
        family: &D,
        y: &Array1<f64>,
    ) -> Self {
        let d = model.diagnostics(family, y).unwrap();
        let k_bic = (y.len() as f64).ln();
        Self {
            n_obs: d.n_obs,
            total_edf: fmt3(d.total_edf),
            log_likelihood: fmt(d.log_likelihood),
            aic: fmt(d.aic),
            bic: fmt(d.bic),
            gaic_k2: fmt(model.gaic(family, y, 2.0).unwrap()),
            gaic_bic: fmt(model.gaic(family, y, k_bic).unwrap()),
            pearson_residuals: ResidualSummary::of(&d.pearson_residuals),
            response_residuals: ResidualSummary::of(&d.response_residuals),
        }
    }
}

#[test]
fn diagnostics_gaussian_linear() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(100, 1.0, 5.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    let snap = DiagnosticsSnapshot::from_fit(&model, &Gaussian::new(), &y);
    insta::assert_yaml_snapshot!(snap);
}

#[test]
fn diagnostics_poisson_pspline() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.poisson_data(150, 0.5, 0.3);
    let formula = Formula::new().with_terms("mu", vec![Term::Intercept, pspline("x", 8)]);
    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();
    let snap = DiagnosticsSnapshot::from_fit(&model, &Poisson::new(), &y);
    insta::assert_yaml_snapshot!(snap);
}

/// Centile curves (the signature GAMLSS output) on a fresh grid, snapshotted at the
/// gamlss default percentiles. Locks the response-scale quantile inversion.
#[test]
fn centiles_gaussian_linear() {
    use glissando::DataSet;
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(100, 1.0, 5.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let mut grid = DataSet::new();
    grid.insert_column("x", Array1::linspace(0.0, 1.0, 5));
    let pcts = [2.0, 10.0, 50.0, 90.0, 98.0];
    let centiles = model.centiles(&grid, &Gaussian::new(), &pcts).unwrap();

    // BTreeMap for deterministic key order; values rounded for backend stability.
    let snap: std::collections::BTreeMap<String, Vec<String>> = centiles
        .iter()
        .map(|(k, v)| (k.clone(), v.iter().map(|&x| fmt(x)).collect()))
        .collect();
    insta::assert_yaml_snapshot!(snap);
}
