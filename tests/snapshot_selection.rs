// Characterization snapshots (TEST-4) for model selection: the GAIC information
// table and the likelihood-ratio test over nested Gaussian models. Frozen behavior,
// not expectations (see `tests/regression.rs`); floats are rounded so the snapshot
// survives openblas/pure-rust drift.
//
// First-time creation / deliberate refresh:
//   INSTA_UPDATE=auto cargo test --test snapshot_selection
// then promote the `.snap.new` files by hand (cargo-insta is not installed).
#![cfg(not(feature = "python"))]
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::{pspline, Generator};
use glissando::distributions::Gaussian;
use glissando::selection::{ic_table, lr_test};
use glissando::{Formula, GamlssModel, Term};
use serde::Serialize;

/// 3 significant figures: a penalized smooth's effective df (and the deviance / GAIC
/// that ride on it) drift in the 4th digit between openblas and pure-rust as REML's
/// λ lands slightly differently. Coarse enough to be backend-stable, fine enough to
/// still catch a real change in which model wins.
fn fmt(x: f64) -> String {
    format!("{:.2e}", x)
}

#[derive(Debug, Serialize)]
struct IcRowSnapshot {
    label: String,
    edf: String,
    global_deviance: String,
    gaic: String,
}

#[derive(Debug, Serialize)]
struct LrTestSnapshot {
    lr_stat: String,
    df: String,
    p_value: String,
}

/// Fit three nested models to one Gaussian response: intercept-only, linear mean,
/// and a P-spline mean. GAIC(k=2) should rank the linear model best on data that is
/// genuinely linear (the smooth pays EDF it can't recover in fit).
#[test]
fn ic_table_and_lr_test_nested_gaussian() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(150, 1.0, 5.0, 1.0);

    let f_null = Formula::new()
        .with_terms("mu", vec![Term::Intercept])
        .with_terms("sigma", vec![Term::Intercept]);
    let f_linear = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::linear("x")])
        .with_terms("sigma", vec![Term::Intercept]);
    let f_smooth = Formula::new()
        .with_terms("mu", vec![Term::Intercept, pspline("x", 8)])
        .with_terms("sigma", vec![Term::Intercept]);

    let m_null = GamlssModel::fit(&data, &y, &f_null, &Gaussian::new()).unwrap();
    let m_linear = GamlssModel::fit(&data, &y, &f_linear, &Gaussian::new()).unwrap();
    let m_smooth = GamlssModel::fit(&data, &y, &f_smooth, &Gaussian::new()).unwrap();

    let table = ic_table(
        &[
            ("null", &m_null),
            ("linear", &m_linear),
            ("smooth", &m_smooth),
        ],
        &Gaussian::new(),
        &y,
        2.0,
    )
    .unwrap();
    let table_snap: Vec<IcRowSnapshot> = table
        .iter()
        .map(|r| IcRowSnapshot {
            label: r.label.clone(),
            edf: fmt(r.edf),
            global_deviance: fmt(r.global_deviance),
            gaic: fmt(r.gaic),
        })
        .collect();
    insta::assert_yaml_snapshot!("ic_table_nested_gaussian", table_snap);

    // Null (intercept-only mean) nested in the linear-mean model.
    let lr = lr_test(&m_null, &m_linear, &Gaussian::new(), &y).unwrap();
    let lr_snap = LrTestSnapshot {
        lr_stat: fmt(lr.lr_stat),
        df: fmt(lr.df),
        p_value: fmt(lr.p_value),
    };
    insta::assert_yaml_snapshot!("lr_test_null_in_linear", lr_snap);
}
