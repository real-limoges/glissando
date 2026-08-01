// Per-family end-to-end fit snapshots.
//
// `tests/regression.rs` snapshots three Gaussian fits — 1 of 15 families. This
// file covers the rest, so that a change to any family's `derivatives()` shows
// up as fitted-coefficient drift rather than passing unnoticed.
//
// It is the end-to-end counterpart to `tests/derivative_golden.rs`: that file
// pins what each family *returns*, this one pins where the fitting loop
// *lands*. The generic-link-chain-rule refactor (Altitude #1) must leave every
// snapshot here byte-identical, because it only changes how the default-link
// chain rule is expressed, not what it evaluates to.
//
// Data is generated deterministically from closed-form patterns rather than an
// RNG, so the fixtures do not depend on any random-number implementation.
//
// First-time creation: `INSTA_UPDATE=auto cargo test --test regression_families`.
#![cfg(not(feature = "python"))]
#![cfg(not(target_arch = "wasm32"))]

use glissando::distributions::{Beta, Binomial};
use glissando::distributions::{
    CensorStatus, Censored, Distribution, Gamma, Gaussian, Hurdle, NegativeBinomial, Ocat, Poisson,
    StudentT, Truncated, Weibull, BCCG, BCPE, BCT,
};
use glissando::{DataSet, FitConfig, Formula, GamlssModel, Smooth, Term};

mod common;
use common::Generator;
use ndarray::Array1;
use serde::Serialize;
use std::collections::BTreeMap;

const N: usize = 60;

/// Snapshotted fit summary. Mirrors `tests/regression.rs::ModelSnapshot`;
/// λ is omitted because every fixture here is purely parametric (no penalties),
/// so λ carries no information.
#[derive(Debug, Serialize)]
struct FitSnapshot {
    converged: bool,
    coefficients: BTreeMap<String, Vec<String>>,
    edf: BTreeMap<String, String>,
    log_likelihood: String,
    aic: String,
}

/// 5 significant figures, matching `tests/regression.rs`'s convention: enough
/// to catch real drift, loose enough to survive trailing-digit differences
/// between the openblas and pure-rust backends.
fn fmt(x: f64) -> String {
    format!("{:.4e}", x)
}

fn snapshot<D: Distribution + ?Sized>(
    model: &GamlssModel,
    family: &D,
    y: &Array1<f64>,
) -> FitSnapshot {
    let diag = model.diagnostics(family, y).unwrap();
    FitSnapshot {
        converged: model.converged(),
        coefficients: model
            .models
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.coefficients.0.iter().map(|&c| fmt(c)).collect(),
                )
            })
            .collect(),
        edf: model
            .models
            .iter()
            .map(|(k, v)| (k.clone(), fmt(v.edf)))
            .collect(),
        log_likelihood: fmt(diag.log_likelihood),
        aic: fmt(diag.aic),
    }
}

/// Deterministic covariate on `[0, 1)`.
fn x_grid() -> Array1<f64> {
    (0..N).map(|i| i as f64 / N as f64).collect()
}

fn dataset(x: &Array1<f64>) -> DataSet {
    let mut d = DataSet::new();
    d.insert_column("x", x.clone());
    d
}

/// Deterministic bounded "noise" in `[-1, 1]`, from a non-repeating pattern.
fn jitter(i: usize) -> f64 {
    ((i as f64 * 0.7391).sin() + (i as f64 * 1.317).cos()) * 0.5
}

/// `μ ~ 1 + x`, every remaining parameter intercept-only.
fn formula_for<D: Distribution + ?Sized>(family: &D) -> Formula {
    let mut f = Formula::new();
    f.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );
    for p in family.parameters().iter().skip(1) {
        f.add_terms((*p).to_string(), vec![Term::Intercept]);
    }
    f
}

fn fit_and_snapshot<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    x: &Array1<f64>,
) -> FitSnapshot {
    fit_and_snapshot_with_data(family, y, &dataset(x))
}

fn fit_and_snapshot_with_data<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    data: &DataSet,
) -> FitSnapshot {
    let model = GamlssModel::fit(data, y, &formula_for(family), family)
        .unwrap_or_else(|e| panic!("{} fit failed: {:?}", family.name(), e));
    snapshot(&model, family, y)
}

// ---------------------------------------------------------------------------
// Simple families
// ---------------------------------------------------------------------------

#[test]
fn fit_poisson() {
    let x = x_grid();
    let y: Array1<f64> = (0..N)
        .map(|i| ((2.0 + 3.0 * x[i]) + jitter(i)).max(0.0).round())
        .collect();
    insta::assert_yaml_snapshot!(fit_and_snapshot(&Poisson::new(), &y, &x));
}

#[test]
fn fit_binomial() {
    let x = x_grid();
    let y: Array1<f64> = (0..N)
        .map(|i| {
            let p = 1.0 / (1.0 + (-(4.0 * x[i] - 2.0)).exp());
            if p + 0.25 * jitter(i) > 0.5 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    insta::assert_yaml_snapshot!(fit_and_snapshot(&Binomial::new(1), &y, &x));
}

#[test]
fn fit_gamma() {
    let x = x_grid();
    let y: Array1<f64> = (0..N)
        .map(|i| (1.0 + 2.0 * x[i]) * (1.0 + 0.15 * jitter(i)))
        .collect();
    insta::assert_yaml_snapshot!(fit_and_snapshot(&Gamma::new(), &y, &x));
}

#[test]
fn fit_negative_binomial() {
    // Genuinely overdispersed draws, not smooth jitter. With deterministic
    // jitter the sample variance sits *below* the mean, so σ collapses toward
    // its numerical floor (σ ≈ 1e-10, the Poisson limit) and its coefficient is
    // then determined by where that floor bites — which differs between the
    // openblas and pure-rust backends and made the snapshot backend-dependent.
    // Drawing from NB(μ, σ) with σ = 0.4 keeps σ identifiable and the snapshot
    // stable across both.
    let mut gen = Generator::new(42);
    let x = x_grid();
    let y: Array1<f64> = (0..N)
        .map(|i| common::sample_negative_binomial(&mut gen.rng, 3.0 + 5.0 * x[i], 0.4))
        .collect();
    insta::assert_yaml_snapshot!(fit_and_snapshot(&NegativeBinomial::new(), &y, &x));
}

#[test]
fn fit_beta() {
    let x = x_grid();
    let y: Array1<f64> = (0..N)
        .map(|i| (0.25 + 0.4 * x[i] + 0.06 * jitter(i)).clamp(0.02, 0.98))
        .collect();
    insta::assert_yaml_snapshot!(fit_and_snapshot(&Beta::new(), &y, &x));
}

#[test]
fn fit_weibull() {
    let x = x_grid();
    let y: Array1<f64> = (0..N)
        .map(|i| (1.0 + 1.5 * x[i]) * (1.0 + 0.2 * jitter(i)))
        .collect();
    insta::assert_yaml_snapshot!(fit_and_snapshot(&Weibull::new(), &y, &x));
}

#[test]
fn fit_student_t() {
    // ν must be intercept-only (the KKT projection at the ν floor is exact only
    // for a constant η_ν — `student_t.rs:140-146`), which `formula_for` gives.
    let x = x_grid();
    let y: Array1<f64> = (0..N).map(|i| 1.0 + 2.0 * x[i] + 0.5 * jitter(i)).collect();
    insta::assert_yaml_snapshot!(fit_and_snapshot(&StudentT::new(), &y, &x));
}

// ---------------------------------------------------------------------------
// Box-Cox family
// ---------------------------------------------------------------------------

// The Box-Cox trio is the one place a closed-form fixture does not work. BCT
// and BCPE estimate a fourth, tail-shape parameter (τ), which is unidentifiable
// from smooth deterministic jitter: τ drifts toward its boundary and the fit
// stops at the iteration limit. A non-converged snapshot would pin
// iteration-limit behavior rather than a fitted optimum, so these three draw
// from the actual family via the shared seeded generators — the same
// `Generator::new(42)` pattern `tests/regression.rs` already uses.

#[test]
fn fit_bccg() {
    let (y, data) = Generator::new(42).bccg_data(200, 0.0, 1.0, 0.2, 0.5);
    insta::assert_yaml_snapshot!(fit_and_snapshot_with_data(&BCCG::new(), &y, &data));
}

#[test]
fn fit_bct() {
    // This snapshot records `converged: false`, and that is pre-existing and
    // expected, not a defect in the fixture: near the normal limit τ is weakly
    // identified, so the deviance keeps wiggling above tolerance even though
    // every estimate lands on target. `bct_recovers_known_parameters`
    // (`tests/boxcox_families.rs:32-35`) documents the same behavior and asserts
    // recovery rather than the flag.
    //
    // The estimates here confirm it: σ = exp(-1.634) = 0.195 against a true 0.2,
    // τ = exp(1.916) = 6.8 against a true 6.0, slope 1.06 against a true 1.0.
    // The coefficients are what this file pins, and they are stable.
    let (y, data) = Generator::new(42).bct_data(200, 0.0, 1.0, 0.2, 0.5, 6.0);
    insta::assert_yaml_snapshot!(fit_and_snapshot_with_data(&BCT::new(), &y, &data));
}

#[test]
fn fit_bcpe() {
    let (y, data) = Generator::new(42).bcpe_data(200, 0.0, 1.0, 0.2, 0.5, 2.0);
    insta::assert_yaml_snapshot!(fit_and_snapshot_with_data(&BCPE::new(), &y, &data));
}

// ---------------------------------------------------------------------------
// Ordered categorical
// ---------------------------------------------------------------------------

#[test]
fn fit_ocat() {
    let x = x_grid();
    let y: Array1<f64> = (0..N)
        .map(|i| {
            let latent = -1.0 + 3.0 * x[i] + 0.5 * jitter(i);
            if latent < -0.5 {
                1.0
            } else if latent < 0.5 {
                2.0
            } else if latent < 1.2 {
                3.0
            } else {
                4.0
            }
        })
        .collect();
    insta::assert_yaml_snapshot!(fit_and_snapshot(&Ocat::new(4), &y, &x));
}

// ---------------------------------------------------------------------------
// Structural wrappers
// ---------------------------------------------------------------------------

#[test]
fn fit_censored_gaussian() {
    let x = x_grid();
    let y: Array1<f64> = (0..N).map(|i| 1.0 + 2.0 * x[i] + 0.4 * jitter(i)).collect();
    // Every third row right-censored: a deterministic, reproducible pattern.
    let status = Array1::from_vec(
        (0..N)
            .map(|i| {
                if i % 3 == 0 {
                    CensorStatus::Right
                } else {
                    CensorStatus::Event
                }
            })
            .collect(),
    );
    let family = Censored::new(Box::new(Gaussian::new()), status);
    insta::assert_yaml_snapshot!(fit_and_snapshot(&family, &y, &x));
}

#[test]
fn fit_truncated_gaussian() {
    let x = x_grid();
    let y: Array1<f64> = (0..N).map(|i| 1.0 + 2.0 * x[i] + 0.4 * jitter(i)).collect();
    let lower = Array1::from_elem(N, -1.0);
    let upper = Array1::from_elem(N, 6.0);
    let family = Truncated::new(Box::new(Gaussian::new()), lower, upper);
    insta::assert_yaml_snapshot!(fit_and_snapshot(&family, &y, &x));
}

#[test]
fn fit_hurdle_gamma() {
    let x = x_grid();
    // Every fourth row is a structural zero; the rest are strictly positive.
    let y: Array1<f64> = (0..N)
        .map(|i| {
            if i % 4 == 0 {
                0.0
            } else {
                (1.0 + 2.0 * x[i]) * (1.0 + 0.15 * jitter(i))
            }
        })
        .collect();
    let family = Hurdle::new(Box::new(Gamma::new()));
    insta::assert_yaml_snapshot!(fit_and_snapshot(&family, &y, &x));
}

// ---------------------------------------------------------------------------
// REML criterion
// ---------------------------------------------------------------------------

#[test]
fn fit_gaussian_pspline_reml() {
    // `tests/regression.rs` pins every snapshot to GCV. This covers the default
    // criterion (REML) on a penalized fit, so a change that moves λ selection
    // has somewhere to show up.
    let x = x_grid();
    let y: Array1<f64> = (0..N)
        .map(|i| (2.0 * std::f64::consts::PI * x[i]).sin() + 0.2 * jitter(i))
        .collect();
    let data = dataset(&x);
    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 10,
                degree: 3,
                penalty_order: 2,
                range: None,
            }),
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let family = Gaussian::new();
    let model = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &formula,
        &family,
        FitConfig::default(), // criterion: Reml
    )
    .unwrap();
    insta::assert_yaml_snapshot!(snapshot(&model, &family, &y));
}
