// Golden characterization tables for `Distribution::derivatives`.
//
// PURPOSE. These snapshots freeze the exact `(score, weight)` arrays every
// family returns today, per parameter, at a fixed fixture. They exist to gate
// the generic-link-chain-rule refactor (Altitude #1): that refactor moves the
// `dμ/dη` chain rule out of each family and into `fitting/scoring.rs`, and it
// is *required* to leave the default-link numbers unchanged. Any drift here is
// therefore a defect, not a snapshot to re-accept.
//
// WHY SNAPSHOTS RATHER THAN A FINITE-DIFFERENCE ORACLE. The score `u` already
// has finite-difference coverage in each family's unit tests, but the Fisher
// weight `w` does not, and `w` is where the *squared* chain rule lives. `w`
// cannot be finite-differenced generically: several families deliberately
// return *expected* information (StudentT μ, `student_t.rs:102-110`) or a
// squared-score surrogate (NegBinomial σ, `negative_binomial.rs:94-100`) rather
// than the observed `−∂²l/∂η²`, so a second difference of `loglik_pointwise` is
// simply a different quantity. Characterizing the current values sidesteps the
// theory entirely and is exact.
//
// PRECISION. Values are formatted to 11 significant digits. The refactor
// reassociates floating-point operations — `w_σ = 2.0` becomes
// `σ²·(2/σ²)`, which is not bit-identical — so bit-exactness is the wrong bar.
// Reassociation moves the last ~1-2 digits (~1e-16 relative); a wrong chain
// rule moves the value by orders of magnitude. 11 digits sits far above the
// noise and far below any real algebraic change.
//
// First-time creation: `INSTA_UPDATE=auto cargo test --test derivative_golden`,
// then `cargo insta accept`.
#![cfg(not(feature = "python"))]
#![cfg(not(target_arch = "wasm32"))]

use glissando::distributions::{Beta, Binomial};
use glissando::distributions::{
    CensorStatus, Censored, Distribution, Gamma, Gaussian, Hurdle, NegativeBinomial, Ocat, Poisson,
    StudentT, Truncated, Weibull, BCCG, BCPE, BCT,
};
use ndarray::{array, Array1};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// A frozen `(score, weight)` table for one family at one fixture.
///
/// `y` and `params` are snapshotted alongside the outputs so the fixture is
/// legible in the `.snap` file and a fixture edit is visible in the diff rather
/// than showing up as unexplained output drift.
#[derive(Debug, Serialize)]
struct DerivativeGolden {
    family: String,
    y: Vec<String>,
    params: BTreeMap<String, Vec<String>>,
    /// Score `u` per parameter.
    scores: BTreeMap<String, Vec<String>>,
    /// Fisher / working weight `w` per parameter.
    weights: BTreeMap<String, Vec<String>>,
}

fn fmt(x: f64) -> String {
    format!("{:.10e}", x)
}

fn fmt_vec(v: &Array1<f64>) -> Vec<String> {
    v.iter().map(|&x| fmt(x)).collect()
}

/// Evaluate `family.derivatives` at the fixture and package it for snapshotting.
///
/// Iterates `family.parameters()` rather than the returned map's keys so a
/// family that silently stops emitting a parameter fails loudly here.
fn golden<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    owned: &[(&'static str, Array1<f64>)],
) -> DerivativeGolden {
    let params: HashMap<&str, &Array1<f64>> = owned.iter().map(|(k, v)| (*k, v)).collect();
    let derivs = family
        .derivatives(y, &params)
        .expect("derivatives must succeed on a valid fixture");

    let mut scores = BTreeMap::new();
    let mut weights = BTreeMap::new();
    for &name in family.parameters() {
        let (u, w) = derivs
            .get(name)
            .unwrap_or_else(|| panic!("{}: no derivatives entry for '{}'", family.name(), name));
        assert_eq!(u.len(), y.len(), "{}::{} score length", family.name(), name);
        assert_eq!(
            w.len(),
            y.len(),
            "{}::{} weight length",
            family.name(),
            name
        );
        scores.insert(name.to_string(), fmt_vec(u));
        weights.insert(name.to_string(), fmt_vec(w));
    }

    DerivativeGolden {
        family: family.name().to_string(),
        y: fmt_vec(y),
        params: owned
            .iter()
            .map(|(k, v)| (k.to_string(), fmt_vec(v)))
            .collect(),
        scores,
        weights,
    }
}

// ---------------------------------------------------------------------------
// Simple families
// ---------------------------------------------------------------------------

#[test]
fn golden_gaussian() {
    // Mixed residual signs and a spread of σ, so neither the score nor the
    // constant σ-weight can be reproduced by accident.
    let y = array![-1.5, 0.0, 1.0, 2.5, -0.25, 3.0];
    let owned = [
        ("mu", array![-0.5, 0.25, 0.5, 2.0, 0.0, 2.75]),
        ("sigma", array![1.0, 1.5, 0.8, 0.5, 2.0, 1.25]),
    ];
    insta::assert_yaml_snapshot!(golden(&Gaussian::new(), &y, &owned));
}

#[test]
fn golden_poisson() {
    let y = array![0.0, 1.0, 2.0, 5.0, 9.0, 3.0];
    let owned = [("mu", array![0.5, 1.0, 2.0, 4.0, 8.0, 3.5])];
    insta::assert_yaml_snapshot!(golden(&Poisson::new(), &y, &owned));
}

#[test]
fn golden_binomial() {
    // n_trials = 10, successes spanning both boundaries and the interior.
    let y = array![0.0, 1.0, 5.0, 9.0, 10.0, 3.0];
    let owned = [("mu", array![0.1, 0.2, 0.5, 0.85, 0.95, 0.4])];
    insta::assert_yaml_snapshot!(golden(&Binomial::new(10), &y, &owned));
}

#[test]
fn golden_gamma() {
    let y = array![0.5, 1.0, 2.0, 4.0, 0.25, 3.0];
    let owned = [
        ("mu", array![1.0, 1.5, 2.0, 3.0, 0.5, 2.5]),
        ("sigma", array![0.5, 0.8, 1.0, 1.2, 0.3, 0.9]),
    ];
    insta::assert_yaml_snapshot!(golden(&Gamma::new(), &y, &owned));
}

#[test]
fn golden_negative_binomial() {
    // σ carries the squared-score weight convention (`negative_binomial.rs:100`),
    // which the refactor must reproduce via `i_σ := (∂l/∂σ)²`.
    let y = array![0.0, 1.0, 3.0, 7.0, 12.0, 2.0];
    let owned = [
        ("mu", array![1.0, 2.0, 3.0, 6.0, 10.0, 2.5]),
        ("sigma", array![0.5, 0.8, 1.0, 0.3, 1.5, 0.6]),
    ];
    insta::assert_yaml_snapshot!(golden(&NegativeBinomial::new(), &y, &owned));
}

#[test]
fn golden_beta() {
    let y = array![0.1, 0.25, 0.5, 0.75, 0.9, 0.4];
    let owned = [
        ("mu", array![0.2, 0.3, 0.5, 0.7, 0.85, 0.45]),
        ("phi", array![2.0, 5.0, 10.0, 3.0, 8.0, 4.0]),
    ];
    insta::assert_yaml_snapshot!(golden(&Beta::new(), &y, &owned));
}

#[test]
fn golden_student_t() {
    // ν kept clear of NU_FLOOR so this fixture exercises the interior chain
    // rule, not the KKT boundary projection (covered separately below).
    let y = array![-2.0, -0.5, 0.0, 1.5, 3.0, 0.75];
    let owned = [
        ("mu", array![-1.0, 0.0, 0.25, 1.0, 2.5, 0.5]),
        ("sigma", array![1.0, 1.5, 0.8, 1.2, 2.0, 0.9]),
        ("nu", array![5.0, 8.0, 4.0, 12.0, 6.0, 20.0]),
    ];
    insta::assert_yaml_snapshot!(golden(&StudentT::new(), &y, &owned));
}

#[test]
fn golden_student_t_at_nu_floor() {
    // Pins the aggregate KKT projection at the ν floor (`student_t.rs:147-161`):
    // every row is pinned, so the frozen-vs-lift-off branch is what is being
    // characterized. StudentT keeps this logic as an `eta_derivatives` override,
    // so this table must stay bit-stable across the whole refactor.
    let y = array![-3.0, -1.0, 0.0, 1.0, 4.0, 0.5];
    let owned = [
        ("mu", array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("sigma", array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ("nu", array![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    ];
    insta::assert_yaml_snapshot!(golden(&StudentT::new(), &y, &owned));
}

#[test]
fn golden_weibull() {
    let y = array![0.5, 1.0, 1.5, 2.5, 0.25, 3.5];
    let owned = [
        ("mu", array![1.0, 1.2, 2.0, 2.0, 0.5, 3.0]),
        ("sigma", array![1.0, 1.5, 2.0, 0.8, 2.5, 1.2]),
    ];
    insta::assert_yaml_snapshot!(golden(&Weibull::new(), &y, &owned));
}

// ---------------------------------------------------------------------------
// Box-Cox family
// ---------------------------------------------------------------------------

/// Shared Box-Cox fixture: strictly positive `y`/`μ`, and a `ν` spanning the
/// sign change (ν = 0 is the log-transform limit the code special-cases).
fn boxcox_fixture() -> (Array1<f64>, [(&'static str, Array1<f64>); 3]) {
    let y = array![0.5, 1.0, 1.8, 3.0, 0.75, 2.2];
    let owned = [
        ("mu", array![1.0, 1.2, 2.0, 2.5, 0.8, 2.0]),
        ("sigma", array![0.2, 0.3, 0.15, 0.4, 0.25, 0.35]),
        ("nu", array![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]),
    ];
    (y, owned)
}

#[test]
fn golden_bccg() {
    let (y, owned) = boxcox_fixture();
    insta::assert_yaml_snapshot!(golden(&BCCG::new(), &y, &owned));
}

#[test]
fn golden_bct() {
    let (y, base) = boxcox_fixture();
    let owned = [
        base[0].clone(),
        base[1].clone(),
        base[2].clone(),
        ("tau", array![3.0, 5.0, 8.0, 4.0, 12.0, 6.0]),
    ];
    insta::assert_yaml_snapshot!(golden(&BCT::new(), &y, &owned));
}

#[test]
fn golden_bcpe() {
    let (y, base) = boxcox_fixture();
    let owned = [
        base[0].clone(),
        base[1].clone(),
        base[2].clone(),
        ("tau", array![1.0, 1.5, 2.0, 3.0, 0.8, 4.0]),
    ];
    insta::assert_yaml_snapshot!(golden(&BCPE::new(), &y, &owned));
}

// ---------------------------------------------------------------------------
// Ordered categorical
// ---------------------------------------------------------------------------

#[test]
fn golden_ocat_4_categories() {
    // `params["mu"]` holds η, not μ (identity link on the latent scale);
    // `delta_1` is the first threshold and `delta_2..` are positive increments.
    let y = array![1.0, 2.0, 3.0, 4.0, 2.0, 1.0];
    let owned = [
        ("mu", array![-1.0, -0.25, 0.0, 0.5, 1.0, 0.25]),
        ("delta_1", array![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        ("delta_2", array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ("delta_3", array![1.5, 1.5, 1.5, 1.5, 1.5, 1.5]),
    ];
    insta::assert_yaml_snapshot!(golden(&Ocat::new(4), &y, &owned));
}

#[test]
fn golden_ocat_5_categories() {
    // Closes the `delta_4` coverage gap: with only R=4 fixtures in the unit
    // tests, the fourth threshold's log-link arm had no derivative test at all.
    let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 3.0];
    let owned = [
        ("mu", array![-1.5, -0.5, 0.0, 0.75, 1.5, 0.25]),
        ("delta_1", array![-1.5, -1.5, -1.5, -1.5, -1.5, -1.5]),
        ("delta_2", array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ("delta_3", array![1.25, 1.25, 1.25, 1.25, 1.25, 1.25]),
        ("delta_4", array![0.75, 0.75, 0.75, 0.75, 0.75, 0.75]),
    ];
    insta::assert_yaml_snapshot!(golden(&Ocat::new(5), &y, &owned));
}

// ---------------------------------------------------------------------------
// Structural wrappers
// ---------------------------------------------------------------------------

/// Base fixture shared by the wrapper tables, so a wrapper's table can be read
/// directly against `golden_gaussian`'s.
fn wrapper_base_fixture() -> (Array1<f64>, [(&'static str, Array1<f64>); 2]) {
    let y = array![-1.0, 0.0, 0.5, 1.5, 2.0, 3.0];
    let owned = [
        ("mu", array![0.0, 0.25, 0.5, 1.0, 1.5, 2.0]),
        ("sigma", array![1.0, 1.2, 0.8, 1.5, 1.0, 0.9]),
    ];
    (y, owned)
}

#[test]
fn golden_censored_all_statuses() {
    // Exercises all four `CensorStatus` arms in one table, including `Interval`,
    // which drives the `with_upper` path and the second-derivative difference.
    let (y, owned) = wrapper_base_fixture();
    let status = Array1::from_vec(vec![
        CensorStatus::Event,
        CensorStatus::Right,
        CensorStatus::Left,
        CensorStatus::Interval,
        CensorStatus::Event,
        CensorStatus::Right,
    ]);
    let upper = array![0.0, 0.0, 0.0, 2.5, 0.0, 0.0];
    let family = Censored::with_upper(Box::new(Gaussian::new()), status, upper);
    insta::assert_yaml_snapshot!(golden(&family, &y, &owned));
}

#[test]
fn golden_truncated() {
    let (y, owned) = wrapper_base_fixture();
    let lower = array![-2.0, -2.0, -1.0, -1.0, 0.0, 0.0];
    let upper = array![4.0, 4.0, 5.0, 5.0, 6.0, 6.0];
    let family = Truncated::new(Box::new(Gaussian::new()), lower, upper);
    insta::assert_yaml_snapshot!(golden(&family, &y, &owned));
}

#[test]
fn golden_truncated_over_gamma() {
    // Pins the NUMERIC CDF fallback on the truncation normalizer. Gamma supplies
    // an analytic `cdf_eta_derivatives` for μ only (`gamma.rs`), so σ falls
    // through to `structural.rs::numeric_cdf_grad`, which central-differences on
    // the η scale with a *relative* step for a log link (`σ·e^{±FD_EPS}`).
    // Truncated-over-Gaussian exercises only the analytic path, so without this
    // table the fallback would be unpinned — and it is the path most sensitive
    // to any change in how the perturbation is taken.
    let y = array![0.5, 1.0, 1.5, 2.0, 0.75, 2.5];
    let owned = [
        ("mu", array![1.0, 1.2, 1.5, 2.0, 0.8, 2.2]),
        ("sigma", array![0.5, 0.7, 1.0, 0.6, 0.9, 0.8]),
    ];
    let lower = array![0.1, 0.1, 0.2, 0.2, 0.05, 0.05];
    let upper = array![5.0, 5.0, 6.0, 6.0, 4.0, 4.0];
    let family = Truncated::new(Box::new(Gamma::new()), lower, upper);
    insta::assert_yaml_snapshot!(golden(&family, &y, &owned));
}

#[test]
fn golden_hurdle_over_gamma() {
    // Zeros hit the structural atom (`hurdle.rs:129-130` zeroes the base
    // contribution); positives go through the zero-truncated base.
    let y = array![0.0, 1.0, 0.0, 2.5, 0.5, 3.0];
    let owned = [
        ("mu", array![1.0, 1.5, 2.0, 2.0, 1.0, 2.5]),
        ("sigma", array![0.5, 0.8, 1.0, 0.6, 0.9, 0.7]),
        ("xi", array![0.2, 0.3, 0.5, 0.25, 0.4, 0.15]),
    ];
    let family = Hurdle::new(Box::new(Gamma::new()));
    insta::assert_yaml_snapshot!(golden(&family, &y, &owned));
}
