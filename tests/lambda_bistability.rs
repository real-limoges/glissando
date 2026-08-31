// Integration tests can't run under the `python` feature. PyO3's extension-module linking gets in the way.
#![cfg(not(feature = "python"))]

//! DIAGNOSTIC harness (`#[ignore]`d) for the smoothing-parameter bistability:
//! a P-spline smooth on the control case (`mu_smooth_recovers_nonlinear_mean_control`)
//! sometimes collapses onto its penalty null space (edf → ~2, a straight line)
//! instead of recovering the sine.
//!
//! The companion in-crate test `solver::reml_tests::diagnostic_laml_landscape_control_case`
//! established that the REML/LAML objective is *unimodal* on this data: global
//! min at edf ≈ 13, with the collapse region (edf ≈ 2) carrying a far worse
//! −V_r and sitting on a near-flat high-λ shelf. So collapse is not a competing
//! optimum; it is a gradient optimizer getting stuck on that flat shelf.
//!
//! What I measure here is the *behavioral* consequence: how often the default
//! (REML / L-BFGS) collapses across repeated fits, and whether `Gcv` and the
//! deterministic `FellnerSchall` land on the good interior λ instead.
//!
//! RESOLUTION (kept as a regression diagnostic): the tipping was driven entirely
//! by multi-threaded OpenBLAS reduction-order nondeterminism. This repo now pins
//! BLAS to a single thread for all `cargo` runs (`OPENBLAS_NUM_THREADS=1` in
//! `.cargo/config.toml`), so these harnesses now report a 0/N collapse rate with
//! zero corr spread, and the deterministic recovery assertions in
//! `tests/scale_smooth_recovery.rs` gate the behavior in CI. These repeated-fit
//! harnesses stay `#[ignore]`d (they are heavy) but remain runnable to confirm the
//! spread stays collapsed if the BLAS-threading config ever changes.
//!
//! Run with: `cargo test --test lambda_bistability -- --ignored --nocapture`

use glissando::{
    distributions::Gaussian, DataSet, FitConfig, Formula, GamlssModel, Smooth, SmoothingCriterion,
    Term,
};
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (da, db) = (x - mean_a, y - mean_b);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    cov / (va.sqrt() * vb.sqrt())
}

fn true_curve(x: f64) -> f64 {
    -0.7 + 0.8 * (2.0 * std::f64::consts::PI * x).sin()
}

/// Build the control-case data (seed 7, n = 8000): nonlinear mean, constant noise.
fn control_data() -> (DataSet, Array1<f64>, Vec<f64>) {
    let n = 8_000usize;
    let mut rng = StdRng::seed_from_u64(7);
    let x_vals: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| Normal::new(true_curve(x), 0.2).unwrap().sample(&mut rng))
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x_vals.clone()));
    (data, Array1::from_vec(y_vals), x_vals)
}

fn smooth_term() -> Term {
    Term::Smooth(Smooth::PSpline1D {
        col_name: "x".to_string(),
        n_splines: 15,
        degree: 3,
        penalty_order: 2,
        range: None,
    })
}

fn control_formula() -> Formula {
    let mut formula = Formula::new();
    formula.add_terms("mu".to_string(), vec![Term::Intercept, smooth_term()]);
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);
    formula
}

/// σ-smooth case (the one reported as genuinely flaky): constant mean, the
/// nonlinear curve lives on log σ.
fn sigma_data() -> (DataSet, Array1<f64>, Vec<f64>) {
    let n = 8_000usize;
    let mut rng = StdRng::seed_from_u64(7);
    let x_vals: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| {
            Normal::new(5.0, true_curve(x).exp())
                .unwrap()
                .sample(&mut rng)
        })
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x_vals.clone()));
    (data, Array1::from_vec(y_vals), x_vals)
}

fn sigma_formula() -> Formula {
    let mut formula = Formula::new();
    formula.add_terms("mu".to_string(), vec![Term::Intercept]);
    formula.add_terms("sigma".to_string(), vec![Term::Intercept, smooth_term()]);
    formula
}

/// Fit the σ-smooth case with `criterion`; return (sigma total edf, sigma smooth-term edf, corr(log σ̂, truth), warns).
fn fit_sigma_once(criterion: SmoothingCriterion) -> (f64, f64, f64, usize) {
    let (data, y, x_vals) = sigma_data();
    let formula = sigma_formula();
    let cfg = FitConfig {
        criterion,
        ..FitConfig::default()
    };
    let model = GamlssModel::fit_with_config(&data, &y, None, &formula, &Gaussian::new(), cfg)
        .expect("fit failed");
    let sigma = &model.models["sigma"];
    let smooth_edf = *sigma.term_edf.last().unwrap();
    let preds = model.predict(&data, &Gaussian::new()).unwrap();
    let fitted: Vec<f64> = preds["sigma"].iter().map(|s| s.ln()).collect();
    let truth: Vec<f64> = x_vals.iter().map(|&x| true_curve(x)).collect();
    (
        sigma.edf,
        smooth_edf,
        correlation(&fitted, &truth),
        model.diagnostics.warnings.len(),
    )
}

/// Fit once with `criterion` and return (mu total edf, mu smooth-term edf, corr, n_warnings).
fn fit_once(criterion: SmoothingCriterion) -> (f64, f64, f64, usize) {
    let (data, y, x_vals) = control_data();
    let formula = control_formula();
    let cfg = FitConfig {
        criterion,
        ..FitConfig::default()
    };
    let model = GamlssModel::fit_with_config(&data, &y, None, &formula, &Gaussian::new(), cfg)
        .expect("fit failed");
    let mu = &model.models["mu"];
    let smooth_edf = *mu.term_edf.last().unwrap();
    let preds = model.predict(&data, &Gaussian::new()).unwrap();
    let fitted: Vec<f64> = preds["mu"].to_vec();
    let truth: Vec<f64> = x_vals.iter().map(|&x| true_curve(x)).collect();
    (
        mu.edf,
        smooth_edf,
        correlation(&fitted, &truth),
        model.diagnostics.warnings.len(),
    )
}

/// Q1: collapse frequency / determinism of the default REML path across repeats.
#[test]
#[ignore = "diagnostic: measures REML collapse frequency over repeated fits"]
fn diagnostic_reml_collapse_frequency() {
    const REPEATS: usize = 20;
    eprintln!("REML (default) — {REPEATS} repeated fits of the control case:");
    eprintln!(
        "{:>4}  {:>10}  {:>12}  {:>8}  {:>6}",
        "run", "edf", "smooth_edf", "corr", "warns"
    );
    let mut collapses = 0usize;
    let mut corrs = Vec::new();
    for r in 0..REPEATS {
        let (edf, smooth_edf, corr, warns) = fit_once(SmoothingCriterion::Reml);
        let collapsed = corr <= 0.95;
        if collapsed {
            collapses += 1;
        }
        corrs.push(corr);
        eprintln!(
            "{r:>4}  {edf:>10.3}  {smooth_edf:>12.3}  {corr:>8.4}  {warns:>6}{}",
            if collapsed { "  <-- COLLAPSE" } else { "" }
        );
    }
    let min = corrs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = corrs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "\nREML collapse rate: {collapses}/{REPEATS};  corr range [{min:.4}, {max:.4}]  \
         (spread ⇒ backend-nondeterministic tipping)"
    );
}

/// Q3: cross-method comparison, does GCV / Fellner-Schall land on the interior λ?
#[test]
#[ignore = "diagnostic: compares GCV / REML / Fellner-Schall on the control case"]
fn diagnostic_criterion_comparison() {
    eprintln!("Control case — one fit per criterion (interior recovery ⇒ edf ≈ 13, corr > 0.95):");
    eprintln!(
        "{:>16}  {:>10}  {:>12}  {:>8}  {:>6}",
        "criterion", "edf", "smooth_edf", "corr", "warns"
    );
    for (name, crit) in [
        ("Gcv", SmoothingCriterion::Gcv),
        ("Reml", SmoothingCriterion::Reml),
        ("FellnerSchall", SmoothingCriterion::FellnerSchall),
    ] {
        let (edf, smooth_edf, corr, warns) = fit_once(crit);
        eprintln!("{name:>16}  {edf:>10.3}  {smooth_edf:>12.3}  {corr:>8.4}  {warns:>6}");
    }

    // Collapse rate on the μ-control for REML vs F-S over many repeats. The
    // nondeterminism is BLAS reduction order, so repeats on one machine sample it.
    eprintln!("\nμ-control collapse rate (30 repeats each, collapse ⇒ corr ≤ 0.95):");
    for (name, crit) in [
        ("Reml", SmoothingCriterion::Reml),
        ("FellnerSchall", SmoothingCriterion::FellnerSchall),
    ] {
        let mut collapses = 0usize;
        let mut min = f64::INFINITY;
        for _ in 0..30 {
            let (_e, _se, corr, _w) = fit_once(crit);
            if corr <= 0.95 {
                collapses += 1;
            }
            min = min.min(corr);
        }
        eprintln!("  {name:>14}: {collapses:>2}/30 collapsed,  worst corr = {min:.4}");
    }
}

/// Q1': collapse frequency on the genuinely-flaky σ-smooth case, per criterion.
#[test]
#[ignore = "diagnostic: σ-smooth collapse frequency across criteria"]
fn diagnostic_sigma_collapse_frequency() {
    const REPEATS: usize = 12;
    for (name, crit) in [
        ("Gcv", SmoothingCriterion::Gcv),
        ("Reml", SmoothingCriterion::Reml),
        ("FellnerSchall", SmoothingCriterion::FellnerSchall),
    ] {
        let mut collapses = 0usize;
        let mut corrs = Vec::new();
        for _ in 0..REPEATS {
            let (_edf, _se, corr, _w) = fit_sigma_once(crit);
            if corr <= 0.8 {
                collapses += 1;
            }
            corrs.push(corr);
        }
        let min = corrs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = corrs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        eprintln!(
            "{name:>16}: collapse {collapses:>2}/{REPEATS},  corr range [{min:.4}, {max:.4}]"
        );
    }
}
