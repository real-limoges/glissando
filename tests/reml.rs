// REML smoothing-parameter selection — integration tests.
//
// Validates the end-to-end behaviour of `SmoothingCriterion::Reml` on Gaussian
// and Poisson P-spline models:
//
//   - Default `FitConfig` uses REML.
//   - REML fits converge and produce finite, well-conditioned output.
//   - EDF lands in the interior of the feasible range (not pinned at 0 or p).
//   - REML and GCV agree closely on well-identified data — they shouldn't be
//     identical (different criteria), but the difference should be modest.
//
// Golden-value comparison vs `mgcv::gam(method="REML")` is handled by the
// existing benchmark harness (`benchmark/run_comparison.sh` + the ignored
// `tests/mgcv_reference.rs`); duplicating that here would re-implement the
// same workflow.
#![cfg(not(feature = "python"))]
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::{pspline, Generator};
use glissando::{
    distributions::{Gaussian, Poisson},
    DataSet, FitConfig, Formula, GamlssModel, SmoothingCriterion, Term,
};
use ndarray::Array1;

/// Wiggly sinusoidal Gaussian data: a regime where the basis is genuinely needed
/// and REML/GCV pick comparable smoothing. Linear-trend data lives in the order-2
/// penalty's null space, so on such data REML correctly drives λ to ∞ (EDF ≈ 2)
/// while GCV undersmooths — the criteria *should* disagree there.
fn sinusoidal_gaussian(n: usize, noise: f64, seed: u64) -> (Array1<f64>, DataSet) {
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0, noise).unwrap();
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| {
            let mu = (2.0 * std::f64::consts::PI * t).sin() + 0.5 * (4.0 * t).cos();
            mu + noise_dist.sample(&mut rng)
        })
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));
    (Array1::from_vec(y), data)
}

#[test]
fn default_fit_config_uses_reml() {
    let cfg = FitConfig::default();
    assert_eq!(cfg.criterion, SmoothingCriterion::Reml);
}

#[test]
fn reml_fits_gaussian_pspline_end_to_end() {
    // Sinusoidal (not linear) truth: the smooth is genuinely identified, so REML
    // settles on an interior λ and EDF lands in (1, p). Linear-trend data lives in
    // the order-2 penalty's null space, leaving the REML objective flat in λ — the
    // smoothing parameter is then unidentified and the outer loop's convergence
    // becomes BLAS-dependent. (The intended null-space behaviour is covered by
    // `reml_correctly_smooths_linear_trend_to_null_space`.)
    let (y, data) = sinusoidal_gaussian(200, 0.2, 42);

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, pspline("x", 10)])
        .with_terms("sigma", vec![Term::Intercept]);

    let cfg = FitConfig {
        criterion: SmoothingCriterion::Reml,
        ..FitConfig::default()
    };
    let model = GamlssModel::fit_with_config(&data, &y, &formula, &Gaussian::new(), cfg).unwrap();

    assert!(model.converged(), "REML fit failed to converge");

    let mu = &model.models["mu"];
    assert!(mu.coefficients.0.iter().all(|c| c.is_finite()));
    assert!(mu.edf > 1.0, "EDF too small (over-smoothed): {}", mu.edf);
    // mu has 1 intercept + (n_splines - 1) sum-to-zero spline columns ≈ 10 cols total.
    assert!(
        mu.edf < 10.0,
        "EDF at basis ceiling (unpenalized): {}",
        mu.edf
    );
    assert!(mu.lambdas[0] > 0.0 && mu.lambdas[0].is_finite());
}

#[test]
fn reml_fits_poisson_pspline_end_to_end() {
    let mut rng = Generator::new(7);
    let (y, data) = rng.poisson_data(150, 0.5, 0.3);

    let formula = Formula::new().with_terms("mu", vec![Term::Intercept, pspline("x", 8)]);
    let cfg = FitConfig {
        criterion: SmoothingCriterion::Reml,
        ..FitConfig::default()
    };
    let model = GamlssModel::fit_with_config(&data, &y, &formula, &Poisson::new(), cfg).unwrap();

    assert!(model.converged(), "REML Poisson fit failed to converge");
    let mu = &model.models["mu"];
    assert!(mu.coefficients.0.iter().all(|c| c.is_finite()));
    assert!(
        mu.edf > 1.0 && mu.edf < 8.0,
        "Poisson EDF out of range: {}",
        mu.edf
    );
    assert!(mu.lambdas[0] > 0.0 && mu.lambdas[0].is_finite());
}

#[test]
fn reml_correctly_smooths_linear_trend_to_null_space() {
    // Order-2 P-spline penalty has a 2-dim null space (constants + linear).
    // When the truth is linear, REML correctly drives λ very high so the fit
    // collapses to its null space, giving EDF ≈ 2.  GCV is documented to
    // undersmooth in this regime — we don't assert anything about GCV here,
    // only that REML lands at the principled solution.
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(150, 1.0, 5.0, 1.0);

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, pspline("x", 10)])
        .with_terms("sigma", vec![Term::Intercept]);

    let reml = GamlssModel::fit_with_config(
        &data,
        &y,
        &formula,
        &Gaussian::new(),
        FitConfig {
            criterion: SmoothingCriterion::Reml,
            ..FitConfig::default()
        },
    )
    .unwrap();

    let edf = reml.models["mu"].edf;
    assert!(
        edf < 3.0,
        "REML should collapse to penalty null space on linear data (EDF ≈ 2), got {}",
        edf
    );
}

#[test]
fn reml_and_gcv_agree_on_wiggly_truth() {
    // On data that genuinely needs the basis (sinusoidal truth), both criteria
    // should pick comparable smoothing and produce indistinguishable predictions.
    let (y, data) = sinusoidal_gaussian(200, 0.2, 42);

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, pspline("x", 15)])
        .with_terms("sigma", vec![Term::Intercept]);

    let reml = GamlssModel::fit_with_config(
        &data,
        &y,
        &formula,
        &Gaussian::new(),
        FitConfig {
            criterion: SmoothingCriterion::Reml,
            ..FitConfig::default()
        },
    )
    .unwrap();
    let gcv = GamlssModel::fit_with_config(
        &data,
        &y,
        &formula,
        &Gaussian::new(),
        FitConfig {
            criterion: SmoothingCriterion::Gcv,
            ..FitConfig::default()
        },
    )
    .unwrap();

    let edf_diff = (reml.models["mu"].edf - gcv.models["mu"].edf).abs();
    assert!(
        edf_diff < 3.0,
        "REML EDF ({}) and GCV EDF ({}) disagree by more than 3.0 on wiggly truth",
        reml.models["mu"].edf,
        gcv.models["mu"].edf
    );

    let pred_reml = reml.predict(&data, &Gaussian::new()).unwrap();
    let pred_gcv = gcv.predict(&data, &Gaussian::new()).unwrap();
    let max_abs: f64 = pred_reml["mu"]
        .iter()
        .zip(pred_gcv["mu"].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs < 0.3,
        "REML/GCV μ predictions differ by more than 0.3 in abs (max {})",
        max_abs
    );
}

#[test]
fn fellner_schall_fits_gaussian_pspline_end_to_end() {
    // Sinusoidal truth so F-S has a finite interior optimum.  On linear-trend
    // data, the order-2 P-spline penalty's null space already captures the
    // truth, and any LAML-target optimizer (REML or F-S) will legitimately
    // drive λ to its ceiling — that's covered by REML's own null-space test.
    let (y, data) = sinusoidal_gaussian(200, 0.2, 42);

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, pspline("x", 15)])
        .with_terms("sigma", vec![Term::Intercept]);

    let cfg = FitConfig {
        criterion: SmoothingCriterion::FellnerSchall,
        ..FitConfig::default()
    };
    let model = GamlssModel::fit_with_config(&data, &y, &formula, &Gaussian::new(), cfg).unwrap();

    assert!(model.converged(), "F-S fit failed to converge");
    let mu = &model.models["mu"];
    assert!(mu.coefficients.0.iter().all(|c| c.is_finite()));
    assert!(
        mu.edf > 2.0 && mu.edf < 15.0,
        "F-S EDF out of range: {}",
        mu.edf
    );
    assert!(mu.lambdas[0] > 0.0 && mu.lambdas[0].is_finite());
}

#[test]
fn fellner_schall_fits_poisson_pspline_end_to_end() {
    let mut rng = Generator::new(7);
    let (y, data) = rng.poisson_data(150, 0.5, 0.3);

    let formula = Formula::new().with_terms("mu", vec![Term::Intercept, pspline("x", 8)]);
    let cfg = FitConfig {
        criterion: SmoothingCriterion::FellnerSchall,
        ..FitConfig::default()
    };
    let model = GamlssModel::fit_with_config(&data, &y, &formula, &Poisson::new(), cfg).unwrap();

    assert!(model.converged(), "F-S Poisson fit failed to converge");
    let mu = &model.models["mu"];
    assert!(mu.coefficients.0.iter().all(|c| c.is_finite()));
    assert!(
        mu.edf > 1.0 && mu.edf < 8.0,
        "F-S Poisson EDF out of range: {}",
        mu.edf
    );
    assert!(mu.lambdas[0] > 0.0 && mu.lambdas[0].is_finite());
}

#[test]
fn fellner_schall_and_reml_converge_to_similar_lambda() {
    // F-S and REML optimize the *same* objective (LAML); on a well-behaved
    // wiggly truth they should land within ~one order of magnitude in λ and
    // within ~1 EDF.  Tighter agreement than REML/GCV because the target
    // is identical, only the optimizer differs.
    let (y, data) = sinusoidal_gaussian(200, 0.2, 42);

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, pspline("x", 15)])
        .with_terms("sigma", vec![Term::Intercept]);

    let reml = GamlssModel::fit_with_config(
        &data,
        &y,
        &formula,
        &Gaussian::new(),
        FitConfig {
            criterion: SmoothingCriterion::Reml,
            ..FitConfig::default()
        },
    )
    .unwrap();
    let fs = GamlssModel::fit_with_config(
        &data,
        &y,
        &formula,
        &Gaussian::new(),
        FitConfig {
            criterion: SmoothingCriterion::FellnerSchall,
            ..FitConfig::default()
        },
    )
    .unwrap();

    let edf_diff = (reml.models["mu"].edf - fs.models["mu"].edf).abs();
    assert!(
        edf_diff < 1.5,
        "F-S EDF ({}) and REML EDF ({}) disagree by more than 1.5",
        fs.models["mu"].edf,
        reml.models["mu"].edf
    );

    let log_ratio =
        (fs.models["mu"].lambdas[0].ln() - reml.models["mu"].lambdas[0].ln()).abs() / 10f64.ln();
    assert!(
        log_ratio < 1.0,
        "F-S λ ({}) and REML λ ({}) disagree by more than 1 order of magnitude",
        fs.models["mu"].lambdas[0],
        reml.models["mu"].lambdas[0]
    );
}

#[test]
fn fellner_schall_dispatch_is_distinct_from_gcv() {
    // Proves the F-S match arm is wired (not aliased to GCV).
    let mut rng = Generator::new(123);
    let (y, data) = rng.linear_gaussian(80, 1.0, 5.0, 1.0);
    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, pspline("x", 8)])
        .with_terms("sigma", vec![Term::Intercept]);

    let fs = GamlssModel::fit_with_config(
        &data,
        &y,
        &formula,
        &Gaussian::new(),
        FitConfig {
            criterion: SmoothingCriterion::FellnerSchall,
            ..FitConfig::default()
        },
    )
    .unwrap();
    let gcv = GamlssModel::fit_with_config(
        &data,
        &y,
        &formula,
        &Gaussian::new(),
        FitConfig {
            criterion: SmoothingCriterion::Gcv,
            ..FitConfig::default()
        },
    )
    .unwrap();

    assert!(
        (fs.models["mu"].lambdas[0] - gcv.models["mu"].lambdas[0]).abs() > 1e-12,
        "F-S and GCV produced identical λ — dispatch may not be wired"
    );
}

#[test]
fn criterion_dispatch_is_observable() {
    // Sanity: changing the criterion should affect *something* — either λ or
    // iteration count. If they're byte-for-byte identical the dispatch is dead.
    let mut rng = Generator::new(123);
    let (y, data) = rng.linear_gaussian(80, 1.0, 5.0, 1.0);

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, pspline("x", 8)])
        .with_terms("sigma", vec![Term::Intercept]);

    let reml = GamlssModel::fit_with_config(
        &data,
        &y,
        &formula,
        &Gaussian::new(),
        FitConfig {
            criterion: SmoothingCriterion::Reml,
            ..FitConfig::default()
        },
    )
    .unwrap();
    let gcv = GamlssModel::fit_with_config(
        &data,
        &y,
        &formula,
        &Gaussian::new(),
        FitConfig {
            criterion: SmoothingCriterion::Gcv,
            ..FitConfig::default()
        },
    )
    .unwrap();

    let lambdas_differ = (reml.models["mu"].lambdas[0] - gcv.models["mu"].lambdas[0]).abs() > 1e-12;
    let iters_differ = reml.diagnostics.iterations != gcv.diagnostics.iterations;
    assert!(
        lambdas_differ || iters_differ,
        "REML and GCV produced identical λ and identical iteration count — \
         dispatch may not be wired through"
    );
}
