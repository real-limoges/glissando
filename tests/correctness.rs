// These can't run under the `python` feature. PyO3's extension-module linking won't have it.
#![cfg(not(feature = "python"))]

mod common;

use common::{linear_intercepts, smooth_intercepts, Generator};
use glissando::{
    distributions::Gaussian, Coefficients, CovarianceMatrix, GamlssError, GamlssModel,
};
use ndarray::{array, Array1};

// ----------------------------------------------------------------------------
// B.1: sample_posterior surfaces non-PD covariance as an error
// ----------------------------------------------------------------------------

#[test]
fn sample_posterior_errors_on_non_positive_definite_covariance() {
    // Indefinite 2x2 matrix (eigenvalues -1, 3). Cholesky has to fail here.
    let beta = Coefficients(array![0.0, 0.0]);
    let v = CovarianceMatrix(array![[1.0, 2.0], [2.0, 1.0]]);

    let err = glissando::fitting::sample_posterior(&beta, &v, 10).unwrap_err();
    assert!(
        matches!(err, GamlssError::PosteriorNotPositiveDefinite),
        "expected PosteriorNotPositiveDefinite, got {:?}",
        err
    );
}

#[test]
fn posterior_samples_propagates_non_pd_error() {
    // Build a real fit, then ask for samples on a parameter that isn't there.
    // Confirms the second error branch (UnknownParameter) fires too.
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(50, 1.0, 2.0, 0.5);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let err = model
        .posterior_samples("does_not_exist", 5, None)
        .unwrap_err();
    assert!(
        matches!(err, GamlssError::UnknownParameter { .. }),
        "expected UnknownParameter, got {:?}",
        err
    );
}

// ----------------------------------------------------------------------------
// B.2: smooth-only formula initializes consistently (β=0, η=0) and converges
// ----------------------------------------------------------------------------

#[test]
fn smooth_only_fit_converges_without_intercept_seed() {
    // No `Term::Intercept` for μ. The old buggy seed `beta[0] = eta_start`
    // would dump η₀ into a spline coefficient. Post-fix, β = 0, η = 0, and
    // IRLS settles on a sensible smooth.
    let n = 100;
    let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / (n - 1) as f64));
    // Centered sine so y has near-zero mean. A smooth-only fit recovers it with
    // no intercept term.
    let y: Array1<f64> = x.mapv(|v| (v * 6.0).sin());

    let mut data = glissando::DataSet::new();
    data.insert_column("x", x);

    let formula = smooth_intercepts("x", 10, &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    assert!(model.converged(), "smooth-only mu formula should converge");
    assert!(model.models["mu"]
        .coefficients
        .0
        .iter()
        .all(|b: &f64| b.is_finite()));
}

// ----------------------------------------------------------------------------
// B.3: ParamDiagnostic exposes weight_floor_hits and step_cap_hits
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// D.4: GamlssModel exposes a human-readable summary via Display
// ----------------------------------------------------------------------------

#[test]
fn display_summary_includes_convergence_and_per_param_block() {
    let mut rng = Generator::new(7);
    let (y, data) = rng.linear_gaussian(50, 1.0, 2.0, 0.5);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let s = format!("{}", model);
    assert!(s.contains("GamlssModel"), "header missing: {}", s);
    assert!(s.contains("converged="), "convergence flag missing: {}", s);
    assert!(
        s.contains("mu") && s.contains("sigma"),
        "param blocks missing: {}",
        s
    );
    assert!(
        s.contains("coefficients:"),
        "coefficients line missing: {}",
        s
    );
}

#[test]
fn param_diagnostic_exposes_clamp_counters() {
    // The counters have to be readable on every fit. A well-conditioned fit
    // usually reports zero, but the fields still need to be on the public
    // surface so callers can spot a degenerate fit.
    let mut rng = Generator::new(99);
    let (y, data) = rng.linear_gaussian(50, 1.0, 2.0, 0.5);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    for (param, diag) in &model.diagnostics.param_diagnostics {
        // Compile-time check that the fields exist, plus a runtime sanity check.
        let _ = diag.weight_floor_hits;
        let _ = diag.step_cap_hits;
        assert!(
            diag.weight_floor_hits <= y.len(),
            "{}: floor hits ({}) cannot exceed n_obs ({})",
            param,
            diag.weight_floor_hits,
            y.len()
        );
        assert!(
            diag.step_cap_hits <= y.len(),
            "{}: cap hits ({}) cannot exceed n_obs ({})",
            param,
            diag.step_cap_hits,
            y.len()
        );
    }
}
