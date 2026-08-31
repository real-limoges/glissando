// These can't run under the `python` feature. PyO3's extension-module linking won't have it.
#![cfg(not(feature = "python"))]

mod common;

use common::{intercept_only, linear, linear_intercepts, smooth_intercepts, Generator};
use glissando::{
    distributions::{Gaussian, Poisson},
    selection::{ic_table, lr_test, step_gaic, Direction, StepResult, StepScope},
    FitConfig, Formula, GamlssError, GamlssModel, Term,
};

// ----------------------------------------------------------------------------
// INFER-7: ic_table
// ----------------------------------------------------------------------------

#[test]
fn ic_table_ranks_better_fit_below_worse_fit() {
    let mut rng = Generator::new(42);
    // y depends on x1, so the null (intercept-only) model fits worse than mu ~ x1.
    let (y, data) = rng.gaussian_with_noise_columns(200, 4.0, 1.0, 0.5);

    let null = intercept_only(&["mu", "sigma"]);
    let with_x1 = linear_intercepts("x1", &["mu", "sigma"]);
    let m_null = GamlssModel::fit(&data, &y, &null, &Gaussian::new()).unwrap();
    let m_x1 = GamlssModel::fit(&data, &y, &with_x1, &Gaussian::new()).unwrap();

    let rows = ic_table(
        &[("null", &m_null), ("with_x1", &m_x1)],
        &Gaussian::new(),
        &y,
        2.0,
    )
    .unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].label, "null");
    assert_eq!(rows[1].label, "with_x1");
    // The predictive model wins on deviance and GAIC even though it costs EDF.
    assert!(
        rows[1].global_deviance < rows[0].global_deviance,
        "with_x1 deviance {} should be < null {}",
        rows[1].global_deviance,
        rows[0].global_deviance
    );
    assert!(
        rows[1].gaic < rows[0].gaic,
        "with_x1 GAIC {} should be < null {}",
        rows[1].gaic,
        rows[0].gaic
    );
    // global_deviance == -2·loglik, and GAIC(2) == deviance + 2·edf. Both hold.
    for r in &rows {
        assert!((r.gaic - (r.global_deviance + 2.0 * r.edf)).abs() < 1e-9);
        assert!(r.edf > 0.0);
    }
}

// ----------------------------------------------------------------------------
// INFER-7: lr_test
// ----------------------------------------------------------------------------

/// `mu ~ 1 + Linear(x1) + Linear(noise_col)`, `sigma ~ 1`: the correct model
/// (`linear_intercepts("x1", ..)`) with one extra noise predictor bolted onto `mu`.
fn mu_x1_plus_noise(noise_col: &str) -> Formula {
    let mut f = Formula::new();
    f.add_terms(
        "mu".to_string(),
        vec![Term::Intercept, linear("x1"), linear(noise_col)],
    );
    f.add_terms("sigma".to_string(), vec![Term::Intercept]);
    f
}

#[test]
fn lr_test_has_power_for_a_genuine_term() {
    let mut rng = Generator::new(7);
    let (y, data) = rng.gaussian_with_noise_columns(300, 4.0, 1.0, 0.5);

    let null = intercept_only(&["mu", "sigma"]);
    let alt = linear_intercepts("x1", &["mu", "sigma"]);
    let small = GamlssModel::fit(&data, &y, &null, &Gaussian::new()).unwrap();
    let big = GamlssModel::fit(&data, &y, &alt, &Gaussian::new()).unwrap();

    let test = lr_test(&small, &big, &Gaussian::new(), &y).unwrap();
    assert!(test.df > 0.0, "df should be positive, got {}", test.df);
    assert!(
        test.lr_stat > 50.0,
        "adding a genuine predictor should give a large LR, got {}",
        test.lr_stat
    );
    assert!(
        test.p_value < 1e-6,
        "genuine predictor should give a tiny p-value, got {}",
        test.p_value
    );
}

#[test]
fn lr_test_noise_term_is_weaker_than_genuine_term() {
    let mut rng = Generator::new(99);
    let (y, data) = rng.gaussian_with_noise_columns(300, 4.0, 1.0, 0.5);

    // Correct model mu ~ x1, then pile the pure-noise column x2 on top.
    let correct = linear_intercepts("x1", &["mu", "sigma"]);
    let with_noise = mu_x1_plus_noise("x2");
    let small = GamlssModel::fit(&data, &y, &correct, &Gaussian::new()).unwrap();
    let big = GamlssModel::fit(&data, &y, &with_noise, &Gaussian::new()).unwrap();

    let test = lr_test(&small, &big, &Gaussian::new(), &y).unwrap();
    // A noise column barely nudges the deviance: small LR, non-tiny p-value.
    assert!(
        test.lr_stat < 10.0,
        "noise column should give a small LR, got {}",
        test.lr_stat
    );
    assert!(
        test.p_value > 1e-3,
        "noise column should give a non-significant p-value, got {}",
        test.p_value
    );
}

#[test]
fn lr_test_rejects_misordered_pair() {
    let mut rng = Generator::new(5);
    let (y, data) = rng.gaussian_with_noise_columns(150, 4.0, 1.0, 0.5);

    let null = intercept_only(&["mu", "sigma"]);
    let alt = linear_intercepts("x1", &["mu", "sigma"]);
    let small = GamlssModel::fit(&data, &y, &null, &Gaussian::new()).unwrap();
    let big = GamlssModel::fit(&data, &y, &alt, &Gaussian::new()).unwrap();

    // Swap the arguments: `big` passed as small, `small` as big, so edf decreases.
    let err = lr_test(&big, &small, &Gaussian::new(), &y).unwrap_err();
    assert!(
        matches!(err, GamlssError::Input(_)),
        "mis-ordered pair should return Input, got {:?}",
        err
    );
}

// ----------------------------------------------------------------------------
// INFER-4: step_gaic
// ----------------------------------------------------------------------------

/// Sorted term names on `mu` in a selected model.
fn mu_term_names(r: &StepResult) -> Vec<String> {
    let mut names: Vec<String> = r
        .formula
        .get("mu")
        .expect("mu present")
        .iter()
        .map(|t| t.term_name())
        .collect();
    names.sort();
    names
}

/// Candidate scope: all three predictors are eligible to add or drop on `mu`.
fn mu_scope() -> Vec<StepScope> {
    vec![StepScope {
        param: "mu".to_string(),
        candidates: vec![linear("x1"), linear("x2"), linear("x3")],
    }]
}

#[test]
fn step_gaic_forward_selects_signal_rejects_noise() {
    let mut rng = Generator::new(2024);
    let (y, data) = rng.gaussian_with_noise_columns(300, 4.0, 1.0, 0.5);

    let start = intercept_only(&["mu", "sigma"]);
    // BIC penalty (k = log n) is the parsimonious knob: it reliably keeps the
    // genuine predictor and rejects the pure-noise columns. (AIC's k = 2 is loose
    // enough that a noise column's deviance drop sometimes clears the penalty.
    // That's expected, not a selection bug.)
    let k = (y.len() as f64).ln();
    let result = step_gaic(
        &data,
        &y,
        &Gaussian::new(),
        start,
        &mu_scope(),
        k,
        Direction::Forward,
        FitConfig::default(),
    )
    .unwrap();

    let names = mu_term_names(&result);
    assert!(
        names.contains(&"x1".to_string()),
        "x1 should be selected, got {:?}",
        names
    );
    assert!(
        !names.contains(&"x2".to_string()),
        "noise x2 should be rejected, got {:?}",
        names
    );
    assert!(
        !names.contains(&"x3".to_string()),
        "noise x3 should be rejected, got {:?}",
        names
    );
    assert!(
        !result.trace.is_empty(),
        "at least one accepted move expected"
    );
}

#[test]
fn step_gaic_backward_drops_noise_keeps_signal() {
    let mut rng = Generator::new(2024);
    let (y, data) = rng.gaussian_with_noise_columns(300, 4.0, 1.0, 0.5);

    // Start from the full model: mu ~ 1 + x1 + x2 + x3.
    let full = Formula::new()
        .with_terms(
            "mu",
            vec![Term::Intercept, linear("x1"), linear("x2"), linear("x3")],
        )
        .with_terms("sigma", vec![Term::Intercept]);

    let k = (y.len() as f64).ln(); // BIC: parsimonious, drops the noise columns.
    let result = step_gaic(
        &data,
        &y,
        &Gaussian::new(),
        full,
        &mu_scope(),
        k,
        Direction::Backward,
        FitConfig::default(),
    )
    .unwrap();

    let names = mu_term_names(&result);
    assert!(
        names.contains(&"x1".to_string()),
        "x1 should be kept, got {:?}",
        names
    );
    assert!(
        !names.contains(&"x2".to_string()),
        "noise x2 should be dropped, got {:?}",
        names
    );
    assert!(
        !names.contains(&"x3".to_string()),
        "noise x3 should be dropped, got {:?}",
        names
    );
}

#[test]
fn step_gaic_is_deterministic() {
    let mut rng = Generator::new(13);
    let (y, data) = rng.gaussian_with_noise_columns(250, 3.0, 1.0, 0.6);
    let start = intercept_only(&["mu", "sigma"]);

    let run = || {
        step_gaic(
            &data,
            &y,
            &Gaussian::new(),
            start.clone(),
            &mu_scope(),
            2.0,
            Direction::Both,
            FitConfig::default(),
        )
        .unwrap()
    };
    let r1 = run();
    let r2 = run();

    assert_eq!(
        r1.trace.len(),
        r2.trace.len(),
        "trace length must be reproducible"
    );
    for (a, b) in r1.trace.iter().zip(r2.trace.iter()) {
        assert_eq!(a.move_, b.move_, "move order must be reproducible");
        assert!((a.gaic - b.gaic).abs() < 1e-12, "gaic must be reproducible");
    }
    assert_eq!(mu_term_names(&r1), mu_term_names(&r2));
}

#[test]
fn step_gaic_trace_is_monotone_decreasing() {
    let mut rng = Generator::new(2024);
    let (y, data) = rng.gaussian_with_noise_columns(300, 4.0, 1.0, 0.5);
    let start = intercept_only(&["mu", "sigma"]);

    let result = step_gaic(
        &data,
        &y,
        &Gaussian::new(),
        start,
        &mu_scope(),
        2.0,
        Direction::Forward,
        FitConfig::default(),
    )
    .unwrap();

    // Every accepted move strictly lowered GAIC.
    for w in result.trace.windows(2) {
        assert!(
            w[1].gaic < w[0].gaic,
            "trace GAIC must strictly decrease: {} then {}",
            w[0].gaic,
            w[1].gaic
        );
    }
}

// ============================================================================
// Exhaustive hardening: lr_test
// ============================================================================

#[test]
fn lr_test_p_value_matches_chi_squared_survival() {
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    let mut rng = Generator::new(7);
    let (y, data) = rng.gaussian_with_noise_columns(300, 4.0, 1.0, 0.5);
    let null = intercept_only(&["mu", "sigma"]);
    let alt = linear_intercepts("x1", &["mu", "sigma"]);
    let small = GamlssModel::fit(&data, &y, &null, &Gaussian::new()).unwrap();
    let big = GamlssModel::fit(&data, &y, &alt, &Gaussian::new()).unwrap();

    let test = lr_test(&small, &big, &Gaussian::new(), &y).unwrap();
    // Adding one linear term to mu costs ~1 df.
    assert!((test.df - 1.0).abs() < 0.05, "df {}", test.df);
    assert!(
        test.lr_stat > 3.84,
        "lr_stat {} should clear chi2_1,0.95",
        test.lr_stat
    );
    // Known-value: the p-value is the χ²_df survival function evaluated at lr_stat.
    let expected = 1.0 - ChiSquared::new(test.df).unwrap().cdf(test.lr_stat);
    assert!(
        (test.p_value - expected).abs() < 1e-9,
        "gamma_ur p {} vs statrs ChiSquared {}",
        test.p_value,
        expected
    );
}

#[test]
fn lr_test_handles_fractional_df_from_a_penalized_smooth() {
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    let mut rng = Generator::new(21);
    // A genuinely nonlinear (sinusoidal) mean so the P-spline can't collapse to a
    // straight line. Its effective df stays solidly fractional on either
    // linear-algebra backend.
    let (y, data) = rng.sinusoidal_gaussian(200, 0.3);
    let linear_mu = linear_intercepts("x", &["mu", "sigma"]);
    let smooth_mu = smooth_intercepts("x", 12, &["mu", "sigma"]);
    let small = GamlssModel::fit(&data, &y, &linear_mu, &Gaussian::new()).unwrap();
    let big = GamlssModel::fit(&data, &y, &smooth_mu, &Gaussian::new()).unwrap();

    let test = lr_test(&small, &big, &Gaussian::new(), &y).unwrap();
    // A penalized smooth has non-integer effective df, hence fractional ν.
    assert!(test.df > 0.0);
    assert!(
        (test.df - test.df.round()).abs() > 1e-6,
        "expected fractional df, got {}",
        test.df
    );
    // The χ² survival via gamma_ur still matches statrs at fractional df.
    let expected = 1.0 - ChiSquared::new(test.df).unwrap().cdf(test.lr_stat.max(0.0));
    assert!((test.p_value - expected).abs() < 1e-9);
    assert!((0.0..=1.0).contains(&test.p_value));
}

#[test]
fn lr_test_error_message_names_the_nesting_requirement() {
    let mut rng = Generator::new(5);
    let (y, data) = rng.gaussian_with_noise_columns(150, 4.0, 1.0, 0.5);
    let null = intercept_only(&["mu", "sigma"]);
    let alt = linear_intercepts("x1", &["mu", "sigma"]);
    let small = GamlssModel::fit(&data, &y, &null, &Gaussian::new()).unwrap();
    let big = GamlssModel::fit(&data, &y, &alt, &Gaussian::new()).unwrap();

    let err = lr_test(&big, &small, &Gaussian::new(), &y).unwrap_err();
    match err {
        GamlssError::Input(msg) => {
            assert!(
                msg.contains("nested"),
                "message should explain nesting: {}",
                msg
            );
            assert!(
                msg.contains("edf_big"),
                "message should name the edf ordering: {}",
                msg
            );
        }
        other => panic!("expected Input, got {:?}", other),
    }
}

// ============================================================================
// Exhaustive hardening: ic_table
// ============================================================================

#[test]
fn ic_table_ranks_three_models() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.gaussian_with_noise_columns(300, 4.0, 1.0, 0.5);
    let m_null = GamlssModel::fit(
        &data,
        &y,
        &intercept_only(&["mu", "sigma"]),
        &Gaussian::new(),
    )
    .unwrap();
    let m_x1 = GamlssModel::fit(
        &data,
        &y,
        &linear_intercepts("x1", &["mu", "sigma"]),
        &Gaussian::new(),
    )
    .unwrap();
    let m_full = {
        let f = Formula::new()
            .with_terms(
                "mu",
                vec![Term::Intercept, linear("x1"), linear("x2"), linear("x3")],
            )
            .with_terms("sigma", vec![Term::Intercept]);
        GamlssModel::fit(&data, &y, &f, &Gaussian::new()).unwrap()
    };

    let rows = ic_table(
        &[("null", &m_null), ("x1", &m_x1), ("full", &m_full)],
        &Gaussian::new(),
        &y,
        (y.len() as f64).ln(), // BIC
    )
    .unwrap();
    assert_eq!(rows.len(), 3);
    // The single-signal model wins under BIC: lower than both the underfit null
    // and the overfit full model.
    let gaic = |label: &str| rows.iter().find(|r| r.label == label).unwrap().gaic;
    assert!(
        gaic("x1") < gaic("null"),
        "x1 {} vs null {}",
        gaic("x1"),
        gaic("null")
    );
    assert!(
        gaic("x1") < gaic("full"),
        "x1 {} vs full {}",
        gaic("x1"),
        gaic("full")
    );
}

#[test]
fn ic_table_compares_non_nested_families() {
    // Count data: Poisson and Gaussian are non-nested, but ic_table still ranks them.
    let mut rng = Generator::new(13);
    let (y, data) = rng.poisson_data(300, 0.5, 0.3);
    let formula_p = linear_intercepts("x", &["mu"]);
    let formula_g = linear_intercepts("x", &["mu", "sigma"]);
    let m_pois = GamlssModel::fit(&data, &y, &formula_p, &Poisson::new()).unwrap();
    let m_gauss = GamlssModel::fit(&data, &y, &formula_g, &Gaussian::new()).unwrap();

    // Each model scored under its own family. The IC is comparable as −2·ll + k·edf.
    let pois_row = &ic_table(&[("poisson", &m_pois)], &Poisson::new(), &y, 2.0).unwrap()[0];
    let gauss_row = &ic_table(&[("gaussian", &m_gauss)], &Gaussian::new(), &y, 2.0).unwrap()[0];
    assert!(pois_row.global_deviance.is_finite());
    assert!(gauss_row.global_deviance.is_finite());
    // The correctly-specified Poisson fits the counts better than the Gaussian does.
    assert!(
        pois_row.gaic < gauss_row.gaic,
        "Poisson GAIC {} should beat Gaussian {}",
        pois_row.gaic,
        gauss_row.gaic
    );
}

// ============================================================================
// Exhaustive hardening: step_gaic directions, parameters, families
// ============================================================================

#[test]
fn step_gaic_both_adds_signal_and_drops_noise() {
    let mut rng = Generator::new(2024);
    let (y, data) = rng.gaussian_with_noise_columns(300, 4.0, 1.0, 0.5);
    // Start with the noise term present and the signal absent.
    let start = Formula::new()
        .with_terms("mu", vec![Term::Intercept, linear("x2")])
        .with_terms("sigma", vec![Term::Intercept]);
    let result = step_gaic(
        &data,
        &y,
        &Gaussian::new(),
        start,
        &mu_scope(),
        (y.len() as f64).ln(),
        Direction::Both,
        FitConfig::default(),
    )
    .unwrap();

    let names = mu_term_names(&result);
    assert!(
        names.contains(&"x1".to_string()),
        "Both should add x1, got {:?}",
        names
    );
    assert!(
        !names.contains(&"x2".to_string()),
        "Both should drop noise x2, got {:?}",
        names
    );
}

#[test]
fn step_gaic_selects_a_term_on_sigma() {
    let mut rng = Generator::new(31);
    // Heteroskedastic: σ depends on x, so a term on sigma is genuine signal.
    let (y, data) = rng.heteroskedastic_gaussian(400);
    let start = linear_intercepts("x", &["mu", "sigma"]); // mu~x, sigma~1
    let scope = vec![StepScope {
        param: "sigma".to_string(),
        candidates: vec![linear("x")],
    }];
    let result = step_gaic(
        &data,
        &y,
        &Gaussian::new(),
        start,
        &scope,
        (y.len() as f64).ln(),
        Direction::Forward,
        FitConfig::default(),
    )
    .unwrap();

    let sigma_names: Vec<String> = result
        .formula
        .get("sigma")
        .unwrap()
        .iter()
        .map(|t| t.term_name())
        .collect();
    assert!(
        sigma_names.contains(&"x".to_string()),
        "sigma should gain x, got {:?}",
        sigma_names
    );
}

#[test]
fn step_gaic_selects_on_both_mu_and_sigma() {
    let mut rng = Generator::new(77);
    let (y, data) = rng.heteroskedastic_gaussian(400);
    let start = intercept_only(&["mu", "sigma"]);
    let scope = vec![
        StepScope {
            param: "mu".to_string(),
            candidates: vec![linear("x")],
        },
        StepScope {
            param: "sigma".to_string(),
            candidates: vec![linear("x")],
        },
    ];
    let result = step_gaic(
        &data,
        &y,
        &Gaussian::new(),
        start,
        &scope,
        (y.len() as f64).ln(),
        Direction::Forward,
        FitConfig::default(),
    )
    .unwrap();

    let has = |param: &str, term: &str| {
        result
            .formula
            .get(param)
            .unwrap()
            .iter()
            .any(|t| t.term_name() == term)
    };
    assert!(has("mu", "x"), "mu should gain x");
    assert!(has("sigma", "x"), "sigma should gain x");
}

#[test]
fn step_gaic_recovers_signal_for_a_poisson_family() {
    let mut rng = Generator::new(2024);
    let (y, data) = rng.poisson_with_noise_columns(400, 1.5, 0.5);
    let start = intercept_only(&["mu"]);
    let scope = vec![StepScope {
        param: "mu".to_string(),
        candidates: vec![linear("x1"), linear("x2")],
    }];
    let result = step_gaic(
        &data,
        &y,
        &Poisson::new(),
        start,
        &scope,
        (y.len() as f64).ln(),
        Direction::Forward,
        FitConfig::default(),
    )
    .unwrap();

    let names = mu_term_names(&result);
    assert!(
        names.contains(&"x1".to_string()),
        "Poisson step should select x1, got {:?}",
        names
    );
    assert!(
        !names.contains(&"x2".to_string()),
        "Poisson step should reject noise x2, got {:?}",
        names
    );
}

// ============================================================================
// Exhaustive hardening: step_gaic degenerate / no-op paths
// ============================================================================

#[test]
fn step_gaic_empty_scope_is_a_noop() {
    let mut rng = Generator::new(9);
    let (y, data) = rng.gaussian_with_noise_columns(120, 4.0, 1.0, 0.5);
    let start = linear_intercepts("x1", &["mu", "sigma"]);
    let result = step_gaic(
        &data,
        &y,
        &Gaussian::new(),
        start,
        &[], // no candidate moves
        2.0,
        Direction::Both,
        FitConfig::default(),
    )
    .unwrap();
    assert!(
        result.trace.is_empty(),
        "empty scope should accept no moves"
    );
    assert!(result.model.converged());
}

#[test]
fn step_gaic_forward_with_all_candidates_present_is_a_noop() {
    let mut rng = Generator::new(9);
    let (y, data) = rng.gaussian_with_noise_columns(120, 4.0, 1.0, 0.5);
    // x1 is already in the starting formula, and forward can only add absent terms.
    let start = linear_intercepts("x1", &["mu", "sigma"]);
    let scope = vec![StepScope {
        param: "mu".to_string(),
        candidates: vec![linear("x1")],
    }];
    let result = step_gaic(
        &data,
        &y,
        &Gaussian::new(),
        start,
        &scope,
        2.0,
        Direction::Forward,
        FitConfig::default(),
    )
    .unwrap();
    assert!(result.trace.is_empty(), "no absent candidate ⇒ no move");
}

#[test]
fn step_gaic_rejects_a_non_improving_move() {
    let mut rng = Generator::new(9);
    let (y, data) = rng.gaussian_with_noise_columns(300, 4.0, 1.0, 0.5);
    // Only a pure-noise candidate is on offer, and under BIC it never beats the penalty.
    let start = linear_intercepts("x1", &["mu", "sigma"]); // already correct
    let scope = vec![StepScope {
        param: "mu".to_string(),
        candidates: vec![linear("x2")],
    }];
    let result = step_gaic(
        &data,
        &y,
        &Gaussian::new(),
        start,
        &scope,
        (y.len() as f64).ln(),
        Direction::Forward,
        FitConfig::default(),
    )
    .unwrap();
    // The EPS guard rejects any move that doesn't lower GAIC, so the trace is empty.
    assert!(
        result.trace.is_empty(),
        "noise term should not be accepted, trace {:?}",
        result.trace.iter().map(|r| &r.move_).collect::<Vec<_>>()
    );
}

#[test]
fn step_gaic_is_deterministic_at_formula_level_across_seeds_and_k() {
    for seed in [1u64, 2, 3] {
        for &k in &[2.0_f64, 5.0] {
            let mut rng = Generator::new(seed);
            let (y, data) = rng.gaussian_with_noise_columns(250, 3.0, 1.0, 0.6);
            let run = || {
                step_gaic(
                    &data,
                    &y,
                    &Gaussian::new(),
                    intercept_only(&["mu", "sigma"]),
                    &mu_scope(),
                    k,
                    Direction::Both,
                    FitConfig::default(),
                )
                .unwrap()
            };
            let r1 = run();
            let r2 = run();
            assert_eq!(r1.trace.len(), r2.trace.len());
            for (a, b) in r1.trace.iter().zip(r2.trace.iter()) {
                assert_eq!(a.move_, b.move_);
                assert!((a.gaic - b.gaic).abs() < 1e-12);
            }
            // Formula-level replayability, not just matching trace structure.
            assert_eq!(
                mu_term_names(&r1),
                mu_term_names(&r2),
                "seed {} k {}",
                seed,
                k
            );
        }
    }
}
