//! DATA-4 missing-data handling, public-API integration tests.
//!
//! `NaAction::DropRows` (the default) drops any row with a non-finite value in
//! the response or a referenced column, à la R's `na.omit`. The fit it produces
//! has to equal a fit on the manually pre-filtered data. `NaAction::Fail` keeps
//! the historical hard-error behavior.

// Integration tests can't run under the `python` feature. PyO3 linking gets in the way.
#![cfg(not(feature = "python"))]

use glissando::distributions::Gaussian;
use glissando::{DataSet, FitConfig, Formula, GamlssError, GamlssModel, NaAction, Term};
use ndarray::Array1;

fn formula() -> Formula {
    Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::linear("x")])
        .with_terms("sigma", vec![Term::Intercept])
}

/// Dropping incomplete rows gives me exactly the fit I'd get by removing those
/// rows by hand before calling `fit`. Same answer, no surprises.
#[test]
fn drop_rows_equals_manual_prefilter() {
    // Rows 2 and 5 each carry a missing value: NaN in y, Inf in x respectively.
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let y = vec![1.0, 1.8, f64::NAN, 3.4, 4.1, 5.0, 5.9, 6.8];
    let x_bad = {
        let mut v = x.clone();
        v[5] = f64::INFINITY;
        v
    };

    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x_bad));
    let model_auto =
        GamlssModel::fit(&data, &Array1::from_vec(y.clone()), &formula(), &Gaussian).unwrap();

    // Manual pre-filter: keep the rows where both y and x are finite, which drops 2 and 5.
    let keep: Vec<usize> = (0..x.len())
        .filter(|&i| y[i].is_finite() && x[i].is_finite() && i != 5)
        .collect();
    let x_clean: Vec<f64> = keep.iter().map(|&i| x[i]).collect();
    let y_clean: Vec<f64> = keep.iter().map(|&i| y[i]).collect();
    let mut data_clean = DataSet::new();
    data_clean.insert_column("x", Array1::from_vec(x_clean));
    let model_manual = GamlssModel::fit(
        &data_clean,
        &Array1::from_vec(y_clean),
        &formula(),
        &Gaussian,
    )
    .unwrap();

    let beta_auto = &model_auto.models["mu"].coefficients.0;
    let beta_manual = &model_manual.models["mu"].coefficients.0;
    for (a, b) in beta_auto.iter().zip(beta_manual.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "drop-rows fit differs from manual pre-filter: {a} vs {b}"
        );
    }
}

/// An unrelated column with missing values drops no rows at all. Only the
/// formula-referenced variables get a vote (R `na.omit` over the model frame).
#[test]
fn unreferenced_column_missing_does_not_drop_rows() {
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
    let mut junk = vec![0.0; n];
    junk[3] = f64::NAN; // missing, but in a column the formula never references, so it shouldn't matter

    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));
    data.insert_column("junk", Array1::from_vec(junk));

    let model = GamlssModel::fit(&data, &Array1::from_vec(y), &formula(), &Gaussian).unwrap();
    // All rows survived, so a clean linear fit gets the slope right back.
    let slope = model.models["mu"].coefficients.0[1];
    assert!((slope - 3.0).abs() < 1e-6, "slope {slope}");
}

/// `NaAction::Fail` rejects a missing value outright rather than quietly dropping its row.
#[test]
fn fail_action_errors_on_missing() {
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]));
    let y = Array1::from_vec(vec![1.0, f64::NAN, 3.0, 4.0]);

    let err = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &formula(),
        &Gaussian,
        FitConfig::default().with_na_action(NaAction::Fail),
    )
    .unwrap_err();
    assert!(
        matches!(err, GamlssError::NonFiniteValues { .. }),
        "Fail action should reject missing values, got {err:?}"
    );
}

/// Dropping every row (all of them incomplete) is an error, not an empty fit. Fail loud.
#[test]
fn all_rows_missing_is_empty_data_error() {
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0]));
    let y = Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN]);

    let err = GamlssModel::fit(&data, &y, &formula(), &Gaussian).unwrap_err();
    assert!(
        matches!(err, GamlssError::EmptyData),
        "all-missing should yield EmptyData, got {err:?}"
    );
}
