//! Input validation for model fitting.
//!
//! Checks datasets, response variables, and formulas for consistency, finite values,
//! and correct dimensionality before fitting begins.

use crate::distributions::Distribution;
use crate::error::GamlssError;
use crate::types::{DataSet, Formula};
use ndarray::prelude::*;
use std::collections::HashSet;

/// Owned, row-aligned model frame returned by [`drop_incomplete_rows`]: the
/// filtered response, referenced columns, and (optional) prior weights.
type CompleteFrame = (DataSet, Array1<f64>, Option<Array1<f64>>);

/// Drop every row carrying a missing (non-finite) value in `y`, a
/// formula-referenced column, or — when present — its prior weight, returning
/// owned, row-aligned copies of the response, the referenced columns, and the
/// weights (DATA-4, `NaAction::DropRows`). This is R's `na.omit` over the model
/// frame: only variables the formula references participate in the completeness
/// test, so an unrelated column with missing entries never drops a row.
///
/// The returned `DataSet` carries only the referenced columns (all the fitter
/// needs); other columns are dropped from the working copy. Errors with
/// [`GamlssError::EmptyData`] if no complete row remains.
pub fn drop_incomplete_rows(
    y: &Array1<f64>,
    data: &DataSet,
    formula: &Formula,
    weights: Option<&Array1<f64>>,
) -> Result<CompleteFrame, GamlssError> {
    // Columns the formula actually references — the only ones that can mask a row.
    let mut referenced: HashSet<&str> = HashSet::new();
    for terms in formula.values() {
        for term in terms {
            for col in term.column_names() {
                referenced.insert(col);
            }
        }
    }

    // Only the response and referenced columns participate in the completeness
    // test. Weights are deliberately excluded: a non-finite or negative weight is
    // a user error, not missing data, and is rejected by `validate_inputs` rather
    // than silently dropping the row.
    let n = y.len();
    let mut keep = vec![true; n];
    for (i, &yi) in y.iter().enumerate() {
        if !yi.is_finite() {
            keep[i] = false;
        }
    }
    for col in &referenced {
        if let Some(arr) = data.get(*col) {
            for (i, &v) in arr.iter().enumerate() {
                if i < n && !v.is_finite() {
                    keep[i] = false;
                }
            }
        }
    }

    let kept_idx: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    if kept_idx.is_empty() {
        return Err(GamlssError::EmptyData);
    }
    let take =
        |arr: &Array1<f64>| -> Array1<f64> { Array1::from_iter(kept_idx.iter().map(|&i| arr[i])) };

    let mut filtered = DataSet::new();
    for col in &referenced {
        if let Some(arr) = data.get(*col) {
            filtered.insert_column(*col, take(arr));
        }
    }
    let y_filtered = take(y);
    // Subset weights only when their length matches `y`; a mismatched length is
    // left untouched so `validate_inputs` surfaces the length error downstream.
    let w_filtered = match weights {
        Some(w) if w.len() == n => Some(take(w)),
        other => other.cloned(),
    };
    Ok((filtered, y_filtered, w_filtered))
}

/// Validates input data, formula, and optional prior weights for model fitting.
///
/// Checks that:
/// - Dataset is not empty
/// - Response variable exists
/// - Response variable contains only finite values
/// - All parameters in the distribution have formulas
/// - All variables referenced in formulas exist in the data
/// - All numeric variables contain only finite values
/// - If `weights` is `Some`, its length matches `y`, all values are finite and ≥ 0
pub fn validate_inputs<D: Distribution + ?Sized>(
    y: &Array1<f64>,
    data: &DataSet,
    formula: &Formula,
    family: &D,
    weights: Option<&Array1<f64>>,
) -> Result<(), GamlssError> {
    // Check dataset is not empty
    if y.is_empty() {
        return Err(GamlssError::EmptyData);
    }

    let n_obs = y.len();

    // Validate response variable is finite
    let non_finite_count = y.iter().filter(|v| !v.is_finite()).count();
    if non_finite_count > 0 {
        return Err(GamlssError::NonFiniteValues {
            name: "y (response)".to_string(),
            count: non_finite_count,
        });
    }

    // Check all parameters have formulas
    for param in family.parameters() {
        if !formula.contains_key(*param) {
            return Err(GamlssError::MissingFormula {
                param: param.to_string(),
            });
        }
    }

    // Collect all referenced column names from formulas
    let mut referenced_columns: HashSet<&str> = HashSet::new();
    for terms in formula.values() {
        for term in terms {
            for col in term.column_names() {
                referenced_columns.insert(col);
            }
        }
    }

    // Check all referenced columns exist in data
    for col in &referenced_columns {
        if !data.contains_key(*col) {
            return Err(GamlssError::MissingVariable {
                name: col.to_string(),
            });
        }
    }

    // `DataSet` enforces that all columns share a length internally; we only need to
    // check that length agrees with `y`. `n_obs()` returns `None` for empty datasets,
    // which is fine — formulas may only reference `Intercept`, which doesn't need data.
    if let Some(data_n_obs) = data.n_obs() {
        if data_n_obs != n_obs {
            return Err(GamlssError::Input(format!(
                "Dataset has {} observations but response has {}",
                data_n_obs, n_obs,
            )));
        }
    }

    // Check for non-finite values in every column.
    for (name, arr) in data.iter() {
        let non_finite_count = arr.iter().filter(|v| !v.is_finite()).count();
        if non_finite_count > 0 {
            return Err(GamlssError::NonFiniteValues {
                name: name.clone(),
                count: non_finite_count,
            });
        }
    }

    // Validate prior weights when provided.
    if let Some(w) = weights {
        if w.len() != n_obs {
            return Err(GamlssError::Input(format!(
                "weights length {} != y length {}",
                w.len(),
                n_obs,
            )));
        }
        if w.iter().any(|&v| !v.is_finite() || v < 0.0) {
            return Err(GamlssError::Input(
                "weights must be finite and non-negative".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::Gaussian;
    use crate::terms::Term;

    fn gaussian_formula() -> Formula {
        Formula::new()
            .with_terms("mu", vec![Term::Intercept])
            .with_terms("sigma", vec![Term::Intercept])
    }

    fn data_with(name: &str, values: Vec<f64>) -> DataSet {
        let mut d = DataSet::new();
        d.insert_column(name, Array1::from_vec(values));
        d
    }

    #[test]
    fn rejects_empty_response() {
        let y = Array1::<f64>::zeros(0);
        let data = DataSet::new();
        let f = gaussian_formula();
        let err = validate_inputs(&y, &data, &f, &Gaussian, None).unwrap_err();
        assert!(matches!(err, GamlssError::EmptyData));
    }

    #[test]
    fn rejects_non_finite_response() {
        let y = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        let data = DataSet::new();
        let f = gaussian_formula();
        let err = validate_inputs(&y, &data, &f, &Gaussian, None).unwrap_err();
        match err {
            GamlssError::NonFiniteValues { count, .. } => assert_eq!(count, 1),
            other => panic!("expected NonFiniteValues, got {:?}", other),
        }
    }

    #[test]
    fn rejects_missing_formula_for_parameter() {
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let data = DataSet::new();
        let f = Formula::new().with_terms("mu", vec![Term::Intercept]);
        let err = validate_inputs(&y, &data, &f, &Gaussian, None).unwrap_err();
        match err {
            GamlssError::MissingFormula { param } => assert_eq!(param, "sigma"),
            other => panic!("expected MissingFormula, got {:?}", other),
        }
    }

    #[test]
    fn rejects_missing_referenced_column() {
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let data = DataSet::new();
        let f = Formula::new()
            .with_terms(
                "mu",
                vec![Term::Linear {
                    col_name: "x".to_string(),
                }],
            )
            .with_terms("sigma", vec![Term::Intercept]);
        let err = validate_inputs(&y, &data, &f, &Gaussian, None).unwrap_err();
        match err {
            GamlssError::MissingVariable { name } => assert_eq!(name, "x"),
            other => panic!("expected MissingVariable, got {:?}", other),
        }
    }

    #[test]
    fn rejects_dataset_length_mismatched_to_response() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let data = data_with("x", vec![1.0, 2.0]); // length 2, y is length 3
        let f = Formula::new()
            .with_terms(
                "mu",
                vec![Term::Linear {
                    col_name: "x".to_string(),
                }],
            )
            .with_terms("sigma", vec![Term::Intercept]);
        let err = validate_inputs(&y, &data, &f, &Gaussian, None).unwrap_err();
        match err {
            GamlssError::Input(s) => {
                assert!(s.contains("2 observations") && s.contains("response has 3"))
            }
            other => panic!("expected Input, got {:?}", other),
        }
    }

    #[test]
    fn rejects_non_finite_in_covariate() {
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let data = data_with("x", vec![1.0, f64::INFINITY]);
        let f = Formula::new()
            .with_terms(
                "mu",
                vec![Term::Linear {
                    col_name: "x".to_string(),
                }],
            )
            .with_terms("sigma", vec![Term::Intercept]);
        let err = validate_inputs(&y, &data, &f, &Gaussian, None).unwrap_err();
        match err {
            GamlssError::NonFiniteValues { name, count } => {
                assert_eq!(name, "x");
                assert_eq!(count, 1);
            }
            other => panic!("expected NonFiniteValues, got {:?}", other),
        }
    }

    #[test]
    fn accepts_well_formed_input() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let data = data_with("x", vec![0.5, 1.0, 1.5]);
        let f = Formula::new()
            .with_terms(
                "mu",
                vec![
                    Term::Intercept,
                    Term::Linear {
                        col_name: "x".to_string(),
                    },
                ],
            )
            .with_terms("sigma", vec![Term::Intercept]);
        validate_inputs(&y, &data, &f, &Gaussian, None).unwrap();
    }

    // Property tests — proptest is non-wasm only (the dev-dep is gated likewise).
    #[cfg(not(target_arch = "wasm32"))]
    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// For any `y` containing at least one NaN or ±∞, validate_inputs
            /// must return `NonFiniteValues` with `count` equal to the number
            /// of non-finite entries — never silently accept the input.
            #[test]
            fn rejects_any_non_finite_y(
                finite_vals in proptest::collection::vec(-1e6f64..1e6, 1..32),
                n_nan in 0usize..6,
                n_inf in 0usize..6,
            ) {
                // Build a y by interleaving finite and non-finite entries.
                let mut y_vec = finite_vals.clone();
                for _ in 0..n_nan { y_vec.push(f64::NAN); }
                for _ in 0..n_inf { y_vec.push(f64::INFINITY); }
                prop_assume!(!y_vec.is_empty());
                let total_nonfinite = n_nan + n_inf;

                let y = Array1::from_vec(y_vec);
                let data = DataSet::new();
                let f = gaussian_formula();
                let result = validate_inputs(&y, &data, &f, &Gaussian, None);

                if total_nonfinite == 0 {
                    prop_assert!(result.is_ok());
                } else {
                    match result {
                        Err(GamlssError::NonFiniteValues { count, .. }) => {
                            prop_assert_eq!(count, total_nonfinite);
                        }
                        other => prop_assert!(false, "expected NonFiniteValues, got {:?}", other),
                    }
                }
            }
        }
    }
}
