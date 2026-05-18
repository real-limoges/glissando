//! Input validation for model fitting.
//!
//! Checks datasets, response variables, and formulas for consistency, finite values,
//! and correct dimensionality before fitting begins.

use crate::distributions::Distribution;
use crate::error::GamlssError;
use crate::types::{DataSet, Formula};
use ndarray::prelude::*;
use std::collections::HashSet;

/// Validates input data and formula for model fitting.
///
/// Checks that:
/// - Dataset is not empty
/// - Response variable exists
/// - Response variable contains only finite values
/// - All parameters in the distribution have formulas
/// - All variables referenced in formulas exist in the data
/// - All numeric variables contain only finite values
pub fn validate_inputs<D: Distribution + ?Sized>(
    y: &Array1<f64>,
    data: &DataSet,
    formula: &Formula,
    family: &D,
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
        let err = validate_inputs(&y, &data, &f, &Gaussian).unwrap_err();
        assert!(matches!(err, GamlssError::EmptyData));
    }

    #[test]
    fn rejects_non_finite_response() {
        let y = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        let data = DataSet::new();
        let f = gaussian_formula();
        let err = validate_inputs(&y, &data, &f, &Gaussian).unwrap_err();
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
        let err = validate_inputs(&y, &data, &f, &Gaussian).unwrap_err();
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
        let err = validate_inputs(&y, &data, &f, &Gaussian).unwrap_err();
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
        let err = validate_inputs(&y, &data, &f, &Gaussian).unwrap_err();
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
        let err = validate_inputs(&y, &data, &f, &Gaussian).unwrap_err();
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
        validate_inputs(&y, &data, &f, &Gaussian).unwrap();
    }
}
