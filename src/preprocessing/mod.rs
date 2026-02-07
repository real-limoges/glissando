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
pub fn validate_inputs<D: Distribution>(
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

    // Validate all variables have correct length and contain finite values
    for (name, arr) in data.iter() {
        // Check length matches response
        if arr.len() != n_obs {
            return Err(GamlssError::Input(format!(
                "Variable '{}' has {} observations but response has {}",
                name,
                arr.len(),
                n_obs
            )));
        }

        // Check for non-finite values
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
