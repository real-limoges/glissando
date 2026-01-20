use crate::distributions::Distribution;
use crate::error::GamlssError;
use crate::terms::Term;
use ndarray::prelude::*;
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

#[derive(thiserror::Error, Debug)]
pub enum PreprocessingError {
    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Column '{0}' contains Null values. Linear algebra cannot handle Nulls.")]
    NullValues(String),
    #[error("Unsupported data type in column '{0}'")]
    UnsupportedType(String),
}

pub fn series_to_array1(series: &Series) -> Result<Array1<f64>, PreprocessingError> {
    let casted = series.cast(&DataType::Float64)?;
    let ca = casted.f64()?;

    if ca.null_count() > 0 {
        return Err(PreprocessingError::NullValues(series.name().to_string()));
    }
    let vec_data: Vec<f64> = ca.into_no_null_iter().collect();

    Ok(Array1::from(vec_data))
}

pub fn validate_inputs<D: Distribution>(
    data: &DataFrame,
    y_name: &str,
    formula: &HashMap<String, Vec<Term>>,
    family: &D,
) -> Result<(), GamlssError> {
    if data.height() == 0 {
        return Err(GamlssError::EmptyData);
    }

    let available_columns: HashSet<String> = data
        .get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    if !available_columns.contains(y_name) {
        return Err(GamlssError::MissingColumn {
            column: y_name.to_string(),
        });
    }
    validate_column_finite(data, y_name)?;

    for param in family.parameters() {
        if !formula.contains_key(*param) {
            return Err(GamlssError::MissingFormula {
                param: param.to_string(),
            });
        }
    }

    let mut referenced_columns: HashSet<&str> = HashSet::new();
    for terms in formula.values() {
        for term in terms {
            for col in term.column_names() {
                referenced_columns.insert(col);
            }
        }
    }

    for col in &referenced_columns {
        if !available_columns.contains(*col) {
            return Err(GamlssError::MissingColumn {
                column: col.to_string(),
            });
        }
    }

    for col in &referenced_columns {
        let series = data.column(col).map_err(|_| GamlssError::MissingColumn {
            column: col.to_string(),
        })?;

        if series.dtype().is_primitive_numeric() {
            validate_column_finite(data, col)?;
        }
    }

    Ok(())
}

fn validate_column_finite(data: &DataFrame, col_name: &str) -> Result<(), GamlssError> {
    let series = data
        .column(col_name)
        .map_err(|_| GamlssError::MissingColumn {
            column: col_name.to_string(),
        })?;

    let casted = match series.cast(&DataType::Float64) {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };

    let ca = match casted.f64() {
        Ok(ca) => ca,
        Err(_) => return Ok(()),
    };

    let non_finite_count = ca
        .into_iter()
        .filter(|opt| match opt {
            Some(v) => !v.is_finite(),
            None => true,
        })
        .count();

    if non_finite_count > 0 {
        return Err(GamlssError::NonFiniteValues {
            column: col_name.to_string(),
            count: non_finite_count,
        });
    }

    Ok(())
}
