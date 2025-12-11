use ndarray::prelude::*;
use polars::prelude::*;
use std::collections::HashMap;
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
