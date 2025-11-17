#![recursion_limit = "1024"]
#![allow(dead_code, unused_variables, unused_imports)]
mod error;
mod fitting;
pub mod families;
mod splines;
mod types;
mod terms;


// for end user
pub use error::GamlssError;
// pub mod families;
pub use terms::{Term, Smooth};
pub use types::*;

//

use families::Distribution;
use ndarray::{Array, Array1};
use polars::prelude::{DataFrame, PolarsError};
use std::collections::HashMap;
// use crate::fitting::ModelParameter;

#[derive(Debug)]
pub struct FittedParameter {
    pub coefficients: Coefficients,
    pub covariance: CovarianceMatrix,
    pub terms: Vec<Term>,
    pub eta: Array1<f64>,
    pub fitted_values: Array1<f64>,
}

#[derive(Debug)]
pub struct GamlssModel {
    pub models: HashMap<String, FittedParameter>,
}

impl GamlssModel {
    pub fn fit<D: Distribution> (
        data: &DataFrame,
        y_name: &str,
        formula: &HashMap<String, Vec<Term>>,
        family: &D,
    ) -> Result<Self, GamlssError> {

        let y_series = data.column(y_name)
            .map_err(|e| GamlssError::Input(format!("Target Column '{}' not found: {}", y_name, e)))?;

        let y_vec = y_series.f64()?
            .to_ndarray()?
            .to_shape(y_series.len())?;

        let y_vector = Array1::from_vec(y_vec.to_vec());

        let fitted_models = fitting::fit_gamlss(
            data, &y_vector, formula, family
        )?;

        Ok(Self {
            models: fitted_models,
        })
    }

    // I know that the sampling is going to change quite radically, so I'm just commenting this
    // out for right now
    pub fn posterior_samples(&self, n_samples: usize) -> Vec<Coefficients> {
        todo!();
        // fitting::inference::sample_posterior(&self.coefficients, &self.covariance, n_samples)
        //     .into_iter()
        //     .map(Coefficients)
        //     .collect()
    }
}