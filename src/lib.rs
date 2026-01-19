#![recursion_limit = "1024"]
#![allow(dead_code, unused_variables, unused_imports)]
pub mod distributions;
mod error;
pub mod fitting;
mod math;
pub mod preprocessing;
mod splines;
mod terms;
mod types;

pub use error::GamlssError;
pub use fitting::{FitConfig, FitDiagnostics};
pub use terms::{Smooth, Term};
pub use types::*;

use distributions::Distribution;
use ndarray::{Array, Array1};
use polars::prelude::{DataFrame, PolarsError};
use preprocessing::validate_inputs;
use std::collections::HashMap;

#[derive(Debug)]
pub struct GamlssModel {
    pub models: HashMap<String, fitting::FittedParameter>,
    pub diagnostics: FitDiagnostics,
}

impl GamlssModel {
    pub fn fit<D: Distribution>(
        data: &DataFrame,
        y_name: &str,
        formula: &HashMap<String, Vec<Term>>,
        family: &D,
    ) -> Result<Self, GamlssError> {
        Self::fit_with_config(data, y_name, formula, family, FitConfig::default())
    }

    pub fn fit_with_config<D: Distribution>(
        data: &DataFrame,
        y_name: &str,
        formula: &HashMap<String, Vec<Term>>,
        family: &D,
        config: FitConfig,
    ) -> Result<Self, GamlssError> {
        validate_inputs(data, y_name, formula, family)?;

        let y_series = data.column(y_name).map_err(|e| {
            GamlssError::Input(format!("Target Column '{}' not found: {}", y_name, e))
        })?;

        let binding = y_series.f64()?.to_ndarray()?;
        let y_vec = binding
            .to_shape(y_series.len())
            .map_err(|e| GamlssError::Shape(e.to_string()))?;

        let y_vector = Array1::from_vec(y_vec.to_vec());

        let (fitted_models, diagnostics) =
            fitting::fit_gamlss(data, &y_vector, formula, family, &config)?;

        Ok(Self {
            models: fitted_models,
            diagnostics,
        })
    }

    pub fn converged(&self) -> bool {
        self.diagnostics.converged
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
