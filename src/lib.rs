#![recursion_limit = "1024"]
//! Generalized Additive Models for Location, Scale, and Shape (GAMLSS) in Rust.
//!
//! GAMLSS extends traditional regression by modeling multiple distribution parameters
//! (mean, variance, shape) as functions of predictors using the Rigby-Stasinopoulos
//! algorithm with penalized B-splines for nonlinear effects.
//!
//! # Quick start
//!
//! ```
//! use glissando::{GamlssModel, DataSet, Formula, Term};
//! use glissando::distributions::Gaussian;
//! use ndarray::Array1;
//!
//! let y = Array1::from_vec(vec![2.1, 4.0, 5.9, 8.1, 10.0]);
//! let mut data = DataSet::new();
//! data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
//!
//! let formula = Formula::new()
//!     .with_terms("mu", vec![Term::Intercept, Term::Linear { col_name: "x".to_string() }])
//!     .with_terms("sigma", vec![Term::Intercept]);
//!
//! let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
//! assert!(model.converged());
//! ```

pub mod distributions;
mod error;
#[cfg(feature = "python")]
mod ffi;
pub mod fitting;
#[cfg(feature = "serialization")]
pub mod json;
mod linalg;
mod math;
pub mod preprocessing;
#[cfg(feature = "python")]
mod python;
mod splines;
mod terms;
mod types;
#[cfg(feature = "wasm")]
pub mod wasm;

/// Re-export of the exact `ndarray` major this crate is built against. Construct
/// or consume `Array1`/`Array2` through `glissando::ndarray::…` so the types
/// unify with this crate's public API (`predict`, `predict_with_se`) without
/// pinning a matching `ndarray` version yourself.
pub use ndarray;

pub use error::GamlssError;
pub use fitting::diagnostics::{self, ModelDiagnostics};
pub use fitting::{FitConfig, FitDiagnostics, ParamDiagnostic, SmoothingCriterion};
pub use terms::{Smooth, Term};
pub use types::{Coefficients, CovarianceMatrix, DataSet, Formula};

use distributions::Distribution;
use fitting::assembler::assemble_model_matrices;
use indexmap::IndexMap;
use ndarray::{Array1, Array2};
use preprocessing::validate_inputs;
use std::collections::HashMap;
use std::fmt;

/// A fitted GAMLSS model containing per-parameter results and convergence diagnostics.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GamlssModel {
    /// Fitted results keyed by parameter name, in `family.parameters()` order.
    /// `IndexMap` preserves insertion order so iteration over `models` and
    /// `predict_samples` is deterministic.
    pub models: IndexMap<String, fitting::FittedParameter>,
    /// Convergence diagnostics from the RS algorithm.
    pub diagnostics: FitDiagnostics,
}

impl GamlssModel {
    /// Fits a GAMLSS model with default configuration.
    ///
    /// # Errors
    ///
    /// - [`GamlssError::EmptyData`] / [`GamlssError::NonFiniteValues`] /
    ///   [`GamlssError::MissingVariable`] / [`GamlssError::MissingFormula`] /
    ///   [`GamlssError::Input`] — input validation failures
    /// - [`GamlssError::Convergence`] — RS outer loop did not converge in
    ///   `FitConfig::max_iterations`
    /// - [`GamlssError::Linalg`] — singular `X'WX + Σλ·S`, Cholesky failure, etc.
    /// - [`GamlssError::Optimization`] — L-BFGS smoothing-parameter search failed
    /// - [`GamlssError::UnknownParameter`] — formula names a parameter the family
    ///   does not expose
    pub fn fit<D: Distribution + ?Sized>(
        data: &DataSet,
        y: &Array1<f64>,
        formula: &Formula,
        family: &D,
    ) -> Result<Self, GamlssError> {
        Self::fit_with_config(data, y, None, formula, family, FitConfig::default())
    }

    /// Fits a GAMLSS model with per-observation prior weights and default configuration.
    ///
    /// Each observation's likelihood contribution is scaled by its corresponding weight
    /// before fitting.  This replicates `mgcv::bam(..., weights=w)` semantics: the
    /// prior weights multiply the IRLS working weights (they do not replace them).
    ///
    /// `weights` must be finite, non-negative, and the same length as `y`.  A weight
    /// of zero effectively excludes that observation.
    ///
    /// # Errors
    ///
    /// Same error set as [`GamlssModel::fit`], plus [`GamlssError::Input`] if the
    /// weights vector fails validation.
    pub fn fit_weighted<D: Distribution + ?Sized>(
        data: &DataSet,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        formula: &Formula,
        family: &D,
    ) -> Result<Self, GamlssError> {
        Self::fit_with_config(
            data,
            y,
            Some(weights),
            formula,
            family,
            FitConfig::default(),
        )
    }

    /// Fits a GAMLSS model with custom iteration limits, tolerance, and optional prior weights.
    ///
    /// `weights` follows the same semantics as [`GamlssModel::fit_weighted`].  Pass
    /// `None` for unweighted fitting (equivalent to all-ones weights).
    ///
    /// # Errors
    ///
    /// Same error set as [`GamlssModel::fit`].
    pub fn fit_with_config<D: Distribution + ?Sized>(
        data: &DataSet,
        y: &Array1<f64>,
        weights: Option<&Array1<f64>>,
        formula: &Formula,
        family: &D,
        config: FitConfig,
    ) -> Result<Self, GamlssError> {
        validate_inputs(y, data, formula, family, weights)?;

        let (fitted_models, diagnostics) =
            fitting::fit_gamlss(data, y, weights, formula, family, &config)?;

        Ok(Self {
            models: fitted_models,
            diagnostics,
        })
    }

    pub fn converged(&self) -> bool {
        self.diagnostics.converged
    }

    /// Returns the linear-predictor design matrix X for `new_data` and the named
    /// distribution parameter (`predict(type="lpmatrix")` in mgcv). Column order
    /// matches `coefficients` and `term_index_map`.
    ///
    /// # Errors
    ///
    /// - [`GamlssError::UnknownParameter`] if `param` is not in the fitted model.
    /// - [`GamlssError::Input`] if `new_data` has no columns.
    /// - [`GamlssError::MissingVariable`] if `new_data` is missing a column the formula references.
    pub fn design_matrix(
        &self,
        new_data: &DataSet,
        param: &str,
    ) -> Result<Array2<f64>, GamlssError> {
        let fitted = self
            .models
            .get(param)
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: "<fitted model>".to_string(),
                param: param.to_string(),
            })?;
        let n_obs = new_data
            .n_obs()
            .ok_or_else(|| GamlssError::Input("new_data has no columns".into()))?;
        let (x_matrix, _, _, _) = assemble_model_matrices(new_data, n_obs, &fitted.terms)?;
        Ok(x_matrix.0)
    }

    /// Returns the `p × p` posterior covariance matrix `V = (X'WX + Σλ·S)⁻¹` for
    /// the named distribution parameter.
    ///
    /// # Errors
    ///
    /// Returns [`GamlssError::UnknownParameter`] if `param` is not in the fitted model.
    pub fn covariance_matrix(&self, param: &str) -> Result<&types::CovarianceMatrix, GamlssError> {
        self.models
            .get(param)
            .map(|f| &f.covariance)
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: "<fitted model>".to_string(),
                param: param.to_string(),
            })
    }

    /// Returns the term → coefficient column block map for the named distribution parameter.
    ///
    /// Each entry is `(term_name, first_col, last_col_exclusive)`. Column order matches
    /// `design_matrix` and `coefficients`.
    ///
    /// # Errors
    ///
    /// Returns [`GamlssError::UnknownParameter`] if `param` is not in the fitted model.
    pub fn term_index_map(&self, param: &str) -> Result<&[(String, usize, usize)], GamlssError> {
        self.models
            .get(param)
            .map(|f| f.term_blocks.as_slice())
            .ok_or_else(|| GamlssError::UnknownParameter {
                distribution: "<fitted model>".to_string(),
                param: param.to_string(),
            })
    }

    /// Serializes the model to JSON, including the distribution name for later deserialization.
    ///
    /// # Errors
    ///
    /// Returns `GamlssError::Input` if serialization fails.
    #[cfg(feature = "serde")]
    pub fn to_json<D: Distribution + ?Sized>(&self, family: &D) -> Result<String, GamlssError> {
        let wrapper = SerializedModel {
            distribution: family.name().to_string(),
            model: self,
        };
        serde_json::to_string(&wrapper).map_err(|e| GamlssError::Input(e.to_string()))
    }

    /// Deserializes a model from JSON, returning the model and distribution name.
    ///
    /// # Errors
    ///
    /// Returns `GamlssError::Input` if deserialization fails.
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<(Self, String), GamlssError> {
        let wrapper: OwnedSerializedModel =
            serde_json::from_str(json).map_err(|e| GamlssError::Input(e.to_string()))?;
        Ok((wrapper.model, wrapper.distribution))
    }

    /// Predict fitted values for new data.
    ///
    /// Returns a map of parameter name → response-scale fitted values. For
    /// standard errors see [`predict_with_se`](Self::predict_with_se).
    ///
    /// # Examples
    ///
    /// ```
    /// use glissando::{GamlssModel, DataSet, Formula, Term};
    /// use glissando::distributions::Gaussian;
    /// use ndarray::Array1;
    ///
    /// let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let mut data = DataSet::new();
    /// data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    /// let formula = Formula::new()
    ///     .with_terms("mu", vec![Term::Intercept, Term::Linear { col_name: "x".into() }])
    ///     .with_terms("sigma", vec![Term::Intercept]);
    /// let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    ///
    /// let mut new_data = DataSet::new();
    /// new_data.insert_column("x", Array1::from_vec(vec![6.0, 7.0]));
    /// let preds = model.predict(&new_data, &Gaussian::new()).unwrap();
    /// assert_eq!(preds["mu"].len(), 2);
    /// ```
    ///
    /// # Errors
    ///
    /// - [`GamlssError::Input`] — `new_data` has no columns
    /// - [`GamlssError::MissingVariable`] — formula references a column absent
    ///   from `new_data`
    /// - [`GamlssError::UnknownParameter`] — fitted model contains a parameter
    ///   the family does not expose (should be unreachable on a model produced
    ///   by `fit`)
    /// - [`GamlssError::Shape`] — design-matrix assembly produced an incompatible
    ///   shape (also unreachable on a well-formed model)
    pub fn predict<D: Distribution + ?Sized>(
        &self,
        new_data: &DataSet,
        family: &D,
    ) -> Result<HashMap<String, Array1<f64>>, GamlssError> {
        let n_obs = new_data
            .n_obs()
            .ok_or_else(|| GamlssError::Input("new_data has no columns".into()))?;
        let mut predictions = HashMap::new();

        for (param_name, fitted_param) in &self.models {
            let (x_matrix, _, _, _) =
                assemble_model_matrices(new_data, n_obs, &fitted_param.terms)?;
            let eta = x_matrix.0.dot(&fitted_param.coefficients.0);
            let link = family.default_link(param_name)?;
            let fitted = eta.mapv(|e| link.inv_link(e));

            predictions.insert(param_name.clone(), fitted);
        }

        Ok(predictions)
    }

    /// Predict fitted values with standard errors (`se = √diag(X V X')`).
    ///
    /// Returns one [`PredictionResult`] per parameter. For bare fitted values
    /// without SEs, prefer [`predict`](Self::predict).
    ///
    /// # Errors
    ///
    /// Same error set as [`GamlssModel::predict`].
    pub fn predict_with_se<D: Distribution + ?Sized>(
        &self,
        new_data: &DataSet,
        family: &D,
    ) -> Result<HashMap<String, PredictionResult>, GamlssError> {
        let n_obs = new_data
            .n_obs()
            .ok_or_else(|| GamlssError::Input("new_data has no columns".into()))?;
        let mut results = HashMap::new();

        for (param_name, fitted_param) in &self.models {
            let (x_matrix, _, _, _) =
                assemble_model_matrices(new_data, n_obs, &fitted_param.terms)?;
            let eta = x_matrix.0.dot(&fitted_param.coefficients.0);

            let v = &fitted_param.covariance.0;
            let se_eta: Array1<f64> = x_matrix
                .0
                .rows()
                .into_iter()
                .map(|x_i| {
                    let v_x_i = v.dot(&x_i);
                    x_i.dot(&v_x_i).max(0.0).sqrt()
                })
                .collect();

            let link = family.default_link(param_name)?;
            let fitted = eta.view().mapv(|e| link.inv_link(e));

            results.insert(
                param_name.clone(),
                PredictionResult {
                    fitted,
                    eta,
                    se_eta,
                },
            );
        }

        Ok(results)
    }

    /// Compute aggregate diagnostics (residuals, log-likelihood, AIC, BIC, EDF)
    /// for this fitted model against the observed response `y`.
    ///
    /// # Errors
    ///
    /// - [`GamlssError::UnknownParameter`] — family requires a parameter not
    ///   present in `self.models`
    /// - [`GamlssError::Input`] — family's variance or log-density evaluation
    ///   rejects the parameter values (e.g. invalid support)
    pub fn diagnostics<D: Distribution + ?Sized>(
        &self,
        family: &D,
        y: &Array1<f64>,
    ) -> Result<ModelDiagnostics, GamlssError> {
        fitting::diagnostics::compute(&self.models, family, y)
    }

    /// Sample from the posterior distribution of coefficients for a given parameter.
    ///
    /// Uses Cholesky decomposition of the covariance matrix to generate samples
    /// from the approximate posterior N(beta_hat, V_beta).
    ///
    /// Pass `seed = Some(s)` for reproducible samples; `seed = None` uses an
    /// unseeded thread-local RNG.
    ///
    /// # Errors
    ///
    /// Returns [`GamlssError::UnknownParameter`] if `param_name` is not in the fitted
    /// model, or [`GamlssError::PosteriorNotPositiveDefinite`] if the covariance is
    /// not positive definite (degenerate fit).
    pub fn posterior_samples(
        &self,
        param_name: &str,
        n_samples: usize,
        seed: Option<u64>,
    ) -> Result<Vec<Coefficients>, GamlssError> {
        let fitted_param =
            self.models
                .get(param_name)
                .ok_or_else(|| GamlssError::UnknownParameter {
                    distribution: "<fitted model>".to_string(),
                    param: param_name.to_string(),
                })?;

        let samples = fitting::sample_posterior_seeded(
            &fitted_param.coefficients,
            &fitted_param.covariance,
            n_samples,
            seed,
        )?;

        Ok(samples.into_iter().map(Coefficients).collect())
    }

    /// Generate prediction samples by sampling from posterior and propagating through predictions.
    ///
    /// For each posterior sample of coefficients, computes predictions on new data.
    /// Returns samples of fitted values on the response scale.
    ///
    /// Pass `seed = Some(s)` for reproducible samples across calls; `seed = None`
    /// uses an unseeded thread-local RNG.
    ///
    /// # Errors
    ///
    /// In addition to the error set of [`GamlssModel::predict`], this can return
    /// [`GamlssError::PosteriorNotPositiveDefinite`] if any parameter's
    /// covariance matrix fails Cholesky factorization.
    pub fn predict_samples<D: Distribution + ?Sized>(
        &self,
        new_data: &DataSet,
        family: &D,
        n_samples: usize,
        seed: Option<u64>,
    ) -> Result<HashMap<String, Vec<Array1<f64>>>, GamlssError> {
        let n_obs = new_data
            .n_obs()
            .ok_or_else(|| GamlssError::Input("new_data has no columns".into()))?;
        let mut results = HashMap::new();

        for (param_name, fitted_param) in &self.models {
            let (x_matrix, _, _, _) =
                assemble_model_matrices(new_data, n_obs, &fitted_param.terms)?;

            let beta_samples = fitting::sample_posterior_seeded(
                &fitted_param.coefficients,
                &fitted_param.covariance,
                n_samples,
                seed,
            )?;

            let link = family.default_link(param_name)?;

            let prediction_samples: Vec<Array1<f64>> = beta_samples
                .iter()
                .map(|beta| {
                    let eta = x_matrix.0.dot(beta);
                    eta.mapv(|e| link.inv_link(e))
                })
                .collect();

            results.insert(param_name.clone(), prediction_samples);
        }

        Ok(results)
    }
}

/// Prediction output containing fitted values, linear predictor, and standard errors.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PredictionResult {
    /// Response-scale: link⁻¹(eta).
    pub fitted: Array1<f64>,
    pub eta: Array1<f64>,
    /// Standard errors on the linear-predictor scale.
    pub se_eta: Array1<f64>,
}

/// R-style summary: convergence status, per-parameter EDF, λ values, and the
/// head of each coefficient vector. The full coefficient/covariance state is
/// still available on the `models` field.
impl fmt::Display for GamlssModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "GamlssModel: converged={}, iterations={}, final_change={:.3e}",
            self.diagnostics.converged, self.diagnostics.iterations, self.diagnostics.final_change,
        )?;
        for (name, fitted) in &self.models {
            let lambdas = if fitted.lambdas.is_empty() {
                "[]".to_string()
            } else {
                let inner = fitted
                    .lambdas
                    .iter()
                    .map(|l| format!("{:.4}", l))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{}]", inner)
            };
            writeln!(f, "  {} (edf={:.3}, lambdas={})", name, fitted.edf, lambdas)?;
            let coeffs = &fitted.coefficients.0;
            let preview: Vec<String> = coeffs.iter().take(6).map(|c| format!("{:.4}", c)).collect();
            let tail = if coeffs.len() > 6 {
                format!(", … ({} more)", coeffs.len() - 6)
            } else {
                String::new()
            };
            writeln!(f, "    coefficients: [{}{}]", preview.join(", "), tail)?;
        }
        Ok(())
    }
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize)]
struct SerializedModel<'a> {
    distribution: String,
    model: &'a GamlssModel,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct OwnedSerializedModel {
    distribution: String,
    model: GamlssModel,
}
