//! Python bindings via PyO3.
//!
//! Exposes model fitting and prediction over NumPy arrays. Term and family parsing
//! lives in `terms::py_parse` and `distributions` so this file stays a thin
//! marshalling layer between Python and the core Rust API.

use ndarray::Array1;
use numpy::{PyArray2, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use crate::distributions::{
    Beta, Binomial, Gamma, Gaussian, NegativeBinomial, Ocat, Poisson, StudentT,
};
use crate::ffi::FamilyType;
use crate::fitting::selection::{self, Direction, StepScope};
use crate::fitting::{FitConfig, SmoothingCriterion};
use crate::terms::py_parse;
use crate::{DataSet, Formula, GamlssModel};

// Stateless distribution wrappers — the Python class carries no data.
macro_rules! py_distribution {
    ($py_name:ident, $name:expr) => {
        #[pyclass(name = $name, frozen)]
        struct $py_name;

        #[pymethods]
        impl $py_name {
            #[new]
            fn new() -> Self {
                Self
            }
        }
    };
}

py_distribution!(PyGaussian, "Gaussian");
py_distribution!(PyPoisson, "Poisson");
py_distribution!(PyGamma, "Gamma");
py_distribution!(PyNegativeBinomial, "NegativeBinomial");
py_distribution!(PyBeta, "Beta");
py_distribution!(PyStudentT, "StudentT");

/// Binomial carries `n_trials` state, so it gets a manual `pyclass`.
#[pyclass(name = "Binomial", frozen)]
struct PyBinomial {
    n_trials: Vec<f64>,
}

#[pymethods]
impl PyBinomial {
    #[new]
    fn new(n_trials: Vec<f64>) -> Self {
        Self { n_trials }
    }
}

/// Ocat carries `n_categories` state, so it gets a manual `pyclass`.
#[pyclass(name = "Ocat", frozen)]
struct PyOcat {
    n_categories: usize,
}

#[pymethods]
impl PyOcat {
    #[new]
    fn new(n_categories: usize) -> PyResult<Self> {
        if !(2..=5).contains(&n_categories) {
            return Err(PyValueError::new_err("Ocat: n_categories must be 2–5"));
        }
        Ok(Self { n_categories })
    }
}

fn py_dict_to_dataset(py_dict: &Bound<'_, PyDict>) -> PyResult<DataSet> {
    let mut dataset = DataSet::new();
    for (key, value) in py_dict.iter() {
        let col_name: String = key.extract()?;
        let array: PyReadonlyArray1<f64> = value.extract()?;
        dataset
            .try_insert_column(col_name, array.as_array().to_owned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    }
    Ok(dataset)
}

fn py_dict_to_formula(py_dict: &Bound<'_, PyDict>) -> PyResult<Formula> {
    let mut formula = Formula::new();
    for (param, terms) in py_dict.iter() {
        let param_name: String = param.extract()?;
        let term_list: &Bound<pyo3::types::PyList> = terms.cast()?;
        formula.add_terms(param_name, py_parse::parse_terms(term_list)?);
    }
    Ok(formula)
}

/// Parse a `{param: [term, ...]}` dict into the candidate scope for stepwise
/// selection — each value uses the same term encoding as a formula.
fn py_dict_to_scope(scope: &Bound<'_, PyDict>) -> PyResult<Vec<StepScope>> {
    let mut out = Vec::with_capacity(scope.len());
    for (param, cands) in scope.iter() {
        let param_name: String = param.extract()?;
        let term_list: &Bound<pyo3::types::PyList> = cands.cast()?;
        out.push(StepScope {
            param: param_name,
            candidates: py_parse::parse_terms(term_list)?,
        });
    }
    Ok(out)
}

/// Build a [`FitConfig`] from an optional Python config dict (same keys as
/// `GamlssModel.fit_with_config`). Missing keys keep their defaults.
fn parse_fit_config(config: &Bound<'_, PyDict>) -> PyResult<FitConfig> {
    let mut fit_config = FitConfig::default();
    if let Some(v) = config.get_item("max_iterations")? {
        fit_config.max_iterations = v.extract()?;
    }
    if let Some(v) = config.get_item("tolerance")? {
        fit_config.tolerance = v.extract()?;
    }
    if let Some(v) = config.get_item("criterion")? {
        let s: String = v.extract()?;
        fit_config.criterion = match s.to_ascii_lowercase().as_str() {
            "gcv" => SmoothingCriterion::Gcv,
            "reml" => SmoothingCriterion::Reml,
            "fellner_schall" => SmoothingCriterion::FellnerSchall,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown criterion '{}', expected 'gcv', 'reml', or 'fellner_schall'",
                    other
                )))
            }
        };
    }
    if let Some(v) = config.get_item("step_halving")? {
        fit_config.step_halving = v.extract()?;
    }
    if let Some(v) = config.get_item("gd_tolerance")? {
        fit_config.gd_tolerance = v.extract()?;
    }
    Ok(fit_config)
}

/// Parse the `direction` string for stepwise selection.
fn parse_direction(direction: &str) -> PyResult<Direction> {
    match direction.to_ascii_lowercase().as_str() {
        "forward" => Ok(Direction::Forward),
        "backward" => Ok(Direction::Backward),
        "both" => Ok(Direction::Both),
        other => Err(PyValueError::new_err(format!(
            "Unknown direction '{}', expected 'forward', 'backward', or 'both'",
            other
        ))),
    }
}

fn extract_family(family_obj: &Bound<'_, PyAny>) -> PyResult<FamilyType> {
    if family_obj.extract::<PyRef<PyGaussian>>().is_ok() {
        return Ok(FamilyType::Gaussian(Gaussian::new()));
    }
    if family_obj.extract::<PyRef<PyPoisson>>().is_ok() {
        return Ok(FamilyType::Poisson(Poisson::new()));
    }
    if let Ok(b) = family_obj.extract::<PyRef<PyBinomial>>() {
        return Ok(FamilyType::Binomial(Binomial::with_trials(
            Array1::from_vec(b.n_trials.clone()),
        )));
    }
    if family_obj.extract::<PyRef<PyGamma>>().is_ok() {
        return Ok(FamilyType::Gamma(Gamma::new()));
    }
    if family_obj.extract::<PyRef<PyNegativeBinomial>>().is_ok() {
        return Ok(FamilyType::NegativeBinomial(NegativeBinomial::new()));
    }
    if family_obj.extract::<PyRef<PyBeta>>().is_ok() {
        return Ok(FamilyType::Beta(Beta::new()));
    }
    if family_obj.extract::<PyRef<PyStudentT>>().is_ok() {
        return Ok(FamilyType::StudentT(StudentT::new()));
    }
    if let Ok(o) = family_obj.extract::<PyRef<PyOcat>>() {
        return Ok(FamilyType::Ocat(Ocat::new(o.n_categories)));
    }

    Err(PyValueError::new_err(
        "Unknown distribution type. Use Gaussian(), Poisson(), Binomial(), Gamma(), NegativeBinomial(), Beta(), StudentT(), or Ocat(n_categories)",
    ))
}

fn fit_with_family(
    family: &FamilyType,
    data: &DataSet,
    y: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    formula: &Formula,
) -> PyResult<GamlssModel> {
    GamlssModel::fit_with_config(
        data,
        y,
        weights,
        formula,
        family.as_distribution(),
        Default::default(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
}

fn predict_with_family(
    family: &FamilyType,
    model: &GamlssModel,
    new_data: &DataSet,
) -> PyResult<HashMap<String, Array1<f64>>> {
    model
        .predict(new_data, family.as_distribution())
        .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))
}

#[pyclass(name = "GamlssModel")]
struct PyGamlssModel {
    inner: GamlssModel,
    family: FamilyType,
}

#[pymethods]
impl PyGamlssModel {
    /// Fit a GAMLSS model.
    ///
    /// Parameters
    /// ----------
    /// data : dict
    ///     Dictionary mapping column names to 1D arrays
    /// y : array
    ///     Response variable (1D array)
    /// formula : dict
    ///     Dictionary mapping parameter names to lists of terms.
    ///     Each term is a tuple: ('intercept',), ('linear', 'x'),
    ///     ('smooth', 'x', {'n_splines': 10}), or ('random', 'group')
    /// family : Distribution
    ///     Distribution object (e.g., Gaussian(), Poisson())
    ///
    /// Returns
    /// -------
    /// GamlssModel
    ///     Fitted model object
    #[staticmethod]
    #[pyo3(signature = (data, y, formula, family, weights=None))]
    fn fit(
        data: &Bound<PyDict>,
        y: PyReadonlyArray1<f64>,
        formula: &Bound<PyDict>,
        family: &Bound<PyAny>,
        weights: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let dataset = py_dict_to_dataset(data)?;
        let y_array = y.as_array().to_owned();
        let w_array = weights.as_ref().map(|w| w.as_array().to_owned());
        let rust_formula = py_dict_to_formula(formula)?;
        let family_type = extract_family(family)?;

        let model = fit_with_family(
            &family_type,
            &dataset,
            &y_array,
            w_array.as_ref(),
            &rust_formula,
        )?;

        Ok(Self {
            inner: model,
            family: family_type,
        })
    }

    /// Fit a GAMLSS model with custom configuration.
    ///
    /// Parameters
    /// ----------
    /// data, y, formula, family : same as `fit`
    /// config : dict
    ///     Optional config keys:
    ///         `max_iterations` (int)
    ///         `tolerance` (float)
    ///         `criterion` ("reml", "gcv", or "fellner_schall"; default "reml")
    ///         `step_halving` (bool; default True) — monotone-descent line search
    ///         `gd_tolerance` (float; default 1e-3) — global-deviance convergence tol
    #[staticmethod]
    #[pyo3(signature = (data, y, formula, family, config, weights=None))]
    fn fit_with_config(
        data: &Bound<PyDict>,
        y: PyReadonlyArray1<f64>,
        formula: &Bound<PyDict>,
        family: &Bound<PyAny>,
        config: &Bound<PyDict>,
        weights: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let dataset = py_dict_to_dataset(data)?;
        let y_array = y.as_array().to_owned();
        let w_array = weights.as_ref().map(|w| w.as_array().to_owned());
        let rust_formula = py_dict_to_formula(formula)?;
        let family_type = extract_family(family)?;

        let fit_config = parse_fit_config(config)?;

        let model = GamlssModel::fit_with_config(
            &dataset,
            &y_array,
            w_array.as_ref(),
            &rust_formula,
            family_type.as_distribution(),
            fit_config,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))?;

        Ok(Self {
            inner: model,
            family: family_type,
        })
    }

    /// Predict fitted values for new data. Returns `{param_name: array}`.
    fn predict(&self, py: Python<'_>, new_data: &Bound<PyDict>) -> PyResult<Py<PyDict>> {
        let dataset = py_dict_to_dataset(new_data)?;
        let predictions = predict_with_family(&self.family, &self.inner, &dataset)?;

        let py_dict = PyDict::new(py);
        for (param_name, values) in predictions {
            py_dict.set_item(param_name, values.to_pyarray(py))?;
        }
        Ok(py_dict.into())
    }

    /// Predict with standard errors on the linear-predictor scale.
    ///
    /// Returns `{param_name: {"fitted": array, "eta": array, "se_eta": array}}`.
    fn predict_with_se(&self, py: Python<'_>, new_data: &Bound<PyDict>) -> PyResult<Py<PyDict>> {
        let dataset = py_dict_to_dataset(new_data)?;
        let results = self
            .inner
            .predict_with_se(&dataset, self.family.as_distribution())
            .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;

        let py_dict = PyDict::new(py);
        for (param_name, pr) in results {
            let inner = PyDict::new(py);
            inner.set_item("fitted", pr.fitted.to_pyarray(py))?;
            inner.set_item("eta", pr.eta.to_pyarray(py))?;
            inner.set_item("se_eta", pr.se_eta.to_pyarray(py))?;
            py_dict.set_item(param_name, inner)?;
        }
        Ok(py_dict.into())
    }

    /// Generate `n_samples` prediction samples from the approximate posterior.
    ///
    /// Returns `{param_name: list[array]}` — each list has `n_samples` arrays
    /// of length n_obs.
    ///
    /// Pass an integer `seed` for reproducible samples; omit or pass `None` for
    /// non-deterministic (unseeded) sampling.
    #[pyo3(signature = (new_data, n_samples, seed = None))]
    fn predict_samples(
        &self,
        py: Python<'_>,
        new_data: &Bound<PyDict>,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<Py<PyDict>> {
        let dataset = py_dict_to_dataset(new_data)?;
        let results = self
            .inner
            .predict_samples(&dataset, self.family.as_distribution(), n_samples, seed)
            .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;

        let py_dict = PyDict::new(py);
        for (param_name, samples) in results {
            let list = PyList::empty(py);
            for s in samples {
                list.append(s.to_pyarray(py))?;
            }
            py_dict.set_item(param_name, list)?;
        }
        Ok(py_dict.into())
    }

    /// Coefficient vector for a fitted parameter, as a numpy array.
    fn coefficients<'py>(
        &self,
        py: Python<'py>,
        param: &str,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let fitted = self.inner.models.get(param).ok_or_else(|| {
            PyKeyError::new_err(format!(
                "Parameter '{}' not found. Available: {:?}",
                param,
                self.inner.models.keys().collect::<Vec<_>>()
            ))
        })?;
        Ok(fitted.coefficients.0.to_pyarray(py))
    }

    /// Fitted-values vector for a fitted parameter, as a numpy array.
    fn fitted_values<'py>(
        &self,
        py: Python<'py>,
        param: &str,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let fitted = self.inner.models.get(param).ok_or_else(|| {
            PyKeyError::new_err(format!(
                "Parameter '{}' not found. Available: {:?}",
                param,
                self.inner.models.keys().collect::<Vec<_>>()
            ))
        })?;
        Ok(fitted.fitted_values.to_pyarray(py))
    }

    fn converged(&self) -> bool {
        self.inner.converged()
    }

    /// Returns the `(n_obs × n_coeffs)` linear-predictor design matrix for `param`
    /// as a numpy float64 array. Equivalent to mgcv's `predict(type="lpmatrix")`.
    fn design_matrix<'py>(
        &self,
        py: Python<'py>,
        new_data: &Bound<'_, PyDict>,
        param: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let dataset = py_dict_to_dataset(new_data)?;
        let x = self
            .inner
            .design_matrix(&dataset, param)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(x.to_pyarray(py))
    }

    /// Returns the `(p × p)` posterior covariance matrix `V = (X'WX + Σλ·S)⁻¹`
    /// for `param` as a numpy float64 array.
    fn covariance_matrix<'py>(
        &self,
        py: Python<'py>,
        param: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let v = self
            .inner
            .covariance_matrix(param)
            .map_err(|e| PyKeyError::new_err(e.to_string()))?;
        Ok(v.0.to_pyarray(py))
    }

    /// Returns the term → coefficient column block map for `param` as a Python dict.
    ///
    /// Each key is the mgcv-style term name; each value is a `(first_col, last_col_exclusive)`
    /// tuple of ints. Column order matches `design_matrix` and `coefficients`.
    fn term_index_map(&self, py: Python<'_>, param: &str) -> PyResult<Py<pyo3::types::PyDict>> {
        let blocks = self
            .inner
            .term_index_map(param)
            .map_err(|e| PyKeyError::new_err(e.to_string()))?;
        let d = pyo3::types::PyDict::new(py);
        for (name, first, last) in blocks {
            d.set_item(name, (*first, *last))?;
        }
        Ok(d.into())
    }

    /// Predict the `(n_obs × R)` category-probability matrix for an ordered-categorical model.
    ///
    /// Returns a 2D numpy array of shape `(n_obs, R)` where each row sums to 1.
    /// Raises `RuntimeError` if the model was not fitted with an `Ocat` family.
    fn predict_class_probabilities<'py>(
        &self,
        py: Python<'py>,
        new_data: &Bound<'_, PyDict>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let ocat = self.family.as_ocat().ok_or_else(|| {
            PyRuntimeError::new_err(
                "predict_class_probabilities requires a model fitted with Ocat family",
            )
        })?;
        let dataset = py_dict_to_dataset(new_data)?;
        let probs = self
            .inner
            .predict_class_probabilities(&dataset, ocat)
            .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;
        Ok(probs.to_pyarray(py))
    }

    /// Randomized normalized quantile residuals (gamlss's default residual), as a
    /// numpy array. Standard normal if the model is correct, regardless of family.
    ///
    /// `seed` makes the discrete-family randomization reproducible (ignored for
    /// continuous families). Evaluate against the response `y` the model was fit on.
    #[pyo3(signature = (y, seed=None))]
    fn quantile_residuals<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let y_array = y.as_array().to_owned();
        let residuals = self
            .inner
            .quantile_residuals(self.family.as_distribution(), &y_array, seed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(residuals.to_pyarray(py))
    }

    /// Response-scale centile curves for `new_data`. Returns `{"C<pct>": array}`.
    ///
    /// `percentiles` are centile levels in percent (gamlss defaults:
    /// 0.4, 2, 10, 25, 50, 75, 90, 98, 99.6).
    fn centiles(
        &self,
        py: Python<'_>,
        new_data: &Bound<PyDict>,
        percentiles: Vec<f64>,
    ) -> PyResult<Py<PyDict>> {
        let dataset = py_dict_to_dataset(new_data)?;
        let curves = self
            .inner
            .centiles(&dataset, self.family.as_distribution(), &percentiles)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let py_dict = PyDict::new(py);
        for (key, values) in curves {
            py_dict.set_item(key, values.to_pyarray(py))?;
        }
        Ok(py_dict.into())
    }

    /// Per-observation quantile prediction: `p[i]` is the centile level for row
    /// `i`, in `(0, 1)`. Returns a numpy array of predicted responses.
    fn quantile_prediction<'py>(
        &self,
        py: Python<'py>,
        new_data: &Bound<PyDict>,
        p: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let dataset = py_dict_to_dataset(new_data)?;
        let p_array = p.as_array().to_owned();
        let predicted = self
            .inner
            .quantile_prediction(&dataset, self.family.as_distribution(), &p_array)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(predicted.to_pyarray(py))
    }

    /// Generalized AIC at penalty `k`: `−2·loglik + k·EDF`.
    ///
    /// `k = 2` is AIC, `k = log(n)` is BIC. Evaluate against the same response
    /// `y` the model was fit on.
    fn gaic(&self, y: PyReadonlyArray1<f64>, k: f64) -> PyResult<f64> {
        let y_array = y.as_array().to_owned();
        self.inner
            .gaic(self.family.as_distribution(), &y_array, k)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Likelihood-ratio test treating `self` as the nested (small) model and
    /// `bigger` as the alternative (big) model. Returns a dict
    /// `{"lr_stat", "df", "p_value"}`.
    ///
    /// Raises `ValueError` if the pair is mis-ordered or non-nested
    /// (`edf_big ≤ edf_small`). Both models must share this model's family.
    fn lr_test(
        &self,
        py: Python<'_>,
        bigger: &PyGamlssModel,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyDict>> {
        let y_array = y.as_array().to_owned();
        let result = selection::lr_test(
            &self.inner,
            &bigger.inner,
            self.family.as_distribution(),
            &y_array,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let d = PyDict::new(py);
        d.set_item("lr_stat", result.lr_stat)?;
        d.set_item("df", result.df)?;
        d.set_item("p_value", result.p_value)?;
        Ok(d.into())
    }

    /// Information-criterion comparison table over `(label, model)` pairs, ranked
    /// in input order. Returns a list of dicts
    /// `{"label", "edf", "global_deviance", "gaic"}`.
    ///
    /// All models must be fit to the same response `y`; the family is taken from
    /// the first model. Raises `ValueError` if `models` is empty.
    #[staticmethod]
    fn ic_table(
        py: Python<'_>,
        models: Vec<(String, Py<PyGamlssModel>)>,
        y: PyReadonlyArray1<f64>,
        k: f64,
    ) -> PyResult<Py<PyList>> {
        if models.is_empty() {
            return Err(PyValueError::new_err(
                "ic_table requires at least one model",
            ));
        }
        let y_array = y.as_array().to_owned();
        // Borrow every model for the duration of the call.
        let borrowed: Vec<(String, PyRef<PyGamlssModel>)> = models
            .iter()
            .map(|(label, m)| Ok((label.clone(), m.borrow(py))))
            .collect::<PyResult<_>>()?;
        let family = borrowed[0].1.family.as_distribution();
        let pairs: Vec<(&str, &GamlssModel)> = borrowed
            .iter()
            .map(|(label, m)| (label.as_str(), &m.inner))
            .collect();
        let rows = selection::ic_table(&pairs, family, &y_array, k)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = PyList::empty(py);
        for r in &rows {
            let d = PyDict::new(py);
            d.set_item("label", &r.label)?;
            d.set_item("edf", r.edf)?;
            d.set_item("global_deviance", r.global_deviance)?;
            d.set_item("gaic", r.gaic)?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    /// Greedy stepwise term selection by GAIC(`k`) — the `stepGAIC` analog.
    ///
    /// Parameters
    /// ----------
    /// data, y, family : same as `fit`
    /// start : dict
    ///     Starting formula (e.g. intercept-only).
    /// scope : dict
    ///     `{param: [term, ...]}` of terms eligible to add/drop, same term
    ///     encoding as a formula.
    /// k : float
    ///     GAIC penalty (`2` ≡ AIC, `log(n)` ≡ BIC).
    /// direction : str
    ///     `"forward"`, `"backward"`, or `"both"` (default `"both"`).
    /// config : dict, optional
    ///     Same keys as `fit_with_config`.
    ///
    /// Returns
    /// -------
    /// dict
    ///     `{"model": GamlssModel, "trace": [{"move", "gaic", "edf"}, ...]}`.
    #[staticmethod]
    #[pyo3(signature = (data, y, family, start, scope, k, direction="both", config=None))]
    #[allow(clippy::too_many_arguments)]
    fn step_gaic(
        py: Python<'_>,
        data: &Bound<PyDict>,
        y: PyReadonlyArray1<f64>,
        family: &Bound<PyAny>,
        start: &Bound<PyDict>,
        scope: &Bound<PyDict>,
        k: f64,
        direction: &str,
        config: Option<&Bound<PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let dataset = py_dict_to_dataset(data)?;
        let y_array = y.as_array().to_owned();
        let family_type = extract_family(family)?;
        let start_formula = py_dict_to_formula(start)?;
        let scope_vec = py_dict_to_scope(scope)?;
        let dir = parse_direction(direction)?;
        let fit_config = match config {
            Some(c) => parse_fit_config(c)?,
            None => FitConfig::default(),
        };

        let result = selection::step_gaic(
            &dataset,
            &y_array,
            family_type.as_distribution(),
            start_formula,
            &scope_vec,
            k,
            dir,
            fit_config,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let trace = PyList::empty(py);
        for r in &result.trace {
            let d = PyDict::new(py);
            d.set_item("move", &r.move_)?;
            d.set_item("gaic", r.gaic)?;
            d.set_item("edf", r.edf)?;
            trace.append(d)?;
        }
        let model = PyGamlssModel {
            inner: result.model,
            family: family_type,
        };
        let out = PyDict::new(py);
        out.set_item("model", Py::new(py, model)?)?;
        out.set_item("trace", trace)?;
        Ok(out.into())
    }
}

#[pymodule]
fn glissando(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyGamlssModel>()?;
    m.add_class::<PyGaussian>()?;
    m.add_class::<PyPoisson>()?;
    m.add_class::<PyBinomial>()?;
    m.add_class::<PyGamma>()?;
    m.add_class::<PyNegativeBinomial>()?;
    m.add_class::<PyBeta>()?;
    m.add_class::<PyStudentT>()?;
    m.add_class::<PyOcat>()?;
    Ok(())
}
