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

use crate::distributions::{Beta, Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, StudentT};
use crate::ffi::FamilyType;
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

    Err(PyValueError::new_err(
        "Unknown distribution type. Use Gaussian(), Poisson(), Binomial(), Gamma(), NegativeBinomial(), Beta(), or StudentT()",
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
    Ok(())
}
