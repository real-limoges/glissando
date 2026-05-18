//! Python bindings via PyO3.
//!
//! Exposes model fitting and prediction over NumPy arrays. Term and family parsing
//! lives in `terms::py_parse` and `distributions` so this file stays a thin
//! marshalling layer between Python and the core Rust API.

use ndarray::Array1;
use numpy::{PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::terms::py_parse;
use crate::{distributions::*, DataSet, Formula, GamlssError, GamlssModel};

/// Internal enum dispatching to a concrete Distribution while preserving its concrete type.
enum FamilyType {
    Gaussian(Gaussian),
    Poisson(Poisson),
    Binomial(Binomial),
    Gamma(Gamma),
    NegativeBinomial(NegativeBinomial),
    Beta(Beta),
    StudentT(StudentT),
}

impl FamilyType {
    fn as_distribution(&self) -> &dyn Distribution {
        match self {
            FamilyType::Gaussian(d) => d,
            FamilyType::Poisson(d) => d,
            FamilyType::Binomial(d) => d,
            FamilyType::Gamma(d) => d,
            FamilyType::NegativeBinomial(d) => d,
            FamilyType::Beta(d) => d,
            FamilyType::StudentT(d) => d,
        }
    }

    fn fit_model(
        &self,
        data: &DataSet,
        y: &Array1<f64>,
        formula: &Formula,
    ) -> Result<GamlssModel, GamlssError> {
        GamlssModel::fit(data, y, formula, self.as_distribution())
    }

    fn predict(
        &self,
        model: &GamlssModel,
        new_data: &DataSet,
    ) -> Result<HashMap<String, Array1<f64>>, GamlssError> {
        model.predict(new_data, self.as_distribution())
    }
}

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
    fn fit(
        data: &Bound<PyDict>,
        y: PyReadonlyArray1<f64>,
        formula: &Bound<PyDict>,
        family: &Bound<PyAny>,
    ) -> PyResult<Self> {
        let dataset = py_dict_to_dataset(data)?;
        let y_array = y.as_array().to_owned();
        let rust_formula = py_dict_to_formula(formula)?;
        let family_type = extract_family(family)?;

        let model = family_type
            .fit_model(&dataset, &y_array, &rust_formula)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))?;

        Ok(Self {
            inner: model,
            family: family_type,
        })
    }

    /// Predict fitted values for new data. Returns `{param_name: array}`.
    fn predict(&self, py: Python<'_>, new_data: &Bound<PyDict>) -> PyResult<Py<PyDict>> {
        let dataset = py_dict_to_dataset(new_data)?;
        let predictions = self
            .family
            .predict(&self.inner, &dataset)
            .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;

        let py_dict = PyDict::new(py);
        for (param_name, values) in predictions {
            py_dict.set_item(param_name, values.to_pyarray(py))?;
        }
        Ok(py_dict.into())
    }

    fn converged(&self) -> bool {
        self.inner.converged()
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
