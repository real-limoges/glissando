use ndarray::Array1;
use numpy::{PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;

use crate::{distributions::*, DataSet, Formula, GamlssError, GamlssModel, Smooth, Term};

// Internal enum to dispatch to concrete distribution types
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
    fn fit_model(
        &self,
        data: &DataSet,
        y: &Array1<f64>,
        formula: &Formula,
    ) -> Result<GamlssModel, GamlssError> {
        match self {
            FamilyType::Gaussian(d) => GamlssModel::fit(data, y, formula, d),
            FamilyType::Poisson(d) => GamlssModel::fit(data, y, formula, d),
            FamilyType::Binomial(d) => GamlssModel::fit(data, y, formula, d),
            FamilyType::Gamma(d) => GamlssModel::fit(data, y, formula, d),
            FamilyType::NegativeBinomial(d) => GamlssModel::fit(data, y, formula, d),
            FamilyType::Beta(d) => GamlssModel::fit(data, y, formula, d),
            FamilyType::StudentT(d) => GamlssModel::fit(data, y, formula, d),
        }
    }

    fn predict(
        &self,
        model: &GamlssModel,
        new_data: &DataSet,
    ) -> Result<HashMap<String, Array1<f64>>, GamlssError> {
        match self {
            FamilyType::Gaussian(d) => model.predict(new_data, d),
            FamilyType::Poisson(d) => model.predict(new_data, d),
            FamilyType::Binomial(d) => model.predict(new_data, d),
            FamilyType::Gamma(d) => model.predict(new_data, d),
            FamilyType::NegativeBinomial(d) => model.predict(new_data, d),
            FamilyType::Beta(d) => model.predict(new_data, d),
            FamilyType::StudentT(d) => model.predict(new_data, d),
        }
    }
}

// Distribution wrapper classes
#[pyclass(name = "Gaussian", frozen)]
struct PyGaussian;

#[pymethods]
impl PyGaussian {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "Poisson", frozen)]
struct PyPoisson;

#[pymethods]
impl PyPoisson {
    #[new]
    fn new() -> Self {
        Self
    }
}

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

#[pyclass(name = "Gamma", frozen)]
struct PyGamma;

#[pymethods]
impl PyGamma {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "NegativeBinomial", frozen)]
struct PyNegativeBinomial;

#[pymethods]
impl PyNegativeBinomial {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "Beta", frozen)]
struct PyBeta;

#[pymethods]
impl PyBeta {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(name = "StudentT", frozen)]
struct PyStudentT;

#[pymethods]
impl PyStudentT {
    #[new]
    fn new() -> Self {
        Self
    }
}

// Helper functions
fn py_dict_to_dataset(_py: Python, py_dict: &Bound<PyDict>) -> PyResult<DataSet> {
    let mut dataset = DataSet::new();

    for (key, value) in py_dict.iter() {
        let col_name: String = key.extract()?;
        let array: PyReadonlyArray1<f64> = value.extract()?;
        dataset.insert_column(col_name, array.as_array().to_owned());
    }

    Ok(dataset)
}

#[allow(deprecated)]
fn py_dict_to_formula(py: Python, py_dict: &Bound<PyDict>) -> PyResult<Formula> {
    let mut formula = Formula::new();

    for (param, terms) in py_dict.iter() {
        let param_name: String = param.extract()?;
        // Use borrowed downcast for iterator values
        let term_list: &Bound<pyo3::types::PyList> = terms.downcast()?;
        let rust_terms = parse_terms(py, term_list)?;
        formula.add_terms(param_name, rust_terms);
    }

    Ok(formula)
}

fn parse_terms(py: Python, term_list: &Bound<pyo3::types::PyList>) -> PyResult<Vec<Term>> {
    let mut terms = Vec::new();

    for item in term_list.iter() {
        let term = parse_single_term(py, &item)?;
        terms.push(term);
    }

    Ok(terms)
}

#[allow(deprecated)]
fn parse_single_term(_py: Python, item: &Bound<PyAny>) -> PyResult<Term> {
    // Each term is a tuple: ("intercept",) or ("linear", "x") or ("smooth", "x", {...})
    let tuple: &Bound<PyTuple> = item.downcast()?;

    if tuple.len() == 0 {
        return Err(PyValueError::new_err("Empty term tuple"));
    }

    let term_type: String = tuple.get_item(0)?.extract()?;

    match term_type.as_str() {
        "intercept" => Ok(Term::Intercept),

        "linear" => {
            if tuple.len() != 2 {
                return Err(PyValueError::new_err(
                    "Linear term requires: ('linear', 'col_name')",
                ));
            }
            let col_name: String = tuple.get_item(1)?.extract()?;
            Ok(Term::Linear { col_name })
        }

        "smooth" => {
            if tuple.len() < 2 {
                return Err(PyValueError::new_err(
                    "Smooth term requires at least: ('smooth', 'col_name')",
                ));
            }
            let col_name: String = tuple.get_item(1)?.extract()?;

            // Optional kwargs dict for smooth parameters
            #[allow(deprecated)]
            let (n_splines, degree, penalty_order) = if tuple.len() >= 3 {
                let kwargs_item = tuple.get_item(2)?;
                let kwargs: &Bound<PyDict> = kwargs_item.downcast()?;
                let n_splines = kwargs
                    .get_item("n_splines")?
                    .map(|v| v.extract::<usize>())
                    .transpose()?
                    .unwrap_or(10);
                let degree = kwargs
                    .get_item("degree")?
                    .map(|v| v.extract::<usize>())
                    .transpose()?
                    .unwrap_or(3);
                let penalty_order = kwargs
                    .get_item("penalty_order")?
                    .map(|v| v.extract::<usize>())
                    .transpose()?
                    .unwrap_or(2);
                (n_splines, degree, penalty_order)
            } else {
                (10, 3, 2)
            };

            Ok(Term::Smooth(Smooth::PSpline1D {
                col_name,
                n_splines,
                degree,
                penalty_order,
            }))
        }

        "random" => {
            if tuple.len() != 2 {
                return Err(PyValueError::new_err(
                    "Random effect requires: ('random', 'col_name')",
                ));
            }
            let col_name: String = tuple.get_item(1)?.extract()?;
            Ok(Term::Smooth(Smooth::RandomEffect { col_name }))
        }

        _ => Err(PyValueError::new_err(format!(
            "Unknown term type: {}. Use 'intercept', 'linear', 'smooth', or 'random'",
            term_type
        ))),
    }
}

fn extract_family(_py: Python, family_obj: &Bound<PyAny>) -> PyResult<FamilyType> {
    // Try each distribution type
    if family_obj.extract::<PyRef<PyGaussian>>().is_ok() {
        return Ok(FamilyType::Gaussian(Gaussian::new()));
    }
    if family_obj.extract::<PyRef<PyPoisson>>().is_ok() {
        return Ok(FamilyType::Poisson(Poisson::new()));
    }
    if let Ok(b) = family_obj.extract::<PyRef<PyBinomial>>() {
        let n_trials = Array1::from_vec(b.n_trials.clone());
        return Ok(FamilyType::Binomial(Binomial::with_trials(n_trials)));
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
        "Unknown distribution type. Use Gaussian(), Poisson(), Binomial(), Gamma(), NegativeBinomial(), Beta(), or StudentT()"
    ))
}

// Main model wrapper
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
    ///
    /// Examples
    /// --------
    /// >>> model = GamlssModel.fit(
    /// ...     data={'x': [1.0, 2.0, 3.0, 4.0, 5.0]},
    /// ...     y=[2.1, 4.0, 5.9, 8.1, 10.0],
    /// ...     formula={
    /// ...         'mu': [('intercept',), ('linear', 'x')],
    /// ...         'sigma': [('intercept',)]
    /// ...     },
    /// ...     family=Gaussian()
    /// ... )
    #[staticmethod]
    fn fit(
        py: Python,
        data: &Bound<PyDict>,
        y: PyReadonlyArray1<f64>,
        formula: &Bound<PyDict>,
        family: &Bound<PyAny>,
    ) -> PyResult<Self> {
        // Convert inputs
        let dataset = py_dict_to_dataset(py, data)?;
        let y_array = y.as_array().to_owned();
        let rust_formula = py_dict_to_formula(py, formula)?;
        let family_type = extract_family(py, family)?;

        // Fit model
        let model = family_type
            .fit_model(&dataset, &y_array, &rust_formula)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))?;

        Ok(Self {
            inner: model,
            family: family_type,
        })
    }

    /// Predict fitted values for new data.
    ///
    /// Parameters
    /// ----------
    /// new_data : dict
    ///     Dictionary mapping column names to 1D arrays
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary mapping parameter names to predicted values (1D arrays)
    fn predict(&self, py: Python, new_data: &Bound<PyDict>) -> PyResult<Py<PyDict>> {
        let dataset = py_dict_to_dataset(py, new_data)?;

        let predictions = self
            .family
            .predict(&self.inner, &dataset)
            .map_err(|e| PyRuntimeError::new_err(format!("Prediction failed: {}", e)))?;

        // Convert HashMap<String, Array1<f64>> to Python dict
        let py_dict = PyDict::new(py);
        for (param_name, values) in predictions {
            let py_array = values.to_pyarray(py);
            py_dict.set_item(param_name, py_array)?;
        }

        Ok(py_dict.into())
    }

    /// Check if the model converged.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the model converged, False otherwise
    fn converged(&self) -> bool {
        self.inner.converged()
    }
}

#[pymodule]
fn gamlss_rs(m: &Bound<PyModule>) -> PyResult<()> {
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
