//! Model formula terms: building blocks for specifying regression terms in GAMLSS.
//!
//! A formula consists of terms that specify how each distribution parameter (μ, σ, ν, ...)
//! depends on predictor variables. Terms can be parametric (intercept, linear) or semiparametric
//! (smooth with penalties, random effects).

/// A single term in a model formula: intercept, linear effect, or smooth.
///
/// Terms are combined into a `Formula` to specify which predictors affect each parameter.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Term {
    /// Intercept term (no column needed).
    Intercept,
    /// Linear effect of a column.
    Linear { col_name: String },
    /// Smooth effect: P-spline, tensor product, or random effect.
    Smooth(Smooth),
}

impl Term {
    /// Returns the column names referenced by this term.
    pub fn column_names(&self) -> Vec<&str> {
        match self {
            Term::Intercept => vec![],
            Term::Linear { col_name } => vec![col_name.as_str()],
            Term::Smooth(smooth) => smooth.column_names(),
        }
    }
}

/// Smooth term specification for nonlinear effects and random intercepts.
///
/// Smooth terms enable flexible, data-driven modeling through penalized basis expansions.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Smooth {
    /// 1D P-spline: flexible univariate smooth effect.
    /// The smoothing parameter (λ) balances fit vs. smoothness.
    PSpline1D {
        /// Column name for the predictor variable.
        col_name: String,
        /// Number of B-spline basis functions (typical: 5-50).
        n_splines: usize,
        /// Polynomial degree of the B-spline basis (typical: 2-3).
        degree: usize,
        /// Penalty matrix order: 1 = linear trends, 2 = constant second differences (typical).
        penalty_order: usize,
    },
    /// 2D tensor product of two P-spline bases for interaction terms.
    /// Suitable for modeling smooth interactions: f(x₁, x₂).
    TensorProduct {
        /// Column name for the first predictor.
        col_name_1: String,
        /// Number of basis functions for the first marginal basis.
        n_splines_1: usize,
        /// Penalty order for the first margin.
        penalty_order_1: usize,

        /// Column name for the second predictor.
        col_name_2: String,
        /// Number of basis functions for the second marginal basis.
        n_splines_2: usize,
        /// Penalty order for the second margin.
        penalty_order_2: usize,

        /// Polynomial degree shared across both marginal bases.
        degree: usize,
    },
    /// Random intercept term indexed by a grouping variable.
    /// Assumes each group has its own random intercept ~ N(0, σ²_u).
    RandomEffect {
        /// Column name containing group identifiers (e.g., subject ID).
        col_name: String,
    },
}

impl Smooth {
    /// Returns the column names referenced by this smooth term.
    pub fn column_names(&self) -> Vec<&str> {
        match self {
            Smooth::PSpline1D { col_name, .. } => vec![col_name.as_str()],
            Smooth::TensorProduct {
                col_name_1,
                col_name_2,
                ..
            } => vec![col_name_1.as_str(), col_name_2.as_str()],
            Smooth::RandomEffect { col_name } => vec![col_name.as_str()],
        }
    }
}

/// Parsing utilities for the Python FFI: lifts term parsing out of `python.rs`
/// so the FFI surface stays a thin marshalling layer.
#[cfg(feature = "python")]
pub(crate) mod py_parse {
    use super::{Smooth, Term};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyTuple};

    /// Defaults for `Smooth::PSpline1D` when omitted from the Python tuple's kwargs.
    const DEFAULT_N_SPLINES: usize = 10;
    const DEFAULT_DEGREE: usize = 3;
    const DEFAULT_PENALTY_ORDER: usize = 2;

    /// Parse a Python list of term tuples into `Vec<Term>`.
    pub fn parse_terms(term_list: &Bound<'_, pyo3::types::PyList>) -> PyResult<Vec<Term>> {
        term_list
            .iter()
            .map(|item| parse_single_term(&item))
            .collect()
    }

    /// Parse one term tuple. Supported shapes:
    /// - `("intercept",)`
    /// - `("linear", "col_name")`
    /// - `("smooth", "col_name")` or `("smooth", "col_name", {kwargs})`
    /// - `("random", "col_name")`
    fn parse_single_term(item: &Bound<'_, PyAny>) -> PyResult<Term> {
        let tuple: &Bound<PyTuple> = item.cast()?;
        if tuple.is_empty() {
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
            "smooth" => parse_smooth(tuple),
            "random" => {
                if tuple.len() != 2 {
                    return Err(PyValueError::new_err(
                        "Random effect requires: ('random', 'col_name')",
                    ));
                }
                let col_name: String = tuple.get_item(1)?.extract()?;
                Ok(Term::Smooth(Smooth::RandomEffect { col_name }))
            }
            other => Err(PyValueError::new_err(format!(
                "Unknown term type: {}. Use 'intercept', 'linear', 'smooth', or 'random'",
                other
            ))),
        }
    }

    fn parse_smooth(tuple: &Bound<'_, PyTuple>) -> PyResult<Term> {
        if tuple.len() < 2 {
            return Err(PyValueError::new_err(
                "Smooth term requires at least: ('smooth', 'col_name')",
            ));
        }
        let col_name: String = tuple.get_item(1)?.extract()?;

        let (n_splines, degree, penalty_order) = if tuple.len() >= 3 {
            let kwargs_item = tuple.get_item(2)?;
            let kwargs: &Bound<PyDict> = kwargs_item.cast()?;
            (
                kwarg_or(kwargs, "n_splines", DEFAULT_N_SPLINES)?,
                kwarg_or(kwargs, "degree", DEFAULT_DEGREE)?,
                kwarg_or(kwargs, "penalty_order", DEFAULT_PENALTY_ORDER)?,
            )
        } else {
            (DEFAULT_N_SPLINES, DEFAULT_DEGREE, DEFAULT_PENALTY_ORDER)
        };

        Ok(Term::Smooth(Smooth::PSpline1D {
            col_name,
            n_splines,
            degree,
            penalty_order,
        }))
    }

    fn kwarg_or(kwargs: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<usize> {
        Ok(kwargs
            .get_item(key)?
            .map(|v| v.extract::<usize>())
            .transpose()?
            .unwrap_or(default))
    }
}
