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
///
/// # Smooths on scale/shape parameters
///
/// A `Smooth` is valid on any distribution parameter, not just the location
/// (`mu`) — e.g. a `PSpline1D` on `sigma` to model nonlinear heteroskedasticity.
/// The default REML smoothing-parameter selection recovers these scale/shape
/// curves (see `tests/scale_smooth_recovery.rs`). If a smooth carries little
/// signal its penalty can drive it down to a straight line (its penalty null
/// space); when that happens the fit records a message in
/// [`FitDiagnostics::warnings`](crate::FitDiagnostics) and you can inspect
/// [`FittedParameter::term_edf`](crate::fitting::FittedParameter) — a per-term
/// effective-degrees-of-freedom near the term's null-space dimension means the
/// curve collapsed and a `Linear` term would be the honest description.
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
    /// 1D natural cubic regression spline (mgcv `bs = "cr"`).
    ///
    /// Knots are placed at quantiles of the training data (matching mgcv's default).
    /// Natural boundary conditions (`f'' = 0` at the outer knots) ensure linear
    /// extrapolation beyond the data range — preventing the edge curl that
    /// unconstrained B-splines produce.
    ///
    /// The penalty is the exact integrated squared second derivative `∫ [f'']²`,
    /// whose null space is spanned by constants and linear functions (rank k-2).
    CrSpline1D {
        /// Column name for the predictor variable.
        col_name: String,
        /// Number of knots / basis functions (mgcv default: 6).
        k: usize,
        /// Optional point constraint: pin `f(pc) = 0` (e.g. `pc = 0` for
        /// concessions dollars). When set, replaces the sum-to-zero centering
        /// used for identifiability when an intercept is also present.
        #[cfg_attr(feature = "serde", serde(default))]
        pc: Option<f64>,
        /// Quantile knot positions, length `k`.  **Leave empty when building a
        /// formula** — they are resolved once from training data at fit time and
        /// then stored here so prediction reuses the identical basis.
        #[cfg_attr(feature = "serde", serde(default))]
        knots: Vec<f64>,
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
            Smooth::CrSpline1D { col_name, .. } => vec![col_name.as_str()],
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
    /// Default number of knots for `Smooth::CrSpline1D` (matching mgcv default).
    const DEFAULT_CR_K: usize = 6;

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

        // Hoist the kwargs item binding so it outlives the borrow taken by `kwargs`.
        let kwargs_item = if tuple.len() >= 3 {
            Some(tuple.get_item(2)?)
        } else {
            None
        };
        let kwargs: Option<&Bound<PyDict>> =
            kwargs_item.as_ref().map(|item| item.cast()).transpose()?;

        // Dispatch on the `bs` basis type kwarg (default: "ps" = P-spline).
        let bs = if let Some(kw) = kwargs {
            kwarg_str_or(kw, "bs", "ps")?
        } else {
            "ps".to_string()
        };

        match bs.as_str() {
            "cr" => {
                let (k, pc) = if let Some(kw) = kwargs {
                    (kwarg_or(kw, "k", DEFAULT_CR_K)?, kwarg_opt_f64(kw, "pc")?)
                } else {
                    (DEFAULT_CR_K, None)
                };
                Ok(Term::Smooth(Smooth::CrSpline1D {
                    col_name,
                    k,
                    pc,
                    knots: vec![], // resolved from training data at fit time
                }))
            }
            _ => {
                let (n_splines, degree, penalty_order) = if let Some(kw) = kwargs {
                    (
                        kwarg_or(kw, "n_splines", DEFAULT_N_SPLINES)?,
                        kwarg_or(kw, "degree", DEFAULT_DEGREE)?,
                        kwarg_or(kw, "penalty_order", DEFAULT_PENALTY_ORDER)?,
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
        }
    }

    fn kwarg_or(kwargs: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<usize> {
        Ok(kwargs
            .get_item(key)?
            .map(|v| v.extract::<usize>())
            .transpose()?
            .unwrap_or(default))
    }

    fn kwarg_str_or(kwargs: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
        Ok(kwargs
            .get_item(key)?
            .map(|v| v.extract::<String>())
            .transpose()?
            .unwrap_or_else(|| default.to_string()))
    }

    fn kwarg_opt_f64(kwargs: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
        match kwargs.get_item(key)? {
            None => Ok(None),
            Some(v) => {
                if v.is_none() {
                    Ok(None)
                } else {
                    Ok(Some(v.extract::<f64>()?))
                }
            }
        }
    }
}
