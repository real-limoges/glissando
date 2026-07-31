//! Model formula terms: building blocks for specifying regression terms in GAMLSS.
//!
//! A formula consists of terms that specify how each distribution parameter (μ, σ, ν, ...)
//! depends on predictor variables. Terms can be parametric (intercept, linear) or semiparametric
//! (smooth with penalties, random effects).

/// A single term in a model formula: intercept, linear effect, smooth, offset,
/// factor, or interaction.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Term {
    Intercept,
    Linear {
        col_name: String,
    },
    /// P-spline, tensor product, or random effect.
    Smooth(Smooth),
    /// A known per-row term that enters the linear predictor `η` additively with
    /// a fixed coefficient of 1 (e.g. `log(exposure)` for a rate model). It
    /// contributes to `η`, not to `β`, so it occupies no design column.
    Offset {
        col_name: String,
    },
    /// A categorical predictor expanded into dummy columns under a contrast
    /// coding. `levels` and `labels` start empty in a hand-built formula and are
    /// resolved once from the training column at fit time (mirroring how
    /// [`Smooth::CrSpline1D`] resolves its knots), then replayed verbatim at
    /// predict time so the fit and predict design matrices agree.
    Factor {
        col_name: String,
        contrast: Contrast,
        /// Sorted distinct numeric level codes, length `L`. Empty until resolved.
        #[cfg_attr(feature = "serde", serde(default))]
        levels: Vec<f64>,
        /// Optional human-readable level labels aligned with `levels`, used for
        /// readable `col[label]` coefficient naming. Empty ⇒ fall back to the
        /// numeric code in the label.
        #[cfg_attr(feature = "serde", serde(default))]
        labels: Vec<String>,
    },
    /// Row-wise product of two terms' design columns (e.g. `x:z`). The
    /// coefficient count is the product of the two operands' column counts;
    /// factor×continuous and factor×factor both fall out of the same product.
    Interaction(Box<Term>, Box<Term>),
}

/// Contrast coding for a [`Term::Factor`] with `L` levels, producing `L − 1`
/// columns so a factor sharing its parameter with an [`Term::Intercept`] stays
/// identifiable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Contrast {
    /// Treatment (dummy) coding against the first level as baseline — R's
    /// default `contr.treatment`. Column `j` indicates level `j + 1`.
    #[default]
    Treatment,
    /// Sum-to-zero coding (`contr.sum`): each column contrasts a level against
    /// the grand mean, the last level being the negative sum of the others.
    SumToZero,
}

impl Term {
    /// A parametric linear term on `col_name` — e.g. `Term::linear("age")`.
    pub fn linear(col_name: impl Into<String>) -> Self {
        Term::Linear {
            col_name: col_name.into(),
        }
    }

    /// Wraps a [`Smooth`] as a [`Term`] — e.g. `Term::smooth(Smooth::ps("x"))`.
    pub fn smooth(smooth: Smooth) -> Self {
        Term::Smooth(smooth)
    }

    /// An offset term on `col_name` — enters `η` additively with coefficient 1.
    pub fn offset(col_name: impl Into<String>) -> Self {
        Term::Offset {
            col_name: col_name.into(),
        }
    }

    /// A categorical factor on `col_name` with treatment (dummy) contrasts —
    /// R's default. Levels resolve from the data at fit time. Use
    /// [`Term::factor_with`] to choose a different [`Contrast`].
    pub fn factor(col_name: impl Into<String>) -> Self {
        Self::factor_with(col_name, Contrast::Treatment)
    }

    /// A categorical factor on `col_name` with an explicit [`Contrast`].
    pub fn factor_with(col_name: impl Into<String>, contrast: Contrast) -> Self {
        Term::Factor {
            col_name: col_name.into(),
            contrast,
            levels: Vec::new(),
            labels: Vec::new(),
        }
    }

    /// The row-wise interaction (product) of two terms — e.g.
    /// `Term::interaction(Term::factor("region"), Term::linear("x"))`.
    pub fn interaction(left: Term, right: Term) -> Self {
        Term::Interaction(Box::new(left), Box::new(right))
    }

    /// Returns the column names referenced by this term.
    pub fn column_names(&self) -> Vec<&str> {
        match self {
            Term::Intercept => vec![],
            Term::Linear { col_name } => vec![col_name.as_str()],
            Term::Smooth(smooth) => smooth.column_names(),
            Term::Offset { col_name } => vec![col_name.as_str()],
            Term::Factor { col_name, .. } => vec![col_name.as_str()],
            Term::Interaction(left, right) => {
                let mut cols = left.column_names();
                cols.extend(right.column_names());
                cols
            }
        }
    }

    /// Returns an mgcv-style label for this term, used as the key in
    /// `FittedParameter::term_blocks` and downstream `predictors_info` maps.
    ///
    /// - `Intercept`  → `"(intercept)"`
    /// - `Linear(x)`  → `"x"`
    /// - `Smooth`     → delegates to [`Smooth::term_name`]
    pub fn term_name(&self) -> String {
        match self {
            Term::Intercept => "(intercept)".to_string(),
            Term::Linear { col_name } => col_name.clone(),
            Term::Smooth(s) => s.term_name(),
            Term::Offset { col_name } => format!("offset({col_name})"),
            Term::Factor { col_name, .. } => col_name.clone(),
            Term::Interaction(left, right) => {
                format!("{}:{}", left.term_name(), right.term_name())
            }
        }
    }
}

impl std::fmt::Display for Term {
    /// Renders a term in the R/mgcv formula spelling so a parsed formula can
    /// round-trip back to a string (`Intercept` → `1`, `Linear` → the bare name,
    /// `Smooth` → `s(..)`/`te(..)`, `Offset` → `offset(..)`, `Factor` → the bare
    /// name, `Interaction` → `a:b`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Intercept => write!(f, "1"),
            Term::Linear { col_name } => write!(f, "{col_name}"),
            Term::Smooth(s) => write!(f, "{}", s.formula_repr()),
            Term::Offset { col_name } => write!(f, "offset({col_name})"),
            // Rendered as `factor(..)` (not the bare name) so it round-trips: a
            // bare name parses back as `Linear`, whereas `factor(..)` reparses as
            // a `Factor`. Sum-to-zero coding surfaces its contrast hint.
            Term::Factor {
                col_name,
                contrast: Contrast::Treatment,
                ..
            } => write!(f, "factor({col_name})"),
            Term::Factor {
                col_name,
                contrast: Contrast::SumToZero,
                ..
            } => write!(f, "factor({col_name}, sum)"),
            Term::Interaction(left, right) => write!(f, "{left}:{right}"),
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
    /// 1D P-spline smooth.
    PSpline1D {
        col_name: String,
        /// Typical: 5–50.
        n_splines: usize,
        /// Typical: 2–3.
        degree: usize,
        /// 1 = linear trends, 2 = constant second differences.
        penalty_order: usize,
        /// Training-data range `(min, max)` the uniform knot grid is anchored
        /// to. **Leave `None` when building a formula** — it is resolved once
        /// from training data at fit time and stored here so prediction reuses
        /// the identical basis (previously the knots were silently re-derived
        /// from the *prediction* data's range, corrupting out-of-range or
        /// subset predictions).
        #[cfg_attr(feature = "serde", serde(default))]
        range: Option<(f64, f64)>,
    },
    /// 2D tensor product of two P-spline bases: f(x₁, x₂).
    TensorProduct {
        col_name_1: String,
        n_splines_1: usize,
        penalty_order_1: usize,

        col_name_2: String,
        n_splines_2: usize,
        penalty_order_2: usize,

        /// Shared across both marginal bases.
        degree: usize,

        /// Training ranges of the two margins; same fit-time resolution
        /// contract as [`Smooth::PSpline1D`]'s `range`.
        #[cfg_attr(feature = "serde", serde(default))]
        range_1: Option<(f64, f64)>,
        #[cfg_attr(feature = "serde", serde(default))]
        range_2: Option<(f64, f64)>,
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
        col_name: String,
        /// glissando default: 6 (mgcv's own bs="cr" default is k = 10).
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
        col_name: String,
        /// Group levels in column order, resolved once from training data at
        /// fit time (sorted for determinism) and stored so prediction maps
        /// groups to the same coefficient columns. **Leave empty when building
        /// a formula.** Prediction rows with a level not seen in training are
        /// an error, matching mgcv's factor semantics.
        #[cfg_attr(feature = "serde", serde(default))]
        levels: Vec<String>,
    },
}

impl Smooth {
    /// Default number of P-spline basis functions. Shared with the Python FFI
    /// parser (`py_parse`) so the Rust and Python surfaces agree on one default.
    pub const DEFAULT_N_SPLINES: usize = 10;
    /// Default B-spline degree (cubic).
    pub const DEFAULT_DEGREE: usize = 3;
    /// Default difference-penalty order (2 ⇒ penalize curvature).
    pub const DEFAULT_PENALTY_ORDER: usize = 2;
    /// Default CR-spline knot count (glissando default; mgcv's `bs="cr"` default is 10).
    pub const DEFAULT_CR_K: usize = 6;

    /// 1D P-spline on `col_name` with default basis size / degree / penalty order.
    /// Override via [`Smooth::n_splines`], [`Smooth::degree`], [`Smooth::penalty_order`]:
    /// `Smooth::ps("x").n_splines(20)`.
    pub fn ps(col_name: impl Into<String>) -> Self {
        Smooth::PSpline1D {
            col_name: col_name.into(),
            n_splines: Self::DEFAULT_N_SPLINES,
            degree: Self::DEFAULT_DEGREE,
            penalty_order: Self::DEFAULT_PENALTY_ORDER,
            range: None,
        }
    }

    /// 1D natural cubic regression spline on `col_name` with the mgcv-default knot
    /// count. Knots resolve from training data at fit time, so they start empty —
    /// override the count with [`Smooth::k`] and pin a point constraint with
    /// [`Smooth::pc`].
    pub fn cr(col_name: impl Into<String>) -> Self {
        Smooth::CrSpline1D {
            col_name: col_name.into(),
            k: Self::DEFAULT_CR_K,
            pc: None,
            knots: Vec::new(),
        }
    }

    /// Random intercept indexed by the grouping column `col_name`.
    pub fn re(col_name: impl Into<String>) -> Self {
        Smooth::RandomEffect {
            col_name: col_name.into(),
            levels: Vec::new(),
        }
    }

    /// 2D tensor-product smooth f(`col_name_1`, `col_name_2`) with default marginal
    /// basis sizes, penalty orders, and degree.
    pub fn tensor(col_name_1: impl Into<String>, col_name_2: impl Into<String>) -> Self {
        Smooth::TensorProduct {
            col_name_1: col_name_1.into(),
            n_splines_1: Self::DEFAULT_N_SPLINES,
            penalty_order_1: Self::DEFAULT_PENALTY_ORDER,
            col_name_2: col_name_2.into(),
            n_splines_2: Self::DEFAULT_N_SPLINES,
            penalty_order_2: Self::DEFAULT_PENALTY_ORDER,
            degree: Self::DEFAULT_DEGREE,
            range_1: None,
            range_2: None,
        }
    }

    /// Builder: set the P-spline basis size. No-op on non-P-spline smooths.
    pub fn n_splines(mut self, n: usize) -> Self {
        if let Smooth::PSpline1D { n_splines, .. } = &mut self {
            *n_splines = n;
        }
        self
    }

    /// Builder: set the B-spline degree (applies to P-splines and tensor products).
    pub fn degree(mut self, d: usize) -> Self {
        match &mut self {
            Smooth::PSpline1D { degree, .. } | Smooth::TensorProduct { degree, .. } => *degree = d,
            _ => {}
        }
        self
    }

    /// Builder: set the difference-penalty order. No-op on non-P-spline smooths.
    pub fn penalty_order(mut self, order: usize) -> Self {
        if let Smooth::PSpline1D { penalty_order, .. } = &mut self {
            *penalty_order = order;
        }
        self
    }

    /// Builder: set the CR-spline knot count. No-op on non-CR smooths.
    pub fn k(mut self, k: usize) -> Self {
        if let Smooth::CrSpline1D { k: knot_count, .. } = &mut self {
            *knot_count = k;
        }
        self
    }

    /// Builder: pin a CR spline so f(`pc`) = 0. No-op on non-CR smooths.
    pub fn pc(mut self, pc: f64) -> Self {
        if let Smooth::CrSpline1D { pc: point, .. } = &mut self {
            *point = Some(pc);
        }
        self
    }

    /// Returns an mgcv-style label for this smooth term.
    ///
    /// - `PSpline1D(x)` / `CrSpline1D(x)` / `RandomEffect(x)` → `"s(x)"` (with
    ///   an optional `, pc={v}` suffix on point-constrained CR splines).
    /// - `TensorProduct(x1, x2)` → `"te(x1,x2)"`.
    pub fn term_name(&self) -> String {
        match self {
            Smooth::PSpline1D { col_name, .. } => format!("s({col_name})"),
            Smooth::CrSpline1D {
                col_name,
                pc: Some(v),
                ..
            } => format!("s({col_name}), pc={v}"),
            Smooth::CrSpline1D { col_name, .. } => format!("s({col_name})"),
            Smooth::RandomEffect { col_name, .. } => format!("s({col_name})"),
            Smooth::TensorProduct {
                col_name_1,
                col_name_2,
                ..
            } => format!("te({col_name_1},{col_name_2})"),
        }
    }

    /// Renders this smooth in the R/mgcv formula spelling the string parser
    /// accepts, so a parsed formula round-trips. Non-default basis sizes surface
    /// as `k=`; the basis type surfaces as `bs=` for CR / random-effect smooths.
    pub fn formula_repr(&self) -> String {
        match self {
            Smooth::PSpline1D {
                col_name,
                n_splines,
                ..
            } => {
                if *n_splines == Self::DEFAULT_N_SPLINES {
                    format!("s({col_name})")
                } else {
                    format!("s({col_name}, k={n_splines})")
                }
            }
            Smooth::CrSpline1D { col_name, k, .. } => {
                if *k == Self::DEFAULT_CR_K {
                    format!("s({col_name}, bs=\"cr\")")
                } else {
                    format!("s({col_name}, bs=\"cr\", k={k})")
                }
            }
            Smooth::RandomEffect { col_name, .. } => format!("s({col_name}, bs=\"re\")"),
            Smooth::TensorProduct {
                col_name_1,
                col_name_2,
                ..
            } => format!("te({col_name_1},{col_name_2})"),
        }
    }

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
            Smooth::RandomEffect { col_name, .. } => vec![col_name.as_str()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ps_constructor_uses_shared_defaults() {
        match Smooth::ps("x") {
            Smooth::PSpline1D {
                col_name,
                n_splines,
                degree,
                penalty_order,
                range,
            } => {
                assert_eq!(col_name, "x");
                assert_eq!(n_splines, Smooth::DEFAULT_N_SPLINES);
                assert_eq!(degree, Smooth::DEFAULT_DEGREE);
                assert_eq!(penalty_order, Smooth::DEFAULT_PENALTY_ORDER);
                assert_eq!(range, None, "range resolves at fit time");
            }
            other => panic!("expected PSpline1D, got {other:?}"),
        }
    }

    #[test]
    fn ps_builders_override_defaults() {
        match Smooth::ps("x").n_splines(20).degree(2).penalty_order(1) {
            Smooth::PSpline1D {
                n_splines,
                degree,
                penalty_order,
                ..
            } => assert_eq!((n_splines, degree, penalty_order), (20, 2, 1)),
            other => panic!("expected PSpline1D, got {other:?}"),
        }
    }

    #[test]
    fn cr_constructor_defaults_and_builders() {
        match Smooth::cr("d") {
            Smooth::CrSpline1D {
                col_name,
                k,
                pc,
                knots,
            } => {
                assert_eq!(col_name, "d");
                assert_eq!(k, Smooth::DEFAULT_CR_K);
                assert_eq!(pc, None);
                assert!(
                    knots.is_empty(),
                    "knots resolve at fit time, not construction"
                );
            }
            other => panic!("expected CrSpline1D, got {other:?}"),
        }
        match Smooth::cr("d").k(8).pc(0.0) {
            Smooth::CrSpline1D { k, pc, .. } => {
                assert_eq!(k, 8);
                assert_eq!(pc, Some(0.0));
            }
            other => panic!("expected CrSpline1D, got {other:?}"),
        }
    }

    #[test]
    fn re_and_tensor_constructors() {
        assert!(
            matches!(Smooth::re("g"), Smooth::RandomEffect { col_name, .. } if col_name == "g")
        );
        match Smooth::tensor("a", "b") {
            Smooth::TensorProduct {
                col_name_1,
                col_name_2,
                degree,
                ..
            } => {
                assert_eq!((col_name_1.as_str(), col_name_2.as_str()), ("a", "b"));
                assert_eq!(degree, Smooth::DEFAULT_DEGREE);
            }
            other => panic!("expected TensorProduct, got {other:?}"),
        }
    }

    #[test]
    fn term_constructors_match_struct_forms() {
        assert!(matches!(Term::linear("x"), Term::Linear { col_name } if col_name == "x"));
        assert!(matches!(
            Term::smooth(Smooth::ps("x")),
            Term::Smooth(Smooth::PSpline1D { .. })
        ));
    }

    #[test]
    fn builders_are_noops_on_wrong_variant() {
        // `.n_splines` only affects P-splines; a CR spline passes through unchanged.
        assert!(matches!(
            Smooth::cr("x").n_splines(99),
            Smooth::CrSpline1D { k, .. } if k == Smooth::DEFAULT_CR_K
        ));
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

    /// Defaults sourced from the native constructors ([`Smooth::ps`]/[`Smooth::cr`]),
    /// re-aliased here so the Python FFI and the Rust API share one source of truth.
    const DEFAULT_N_SPLINES: usize = Smooth::DEFAULT_N_SPLINES;
    const DEFAULT_DEGREE: usize = Smooth::DEFAULT_DEGREE;
    const DEFAULT_PENALTY_ORDER: usize = Smooth::DEFAULT_PENALTY_ORDER;
    const DEFAULT_CR_K: usize = Smooth::DEFAULT_CR_K;

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
                Ok(Term::Smooth(Smooth::RandomEffect {
                    col_name,
                    levels: vec![], // resolved from training data at fit time
                }))
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
                    range: None, // resolved from training data at fit time
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
