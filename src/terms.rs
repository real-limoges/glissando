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
