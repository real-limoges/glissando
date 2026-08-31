//! `Formula`: a mapping from distribution-parameter names (`"mu"`, `"sigma"`, …)
//! to the list of [`Term`]s describing each parameter's linear predictor.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::terms::Term;

/// A model formula mapping parameter names to term vectors, wrapping
/// `HashMap<String, Vec<Term>>`. The inner field is crate-private; go through the
/// builder methods or `Deref` for read access.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Formula(pub(crate) HashMap<String, Vec<Term>>);

impl Formula {
    /// Creates an empty formula with no parameter terms.
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Builder method: adds terms for a distribution parameter, returning `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use glissando::{Formula, Term};
    ///
    /// let f = Formula::new()
    ///     .with_terms("mu", vec![Term::Intercept])
    ///     .with_terms("sigma", vec![Term::Intercept]);
    /// assert_eq!(f.param_names().len(), 2);
    /// ```
    pub fn with_terms(mut self, param: impl Into<String>, terms: Vec<Term>) -> Self {
        self.0.insert(param.into(), terms);
        self
    }

    /// Adds or replaces terms for a distribution parameter.
    pub fn add_terms(&mut self, param: impl Into<String>, terms: Vec<Term>) {
        self.0.insert(param.into(), terms);
    }

    /// Build a single-parameter formula from an R/mgcv-style string (DATA-5).
    ///
    /// The response on the left of `~` is parsed and then thrown away (glissando
    /// takes the response array separately at fit time), so `"y ~ s(x) + z"` and
    /// `"~ s(x) + z"` mean the same thing here.
    ///
    /// ```
    /// use glissando::{Formula, Term};
    ///
    /// let f = Formula::parse("mu", "y ~ s(x) + region").unwrap();
    /// assert_eq!(f["mu"].len(), 3); // intercept + smooth + factor-or-linear
    /// ```
    ///
    /// # Errors
    ///
    /// [`GamlssError::Input`](crate::GamlssError::Input) on malformed input
    /// (unbalanced calls, unknown smooth/contrast arguments, empty right-hand side).
    pub fn parse(
        param: impl Into<String>,
        formula: &str,
    ) -> Result<Self, crate::error::GamlssError> {
        let (_response, terms) = crate::types::parse_formula_string(formula)?;
        Ok(Self::new().with_terms(param, terms))
    }

    /// Build a multi-parameter formula from `(param, formula_string)` pairs.
    ///
    /// ```
    /// use glissando::Formula;
    ///
    /// let f = Formula::from_strings([
    ///     ("mu", "y ~ s(x) + region"),
    ///     ("sigma", "~ x"),
    /// ]).unwrap();
    /// assert_eq!(f.param_names().len(), 2);
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates the first parse error encountered.
    pub fn from_strings<'a, I>(specs: I) -> Result<Self, crate::error::GamlssError>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut f = Self::new();
        for (param, formula) in specs {
            let (_response, terms) = crate::types::parse_formula_string(formula)?;
            f.add_terms(param, terms);
        }
        Ok(f)
    }

    /// Returns the names of all distribution parameters in this formula.
    pub fn param_names(&self) -> Vec<&String> {
        self.0.keys().collect()
    }
}

impl Deref for Formula {
    type Target = HashMap<String, Vec<Term>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Formula {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<HashMap<String, Vec<Term>>> for Formula {
    fn from(map: HashMap<String, Vec<Term>>) -> Self {
        Self(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formula_with_terms_chains() {
        let f = Formula::new()
            .with_terms("mu", vec![Term::Intercept])
            .with_terms("sigma", vec![Term::Intercept]);
        assert_eq!(f.0.len(), 2);
        assert!(f.0.contains_key("mu"));
        assert!(f.0.contains_key("sigma"));
    }

    #[test]
    fn formula_add_terms_replaces_existing() {
        let mut f = Formula::new();
        f.add_terms("mu", vec![Term::Intercept]);
        f.add_terms(
            "mu",
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".into(),
                },
            ],
        );
        assert_eq!(f.0.get("mu").unwrap().len(), 2);
    }

    #[test]
    fn formula_param_names_includes_added_keys() {
        let f = Formula::new().with_terms("mu", vec![Term::Intercept]);
        let names: Vec<&str> = f.param_names().iter().map(|s| s.as_str()).collect();
        assert_eq!(names, vec!["mu"]);
    }
}
