//! `Formula`: a mapping from distribution parameters ([`Param::Mu`],
//! [`Param::Sigma`], …) to the list of [`Term`]s describing each parameter's linear
//! predictor.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::error::GamlssError;
use crate::terms::Term;
use crate::Param;

/// A model formula mapping parameters to term vectors, wrapping
/// `HashMap<Param, Vec<Term>>`. The inner field is crate-private; go through the
/// builder methods or `Deref` for read access.
///
/// Serialized, it is an object keyed by the parameter names (`{"mu": [...]}`).
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Formula(pub(crate) HashMap<Param, Vec<Term>>);

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
    /// use glissando::{Formula, Param, Term};
    ///
    /// let f = Formula::new()
    ///     .with_terms(Param::Mu, vec![Term::Intercept])
    ///     .with_terms(Param::Sigma, vec![Term::Intercept]);
    /// assert_eq!(f.param_names().len(), 2);
    /// ```
    pub fn with_terms(mut self, param: Param, terms: Vec<Term>) -> Self {
        self.0.insert(param, terms);
        self
    }

    /// Adds or replaces terms for a distribution parameter.
    pub fn add_terms(&mut self, param: Param, terms: Vec<Term>) {
        self.0.insert(param, terms);
    }

    /// Build a single-parameter formula from an R/mgcv-style string.
    ///
    /// The response on the left of `~` is parsed and then thrown away (glissando
    /// takes the response array separately at fit time), so `"y ~ s(x) + z"` and
    /// `"~ s(x) + z"` mean the same thing here.
    ///
    /// ```
    /// use glissando::{Formula, Param};
    ///
    /// let f = Formula::parse(Param::Mu, "y ~ s(x) + region").unwrap();
    /// assert_eq!(f[&Param::Mu].len(), 3); // intercept + smooth + factor-or-linear
    /// ```
    ///
    /// # Errors
    ///
    /// [`GamlssError::Input`] on malformed input (unbalanced calls, unknown
    /// smooth/contrast arguments, empty right-hand side).
    pub fn parse(param: Param, formula: &str) -> Result<Self, GamlssError> {
        let (_response, terms) = crate::types::parse_formula_string(formula)?;
        Ok(Self::new().with_terms(param, terms))
    }

    /// Build a multi-parameter formula from `(param name, formula string)` pairs.
    ///
    /// Both halves are strings, so this is a parsing boundary: each name goes
    /// through [`Param::from_str`](std::str::FromStr::from_str).
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
    /// [`GamlssError::InvalidParamName`] for a name that is no parameter, else the
    /// first formula parse error encountered.
    pub fn from_strings<'a, I>(specs: I) -> Result<Self, GamlssError>
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut f = Self::new();
        for (name, formula) in specs {
            let param: Param = name.parse()?;
            let (_response, terms) = crate::types::parse_formula_string(formula)?;
            f.add_terms(param, terms);
        }
        Ok(f)
    }

    /// Returns the distribution parameters this formula has terms for.
    pub fn param_names(&self) -> Vec<Param> {
        self.0.keys().copied().collect()
    }
}

impl Deref for Formula {
    type Target = HashMap<Param, Vec<Term>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Formula {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<HashMap<Param, Vec<Term>>> for Formula {
    fn from(map: HashMap<Param, Vec<Term>>) -> Self {
        Self(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formula_with_terms_chains() {
        let f = Formula::new()
            .with_terms(Param::Mu, vec![Term::Intercept])
            .with_terms(Param::Sigma, vec![Term::Intercept]);
        assert_eq!(f.0.len(), 2);
        assert!(f.0.contains_key(&Param::Mu));
        assert!(f.0.contains_key(&Param::Sigma));
    }

    #[test]
    fn formula_add_terms_replaces_existing() {
        let mut f = Formula::new();
        f.add_terms(Param::Mu, vec![Term::Intercept]);
        f.add_terms(
            Param::Mu,
            vec![
                Term::Intercept,
                Term::Linear {
                    col_name: "x".into(),
                },
            ],
        );
        assert_eq!(f.0.get(&Param::Mu).unwrap().len(), 2);
    }

    #[test]
    fn formula_param_names_includes_added_keys() {
        let f = Formula::new().with_terms(Param::Mu, vec![Term::Intercept]);
        assert_eq!(f.param_names(), vec![Param::Mu]);
    }
}
