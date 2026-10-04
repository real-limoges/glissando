//! `Param`: the closed set of distribution-parameter names.
//!
//! Every family's parameters are drawn from this enum, so a parameter name is
//! checked once, where a string enters the crate (JSON, Python, WASM,
//! [`Formula::from_strings`](crate::Formula::from_strings)), and is a plain value
//! everywhere after that.
//!
//! The string spellings (`"mu"`, `"delta_1"`, …) live only in [`Param::as_str`];
//! [`Display`](fmt::Display), [`FromStr`] and serde all go through it, so every wire
//! format keeps exactly the spellings it had when parameters were strings.
//!
//! `Param` deliberately does not implement `Ord`. Declaration order (μ, σ, ν, τ) and
//! string order (mu, nu, sigma, tau) disagree, and sorted outputs (prediction JSON,
//! the characterization snapshots) have always used string order; they key by
//! [`Param::as_str`] so that choice stays visible at the call site.

use std::fmt;
use std::str::FromStr;

use crate::error::GamlssError;

/// A distribution parameter.
///
/// `Delta1`..`Delta4` are the [`Ocat`](crate::distributions::Ocat) thresholds; the
/// family supports at most five categories, so four is the most there can be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Param {
    /// Location (`"mu"`).
    Mu,
    /// Scale (`"sigma"`).
    Sigma,
    /// First shape parameter (`"nu"`).
    Nu,
    /// Second shape parameter (`"tau"`).
    Tau,
    /// Beta precision (`"phi"`).
    Phi,
    /// Hurdle zero probability (`"xi"`).
    Xi,
    /// First Ocat threshold (`"delta_1"`).
    Delta1,
    /// Second Ocat threshold increment (`"delta_2"`).
    Delta2,
    /// Third Ocat threshold increment (`"delta_3"`).
    Delta3,
    /// Fourth Ocat threshold increment (`"delta_4"`).
    Delta4,
}

impl Param {
    /// Every variant, in declaration order.
    pub const ALL: [Param; 10] = [
        Param::Mu,
        Param::Sigma,
        Param::Nu,
        Param::Tau,
        Param::Phi,
        Param::Xi,
        Param::Delta1,
        Param::Delta2,
        Param::Delta3,
        Param::Delta4,
    ];

    /// The parameter's name as it appears in formulas, JSON, and Python dicts.
    pub const fn as_str(self) -> &'static str {
        match self {
            Param::Mu => "mu",
            Param::Sigma => "sigma",
            Param::Nu => "nu",
            Param::Tau => "tau",
            Param::Phi => "phi",
            Param::Xi => "xi",
            Param::Delta1 => "delta_1",
            Param::Delta2 => "delta_2",
            Param::Delta3 => "delta_3",
            Param::Delta4 => "delta_4",
        }
    }
}

impl fmt::Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Param {
    type Err = GamlssError;

    /// Parses a parameter name. Fails with [`GamlssError::InvalidParamName`] for a
    /// string that is no parameter of any family; whether a particular family has
    /// the parameter is a separate check.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Param::ALL
            .into_iter()
            .find(|p| p.as_str() == name)
            .ok_or_else(|| GamlssError::InvalidParamName {
                name: name.to_string(),
            })
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Param {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Param {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        name.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_round_trips_through_its_name() {
        for p in Param::ALL {
            assert_eq!(p.as_str().parse::<Param>().unwrap(), p);
            assert_eq!(p.to_string(), p.as_str());
        }
    }

    #[test]
    fn names_are_distinct() {
        let mut names: Vec<&str> = Param::ALL.iter().map(|p| p.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Param::ALL.len());
    }

    #[test]
    fn unknown_name_is_invalid_param_name() {
        for bad in ["zeta", "", "Mu", "delta_0", "delta_5", "delta1"] {
            match bad.parse::<Param>() {
                Err(GamlssError::InvalidParamName { name }) => assert_eq!(name, bad),
                other => panic!("{bad:?} parsed as {other:?}"),
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_uses_the_string_names_as_keys_and_values() {
        use std::collections::BTreeMap;

        let json = serde_json::to_string(&Param::Delta2).unwrap();
        assert_eq!(json, r#""delta_2""#);
        assert_eq!(serde_json::from_str::<Param>(&json).unwrap(), Param::Delta2);

        let map: indexmap::IndexMap<Param, u8> = [(Param::Mu, 1), (Param::Sigma, 2)].into();
        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(json, r#"{"mu":1,"sigma":2}"#);
        let back: indexmap::IndexMap<Param, u8> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, map);

        let err = serde_json::from_str::<BTreeMap<String, Param>>(r#"{"a":"zeta"}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("zeta"), "{err}");
    }
}
