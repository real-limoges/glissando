// The JSON facade sits behind the `serialization` feature; `python` is excluded
// for the usual PyO3 extension-module linking reason, and proptest is non-wasm.
#![cfg(all(
    feature = "serialization",
    not(feature = "python"),
    not(target_arch = "wasm32")
))]

//! Property-based coverage (TEST-1) of the `glissando::json` embedding facade.
//!
//! Two families of property. First, *totality*: every string-in parse entry point
//! returns `Ok`/`Err` on arbitrary input, never panicking, so an embedder can throw
//! untrusted bytes at the boundary safely. Second, a *round-trip* invariant: a
//! fitted model serialized with `to_json` and reloaded with `json::load` predicts
//! the same values it did before, so persistence is lossless where it counts.

mod common;

use common::Generator;
use glissando::distributions::Gaussian;
use glissando::{json, Formula, GamlssModel, Smooth, Term};
use proptest::prelude::*;

/// A grab-bag of tokens that reach the JSON parsers' interesting branches (arrays,
/// objects, numbers, nesting, quotes) far more often than random unicode would.
fn json_alphabet() -> impl Strategy<Value = String> {
    proptest::collection::vec(
        proptest::sample::select(vec![
            "{",
            "}",
            "[",
            "]",
            ":",
            ",",
            "\"",
            "mu",
            "sigma",
            "x",
            "z",
            "Intercept",
            "Linear",
            "col_name",
            "null",
            "1.0",
            "-2",
            "0",
            "1e9",
            "NaN",
            "true",
            " ",
            "~",
            "s(x)",
            "y ~ x",
        ]),
        0..16,
    )
    .prop_map(|parts| parts.concat())
}

proptest! {
    /// `parse_response` is total: `Ok`/`Err`, never a panic.
    #[test]
    fn parse_response_never_panics(s in ".*") {
        let _ = json::parse_response(&s);
    }

    /// `parse_data` is total.
    #[test]
    fn parse_data_never_panics(s in prop_oneof![".*", json_alphabet()]) {
        let _ = json::parse_data(&s);
    }

    /// `parse_formula` is total across both accepted spellings (string-map and
    /// term-list), including malformed ones.
    #[test]
    fn parse_formula_never_panics(s in prop_oneof![".*", json_alphabet()]) {
        let _ = json::parse_formula(&s);
    }

    /// `parse_config` is total.
    #[test]
    fn parse_config_never_panics(s in prop_oneof![".*", json_alphabet()]) {
        let _ = json::parse_config(&s);
    }

    /// `load` (deserialize + rebuild the distribution) is total: arbitrary bytes
    /// yield an error, never a panic.
    #[test]
    fn load_never_panics(s in prop_oneof![".*", json_alphabet()]) {
        let _ = json::load(&s);
    }
}

/// `to_json` → `json::load` preserves predictions for a linear-mean Gaussian fit.
/// Coefficients are not compared byte-for-byte (that needs the `float_roundtrip`
/// feature per the crate docs); predictions within a tight tolerance are the
/// contract embedders actually rely on.
#[test]
fn to_json_load_round_trip_preserves_predictions() {
    let mut rng = Generator::new(7);
    let (y, data) = rng.heteroskedastic_gaussian(200);

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::linear("x")])
        .with_terms("sigma", vec![Term::Intercept, Term::linear("x")]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian).unwrap();

    let saved = model.to_json(&Gaussian).unwrap();
    let (reloaded, family) = json::load(&saved).unwrap();

    let before = model.predict(&data, &Gaussian).unwrap();
    let after = reloaded.predict(&data, family.as_ref()).unwrap();

    for param in ["mu", "sigma"] {
        for (a, b) in before[param].iter().zip(after[param].iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "{param} prediction drifted: {a} vs {b}"
            );
        }
    }
}

/// Same round-trip through a P-spline smooth, whose stored training range and
/// sum-to-zero reparameterization also have to survive serialization.
#[test]
fn to_json_load_round_trip_with_smooth() {
    let mut rng = Generator::new(21);
    let (y, data) = rng.sinusoidal_gaussian(250, 0.3);

    let formula = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::smooth(Smooth::ps("x"))])
        .with_terms("sigma", vec![Term::Intercept]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian).unwrap();

    let saved = model.to_json(&Gaussian).unwrap();
    let (reloaded, family) = json::load(&saved).unwrap();

    let before = model.predict(&data, &Gaussian).unwrap();
    let after = reloaded.predict(&data, family.as_ref()).unwrap();

    for (a, b) in before["mu"].iter().zip(after["mu"].iter()) {
        assert!(
            (a - b).abs() < 1e-9,
            "smooth prediction drifted: {a} vs {b}"
        );
    }
}
