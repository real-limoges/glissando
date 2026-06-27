//! DATA-5 string formula parser — public-API integration tests.
//!
//! The parser is a front-end over `Vec<Term>`: a formula built from a string must
//! fit *identically* to the same formula built with the constructor DSL. A
//! proptest checks that rendering a parsed formula and reparsing it is a fixed
//! point (no drift across the parse ⇄ Display boundary).

// Integration tests cannot run with the `python` feature (PyO3 linking).
#![cfg(not(feature = "python"))]

mod common;

use common::Generator;
use glissando::distributions::Gaussian;
use glissando::{DataSet, Formula, GamlssModel, Smooth, Term};
use ndarray::Array1;

/// A formula parsed from a string fits identically to the builder-built formula.
#[test]
fn parsed_formula_fits_identically_to_builder() {
    let mut rng = Generator::new(11);
    let (y, data) = rng.heteroskedastic_gaussian(300);

    let built = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::linear("x")])
        .with_terms("sigma", vec![Term::Intercept, Term::linear("x")]);
    let parsed = Formula::from_strings([("mu", "y ~ x"), ("sigma", "~ x")]).unwrap();

    let m_built = GamlssModel::fit(&data, &y, &built, &Gaussian).unwrap();
    let m_parsed = GamlssModel::fit(&data, &y, &parsed, &Gaussian).unwrap();

    for param in ["mu", "sigma"] {
        let a = &m_built.models[param].coefficients.0;
        let b = &m_parsed.models[param].coefficients.0;
        assert_eq!(a.len(), b.len(), "{param} coefficient count");
        for (x, z) in a.iter().zip(b.iter()) {
            assert!((x - z).abs() < 1e-12, "{param}: {x} vs {z}");
        }
    }
}

/// A smooth parsed from `s(x)` matches the `Smooth::ps` builder, fitting to the
/// same curve.
#[test]
fn parsed_smooth_matches_builder_smooth() {
    let mut rng = Generator::new(3);
    let (y, data) = rng.heteroskedastic_gaussian(250);

    let built = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::smooth(Smooth::ps("x"))])
        .with_terms("sigma", vec![Term::Intercept]);
    let parsed = Formula::from_strings([("mu", "y ~ s(x)"), ("sigma", "~ 1")]).unwrap();

    let m_built = GamlssModel::fit(&data, &y, &built, &Gaussian).unwrap();
    let m_parsed = GamlssModel::fit(&data, &y, &parsed, &Gaussian).unwrap();

    let p_built = m_built.predict(&data, &Gaussian).unwrap();
    let p_parsed = m_parsed.predict(&data, &Gaussian).unwrap();
    for (a, b) in p_built["mu"].iter().zip(p_parsed["mu"].iter()) {
        assert!((a - b).abs() < 1e-10, "smooth fit drifted: {a} vs {b}");
    }
}

/// The whole-stack ergonomics target: a single rich string with a factor and an
/// interaction parses, fits, and predicts.
#[test]
fn rich_string_formula_fits_end_to_end() {
    let n = 400;
    let g: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let x: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 3.0).collect();
    let y: Vec<f64> = g
        .iter()
        .zip(&x)
        .map(|(&gi, &xi)| 1.0 + 0.5 * gi + 0.7 * xi + 0.4 * gi * xi)
        .collect();

    let mut data = DataSet::new();
    data.insert_column("g", Array1::from_vec(g));
    data.insert_column("x", Array1::from_vec(x));
    let y = Array1::from_vec(y);

    let formula =
        Formula::from_strings([("mu", "y ~ factor(g) + x + factor(g):x"), ("sigma", "~ 1")])
            .unwrap();
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian).unwrap();
    let pred = model.predict(&data, &Gaussian).unwrap();
    for (p, yi) in pred["mu"].iter().zip(y.iter()) {
        assert!((p - yi).abs() < 0.05, "string-formula fit off: {p} vs {yi}");
    }
}

// Property-based round-trip — proptest is non-wasm only.
#[cfg(not(target_arch = "wasm32"))]
mod prop {
    use glissando::parse_formula_string;
    use proptest::prelude::*;

    /// A pool of atomic term spellings the parser supports.
    fn atom() -> impl Strategy<Value = &'static str> {
        proptest::sample::select(vec![
            "x",
            "z",
            "s(x)",
            "s(x, k=12)",
            "s(z, bs=\"cr\")",
            "s(g, bs=\"re\")",
            "te(x, z)",
            "factor(g)",
            "factor(g, sum)",
            "offset(e)",
            "a:b",
            "factor(g):x",
        ])
    }

    proptest! {
        /// Parsing, rendering each term via `Display`, and reparsing is a fixed
        /// point: the term spellings are identical the second time around.
        #[test]
        fn parse_render_reparse_is_stable(
            atoms in proptest::collection::vec(atom(), 1..6),
            suppress in any::<bool>(),
        ) {
            let mut rhs = atoms.join(" + ");
            if suppress {
                rhs = format!("0 + {rhs}");
            }
            let spec = format!("y ~ {rhs}");

            let (_resp, terms) = parse_formula_string(&spec).unwrap();
            let rendered: Vec<String> = terms.iter().map(|t| t.to_string()).collect();

            let has_intercept = rendered.iter().any(|s| s == "1");
            let mut body = rendered.clone();
            if !has_intercept {
                body.insert(0, "0".to_string());
            }
            let (_r2, terms2) = parse_formula_string(&format!("~ {}", body.join(" + "))).unwrap();
            let rerendered: Vec<String> = terms2.iter().map(|t| t.to_string()).collect();

            prop_assert_eq!(rendered, rerendered);
        }
    }
}
