//! Parameter names are checked where they enter the crate: a string that names no
//! parameter fails to parse, and a real parameter the family does not have is
//! rejected before fitting rather than silently ignored.

mod common;

use common::{intercept_only, linear_intercepts, Generator};
use glissando::distributions::{Gaussian, Poisson};
use glissando::{
    selection, Direction, FitConfig, Formula, GamlssError, GamlssModel, Param, StepScope, Term,
};

#[test]
fn formula_key_the_family_lacks_is_rejected() {
    let mut rng = Generator::new(7);
    let (y, data) = rng.linear_gaussian(40, 1.0, 2.0, 0.5);
    // Poisson has only mu; a sigma formula used to be dropped without a word.
    let y = y.mapv(|v: f64| v.abs().round());
    let formula = intercept_only(&[Param::Mu, Param::Sigma]);

    match GamlssModel::fit(&data, &y, &formula, &Poisson::new()) {
        Err(GamlssError::UnknownParameter {
            distribution,
            param,
        }) => {
            assert_eq!(distribution, "Poisson");
            assert_eq!(param, "sigma");
        }
        other => panic!("expected UnknownParameter, got {other:?}"),
    }
}

#[test]
fn step_scope_for_a_missing_parameter_is_rejected() {
    let mut rng = Generator::new(8);
    let (y, data) = rng.linear_gaussian(40, 1.0, 2.0, 0.5);
    let start = intercept_only(&[Param::Mu, Param::Sigma]);
    let scope = [StepScope {
        param: Param::Nu,
        candidates: vec![Term::Linear {
            col_name: "x".into(),
        }],
    }];

    let Err(err) = selection::step_gaic(
        &data,
        &y,
        &Gaussian::new(),
        start,
        &scope,
        2.0,
        Direction::Forward,
        FitConfig::default(),
    ) else {
        panic!("step_gaic accepted a scope for nu on Gaussian");
    };
    assert!(
        matches!(&err, GamlssError::UnknownParameter { param, .. } if param == "nu"),
        "expected UnknownParameter for nu, got {err:?}"
    );
}

#[test]
fn from_strings_rejects_a_name_that_is_no_parameter() {
    match Formula::from_strings([("mu", "~ x"), ("zeta", "~ 1")]) {
        Err(GamlssError::InvalidParamName { name }) => assert_eq!(name, "zeta"),
        other => panic!("expected InvalidParamName, got {other:?}"),
    }
}

#[test]
fn fitted_models_are_keyed_in_family_order() {
    let mut rng = Generator::new(9);
    let (y, data) = rng.linear_gaussian(40, 1.0, 2.0, 0.5);
    let formula = linear_intercepts("x", &[Param::Mu, Param::Sigma]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
    let keys: Vec<Param> = model.models.keys().copied().collect();
    assert_eq!(keys, [Param::Mu, Param::Sigma]);
}

#[cfg(feature = "serialization")]
mod json {
    use glissando::{json, GamlssError};

    #[test]
    fn formula_with_an_invalid_name_fails_to_parse() {
        match json::parse_formula(r#"{"mu": "~ x", "zeta": "~ 1"}"#) {
            Err(GamlssError::InvalidParamName { name }) => assert_eq!(name, "zeta"),
            other => panic!("expected InvalidParamName, got {other:?}"),
        }
    }

    #[test]
    fn config_with_an_invalid_link_key_fails_to_parse() {
        let err = json::parse_config(r#"{"links": {"zeta": "log"}}"#).unwrap_err();
        assert!(err.to_string().contains("zeta"), "{err}");
    }

    #[test]
    fn link_keys_round_trip_as_their_names() {
        let config = json::parse_config(r#"{"links": {"sigma": "sqrt"}}"#).unwrap();
        assert_eq!(config.links[&glissando::Param::Sigma], "sqrt");
    }
}
