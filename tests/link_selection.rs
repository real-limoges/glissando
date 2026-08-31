//! Per-parameter link selection (LINK-1..6).
//!
//! Exercises the public surface of the link-selection feature:
//! - `FitConfig::with_link` overrides a parameter's link at fit time;
//! - the override actually changes the fit (probit/cloglog ≠ logit default);
//! - an unknown link name is a typed error;
//! - an unknown *parameter* name is a typed error, and so is an override on a
//!   parameter whose family cannot honor it;
//! - the chosen link is persisted, so a JSON round-trip predicts through the
//!   *overridden* link, not the family default.

use glissando::distributions::{
    link_from_name, Beta, Binomial, Distribution, Hurdle, Ocat, StudentT,
};
use glissando::{DataSet, FitConfig, Formula, GamlssModel, Term};
use ndarray::Array1;

/// Reference `(link, η, inv_link(η), mu_eta(η))` from R `stats::make.link`. These
/// are the canonical closed-form values (`make.link("probit")`, `"cloglog"`,
/// `"cauchit"`, `"sqrt"`, `"inverse"`, and `"1/mu^2"`). The non-Gaussian rows are
/// exact arithmetic; the probit rows were tabulated with an independent `erf`
/// implementation. A sign or formula slip in any link fails this table.
// These are exact R make.link reference values; a few happen to land near √2 / 1/√2,
// which trips clippy's approx_constant lint. They're data, not constant approximations.
#[allow(clippy::approx_constant)]
#[rustfmt::skip]
const MAKE_LINK_ORACLE: &[(&str, f64, f64, f64)] = &[
    ("probit", -2.0, 2.275013194818e-02, 5.399096651319e-02),
    ("probit", -1.0, 1.586552539315e-01, 2.419707245191e-01),
    ("probit", -0.5, 3.085375387260e-01, 3.520653267643e-01),
    ("probit",  0.0, 5.000000000000e-01, 3.989422804014e-01),
    ("probit",  0.5, 6.914624612740e-01, 3.520653267643e-01),
    ("probit",  1.0, 8.413447460685e-01, 2.419707245191e-01),
    ("probit",  2.0, 9.772498680518e-01, 5.399096651319e-02),
    ("cloglog", -2.0, 1.265769815069e-01, 1.182049515931e-01),
    ("cloglog", -1.0, 3.077993724447e-01, 2.546463800436e-01),
    ("cloglog", -0.5, 4.547607881074e-01, 3.307042988904e-01),
    ("cloglog",  0.0, 6.321205588286e-01, 3.678794411714e-01),
    ("cloglog",  0.5, 8.077043544520e-01, 3.170419210779e-01),
    ("cloglog",  1.0, 9.340119641547e-01, 1.793740787340e-01),
    ("cloglog",  2.0, 9.993820210107e-01, 4.566281420128e-03),
    ("cauchit", -2.0, 1.475836176504e-01, 6.366197723676e-02),
    ("cauchit", -1.0, 2.500000000000e-01, 1.591549430919e-01),
    ("cauchit", -0.5, 3.524163823496e-01, 2.546479089470e-01),
    ("cauchit",  0.0, 5.000000000000e-01, 3.183098861838e-01),
    ("cauchit",  0.5, 6.475836176504e-01, 2.546479089470e-01),
    ("cauchit",  1.0, 7.500000000000e-01, 1.591549430919e-01),
    ("cauchit",  2.0, 8.524163823496e-01, 6.366197723676e-02),
    ("sqrt", 0.5, 2.500000000000e-01, 1.000000000000e+00),
    ("sqrt", 1.0, 1.000000000000e+00, 2.000000000000e+00),
    ("sqrt", 2.0, 4.000000000000e+00, 4.000000000000e+00),
    ("sqrt", 4.0, 1.600000000000e+01, 8.000000000000e+00),
    ("inverse", -2.0, -5.000000000000e-01, -2.500000000000e-01),
    ("inverse", -1.0, -1.000000000000e+00, -1.000000000000e+00),
    ("inverse", -0.5, -2.000000000000e+00, -4.000000000000e+00),
    ("inverse",  0.5,  2.000000000000e+00, -4.000000000000e+00),
    ("inverse",  1.0,  1.000000000000e+00, -1.000000000000e+00),
    ("inverse",  2.0,  5.000000000000e-01, -2.500000000000e-01),
    ("inverse_square", 0.5, 1.414213562373e+00, -1.414213562373e+00),
    ("inverse_square", 1.0, 1.000000000000e+00, -5.000000000000e-01),
    ("inverse_square", 2.0, 7.071067811865e-01, -1.767766952966e-01),
    ("inverse_square", 4.0, 5.000000000000e-01, -6.250000000000e-02),
];

#[test]
fn links_match_r_make_link_oracle() {
    for &(name, eta, inv, mu_eta) in MAKE_LINK_ORACLE {
        let link = link_from_name(name).unwrap();
        assert!(
            (link.inv_link(eta) - inv).abs() <= 1e-9 * (1.0 + inv.abs()),
            "{name} inv_link({eta}): {} vs R {inv}",
            link.inv_link(eta)
        );
        assert!(
            (link.mu_eta(eta) - mu_eta).abs() <= 1e-9 * (1.0 + mu_eta.abs()),
            "{name} mu_eta({eta}): {} vs R {mu_eta}",
            link.mu_eta(eta)
        );
    }
}

/// A small, separable binary dataset: y = 1 for x above the midpoint. Enough
/// signal that every bounded link converges to a non-trivial slope.
fn binary_data() -> (DataSet, Array1<f64>) {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 - 0.5).collect();
    // Deterministic-but-noisy labels so the three links land on different βs.
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let p = 1.0 / (1.0 + (-6.0 * xi).exp());
            // threshold against a fixed dither pattern instead of an RNG
            let dither = ((i * 7) % 10) as f64 / 10.0;
            if p > dither {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));
    (data, Array1::from_vec(y))
}

fn formula() -> Formula {
    Formula::new().with_terms(
        "mu",
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".into(),
            },
        ],
    )
}

#[test]
fn overriding_the_link_changes_the_fit() {
    let (data, y) = binary_data();
    let f = formula();

    let logit = GamlssModel::fit(&data, &y, &f, &Binomial::new(1)).unwrap();
    let probit = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &f,
        &Binomial::new(1),
        FitConfig::default().with_link("mu", "probit"),
    )
    .unwrap();
    let cloglog = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &f,
        &Binomial::new(1),
        FitConfig::default().with_link("mu", "cloglog"),
    )
    .unwrap();

    let slope = |m: &GamlssModel| m.models["mu"].coefficients.0[1];
    // All three should catch the positive trend...
    assert!(slope(&logit) > 0.0, "logit slope should be positive");
    assert!(slope(&probit) > 0.0, "probit slope should be positive");
    assert!(slope(&cloglog) > 0.0, "cloglog slope should be positive");
    // ...but on different link scales, so the coefficients must differ.
    assert!(
        (slope(&logit) - slope(&probit)).abs() > 1e-6,
        "probit fit should differ from logit (got {} vs {})",
        slope(&probit),
        slope(&logit)
    );
    assert!(
        (slope(&logit) - slope(&cloglog)).abs() > 1e-6,
        "cloglog fit should differ from logit"
    );

    // The override is recorded on the fitted parameter; the default is not.
    assert_eq!(probit.models["mu"].link.as_deref(), Some("probit"));
    assert_eq!(cloglog.models["mu"].link.as_deref(), Some("cloglog"));
    assert_eq!(logit.models["mu"].link, None);
}

#[test]
fn unknown_link_name_is_a_typed_error() {
    let (data, y) = binary_data();
    let err = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &formula(),
        &Binomial::new(1),
        FitConfig::default().with_link("mu", "not_a_link"),
    );
    assert!(err.is_err(), "an unknown link name must fail the fit");
}

/// Intercept-only formula over the named parameters, enough to reach the link
/// validation that runs first inside `fit_gamlss`.
fn intercepts(params: &[&str]) -> Formula {
    params.iter().fold(Formula::new(), |f, p| {
        f.with_terms(*p, vec![Term::Intercept])
    })
}

/// A dataset in `(0, 1)`, valid for Beta.
fn unit_interval_data() -> (DataSet, Array1<f64>) {
    let n = 40;
    let y: Vec<f64> = (0..n)
        .map(|i| 0.1 + 0.8 * (i as f64) / (n as f64))
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec((0..n).map(|i| i as f64).collect()));
    (data, Array1::from_vec(y))
}

#[test]
fn a_link_override_for_an_unknown_parameter_is_rejected() {
    // Beta's second parameter is `phi`, not `sigma`. Before this check, the
    // override just never matched inside the per-parameter loop and the fit
    // ran to completion under the default links, reporting success for a
    // configuration it had completely ignored.
    let (data, y) = unit_interval_data();
    let err = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &intercepts(&["mu", "phi"]),
        &Beta,
        FitConfig::default().with_link("sigma", "log"),
    )
    .expect_err("an override keyed on a parameter Beta does not have must fail the fit");
    let msg = err.to_string();
    assert!(
        msg.contains("sigma") && msg.contains("phi"),
        "the error should name the bad key and list the real parameters, got: {msg}"
    );
}

#[test]
fn ocat_rejects_every_link_override() {
    // Ocat's `params["mu"]` holds η, and its threshold Jacobian is `exp(η_k)`
    // only under the log link, so no override can be honored.
    let n = 40;
    let y: Array1<f64> = (0..n).map(|i| (i % 3 + 1) as f64).collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec((0..n).map(|i| i as f64).collect()));

    for param in ["mu", "delta_1", "delta_2"] {
        let err = GamlssModel::fit_with_config(
            &data,
            &y,
            None,
            &intercepts(&["mu", "delta_1", "delta_2"]),
            &Ocat::new(3),
            FitConfig::default().with_link(param, "probit"),
        )
        .err()
        // Not `expect_err`: it takes a plain `&str` and would print the brace
        // literally, hiding which of the three parameters regressed.
        .unwrap_or_else(|| panic!("Ocat must reject a link override on {param}"));
        assert!(
            err.to_string().contains("Ocat") && err.to_string().contains(param),
            "the error should name the family and the parameter, got: {err}"
        );
    }
}

#[test]
fn student_t_rejects_an_override_on_nu_only() {
    let n = 60;
    let y: Array1<f64> = (0..n)
        .map(|i| ((i * 37 % 23) as f64 - 11.0) / 5.0)
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec((0..n).map(|i| i as f64).collect()));
    let f = intercepts(&["mu", "sigma", "nu"]);

    let err = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &f,
        &StudentT,
        FitConfig::default().with_link("nu", "log"),
    )
    .expect_err(
        "StudentT must reject an override on nu: its KKT floor projection assumes FlooredLog",
    );
    assert!(
        err.to_string().contains("nu"),
        "the error should name the parameter, got: {err}"
    );

    // The negative control: the refusal is per-parameter, not per-family. σ has
    // no hand-written link and goes through `chain_to_eta` like anything else.
    GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &f,
        &StudentT,
        FitConfig::default().with_link("sigma", "sqrt"),
    )
    .expect("StudentT sigma should still accept an override");
}

#[test]
fn hurdle_answers_for_xi_itself_rather_than_asking_its_base() {
    // `xi` is the wrapper's own parameter and is absent from `base.parameters()`,
    // so a blind delegation would let a refusing base veto a name it has never
    // heard of. Hurdle's ξ atom goes through `chain_to_eta`, so it accepts links.
    let hurdle_over_ocat = Hurdle::new(Box::new(Ocat::new(3)));
    assert!(
        hurdle_over_ocat.allows_link_override("xi"),
        "xi is the wrapper's own parameter and is link-generic"
    );
    assert!(
        !hurdle_over_ocat.allows_link_override("mu"),
        "a base parameter still answers for itself"
    );

    // And the ordinary case still delegates through to an accepting base.
    let hurdle_over_gamma = Hurdle::new(Box::new(glissando::distributions::Gamma));
    assert!(hurdle_over_gamma.allows_link_override("xi"));
    assert!(hurdle_over_gamma.allows_link_override("mu"));
}

/// A family may only refuse a name it actually has. Without this, a family added
/// later could return `false` for a parameter outside `parameters()` and the
/// refusal would be unreachable dead logic that reads as a live constraint.
#[test]
fn refusals_only_ever_name_real_parameters() {
    let families: Vec<Box<dyn Distribution>> = vec![
        Box::new(Ocat::new(3)),
        Box::new(Ocat::new(5)),
        Box::new(StudentT),
        Box::new(Beta),
        Box::new(Binomial::new(1)),
        Box::new(Hurdle::new(Box::new(glissando::distributions::Gamma))),
    ];
    for family in &families {
        for junk in ["not_a_parameter", "", "MU"] {
            assert!(
                !family.parameters().contains(&junk),
                "{} unexpectedly has a parameter named {junk:?}",
                family.name()
            );
        }
        // Every refused name must be one the family really exposes; the fit
        // rejects unknown keys before ever consulting `allows_link_override`.
        let refused: Vec<&str> = family
            .parameters()
            .iter()
            .copied()
            .filter(|p| !family.allows_link_override(p))
            .collect();
        for p in &refused {
            assert!(
                family.default_link(p).is_ok(),
                "{} refuses an override for {p}, which is not one of its parameters",
                family.name()
            );
        }
    }
}

#[cfg(feature = "serialization")]
#[test]
fn json_roundtrip_preserves_overridden_link() {
    let (data, y) = binary_data();
    let family = Binomial::new(1);

    let model = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &formula(),
        &family,
        FitConfig::default().with_link("mu", "probit"),
    )
    .unwrap();
    let preds = model.predict(&data, &family).unwrap();

    let json = model.to_json(&family).unwrap();
    let (reloaded, desc) = GamlssModel::from_json(&json).unwrap();
    // SER-1: Binomial now round-trips through the descriptor (it carries n_trials),
    // where a bare name string previously could not rebuild it.
    assert_eq!(desc.build().unwrap().name(), "Binomial");
    // The persisted link survives the round-trip...
    assert_eq!(reloaded.models["mu"].link.as_deref(), Some("probit"));

    // ...and predict reconstructs the probit link, not the logit default, so
    // predictions match bit-for-bit. (A regression that dropped the persisted
    // link would silently predict through logit and diverge here.)
    let preds2 = reloaded.predict(&data, &family).unwrap();
    for (a, b) in preds["mu"].iter().zip(preds2["mu"].iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "probit predictions must survive JSON round-trip ({a} vs {b})"
        );
    }
}
