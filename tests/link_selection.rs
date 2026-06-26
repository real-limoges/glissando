//! Per-parameter link selection (LINK-1..6).
//!
//! Exercises the public surface of the link-selection feature:
//! - `FitConfig::with_link` overrides a parameter's link at fit time;
//! - the override actually changes the fit (probit/cloglog ≠ logit default);
//! - an unknown link name is a typed error;
//! - the chosen link is persisted, so a JSON round-trip predicts through the
//!   *overridden* link, not the family default.

use glissando::distributions::{link_from_name, Binomial};
use glissando::{DataSet, FitConfig, Formula, GamlssModel, Term};
use ndarray::Array1;

/// Reference `(link, η, inv_link(η), mu_eta(η))` from R `stats::make.link` — these
/// are the canonical closed-form values (`make.link("probit")`, `"cloglog"`,
/// `"cauchit"`, `"sqrt"`, `"inverse"`, and `"1/mu^2"`). The non-Gaussian rows are
/// exact arithmetic; the probit rows were tabulated with an independent `erf`
/// implementation. A sign or formula slip in any link fails this table.
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
    // All three should pick up the positive trend...
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
    let (reloaded, dist_name) = GamlssModel::from_json(&json).unwrap();
    assert_eq!(dist_name, "Binomial");
    // The persisted link survives the round-trip...
    assert_eq!(reloaded.models["mu"].link.as_deref(), Some("probit"));

    // ...and predict reconstructs the probit link, not the logit default — so
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
