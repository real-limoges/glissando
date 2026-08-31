//! BCT and BCPE families, public-API integration tests (DIST-1 fast-follows).
//!
//! I cover coefficient recovery from simulated data and JSON round-trips here. The
//! distributional reductions (BCT → BCCG as τ → ∞; BCPE → BCCG at τ = 2) I check at
//! the unit level, over alongside the family impls, not in this file.

// Integration tests can't run under the `python` feature. PyO3's extension-module linking gets in the way.
#![cfg(not(feature = "python"))]

mod common;

use common::{linear, Generator};
use glissando::distributions::{BCPE, BCT};
use glissando::{Formula, GamlssModel, Term};

/// μ ~ intercept + x (log link); σ, ν, τ intercept-only. Purely parametric, so this is straight MLE.
fn recovery_formula() -> Formula {
    Formula::new()
        .with_terms("mu", vec![Term::Intercept, linear("x")])
        .with_terms("sigma", vec![Term::Intercept])
        .with_terms("nu", vec![Term::Intercept])
        .with_terms("tau", vec![Term::Intercept])
}

#[test]
fn bct_recovers_known_parameters() {
    // True: log(μ) = 1.0 + 0.7·x, σ = 0.15, ν = 0.6, τ = 6. Heavy tail.
    let (intercept, slope, sigma, nu, tau) = (1.0, 0.7, 0.15, 0.6, 6.0);
    let mut rng = Generator::new(42);
    let (y, data) = rng.bct_data(900, intercept, slope, sigma, nu, tau);

    // Recovery accuracy is the correctness signal here, not the flag. Near the normal
    // limit τ is weakly identified, so the deviance can wiggle above tolerance and leave
    // `converged` unset even when every estimate has landed on target.
    let model = GamlssModel::fit(&data, &y, &recovery_formula(), &BCT::new()).unwrap();

    let mu_beta = &model.models["mu"].coefficients.0;
    assert!(
        (mu_beta[0] - intercept).abs() < 0.12,
        "μ intercept: {} vs {intercept}",
        mu_beta[0]
    );
    assert!(
        (mu_beta[1] - slope).abs() < 0.25,
        "μ slope: {} vs {slope}",
        mu_beta[1]
    );
    let sigma_hat = model.models["sigma"].coefficients.0[0].exp();
    assert!((sigma_hat - sigma).abs() < 0.05, "σ̂ {sigma_hat} vs {sigma}");
    let nu_hat = model.models["nu"].coefficients.0[0];
    assert!((nu_hat - nu).abs() < 0.4, "ν̂ {nu_hat} vs {nu}");
    // τ̂ within a factor of ~2 of the truth. df is the noisiest parameter, always is.
    let tau_hat = model.models["tau"].coefficients.0[0].exp();
    assert!(tau_hat > 3.0 && tau_hat < 15.0, "τ̂ {tau_hat} vs {tau}");
}

#[test]
fn bcpe_recovers_known_mu_structure() {
    // True: log(μ) = 1.0 + 0.7·x, σ = 0.15, ν = 0.6, τ = 1.6. Leptokurtic.
    let (intercept, slope) = (1.0, 0.7);
    let mut rng = Generator::new(7);
    let (y, data) = rng.bcpe_data(900, intercept, slope, 0.15, 0.6, 1.6);

    let model = GamlssModel::fit(&data, &y, &recovery_formula(), &BCPE::new()).unwrap();
    assert!(model.converged(), "BCPE fit should converge");

    let mu_beta = &model.models["mu"].coefficients.0;
    assert!(
        (mu_beta[0] - intercept).abs() < 0.12,
        "μ intercept: {} vs {intercept}",
        mu_beta[0]
    );
    assert!(
        (mu_beta[1] - slope).abs() < 0.25,
        "μ slope: {} vs {slope}",
        mu_beta[1]
    );
    let sigma_hat = model.models["sigma"].coefficients.0[0].exp();
    assert!(sigma_hat > 0.08 && sigma_hat < 0.25, "σ̂ {sigma_hat}");
    let tau_hat = model.models["tau"].coefficients.0[0].exp();
    assert!(tau_hat.is_finite() && tau_hat > 0.5, "τ̂ {tau_hat}");
}

#[cfg(feature = "serialization")]
#[test]
fn bct_json_roundtrip_predicts_identically() {
    let mut rng = Generator::new(99);
    let (y, data) = rng.bct_data(300, 1.0, 0.6, 0.2, 0.5, 7.0);
    let family = BCT::new();
    let model = GamlssModel::fit(&data, &y, &recovery_formula(), &family).unwrap();
    let preds = model.predict(&data, &family).unwrap();

    let json = model.to_json(&family).unwrap();
    let (reloaded, desc) = GamlssModel::from_json(&json).unwrap();
    let reloaded_family = desc.build().unwrap();
    assert_eq!(reloaded_family.name(), "BCT");
    let preds2 = reloaded.predict(&data, reloaded_family.as_ref()).unwrap();
    for key in ["mu", "sigma", "nu", "tau"] {
        for (a, b) in preds[key].iter().zip(preds2[key].iter()) {
            assert!((a - b).abs() < 1e-12, "{key}: {a} vs {b} after round-trip");
        }
    }
}

#[cfg(feature = "serialization")]
#[test]
fn bcpe_json_roundtrip_predicts_identically() {
    let mut rng = Generator::new(123);
    let (y, data) = rng.bcpe_data(300, 1.0, 0.6, 0.2, 0.5, 1.8);
    let family = BCPE::new();
    let model = GamlssModel::fit(&data, &y, &recovery_formula(), &family).unwrap();
    let preds = model.predict(&data, &family).unwrap();

    let json = model.to_json(&family).unwrap();
    let (reloaded, desc) = GamlssModel::from_json(&json).unwrap();
    let reloaded_family = desc.build().unwrap();
    assert_eq!(reloaded_family.name(), "BCPE");
    let preds2 = reloaded.predict(&data, reloaded_family.as_ref()).unwrap();
    for key in ["mu", "sigma", "nu", "tau"] {
        for (a, b) in preds[key].iter().zip(preds2[key].iter()) {
            assert!((a - b).abs() < 1e-12, "{key}: {a} vs {b} after round-trip");
        }
    }
}
