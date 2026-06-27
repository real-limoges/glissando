//! Box-Cox–Cole-Green (BCCG) family — public-API integration tests (DIST-1).
//!
//! Covers: coefficient recovery from simulated data; the chosen distribution
//! reduces to the right textbook law at the ν special cases (an *independent*
//! oracle against statrs `Normal`/`LogNormal`); a JSON round-trip predicts through
//! the same family; and LMS centile curves are monotone with C50 = the fitted
//! median μ.

// Integration tests cannot run with the `python` feature (PyO3 extension-module linking).
#![cfg(not(feature = "python"))]

mod common;

use common::{linear, Generator};
use glissando::distributions::{Distribution, BCCG};
use glissando::{Formula, GamlssModel, Term};
use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, LogNormal, Normal};

/// μ ~ intercept + x (log link), σ and ν intercept-only. A purely parametric
/// GAMLSS (no smooths → no penalty), so the fit is plain maximum likelihood.
fn recovery_formula() -> Formula {
    Formula::new()
        .with_terms("mu", vec![Term::Intercept, linear("x")])
        .with_terms("sigma", vec![Term::Intercept])
        .with_terms("nu", vec![Term::Intercept])
}

fn params_view<'a>(
    owned: &'a [(&'static str, Array1<f64>)],
) -> std::collections::HashMap<&'a str, &'a Array1<f64>> {
    owned.iter().map(|(k, v)| (*k, v)).collect()
}

#[test]
fn bccg_recovers_known_parameters() {
    // True: log(μ) = 1.0 + 0.7·x, σ = 0.15, ν = 0.6.
    let (intercept, slope, sigma, nu) = (1.0, 0.7, 0.15, 0.6);
    let mut rng = Generator::new(42);
    let (y, data) = rng.bccg_data(800, intercept, slope, sigma, nu);

    let model = GamlssModel::fit(&data, &y, &recovery_formula(), &BCCG::new()).unwrap();
    assert!(model.converged(), "BCCG fit should converge");

    // μ coefficients are on the log scale: [b0 ≈ intercept, b1 ≈ slope].
    let mu_beta = &model.models["mu"].coefficients.0;
    assert!(
        (mu_beta[0] - intercept).abs() < 0.1,
        "μ intercept: {} vs {}",
        mu_beta[0],
        intercept
    );
    assert!(
        (mu_beta[1] - slope).abs() < 0.2,
        "μ slope: {} vs {}",
        mu_beta[1],
        slope
    );

    // σ (log link, intercept-only): exp(coef) ≈ σ.
    let sigma_hat = model.models["sigma"].coefficients.0[0].exp();
    assert!(
        (sigma_hat - sigma).abs() < 0.05,
        "σ̂ {} vs {}",
        sigma_hat,
        sigma
    );

    // ν (identity link, intercept-only): coef ≈ ν. Skewness is the noisiest
    // parameter, so the band is deliberately wide but still excludes 0 and 1.
    let nu_hat = model.models["nu"].coefficients.0[0];
    assert!((nu_hat - nu).abs() < 0.4, "ν̂ {} vs {}", nu_hat, nu);
}

#[test]
fn bccg_reduces_to_normal_at_nu_one() {
    // At ν = 1 the Box-Cox transform is z = (y/μ − 1)/σ, so BCCG(μ, σ, 1) is a
    // Normal(μ, μσ). Validate cdf/quantile against statrs's *independent* Normal.
    let owned = [
        ("mu", Array1::from_vec(vec![4.0, 4.0, 4.0, 4.0, 4.0])),
        ("sigma", Array1::from_vec(vec![0.2, 0.2, 0.2, 0.2, 0.2])),
        ("nu", Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0])),
    ];
    let p = params_view(&owned);
    let bccg = BCCG::new();
    let normal = Normal::new(4.0, 4.0 * 0.2).unwrap();

    let ys = Array1::from_vec(vec![2.5, 3.5, 4.0, 4.5, 5.5]);
    let cdf = bccg.cdf(&ys, &p).unwrap();
    for (i, &yi) in ys.iter().enumerate() {
        assert!(
            (cdf[i] - normal.cdf(yi)).abs() < 1e-9,
            "cdf at y={yi}: BCCG {} vs Normal {}",
            cdf[i],
            normal.cdf(yi)
        );
    }
    let levels = Array1::from_vec(vec![0.05, 0.25, 0.5, 0.75, 0.95]);
    let q = bccg.quantile(&levels, &p).unwrap();
    for (i, &lvl) in levels.iter().enumerate() {
        assert!(
            (q[i] - normal.inverse_cdf(lvl)).abs() < 1e-7,
            "quantile at p={lvl}: BCCG {} vs Normal {}",
            q[i],
            normal.inverse_cdf(lvl)
        );
    }
}

#[test]
fn bccg_reduces_to_lognormal_at_nu_zero() {
    // At ν = 0 the transform is z = log(y/μ)/σ, so BCCG(μ, σ, 0) is a
    // LogNormal(log μ, σ). Validate against statrs's independent LogNormal.
    let owned = [
        ("mu", Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0, 2.0])),
        ("sigma", Array1::from_vec(vec![0.4, 0.4, 0.4, 0.4, 0.4])),
        ("nu", Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0])),
    ];
    let p = params_view(&owned);
    let bccg = BCCG::new();
    let lognormal = LogNormal::new(2.0_f64.ln(), 0.4).unwrap();

    let ys = Array1::from_vec(vec![1.0, 1.5, 2.0, 3.0, 4.5]);
    let cdf = bccg.cdf(&ys, &p).unwrap();
    for (i, &yi) in ys.iter().enumerate() {
        assert!(
            (cdf[i] - lognormal.cdf(yi)).abs() < 1e-9,
            "cdf at y={yi}: BCCG {} vs LogNormal {}",
            cdf[i],
            lognormal.cdf(yi)
        );
    }
    let levels = Array1::from_vec(vec![0.05, 0.25, 0.5, 0.75, 0.95]);
    let q = bccg.quantile(&levels, &p).unwrap();
    for (i, &lvl) in levels.iter().enumerate() {
        assert!(
            (q[i] - lognormal.inverse_cdf(lvl)).abs() < 1e-7,
            "quantile at p={lvl}: BCCG {} vs LogNormal {}",
            q[i],
            lognormal.inverse_cdf(lvl)
        );
    }
}

#[test]
fn bccg_centiles_are_monotone_and_median_is_mu() {
    let mut rng = Generator::new(7);
    let (y, data) = rng.bccg_data(400, 1.2, 0.5, 0.18, 0.8);
    let model = GamlssModel::fit(&data, &y, &recovery_formula(), &BCCG::new()).unwrap();

    let levels = [5.0, 25.0, 50.0, 75.0, 95.0];
    let centiles = model.centiles(&data, &BCCG::new(), &levels).unwrap();

    // Monotone in the centile level at every row.
    for w in levels.windows(2) {
        let lo = &centiles[&format!("C{}", w[0])];
        let hi = &centiles[&format!("C{}", w[1])];
        for (i, (&l, &h)) in lo.iter().zip(hi.iter()).enumerate() {
            assert!(h > l, "row {i}: C{} {h} should exceed C{} {l}", w[1], w[0]);
        }
    }

    // C50 is the fitted median μ exactly (p = 0.5 ⇒ z_p = 0 ⇒ y = μ).
    let mu = &model.predict(&data, &BCCG::new()).unwrap()["mu"];
    for (i, (&c, &m)) in centiles["C50"].iter().zip(mu.iter()).enumerate() {
        assert!((c - m).abs() < 1e-6, "row {i}: C50 {c} vs μ {m}");
    }
}

#[cfg(feature = "serialization")]
#[test]
fn bccg_json_roundtrip_predicts_identically() {
    let mut rng = Generator::new(99);
    let (y, data) = rng.bccg_data(300, 1.0, 0.6, 0.2, 0.5);
    let family = BCCG::new();
    let model = GamlssModel::fit(&data, &y, &recovery_formula(), &family).unwrap();
    let preds = model.predict(&data, &family).unwrap();

    let json = model.to_json(&family).unwrap();
    let (reloaded, dist_name) = GamlssModel::from_json(&json).unwrap();
    assert_eq!(dist_name, "BCCG");

    // The reloaded model predicts bit-for-bit — confirms μ/σ/ν and the family all
    // survive the round-trip and predict reconstructs the BCCG transform.
    let reloaded_family = glissando::distributions::from_name(&dist_name).unwrap();
    let preds2 = reloaded.predict(&data, reloaded_family.as_ref()).unwrap();
    for key in ["mu", "sigma", "nu"] {
        for (a, b) in preds[key].iter().zip(preds2[key].iter()) {
            assert!((a - b).abs() < 1e-12, "{key}: {a} vs {b} after round-trip");
        }
    }
}
