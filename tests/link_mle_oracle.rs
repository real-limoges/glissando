// Independent maximum-likelihood oracle for per-parameter link selection.
//
// WHAT THIS PROVES. With only `Intercept` + `Linear` terms there are no
// penalties, so a glissando fit is a plain unpenalized MLE. This file computes
// that same MLE a second way, by directly maximizing `loglik_pointwise` over
// the coefficient vector with a generic Newton optimizer whose gradient and
// Hessian are pure central differences, and asserts the two agree.
//
// WHY IT IS A GENUINE ORACLE. The optimizer never touches
// `Distribution::theta_derivatives`, `Link::mu_eta`, `fitting::scoring`, or the PWLS
// solver. It only evaluates `loglik_pointwise`, which is validated
// independently by each family's own unit tests. So a disagreement localizes
// precisely to the IRLS machinery: the fit did not find the maximum of the
// likelihood it claims to be maximizing.
//
// This is deliberately not a table of R `glm()` constants. R is not installed
// here, and the resulting magic numbers would be unverifiable and frozen to one
// fixture; a live optimizer is reproducible, extends to any family/link pair,
// and runs in CI.
//
// CURRENT STATUS (Altitude #1). Every case below, default link and overridden
// alike, is expected to PASS. These four were the acceptance gate for the
// generic-chain-rule refactor and ran `#[ignore]`d while it was outstanding;
// that refactor has landed, so they are unignored and gate CI like any other
// test. A family now returns its score on the natural parameter scale from
// `Distribution::theta_derivatives` and `distributions::chain_to_eta` applies
// whichever link `fitting::validate_link_overrides` resolved, so an override
// gives IRLS the right `dμ/dη`.
//
// A failure here is a real regression in the IRLS machinery, not an expected
// one. `default_link_fits_reach_the_mle` is the control: if *it* fails, suspect
// the oracle before the fitter. Do not repair a failure by loosening a
// tolerance.
#![cfg(not(feature = "python"))]
#![cfg(not(target_arch = "wasm32"))]

use glissando::distributions::{link_from_name, Distribution, Gamma, Link, Poisson};
use glissando::distributions::{Binomial, Gaussian};
use glissando::{DataSet, FitConfig, Formula, GamlssModel, Term};
use ndarray::Array1;
use std::collections::HashMap;

/// How one distribution parameter is modelled: its link, and whether it carries
/// a slope on `x` in addition to an intercept.
struct ParamSpec {
    name: &'static str,
    link: Box<dyn Link>,
    with_slope: bool,
}

impl ParamSpec {
    fn n_coef(&self) -> usize {
        if self.with_slope {
            2
        } else {
            1
        }
    }
}

/// Map a flat coefficient vector to each parameter's response-scale values.
///
/// Layout matches glissando's: per parameter, `[intercept, slope?]`, in the
/// order `specs` is given (which callers keep aligned with
/// `family.parameters()`).
fn response_scale(specs: &[ParamSpec], beta: &[f64], x: &Array1<f64>) -> Vec<Array1<f64>> {
    let mut out = Vec::with_capacity(specs.len());
    let mut k = 0;
    for spec in specs {
        let b0 = beta[k];
        let b1 = if spec.with_slope { beta[k + 1] } else { 0.0 };
        k += spec.n_coef();
        out.push(x.mapv(|xi| spec.link.inv_link(b0 + b1 * xi)));
    }
    out
}

/// Total log-likelihood at `beta`. The only glissando code this oracle calls.
fn total_loglik<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    x: &Array1<f64>,
    specs: &[ParamSpec],
    beta: &[f64],
) -> f64 {
    let values = response_scale(specs, beta, x);
    let params: HashMap<&str, &Array1<f64>> = specs
        .iter()
        .zip(values.iter())
        .map(|(s, v)| (s.name, v))
        .collect();
    match family.loglik_pointwise(y, &params) {
        Ok(ll) => {
            let s: f64 = ll.sum();
            if s.is_finite() {
                s
            } else {
                f64::NEG_INFINITY
            }
        }
        Err(_) => f64::NEG_INFINITY,
    }
}

/// Central-difference gradient of `total_loglik`.
///
/// The oracle's accuracy is set here, not by the Hessian: the returned estimate
/// is the root of this gradient, so its ~1e-10 relative error is what bounds
/// the recovered coefficients. The Hessian only has to be good enough to
/// converge.
fn gradient<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    x: &Array1<f64>,
    specs: &[ParamSpec],
    beta: &[f64],
) -> Vec<f64> {
    let mut g = vec![0.0; beta.len()];
    let mut b = beta.to_vec();
    for j in 0..beta.len() {
        let h = 1e-6 * beta[j].abs().max(1.0);
        b[j] = beta[j] + h;
        let plus = total_loglik(family, y, x, specs, &b);
        b[j] = beta[j] - h;
        let minus = total_loglik(family, y, x, specs, &b);
        b[j] = beta[j];
        g[j] = (plus - minus) / (2.0 * h);
    }
    g
}

/// Central-difference Hessian, built from differences of `gradient`.
#[allow(clippy::needless_range_loop)] // index arithmetic is the clearest form for matrix assembly
fn hessian<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    x: &Array1<f64>,
    specs: &[ParamSpec],
    beta: &[f64],
) -> Vec<Vec<f64>> {
    let n = beta.len();
    let mut h = vec![vec![0.0; n]; n];
    let mut b = beta.to_vec();
    for j in 0..n {
        let step = 1e-4 * beta[j].abs().max(1.0);
        b[j] = beta[j] + step;
        let gp = gradient(family, y, x, specs, &b);
        b[j] = beta[j] - step;
        let gm = gradient(family, y, x, specs, &b);
        b[j] = beta[j];
        for i in 0..n {
            h[i][j] = (gp[i] - gm[i]) / (2.0 * step);
        }
    }
    // Symmetrize: the two mixed partials differ only by truncation error.
    for i in 0..n {
        for j in 0..i {
            let avg = 0.5 * (h[i][j] + h[j][i]);
            h[i][j] = avg;
            h[j][i] = avg;
        }
    }
    h
}

/// Solve `A z = b` by Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)] // ditto: Gaussian elimination is index-driven
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&r1, &r2| {
            a[r1][col]
                .abs()
                .partial_cmp(&a[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if a[pivot][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut z = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * z[j];
        }
        z[i] = s / a[i][i];
    }
    Some(z)
}

/// Maximize the log-likelihood by damped Newton from `start`.
///
/// Damping is a plain backtracking line search on the log-likelihood itself, so
/// a bad Newton direction (e.g. where the observed Hessian is not negative
/// definite) degrades to a small ascent step rather than diverging.
fn maximize<D: Distribution + ?Sized>(
    family: &D,
    y: &Array1<f64>,
    x: &Array1<f64>,
    specs: &[ParamSpec],
    start: &[f64],
) -> Vec<f64> {
    let mut beta = start.to_vec();
    let mut ll = total_loglik(family, y, x, specs, &beta);
    assert!(
        ll.is_finite(),
        "oracle start point must have finite log-likelihood"
    );

    for _ in 0..500 {
        let g = gradient(family, y, x, specs, &beta);
        let gnorm = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if gnorm < 1e-10 {
            break;
        }
        let h = hessian(family, y, x, specs, &beta);
        // Newton step solves H·d = −g (ascent on a concave surface).
        let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
        let mut dir = match solve(h, neg_g) {
            Some(d) => d,
            None => g.clone(), // singular Hessian: fall back to gradient ascent
        };

        // Where the observed Hessian is indefinite (which happens routinely for
        // a scale parameter far from its optimum), the Newton direction can
        // point *downhill*. Backtracking only shortens a step, it never flips
        // it, so such a direction stalls the search at a non-stationary point.
        // Guard by falling back to steepest ascent whenever the direction is
        // not an ascent direction to first order.
        let slope: f64 = g.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
        if slope.is_nan() || slope <= 0.0 {
            dir = g.clone();
        }

        let mut alpha = 1.0;
        let mut accepted = false;
        for _ in 0..60 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(dir.iter())
                .map(|(b, d)| b + alpha * d)
                .collect();
            let trial_ll = total_loglik(family, y, x, specs, &trial);
            if trial_ll > ll {
                beta = trial;
                ll = trial_ll;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break; // at a maximum to within line-search resolution
        }
    }
    beta
}

/// A tight fit config: the default `tolerance`/`gd_tolerance` of 1e-3 would
/// otherwise bound the comparison well above the oracle's precision.
fn tight() -> FitConfig {
    FitConfig {
        max_iterations: 500,
        tolerance: 1e-11,
        gd_tolerance: 1e-11,
        ..FitConfig::default()
    }
}

fn formula_mu_only() -> Formula {
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

/// `μ ~ 1 + x`, `σ ~ 1`. Two-parameter families need an entry for every
/// parameter or the fit rejects the formula.
fn formula_mu_and_sigma() -> Formula {
    formula_mu_only().with_terms("sigma", vec![Term::Intercept])
}

/// Flatten a fitted model's coefficients in `family.parameters()` order.
fn fitted_coefficients<D: Distribution + ?Sized>(model: &GamlssModel, family: &D) -> Vec<f64> {
    family
        .parameters()
        .iter()
        .flat_map(|p| model.models[*p].coefficients.0.to_vec())
        .collect()
}

/// Assert that the fit reached the maximum of its own likelihood.
///
/// The primary assertion is on the **log-likelihood**, not the coefficients:
/// `ll(fitted) >= ll(oracle)` up to a small slack. That is immune to
/// coefficient ordering, parameterization, and any flat direction in the
/// surface: if an independent optimizer finds a strictly higher likelihood,
/// the fit is definitively not at the MLE, and the size of the gap says how
/// badly.
///
/// The coefficient check follows as a sharper, secondary statement once the
/// likelihoods agree.
#[allow(clippy::too_many_arguments)]
fn assert_reaches_mle<D: Distribution + ?Sized>(
    label: &str,
    family: &D,
    y: &Array1<f64>,
    x: &Array1<f64>,
    specs: &[ParamSpec],
    fitted: &[f64],
    oracle: &[f64],
    tol: f64,
) {
    assert_eq!(fitted.len(), oracle.len(), "{label}: coefficient count");

    let ll_fit = total_loglik(family, y, x, specs, fitted);
    let ll_oracle = total_loglik(family, y, x, specs, oracle);

    // Sanity on the oracle itself: it must sit at a stationary point, otherwise
    // a failure below says nothing about the fit.
    let g = gradient(family, y, x, specs, oracle);
    let gnorm = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        gnorm < 1e-4,
        "{label}: the ORACLE failed to converge (max |grad| = {gnorm:.3e}). \
         Fix the oracle before interpreting this as a defect in the fit."
    );

    let slack = 1e-9 * ll_oracle.abs().max(1.0);
    assert!(
        ll_fit >= ll_oracle - slack,
        "{label}: the fit's log-likelihood is {ll_fit:.12e}, but an independent \
         optimizer reached {ll_oracle:.12e} (higher by {:.3e}). The fit did not \
         reach the maximum of the likelihood it is fitting.\n  \
         fitted coefficients: {fitted:?}\n  oracle coefficients: {oracle:?}",
        ll_oracle - ll_fit
    );

    for (i, (f, o)) in fitted.iter().zip(oracle.iter()).enumerate() {
        let rel = (f - o).abs() / o.abs().max(1.0);
        assert!(
            rel < tol,
            "{label}: coefficient {i} is {f:.12e}, but the maximum-likelihood \
             value is {o:.12e} (relative error {rel:.3e})."
        );
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Deterministic binary data with a clear positive trend and enough overlap
/// that no link separates it perfectly (which would send coefficients to ±∞).
fn binary_data() -> (DataSet, Array1<f64>, Array1<f64>) {
    let n = 60;
    let x: Array1<f64> = (0..n).map(|i| i as f64 / n as f64 - 0.5).collect();
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let p = 1.0 / (1.0 + (-6.0 * x[i]).exp());
            let dither = ((i * 7) % 10) as f64 / 10.0;
            if p > dither {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", x.clone());
    (data, y, x)
}

/// Deterministic counts with a mild trend, kept small enough that a `sqrt`
/// link stays in its valid region.
fn count_data() -> (DataSet, Array1<f64>, Array1<f64>) {
    let n = 60;
    let x: Array1<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let rate = 2.0 + 3.0 * x[i];
            (rate + ((i * 13 % 7) as f64 - 3.0) * 0.4).max(0.0).round()
        })
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", x.clone());
    (data, y, x)
}

/// Strictly positive responses for Gamma.
fn positive_data() -> (DataSet, Array1<f64>, Array1<f64>) {
    let n = 60;
    let x: Array1<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let m = 1.0 + 2.0 * x[i];
            m * (1.0 + ((i * 11 % 9) as f64 - 4.0) * 0.05)
        })
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", x.clone());
    (data, y, x)
}

// ---------------------------------------------------------------------------
// Control: the oracle agrees with glissando on every DEFAULT link
// ---------------------------------------------------------------------------

// If these ever fail, the oracle is broken, not the fitter; check here first
// before believing any of the override tests below.

#[test]
fn default_link_fits_reach_the_mle() {
    // Binomial / logit.
    {
        let (data, y, x) = binary_data();
        let family = Binomial::new(1);
        let model =
            GamlssModel::fit_with_config(&data, &y, None, &formula_mu_only(), &family, tight())
                .unwrap();
        let specs = [ParamSpec {
            name: "mu",
            link: link_from_name("logit").unwrap(),
            with_slope: true,
        }];
        let oracle = maximize(&family, &y, &x, &specs, &[0.0, 0.0]);
        assert_reaches_mle(
            "binomial/logit (default)",
            &family,
            &y,
            &x,
            &specs,
            &fitted_coefficients(&model, &family),
            &oracle,
            1e-5,
        );
    }

    // Poisson / log.
    {
        let (data, y, x) = count_data();
        let family = Poisson::new();
        let model =
            GamlssModel::fit_with_config(&data, &y, None, &formula_mu_only(), &family, tight())
                .unwrap();
        let specs = [ParamSpec {
            name: "mu",
            link: link_from_name("log").unwrap(),
            with_slope: true,
        }];
        let oracle = maximize(&family, &y, &x, &specs, &[0.5, 0.0]);
        assert_reaches_mle(
            "poisson/log (default)",
            &family,
            &y,
            &x,
            &specs,
            &fitted_coefficients(&model, &family),
            &oracle,
            1e-5,
        );
    }

    // Gaussian / identity + log σ: two parameters, so this also checks the
    // oracle's handling of a nuisance parameter.
    {
        let (data, y, x) = positive_data();
        let family = Gaussian::new();
        let model = GamlssModel::fit_with_config(
            &data,
            &y,
            None,
            &formula_mu_and_sigma(),
            &family,
            tight(),
        )
        .unwrap();
        let specs = [
            ParamSpec {
                name: "mu",
                link: link_from_name("identity").unwrap(),
                with_slope: true,
            },
            ParamSpec {
                name: "sigma",
                link: link_from_name("log").unwrap(),
                with_slope: false,
            },
        ];
        let oracle = maximize(&family, &y, &x, &specs, &[1.0, 0.0, -1.0]);
        assert_reaches_mle(
            "gaussian/identity (default)",
            &family,
            &y,
            &x,
            &specs,
            &fitted_coefficients(&model, &family),
            &oracle,
            1e-5,
        );
    }
}

// ---------------------------------------------------------------------------
// The bug: overridden links do not reach the MLE
// ---------------------------------------------------------------------------

#[test]
fn binomial_probit_fit_reaches_the_mle() {
    let (data, y, x) = binary_data();
    let family = Binomial::new(1);
    let model = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &formula_mu_only(),
        &family,
        tight().with_link("mu", "probit"),
    )
    .unwrap();
    let specs = [ParamSpec {
        name: "mu",
        link: link_from_name("probit").unwrap(),
        with_slope: true,
    }];
    let oracle = maximize(&family, &y, &x, &specs, &[0.0, 0.0]);
    assert_reaches_mle(
        "binomial/probit",
        &family,
        &y,
        &x,
        &specs,
        &fitted_coefficients(&model, &family),
        &oracle,
        1e-5,
    );
}

#[test]
fn binomial_cloglog_fit_reaches_the_mle() {
    let (data, y, x) = binary_data();
    let family = Binomial::new(1);
    let model = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &formula_mu_only(),
        &family,
        tight().with_link("mu", "cloglog"),
    )
    .unwrap();
    let specs = [ParamSpec {
        name: "mu",
        link: link_from_name("cloglog").unwrap(),
        with_slope: true,
    }];
    let oracle = maximize(&family, &y, &x, &specs, &[0.0, 0.0]);
    assert_reaches_mle(
        "binomial/cloglog",
        &family,
        &y,
        &x,
        &specs,
        &fitted_coefficients(&model, &family),
        &oracle,
        1e-5,
    );
}

#[test]
fn poisson_sqrt_fit_reaches_the_mle() {
    let (data, y, x) = count_data();
    let family = Poisson::new();
    let model = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &formula_mu_only(),
        &family,
        tight().with_link("mu", "sqrt"),
    )
    .unwrap();
    let specs = [ParamSpec {
        name: "mu",
        link: link_from_name("sqrt").unwrap(),
        with_slope: true,
    }];
    let oracle = maximize(&family, &y, &x, &specs, &[1.4, 0.0]);
    assert_reaches_mle(
        "poisson/sqrt",
        &family,
        &y,
        &x,
        &specs,
        &fitted_coefficients(&model, &family),
        &oracle,
        1e-5,
    );
}

#[test]
fn gamma_inverse_fit_reaches_the_mle() {
    let (data, y, x) = positive_data();
    let family = Gamma::new();
    let model = GamlssModel::fit_with_config(
        &data,
        &y,
        None,
        &formula_mu_and_sigma(),
        &family,
        tight().with_link("mu", "inverse"),
    )
    .unwrap();
    let specs = [
        ParamSpec {
            name: "mu",
            link: link_from_name("inverse").unwrap(),
            with_slope: true,
        },
        ParamSpec {
            name: "sigma",
            link: link_from_name("log").unwrap(),
            with_slope: false,
        },
    ];
    let oracle = maximize(&family, &y, &x, &specs, &[1.0, 0.0, -1.5]);
    assert_reaches_mle(
        "gamma/inverse",
        &family,
        &y,
        &x,
        &specs,
        &fitted_coefficients(&model, &family),
        &oracle,
        1e-5,
    );
}
