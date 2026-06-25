// Behavioral tests for the RS-loop robustness fixes:
//   FIT-1 — step-halving / line-search on the global deviance (monotone descent)
//   FIT-2 — global-deviance outer convergence
//
// The headline guarantees are monotone descent on a stress case and an
// unchanged solution on well-behaved data; both are checked here against the
// public API. The fine-grained mechanics (a single halving lands a lower
// deviance) are covered by unit tests in `src/fitting/scoring.rs`.
#![cfg(not(feature = "python"))]

mod common;

use common::{linear_intercepts, Generator};
use glissando::{distributions::StudentT, DataSet, FitConfig, GamlssModel};
use ndarray::Array1;
use rand::RngExt;

/// Heavy-tailed Student-t data with a handful of extreme outliers — the regime
/// where a raw Fisher step overshoots and the deviance can increase.
fn heavy_tailed_stress() -> (Array1<f64>, DataSet) {
    let mut rng = Generator::new(20240617);
    let n = 200;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let mut y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = 5.0 + 2.0 * xi;
            // ν = 2.5: heavy tails, finite variance barely.
            let t: f64 = rng.rng.sample(rand_distr::StudentT::new(2.5).unwrap());
            mu + 0.8 * t
        })
        .collect();
    // Inject a few gross outliers to provoke overshoot.
    for &i in &[10usize, 90, 150, 175] {
        y_vec[i] += 40.0;
    }
    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));
    (y, data)
}

fn cfg_with(step_halving: bool, max_iterations: usize) -> FitConfig {
    FitConfig {
        step_halving,
        max_iterations,
        ..FitConfig::default()
    }
}

#[test]
fn step_halving_keeps_global_deviance_monotone() {
    // The fit is deterministic, so the global deviance reported after a
    // `k`-iteration run is exactly the deviance at cycle k of the full
    // trajectory. Sweeping k therefore reconstructs the per-cycle GD path; with
    // step-halving on it must be non-increasing (small slack for round-off).
    let (y, data) = heavy_tailed_stress();
    let formula = linear_intercepts("x", &["mu", "sigma", "nu"]);

    let mut prev = f64::INFINITY;
    for k in 1..=15 {
        let cfg = cfg_with(true, k);
        let model =
            GamlssModel::fit_with_config(&data, &y, None, &formula, &StudentT::new(), cfg).unwrap();
        let gd = model
            .diagnostics
            .final_deviance
            .expect("final_deviance set once a cycle runs");
        assert!(
            gd <= prev + 1e-6,
            "global deviance increased at cycle {k}: {gd} > {prev}"
        );
        prev = gd;
    }
}

#[test]
fn step_halving_reaches_no_worse_deviance_than_raw_loop() {
    // On the overshoot-prone stress case, the monotone (step-halving) loop must
    // not end up at a *worse* fit than the raw full-step loop within the same
    // iteration budget — the whole point of damping. In practice the raw loop
    // oscillates to a higher deviance (or fails to settle); damping reaches an
    // at-least-as-good objective.
    let (y, data) = heavy_tailed_stress();
    let formula = linear_intercepts("x", &["mu", "sigma", "nu"]);

    let on = GamlssModel::fit_with_config(&data, &y, None, &formula, &StudentT::new(), cfg_with(true, 200)).unwrap();
    let off = GamlssModel::fit_with_config(&data, &y, None, &formula, &StudentT::new(), cfg_with(false, 200)).unwrap();

    let gd_on = on.diagnostics.final_deviance.unwrap();
    let gd_off = off.diagnostics.final_deviance.unwrap();
    assert!(gd_on.is_finite(), "step-halving deviance must stay finite");
    assert!(
        gd_on <= gd_off + 1e-6,
        "step-halving should be no worse than the raw loop: on={gd_on} off={gd_off}"
    );
}

#[test]
fn step_halving_default_converges_on_heavy_tailed_data() {
    // Heavy tails (ν small) without gross outliers: the regime the guide cites,
    // mild enough that the damped loop converges within the iteration budget.
    let mut rng = Generator::new(99);
    let n = 200;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let y_vec: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = 5.0 + 2.0 * xi;
            let t: f64 = rng.rng.sample(rand_distr::StudentT::new(4.0).unwrap());
            mu + 0.5 * t
        })
        .collect();
    let y = Array1::from_vec(y_vec);
    let mut data = DataSet::new();
    data.insert_column("x".to_string(), Array1::from_vec(x));

    let formula = linear_intercepts("x", &["mu", "sigma", "nu"]);
    let model = GamlssModel::fit(&data, &y, &formula, &StudentT::new()).unwrap();
    assert!(
        model.converged(),
        "heavy-tailed fit should converge with step-halving on"
    );
}

#[test]
fn final_deviance_matches_minus_two_loglik() {
    // FIT-2's reported deviance is exactly −2·loglik of the converged fit, tying
    // the in-loop helper to the public diagnostics path.
    let (y, data) = heavy_tailed_stress();
    let formula = linear_intercepts("x", &["mu", "sigma", "nu"]);
    let model = GamlssModel::fit(&data, &y, &formula, &StudentT::new()).unwrap();

    let gd = model.diagnostics.final_deviance.unwrap();
    let ll = model.diagnostics(&StudentT::new(), &y).unwrap().log_likelihood;
    assert!(
        (gd - (-2.0 * ll)).abs() < 1e-6,
        "final_deviance {gd} != -2*loglik {}",
        -2.0 * ll
    );
}

#[test]
fn step_halving_toggle_is_a_no_op_on_well_behaved_data() {
    // On well-conditioned Gaussian data no halving is needed, so toggling the
    // flag must not move the converged solution (parity guard for FIT-1).
    let mut rng = Generator::new(7);
    let (y, data) = rng.linear_gaussian(200, 1.0, 2.0, 0.5);
    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let on =
        GamlssModel::fit_with_config(&data, &y, None, &formula, &glissando::distributions::Gaussian::new(), cfg_with(true, 200))
            .unwrap();
    let off =
        GamlssModel::fit_with_config(&data, &y, None, &formula, &glissando::distributions::Gaussian::new(), cfg_with(false, 200))
            .unwrap();

    for param in ["mu", "sigma"] {
        let a = &on.models[param].coefficients.0;
        let b = &off.models[param].coefficients.0;
        for (ca, cb) in a.iter().zip(b.iter()) {
            assert!(
                (ca - cb).abs() < 1e-6,
                "{param} coefficient diverged between step_halving on/off: {ca} vs {cb}"
            );
        }
    }
}
