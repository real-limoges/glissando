// Integration tests cannot run with the `python` feature due to PyO3's extension-module linking
#![cfg(not(feature = "python"))]

mod common;

use common::{cr_spline, linear_intercepts, smooth_intercepts, Generator};
use glissando::{
    distributions::{Gaussian, Poisson},
    DataSet, Formula, GamlssModel, Term,
};
use ndarray::Array1;
use rand::RngExt;

#[test]
fn test_predict_on_training_data() {
    // Predictions on training data should match fitted values
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(100, 2.0, 5.0, 1.0);

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // Predict on the same data
    let predictions = model.predict(&data, &Gaussian::new()).unwrap();

    // Check that predictions match fitted values
    let mu_pred = &predictions["mu"];
    let mu_fitted = &model.models["mu"].fitted_values;

    for i in 0..mu_pred.len() {
        let diff = (mu_pred[i] - mu_fitted[i]).abs();
        assert!(
            diff < 1e-10,
            "Prediction should match fitted at index {}: {} vs {}",
            i,
            mu_pred[i],
            mu_fitted[i]
        );
    }
}

#[test]
fn test_predict_on_new_data() {
    // Test prediction on new data points
    let mut rng = Generator::new(123);
    let (y, data) = rng.linear_gaussian(200, 2.0, 5.0, 1.0);

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // Create new data
    let mut new_data = DataSet::new();
    new_data.insert_column("x", Array1::from_vec(vec![0.0, 50.0, 100.0, 150.0, 200.0]));

    let predictions = model.predict(&new_data, &Gaussian::new()).unwrap();
    let mu_pred = &predictions["mu"];

    // For linear model: mu = intercept + slope * x
    // Predictions should follow this pattern
    let coeffs = &model.models["mu"].coefficients.0;
    let intercept = coeffs[0];
    let slope = coeffs[1];

    for (i, &x) in [0.0, 50.0, 100.0, 150.0, 200.0].iter().enumerate() {
        let expected = intercept + slope * x;
        let diff = (mu_pred[i] - expected).abs();
        assert!(
            diff < 1e-10,
            "Prediction at x={} should be {}, got {}",
            x,
            expected,
            mu_pred[i]
        );
    }
}

#[test]
fn test_predict_with_se() {
    let mut rng = Generator::new(456);
    let (y, data) = rng.linear_gaussian(100, 2.0, 5.0, 1.0);

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let results = model.predict_with_se(&data, &Gaussian::new()).unwrap();

    let mu_result = &results["mu"];

    // Check that standard errors are positive
    for i in 0..mu_result.se_eta.len() {
        assert!(
            mu_result.se_eta[i] >= 0.0,
            "SE should be non-negative at index {}",
            i
        );
    }

    // For Gaussian with identity link, fitted should equal eta
    for i in 0..mu_result.fitted.len() {
        let diff = (mu_result.fitted[i] - mu_result.eta[i]).abs();
        assert!(diff < 1e-10, "For identity link, fitted should equal eta");
    }
}

#[test]
fn test_predict_poisson() {
    let mut rng = Generator::new(789);

    let n = 300;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (1.0 + 0.5 * xi).exp();
            let dist = rand_distr::Poisson::new(mu).unwrap();
            rng.rng.sample(dist)
        })
        .collect();

    let y = Array1::from_vec(y);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));

    let formula = linear_intercepts("x", &["mu"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();

    // Predict on training data
    let predictions = model.predict(&data, &Poisson::new()).unwrap();
    let mu_pred = &predictions["mu"];

    // All predictions should be positive (Poisson has log link)
    for i in 0..mu_pred.len() {
        assert!(
            mu_pred[i] > 0.0,
            "Poisson predictions should be positive, got {} at index {}",
            mu_pred[i],
            i
        );
    }
}

#[test]
fn test_posterior_samples() {
    let mut rng = Generator::new(999);
    let (y, data) = rng.linear_gaussian(100, 2.0, 5.0, 1.0);

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // Get posterior samples for mu
    let samples = model.posterior_samples("mu", 100, None).unwrap();

    assert_eq!(samples.len(), 100, "Should have 100 samples");

    // Each sample should have 2 coefficients (intercept + slope)
    for (i, sample) in samples.iter().enumerate() {
        assert_eq!(sample.0.len(), 2, "Sample {} should have 2 coefficients", i);
    }

    // Sample mean should be close to fitted coefficients
    let fitted_coeffs = &model.models["mu"].coefficients.0;
    let mut mean_intercept = 0.0;
    let mut mean_slope = 0.0;
    for sample in &samples {
        mean_intercept += sample.0[0];
        mean_slope += sample.0[1];
    }
    mean_intercept /= samples.len() as f64;
    mean_slope /= samples.len() as f64;

    assert!(
        (mean_intercept - fitted_coeffs[0]).abs() < 1.0,
        "Sample mean intercept should be close to fitted"
    );
    assert!(
        (mean_slope - fitted_coeffs[1]).abs() < 0.1,
        "Sample mean slope should be close to fitted"
    );
}

#[test]
fn test_predict_samples() {
    let mut rng = Generator::new(111);
    let (y, data) = rng.linear_gaussian(50, 2.0, 5.0, 1.0);

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // Get prediction samples
    let pred_samples = model
        .predict_samples(&data, &Gaussian::new(), 50, None)
        .unwrap();

    // Check mu predictions
    let mu_samples = &pred_samples["mu"];
    assert_eq!(mu_samples.len(), 50, "Should have 50 prediction samples");

    // Each sample should have predictions for all observations
    for sample in mu_samples {
        assert_eq!(sample.len(), 50, "Each sample should have 50 predictions");
    }
}

#[test]
fn test_predict_with_smooth() {
    let mut rng = Generator::new(222);

    let n = 200;
    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 / n as f64 * 2.0 * std::f64::consts::PI)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.sin() + rng.rng.sample::<f64, _>(rand_distr::StandardNormal) * 0.2)
        .collect();

    let y = Array1::from_vec(y);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));

    let formula = smooth_intercepts("x", 10, &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // Predict on training data
    let predictions = model.predict(&data, &Gaussian::new()).unwrap();
    let mu_pred = &predictions["mu"];

    // Predictions should capture the sinusoidal pattern
    // Check that predictions at 0, pi, 2*pi are roughly 0, 0, 0 (sin values)
    let idx_0 = 0;
    let idx_pi = n / 2;
    let idx_2pi = n - 1;

    // At x=0, sin(0) = 0
    assert!(
        mu_pred[idx_0].abs() < 0.5,
        "Prediction at x=0 should be near 0, got {}",
        mu_pred[idx_0]
    );

    // At x=pi, sin(pi) = 0
    assert!(
        mu_pred[idx_pi].abs() < 0.5,
        "Prediction at x=pi should be near 0, got {}",
        mu_pred[idx_pi]
    );

    // At x=2*pi, sin(2*pi) = 0
    assert!(
        mu_pred[idx_2pi].abs() < 0.5,
        "Prediction at x=2*pi should be near 0, got {}",
        mu_pred[idx_2pi]
    );
}

// ----------------------------------------------------------------------------
// Phase F.3 — predict_with_se / predict_samples coverage for non-Gaussian link
// ----------------------------------------------------------------------------

#[test]
fn predict_with_se_for_poisson_log_link() {
    // Poisson uses log link, so fitted (response scale) ≠ eta (link scale).
    // Verifies both the SE calculation and the inv_link composition on a
    // distribution where they differ.
    let mut rng = Generator::new(202);
    let n = 200;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (1.0 + 0.5 * xi).exp();
            rng.rng.sample(rand_distr::Poisson::new(mu).unwrap())
        })
        .collect();
    let y = Array1::from_vec(y);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x));

    let formula = linear_intercepts("x", &["mu"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();
    let results = model.predict_with_se(&data, &Poisson::new()).unwrap();
    let mu = &results["mu"];

    assert_eq!(mu.fitted.len(), n);
    assert_eq!(mu.eta.len(), n);
    assert_eq!(mu.se_eta.len(), n);
    for i in 0..n {
        // fitted = exp(eta) for log link, so fitted > 0 and finite.
        assert!(mu.fitted[i] > 0.0 && mu.fitted[i].is_finite());
        // se_eta should be strictly positive (covariance is PD).
        assert!(mu.se_eta[i] > 0.0, "se_eta[{}] = {} ≤ 0", i, mu.se_eta[i]);
        // For log link, fitted ≠ eta.
        assert!(
            (mu.fitted[i] - mu.eta[i]).abs() > 1e-6,
            "log link should make fitted ≠ eta, but at i={} both are {:.6}",
            i,
            mu.fitted[i]
        );
        // fitted ≈ exp(eta) by construction.
        let expected = mu.eta[i].exp();
        assert!(
            (mu.fitted[i] - expected).abs() < 1e-10,
            "fitted[{}] = {} ≠ exp(eta[{}]) = {}",
            i,
            mu.fitted[i],
            i,
            expected
        );
    }
}

#[test]
fn predict_samples_shape_matches_request_for_poisson() {
    let mut rng = Generator::new(303);
    let n = 150;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 2.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = (0.5 + 0.3 * xi).exp();
            rng.rng.sample(rand_distr::Poisson::new(mu).unwrap())
        })
        .collect();
    let y = Array1::from_vec(y);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x.clone()));

    let formula = linear_intercepts("x", &["mu"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Poisson::new()).unwrap();

    for &n_samples in &[1usize, 10, 100] {
        let samples = model
            .predict_samples(&data, &Poisson::new(), n_samples, None)
            .unwrap();
        let mu_samples = &samples["mu"];
        assert_eq!(mu_samples.len(), n_samples, "outer dim should be n_samples");
        for s in mu_samples {
            assert_eq!(s.len(), n, "inner dim should be n_obs");
            // Log link guarantees positive predictions.
            assert!(s.iter().all(|v| *v > 0.0 && v.is_finite()));
        }
    }
}

// ============================================================================
// Guide 1 — design_matrix / covariance_matrix / term_index_map / seed
// ============================================================================

/// design_matrix identity: X · β must equal the fitted linear predictor on
/// the training data (confirming the exported matrix is the fit-time one).
#[test]
fn design_matrix_dot_beta_equals_eta() {
    let mut rng = Generator::new(77);
    let n = 80;
    let (y, data) = rng.linear_gaussian(n, 1.5, 3.0, 0.8);

    // Intercept + CR spline (safe combination — no redundant linear term).
    let mut formula = Formula::new();
    formula.add_terms("mu".to_string(), vec![Term::Intercept, cr_spline("x", 8)]);
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let x = model.design_matrix(&data, "mu").unwrap();
    let beta = &model.models["mu"].coefficients.0;
    let eta_exported = x.dot(beta);
    let eta_fitted = &model.models["mu"].eta;

    assert_eq!(eta_exported.len(), n);
    for i in 0..n {
        let diff = (eta_exported[i] - eta_fitted[i]).abs();
        assert!(
            diff < 1e-9,
            "X·β[{i}] = {} but fitted η = {} (diff = {})",
            eta_exported[i],
            eta_fitted[i],
            diff
        );
    }
}

/// covariance_matrix is symmetric and positive-definite.
///
/// Symmetry is checked directly. PD is confirmed indirectly: `posterior_samples`
/// internally does a Cholesky factorization and returns `PosteriorNotPositiveDefinite`
/// if it fails — so a successful call is our PD certificate.
#[test]
fn covariance_matrix_is_symmetric_and_psd() {
    let mut rng = Generator::new(88);
    let (y, data) = rng.linear_gaussian(60, 2.0, 1.0, 0.5);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    for param in &["mu", "sigma"] {
        let v = model.covariance_matrix(param).unwrap();
        let mat = &v.0;
        let p = mat.nrows();
        assert_eq!(mat.ncols(), p, "covariance must be square for {param}");

        // Symmetry: V[i,j] ≈ V[j,i]
        for i in 0..p {
            for j in 0..p {
                let diff = (mat[[i, j]] - mat[[j, i]]).abs();
                assert!(
                    diff < 1e-10,
                    "V[{i},{j}] != V[{j},{i}] for {param}: diff={diff}"
                );
            }
        }

        // PD: posterior_samples does Cholesky internally; success proves the matrix is PD.
        model
            .posterior_samples(param, 1, Some(0))
            .expect("covariance must be PD");
    }
}

/// term_index_map is non-overlapping, contiguous, starts at 0, and the total
/// width equals the coefficient count and design-matrix column count.
#[test]
fn term_index_map_is_contiguous_and_complete() {
    let mut rng = Generator::new(55);
    let n = 100;
    let (y, data) = rng.linear_gaussian(n, 2.0, 4.0, 1.0);

    let mut formula = Formula::new();
    formula.add_terms("mu".to_string(), vec![Term::Intercept, cr_spline("x", 7)]);
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    for param in &["mu", "sigma"] {
        let blocks = model.term_index_map(param).unwrap();
        let n_coeffs = model.models[*param].coefficients.0.len();
        let x_ncols = model.design_matrix(&data, param).unwrap().ncols();

        assert!(
            !blocks.is_empty(),
            "term_blocks must not be empty for {param}"
        );
        // Starts at 0
        assert_eq!(
            blocks[0].1, 0,
            "first block must start at col 0 for {param}"
        );
        // Contiguous and non-overlapping
        for i in 1..blocks.len() {
            assert_eq!(
                blocks[i].1,
                blocks[i - 1].2,
                "block {i} start ({}) != block {} end ({}) for {param}",
                blocks[i].1,
                i - 1,
                blocks[i - 1].2
            );
        }
        // Total width == n_coeffs == design matrix column count
        let total: usize = blocks.iter().map(|(_, f, l)| l - f).sum();
        assert_eq!(
            total, n_coeffs,
            "sum of block widths != n_coeffs for {param}"
        );
        assert_eq!(
            total, x_ncols,
            "sum of block widths != design_matrix ncols for {param}"
        );
    }
}

/// Linear term name: `Term::Linear { col_name: "x" }` → `"x"`.
/// (Tested separately since combining Linear + CrSpline on the same column
/// causes collinearity in a real fit.)
#[test]
fn linear_term_name_is_col_name() {
    use glissando::Term;
    let t = Term::Linear {
        col_name: "my_var".to_string(),
    };
    assert_eq!(t.term_name(), "my_var");
    let intercept = Term::Intercept;
    assert_eq!(intercept.term_name(), "(intercept)");
}

/// term_name strings match expected mgcv-style labels.
#[test]
fn term_name_strings_are_correct() {
    let mut rng = Generator::new(66);
    let n = 40;
    let (y, data) = rng.linear_gaussian(n, 1.0, 1.0, 0.5);

    // Add a group column for random effect
    let groups: Array1<f64> = Array1::from_iter((0..n).map(|i| (i % 5) as f64));
    let mut data2 = data.clone();
    data2.insert_column("group", groups);

    let mut formula = Formula::new();
    formula.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            cr_spline("x", 6),
            Term::Smooth(glissando::Smooth::RandomEffect {
                col_name: "group".to_string(),
                levels: vec![],
            }),
        ],
    );
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);
    let model = GamlssModel::fit(&data2, &y, &formula, &Gaussian::new()).unwrap();

    let blocks = model.term_index_map("mu").unwrap();
    let names: Vec<&str> = blocks.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(names[0], "(intercept)");
    assert_eq!(names[1], "s(x)");
    assert_eq!(names[2], "s(group)");

    let sigma_blocks = model.term_index_map("sigma").unwrap();
    assert_eq!(sigma_blocks[0].0, "(intercept)");
}

/// Seeded posterior: same seed → identical samples. No seed → differs (probabilistically).
#[test]
fn seeded_predict_samples_are_reproducible() {
    let mut rng = Generator::new(321);
    let (y, data) = rng.linear_gaussian(50, 2.0, 5.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let n_samples = 20;
    let seed = Some(42u64);

    let run1 = model
        .predict_samples(&data, &Gaussian::new(), n_samples, seed)
        .unwrap();
    let run2 = model
        .predict_samples(&data, &Gaussian::new(), n_samples, seed)
        .unwrap();

    // Same seed → identical arrays.
    for s in 0..n_samples {
        for v in 0..run1["mu"][s].len() {
            assert_eq!(
                run1["mu"][s][v], run2["mu"][s][v],
                "seeded runs differ at sample {s}, obs {v}"
            );
        }
    }

    // No seed → different (with overwhelming probability for 20 samples × 50 obs).
    let run_unseeded = model
        .predict_samples(&data, &Gaussian::new(), n_samples, None)
        .unwrap();
    let all_equal = run1["mu"]
        .iter()
        .zip(run_unseeded["mu"].iter())
        .all(|(a, b)| a.iter().zip(b.iter()).all(|(x, y)| x == y));
    assert!(!all_equal, "unseeded run should differ from seeded run");
}

/// Same test through posterior_samples (coefficient-space samples).
#[test]
fn seeded_posterior_samples_are_reproducible() {
    let mut rng = Generator::new(444);
    let (y, data) = rng.linear_gaussian(60, 1.5, 2.0, 0.5);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let run1 = model.posterior_samples("mu", 30, Some(7)).unwrap();
    let run2 = model.posterior_samples("mu", 30, Some(7)).unwrap();

    for (s1, s2) in run1.iter().zip(run2.iter()) {
        for (a, b) in s1.0.iter().zip(s2.0.iter()) {
            assert_eq!(a, b, "seeded posterior samples differ");
        }
    }
}

#[test]
fn test_predict_missing_column_error() {
    let mut rng = Generator::new(333);
    let (y, data) = rng.linear_gaussian(100, 2.0, 5.0, 1.0);

    let formula = linear_intercepts("x", &["mu", "sigma"]);

    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // Try to predict on data missing the 'x' column
    let mut bad_data = DataSet::new();
    bad_data.insert_column("z", Array1::from_vec(vec![1.0, 2.0, 3.0]));

    let result = model.predict(&bad_data, &Gaussian::new());
    assert!(result.is_err(), "Should error when column is missing");
}

/// CrSpline1D knot-persistence: predicting on data with different quantiles must
/// reproduce training fitted values exactly when the training rows are included.
///
/// If the knots were recomputed from `new_data` instead of being stored from
/// training, the basis would silently differ and in-sample predictions would
/// deviate from fitted values — this test catches that.
#[test]
fn cr_spline_prediction_reuses_training_knots() {
    // Training data on [1, 10]
    let n_train = 40;
    let x_train: Array1<f64> = Array1::linspace(1.0, 10.0, n_train);
    let y_train: Array1<f64> = x_train.mapv(|x| 3.0 * x.sin() + 0.5 * x);

    let mut train_data = DataSet::new();
    train_data.insert_column("x", x_train.clone());

    let mut formula = glissando::Formula::new();
    formula.add_terms("mu".to_string(), vec![Term::Intercept, cr_spline("x", 6)]);
    formula.add_terms("sigma".to_string(), vec![Term::Intercept]);

    let model = GamlssModel::fit(&train_data, &y_train, &formula, &Gaussian::new())
        .expect("CrSpline1D fit should succeed");

    // new_data has DIFFERENT range ([5, 25]) — quantiles differ from training.
    // The training rows [1,10] are appended at the end so we can check them.
    let n_extra = 10;
    let x_extra: Array1<f64> = Array1::linspace(5.0, 25.0, n_extra);
    let x_combined: Array1<f64> = Array1::from_iter(x_extra.iter().chain(x_train.iter()).copied());
    let mut new_data = DataSet::new();
    new_data.insert_column("x", x_combined);

    let preds = model.predict(&new_data, &Gaussian::new()).unwrap();
    let mu_pred = &preds["mu"];

    // The last n_train entries of mu_pred correspond to x_train — they must match
    // the model's fitted values (which were computed with the stored training knots).
    let mu_fitted = &model.models["mu"].fitted_values;
    let offset = n_extra;
    for i in 0..n_train {
        let diff = (mu_pred[offset + i] - mu_fitted[i]).abs();
        assert!(
            diff < 1e-10,
            "In-sample prediction differs from fitted value at obs {}: \
             pred={}, fitted={}; knots may not be stable across predict calls",
            i,
            mu_pred[offset + i],
            mu_fitted[i]
        );
    }
}

// --- INFER-2: centiles / quantile prediction ---

#[test]
fn centiles_median_equals_fitted_mu_for_gaussian() {
    let mut rng = Generator::new(42);
    let (y, data) = rng.linear_gaussian(120, 1.0, 3.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let centiles = model.centiles(&data, &Gaussian::new(), &[50.0]).unwrap();
    let fitted = model.predict(&data, &Gaussian::new()).unwrap();
    // For a symmetric family the 50th centile is the fitted mean.
    let c50 = &centiles["C50"];
    let mu = &fitted["mu"];
    for (i, (&c, &m)) in c50.iter().zip(mu.iter()).enumerate() {
        assert!((c - m).abs() < 1e-6, "row {}: C50 {} vs mu {}", i, c, m);
    }
}

#[test]
fn centiles_are_strictly_increasing_in_level() {
    let mut rng = Generator::new(7);
    let (y, data) = rng.linear_gaussian(80, 1.0, 2.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let levels = [2.0, 10.0, 25.0, 50.0, 75.0, 90.0, 98.0];
    let centiles = model.centiles(&data, &Gaussian::new(), &levels).unwrap();
    // At every row the quantile is monotone increasing in the centile level.
    for w in levels.windows(2) {
        let lo_curve = &centiles[&format!("C{}", w[0])];
        let hi_curve = &centiles[&format!("C{}", w[1])];
        for (i, (&lo, &hi)) in lo_curve.iter().zip(hi_curve.iter()).enumerate() {
            assert!(
                hi > lo,
                "row {}: C{} {} should exceed C{} {}",
                i,
                w[1],
                hi,
                w[0],
                lo
            );
        }
    }
}

#[test]
fn centiles_have_nominal_coverage() {
    // Empirical fraction of y below C_α should be ≈ α (the residual property
    // checked from the centile side).
    let mut rng = Generator::new(2024);
    let (y, data) = rng.linear_gaussian(2000, 1.0, 3.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    let centiles = model
        .centiles(&data, &Gaussian::new(), &[10.0, 50.0, 90.0])
        .unwrap();
    for (pct, key) in [(0.10, "C10"), (0.50, "C50"), (0.90, "C90")] {
        let curve = &centiles[key];
        let below = (0..y.len()).filter(|&i| y[i] <= curve[i]).count() as f64 / y.len() as f64;
        assert!(
            (below - pct).abs() < 0.04,
            "{}: empirical {} vs nominal {}",
            key,
            below,
            pct
        );
    }
}

#[test]
fn quantile_prediction_matches_per_row_levels() {
    let mut rng = Generator::new(11);
    let (y, data) = rng.linear_gaussian(60, 1.0, 2.0, 1.0);
    let formula = linear_intercepts("x", &["mu", "sigma"]);
    let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();

    // A constant 0.5 level reproduces the 50th centile (= fitted mu).
    let p = Array1::from_elem(y.len(), 0.5);
    let q = model
        .quantile_prediction(&data, &Gaussian::new(), &p)
        .unwrap();
    let fitted = model.predict(&data, &Gaussian::new()).unwrap();
    for (&qi, &mui) in q.iter().zip(fitted["mu"].iter()) {
        assert!((qi - mui).abs() < 1e-6);
    }
}
