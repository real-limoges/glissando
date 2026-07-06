use glissando::distributions::Ocat;
use glissando::{DataSet, Formula, GamlssModel, Smooth, Term};
use ndarray::{Array1, Array2};

fn make_formula_intercept_only(n_params: usize) -> Formula {
    let mut f = Formula::new();
    f.add_terms("mu".to_string(), vec![Term::Intercept]);
    for k in 1..n_params {
        let param = match k {
            1 => "delta_1",
            2 => "delta_2",
            3 => "delta_3",
            _ => panic!("too many threshold params"),
        };
        f.add_terms(param.to_string(), vec![Term::Intercept]);
    }
    f
}

fn make_formula_with_smooth(n_params: usize) -> Formula {
    let mut f = Formula::new();
    let smooth = Term::Smooth(Smooth::PSpline1D {
        col_name: "x".to_string(),
        n_splines: 10,
        degree: 3,
        penalty_order: 2, range: None,
    });
    f.add_terms("mu".to_string(), vec![Term::Intercept, smooth]);
    for k in 1..n_params {
        let param = match k {
            1 => "delta_1",
            2 => "delta_2",
            3 => "delta_3",
            _ => panic!("too many threshold params"),
        };
        f.add_terms(param.to_string(), vec![Term::Intercept]);
    }
    f
}

fn synthetic_ocat_data(n: usize, seed: u64) -> (Array1<f64>, DataSet) {
    // Simple deterministic "LCG" to avoid pulling in rand for tests.
    let mut state = seed;
    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / (u32::MAX as f64)
    };

    let thresholds = [-1.0_f64, 0.0, 1.2];
    let x_vals: Array1<f64> = (0..n).map(|_| lcg(&mut state) * 4.0 - 2.0).collect();

    let y_vals: Array1<f64> = (0..n)
        .map(|i| {
            let eta = x_vals[i].sin();
            let cum: Vec<f64> = thresholds
                .iter()
                .map(|&t| 1.0 / (1.0 + (-(t - eta)).exp()))
                .collect();
            let p = [cum[0], cum[1] - cum[0], cum[2] - cum[1], 1.0 - cum[2]];
            let u = lcg(&mut state);
            let mut cat = 4usize;
            let mut acc = 0.0;
            for (k, &pk) in p.iter().enumerate() {
                acc += pk;
                if u < acc {
                    cat = k + 1;
                    break;
                }
            }
            cat as f64
        })
        .collect();

    let mut data = DataSet::new();
    data.insert_column("x", x_vals);
    (y_vals, data)
}

// ────────────────────────────────────────────────────────────────────────────
// Basic unit: class probabilities

#[test]
fn category_probs_sum_to_one_r4() {
    let thresholds = vec![-1.0, 0.0, 1.2];
    for eta in [-3.0, -1.0, 0.0, 1.0, 3.0] {
        let probs = Ocat::category_probs(eta, &thresholds);
        assert_eq!(probs.len(), 4);
        let s: f64 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "sum={s} eta={eta}");
        assert!(probs.iter().all(|&p| p > 0.0));
    }
}

#[test]
fn category_probs_monotone_in_eta() {
    let thresholds = vec![-1.0, 0.0, 1.2];
    // Larger eta → more weight in high categories.
    let low = Ocat::category_probs(-3.0, &thresholds);
    let high = Ocat::category_probs(3.0, &thresholds);
    assert!(low[0] > high[0], "P(y=1) should decrease with eta");
    assert!(low[3] < high[3], "P(y=4) should increase with eta");
}

// ────────────────────────────────────────────────────────────────────────────
// Fitting: intercept-only, R=4

#[test]
fn fit_intercept_only_r4_converges() {
    let (y, data) = synthetic_ocat_data(200, 42);
    let family = Ocat::new(4);
    let formula = make_formula_intercept_only(4);
    let model = GamlssModel::fit(&data, &y, &formula, &family).expect("fit failed");
    assert!(model.converged(), "intercept-only R=4 should converge");
}

#[test]
fn fit_intercept_only_r3_converges() {
    // Generate R=3 data (y in {1,2,3}).
    let n = 150;
    let mut state = 7u64;
    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / (u32::MAX as f64)
    };
    let thresholds = [-0.5_f64, 0.5];
    let y: Array1<f64> = (0..n)
        .map(|_| {
            let eta = 0.0_f64;
            let cum = [
                1.0 / (1.0 + (-(thresholds[0] - eta)).exp()),
                1.0 / (1.0 + (-(thresholds[1] - eta)).exp()),
            ];
            let p = [cum[0], cum[1] - cum[0], 1.0 - cum[1]];
            let u = lcg(&mut state);
            let mut cat = 3usize;
            let mut acc = 0.0;
            for (k, &pk) in p.iter().enumerate() {
                acc += pk;
                if u < acc {
                    cat = k + 1;
                    break;
                }
            }
            cat as f64
        })
        .collect();
    let mut data = DataSet::new();
    data.insert_column("x", Array1::zeros(n));
    let family = Ocat::new(3);
    let formula = make_formula_intercept_only(3);
    let model = GamlssModel::fit(&data, &y, &formula, &family).expect("fit failed");
    assert!(model.converged());
}

// ────────────────────────────────────────────────────────────────────────────
// Fitting: with smooth

#[test]
fn fit_with_smooth_r4_converges() {
    let (y, data) = synthetic_ocat_data(300, 123);
    let family = Ocat::new(4);
    let formula = make_formula_with_smooth(4);
    let model = GamlssModel::fit(&data, &y, &formula, &family).expect("fit failed");
    assert!(model.converged(), "smooth R=4 should converge");
}

// ────────────────────────────────────────────────────────────────────────────
// predict_class_probabilities

#[test]
fn predict_class_probabilities_shape_and_sums() {
    let (y, data) = synthetic_ocat_data(200, 99);
    let family = Ocat::new(4);
    let formula = make_formula_with_smooth(4);
    let model = GamlssModel::fit(&data, &y, &formula, &family).expect("fit failed");

    let mut new_data = DataSet::new();
    new_data.insert_column("x", Array1::linspace(-2.0, 2.0, 50));

    let probs: Array2<f64> = model
        .predict_class_probabilities(&new_data, &family)
        .expect("predict_class_probabilities failed");

    assert_eq!(probs.nrows(), 50);
    assert_eq!(probs.ncols(), 4);

    for i in 0..50 {
        let row_sum: f64 = probs.row(i).iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-10, "row {i} sum={row_sum}");
        for j in 0..4 {
            assert!(probs[(i, j)] > 0.0, "row {i} col {j} is zero");
        }
    }
}

#[test]
fn predict_class_probabilities_is_consistent_with_predict() {
    let (y, data) = synthetic_ocat_data(200, 77);
    let family = Ocat::new(4);
    let formula = make_formula_intercept_only(4);
    let model = GamlssModel::fit(&data, &y, &formula, &family).expect("fit failed");

    let mut new_data = DataSet::new();
    new_data.insert_column("x", Array1::zeros(10));

    // predict() gives mu (η) and delta_k; predict_class_probabilities uses the same.
    let param_preds = model.predict(&new_data, &family).expect("predict failed");
    let probs = model
        .predict_class_probabilities(&new_data, &family)
        .expect("predict_class_probabilities failed");

    let eta_mu = &param_preds["mu"];
    // Manually reconstruct thresholds for obs 0 and check against probs.
    let d1 = param_preds["delta_1"][0];
    let d2 = param_preds["delta_2"][0]; // response-scale increment
    let d3 = param_preds["delta_3"][0]; // response-scale increment
    let t = [d1, d1 + d2, d1 + d2 + d3];
    let manual = Ocat::category_probs(eta_mu[0], &t);
    for j in 0..4 {
        assert!(
            (probs[(0, j)] - manual[j]).abs() < 1e-12,
            "col {j}: probs[0,{j}]={} manual={}",
            probs[(0, j)],
            manual[j]
        );
    }
}
