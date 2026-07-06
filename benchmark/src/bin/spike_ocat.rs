//! Phase 0 ocat spike — Candidate A: three independent Binomial(1)/logit models.
//!
//! Fits three cumulative threshold models on train data:
//!   model k: y_k = (y <= k), Binomial(n_trials=1), logit link, formula mu ~ s(x1)+s(x2)
//!
//! Reconstructs the (n × 4) category-probability matrix on the held-out test set:
//!   P(y=1) = μ₁
//!   P(y=2) = μ₂ - μ₁   (clipped to [0, 1] before renormalising)
//!   P(y=3) = μ₃ - μ₂
//!   P(y=4) = 1 - μ₃
//!
//! Also counts "monotonicity violations": rows where the three independent
//! cumulative fits are not monotone increasing (μ₁ ≤ μ₂ ≤ μ₃ is NOT guaranteed
//! because each model is fit independently).
//!
//! Usage:
//!   cargo run -p glissando_benchmark --bin spike_ocat -- \
//!     --train path/to/ocat_train.parquet \
//!     --test  path/to/ocat_test.parquet  \
//!     --output path/to/rust_ocat.json

use glissando::distributions::Binomial;
use glissando::{DataSet, Formula, GamlssModel, Smooth, Term};
use ndarray::Array1;
use polars::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Serialize)]
struct SpikeResult {
    /// (n_test × 4) category probabilities; each row sums to 1.
    probs: Vec<Vec<f64>>,
    /// Rows where the three independent cumulative fits are not monotone
    /// (i.e., μ₁ > μ₂ or μ₂ > μ₃), producing negative raw probabilities
    /// before the clip-and-renormalise step.
    n_violations: usize,
    n_obs: usize,
    models_converged: [bool; 3],
    fit_time_ms: f64,
    error: Option<String>,
}

fn read_parquet(path: &PathBuf) -> DataFrame {
    LazyFrame::scan_parquet(path, Default::default())
        .expect("failed to open parquet")
        .collect()
        .expect("failed to read parquet")
}

fn extract_column(df: &DataFrame, name: &str) -> Array1<f64> {
    let col = df
        .column(name)
        .unwrap_or_else(|_| panic!("column '{}' not found", name));
    let cast = col
        .cast(&DataType::Float64)
        .unwrap_or_else(|_| panic!("column '{}' cannot be cast to f64", name));
    let ca = cast
        .as_materialized_series()
        .f64()
        .unwrap_or_else(|_| panic!("column '{}' cannot be read as f64", name));
    Array1::from_vec(ca.into_no_null_iter().collect())
}

/// P-spline formula mu ~ intercept + s(x1) + s(x2).
/// The explicit intercept matches mgcv's implicit global mean term and is
/// required for correct conditioning of the penalised WLS system — without
/// it the constant component of the two smooths goes through the origin and
/// the 20×20 X'WX block can be singular at the last diagonal entry.
fn make_formula() -> Formula {
    let mk_smooth = |col: &str| {
        Term::Smooth(Smooth::PSpline1D {
            col_name: col.to_string(),
            n_splines: 20,
            degree: 3,
            penalty_order: 2, range: None,
        })
    };
    Formula::new().with_terms(
        "mu",
        vec![Term::Intercept, mk_smooth("x1"), mk_smooth("x2")],
    )
}

fn write_result(path: &PathBuf, result: &SpikeResult) {
    let file = File::create(path).expect("failed to create output file");
    serde_json::to_writer_pretty(BufWriter::new(file), result).expect("failed to write JSON");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut train_path: Option<PathBuf> = None;
    let mut test_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--train" => {
                i += 1;
                train_path = Some(PathBuf::from(&args[i]));
            }
            "--test" => {
                i += 1;
                test_path = Some(PathBuf::from(&args[i]));
            }
            "--output" => {
                i += 1;
                output_path = Some(PathBuf::from(&args[i]));
            }
            _ => {}
        }
        i += 1;
    }

    let train_path = train_path.expect("--train required");
    let test_path = test_path.expect("--test required");
    let output_path = output_path.expect("--output required");

    let train_df = read_parquet(&train_path);
    let test_df = read_parquet(&test_path);

    let start = Instant::now();

    let mut train_data = DataSet::new();
    train_data.insert_column("x1", extract_column(&train_df, "x1"));
    train_data.insert_column("x2", extract_column(&train_df, "x2"));

    let mut test_data = DataSet::new();
    test_data.insert_column("x1", extract_column(&test_df, "x1"));
    test_data.insert_column("x2", extract_column(&test_df, "x2"));

    let n_test = test_df.height();
    let formula = make_formula();
    let family = Binomial::new(1);

    let threshold_cols = ["le1", "le2", "le3"];
    let mut cum_preds: Vec<Array1<f64>> = Vec::with_capacity(3);
    let mut converged = [false; 3];

    for (k, col) in threshold_cols.iter().enumerate() {
        let y_k = extract_column(&train_df, col);
        match GamlssModel::fit(&train_data, &y_k, &formula, &family) {
            Ok(model) => {
                converged[k] = model.converged();
                let preds = model.predict(&test_data, &family).expect("predict failed");
                cum_preds.push(preds["mu"].clone());
            }
            Err(e) => {
                let result = SpikeResult {
                    probs: vec![],
                    n_violations: 0,
                    n_obs: 0,
                    models_converged: converged,
                    fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                    error: Some(format!("model {} ({}): {}", k + 1, col, e)),
                };
                write_result(&output_path, &result);
                return;
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let mut probs: Vec<Vec<f64>> = Vec::with_capacity(n_test);
    let mut n_violations = 0usize;

    for ((&c1, &c2), &c3) in cum_preds[0]
        .iter()
        .take(n_test)
        .zip(cum_preds[1].iter())
        .zip(cum_preds[2].iter())
    {
        // Monotonicity violation: the three independent fits need not be ordered.
        if c1 > c2 || c2 > c3 {
            n_violations += 1;
        }

        // Clip each category prob to [0,1] and renormalise to a valid distribution.
        let p1 = c1.clamp(0.0, 1.0);
        let p2 = (c2 - c1).clamp(0.0, 1.0);
        let p3 = (c3 - c2).clamp(0.0, 1.0);
        let p4 = (1.0 - c3).clamp(0.0, 1.0);
        let total = p1 + p2 + p3 + p4;
        probs.push(if total > 0.0 {
            vec![p1 / total, p2 / total, p3 / total, p4 / total]
        } else {
            vec![0.25, 0.25, 0.25, 0.25]
        });
    }

    let result = SpikeResult {
        probs,
        n_violations,
        n_obs: n_test,
        models_converged: converged,
        fit_time_ms: elapsed,
        error: None,
    };
    write_result(&output_path, &result);

    eprintln!(
        "ocat spike: n_test={}, violations={}/{} ({:.1}%), time={:.0}ms, converged={:?}",
        n_test,
        n_violations,
        n_test,
        100.0 * n_violations as f64 / n_test as f64,
        elapsed,
        converged,
    );
}
