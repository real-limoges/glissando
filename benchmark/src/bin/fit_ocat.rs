//! Glissando Ocat benchmark binary.
//!
//! Fits `Ocat(R=4)` on the spike train data, computes training log-likelihood,
//! and predicts the (n_test × 4) category-probability matrix on the test set.
//!
//! Usage:
//!   cargo run -p glissando_benchmark --bin fit_ocat --release -- \
//!     --train path/to/ocat_train.parquet  \
//!     --test  path/to/ocat_test.parquet   \
//!     --output path/to/glissando_ocat.json \
//!     [--intercept-only]
//!
//! Without `--intercept-only` the mu parameter is modelled with two P-splines
//! (matching the mgcv formula `y ~ s(x1,bs="ps") + s(x2,bs="ps")`).
//! With `--intercept-only` all parameters have intercept-only formulas — this
//! mode is used for the log-likelihood cross-check against mgcv.

use glissando::distributions::Ocat;
use glissando::{DataSet, Formula, GamlssModel, Smooth, Term};
use ndarray::Array1;
use polars::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Serialize)]
struct OcatResult {
    /// (n_test × 4) category probabilities; each row sums to 1.
    probs: Vec<Vec<f64>>,
    /// Unpenalised training log-likelihood Σ log P(y_i = r_i | model).
    loglik_train: f64,
    n_train: usize,
    n_test: usize,
    converged: bool,
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

fn make_formula(intercept_only: bool) -> Formula {
    let mut formula = Formula::new();
    if intercept_only {
        formula.add_terms("mu".to_string(), vec![Term::Intercept]);
    } else {
        let mk_smooth = |col: &str| {
            Term::Smooth(Smooth::PSpline1D {
                col_name: col.to_string(),
                n_splines: 20,
                degree: 3,
                penalty_order: 2, range: None,
            })
        };
        formula.add_terms(
            "mu".to_string(),
            vec![Term::Intercept, mk_smooth("x1"), mk_smooth("x2")],
        );
    }
    for k in 1..=3 {
        let name = match k {
            1 => "delta_1",
            2 => "delta_2",
            3 => "delta_3",
            _ => unreachable!(),
        };
        formula.add_terms(name.to_string(), vec![Term::Intercept]);
    }
    formula
}

fn write_result(path: &PathBuf, result: &OcatResult) {
    let file = File::create(path).expect("failed to create output file");
    serde_json::to_writer_pretty(BufWriter::new(file), result).expect("failed to write JSON");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut train_path: Option<PathBuf> = None;
    let mut test_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut intercept_only = false;

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
            "--intercept-only" => {
                intercept_only = true;
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
    let n_train = train_df.height();
    let n_test = test_df.height();

    let y_train = extract_column(&train_df, "y");

    let mut train_data = DataSet::new();
    let mut test_data = DataSet::new();

    if !intercept_only {
        train_data.insert_column("x1", extract_column(&train_df, "x1"));
        train_data.insert_column("x2", extract_column(&train_df, "x2"));
        test_data.insert_column("x1", extract_column(&test_df, "x1"));
        test_data.insert_column("x2", extract_column(&test_df, "x2"));
    } else {
        // Intercept-only: we still need a column to satisfy DataSet requirements.
        // Insert a dummy constant column of the right length.
        train_data.insert_column("_dummy", Array1::zeros(n_train));
        test_data.insert_column("_dummy", Array1::zeros(n_test));
    }

    let family = Ocat::new(4);
    let formula = make_formula(intercept_only);
    let start = Instant::now();

    let model = match GamlssModel::fit(&train_data, &y_train, &formula, &family) {
        Ok(m) => m,
        Err(e) => {
            let result = OcatResult {
                probs: vec![],
                loglik_train: f64::NAN,
                n_train,
                n_test,
                converged: false,
                fit_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                error: Some(e.to_string()),
            };
            write_result(&output_path, &result);
            return;
        }
    };

    let fit_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    let converged = model.converged();

    // Training log-likelihood.
    let loglik_train = model
        .diagnostics(&family, &y_train)
        .map(|d| d.log_likelihood)
        .unwrap_or(f64::NAN);

    // Predict probabilities on the test set.
    let prob_matrix = match model.predict_class_probabilities(&test_data, &family) {
        Ok(m) => m,
        Err(e) => {
            let result = OcatResult {
                probs: vec![],
                loglik_train,
                n_train,
                n_test,
                converged,
                fit_time_ms,
                error: Some(format!("predict failed: {e}")),
            };
            write_result(&output_path, &result);
            return;
        }
    };

    let probs: Vec<Vec<f64>> = (0..n_test)
        .map(|i| (0..4).map(|j| prob_matrix[(i, j)]).collect())
        .collect();

    let result = OcatResult {
        probs,
        loglik_train,
        n_train,
        n_test,
        converged,
        fit_time_ms,
        error: None,
    };
    write_result(&output_path, &result);

    let mode = if intercept_only {
        "intercept-only"
    } else {
        "smooth"
    };
    eprintln!(
        "fit_ocat ({mode}): n_train={n_train} n_test={n_test} \
         loglik={loglik_train:.4} converged={converged} time={fit_time_ms:.0}ms"
    );
}
