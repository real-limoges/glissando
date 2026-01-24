use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use gamlss_rs::distributions::{
    Beta, Distribution, Gamma, Gaussian, NegativeBinomial, Poisson, StudentT,
};
use ndarray::Array1;
use statrs::function::gamma::digamma as statrs_digamma;
use std::collections::HashMap;

fn generate_test_data(n: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    // Generate realistic parameter values
    let y: Array1<f64> = (0..n).map(|i| (i as f64 % 10.0) + 1.0).collect();
    let mu: Array1<f64> = Array1::from_elem(n, 5.0);
    let sigma: Array1<f64> = Array1::from_elem(n, 1.0);
    let nu: Array1<f64> = Array1::from_elem(n, 10.0);
    (y, mu, sigma, nu)
}

fn bench_poisson_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_derivatives");

    for n in [100, 1_000, 10_000].iter() {
        let (y, mu, _, _) = generate_test_data(*n);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let params: HashMap<&str, &Array1<f64>> = HashMap::from([("mu", &mu)]);
                Poisson.derivatives(black_box(&y), black_box(&params))
            })
        });
    }
    group.finish();
}

fn bench_gaussian_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_derivatives");

    for n in [100, 1_000, 10_000].iter() {
        let (y, mu, sigma, _) = generate_test_data(*n);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let params: HashMap<&str, &Array1<f64>> =
                    HashMap::from([("mu", &mu), ("sigma", &sigma)]);
                Gaussian.derivatives(black_box(&y), black_box(&params))
            })
        });
    }
    group.finish();
}

fn bench_student_t_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("student_t_derivatives");

    for n in [100, 1_000, 10_000].iter() {
        let (y, mu, sigma, nu) = generate_test_data(*n);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let params: HashMap<&str, &Array1<f64>> =
                    HashMap::from([("mu", &mu), ("sigma", &sigma), ("nu", &nu)]);
                StudentT.derivatives(black_box(&y), black_box(&params))
            })
        });
    }
    group.finish();
}

fn bench_gamma_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("gamma_derivatives");

    for n in [100, 1_000, 10_000].iter() {
        let (y, mu, sigma, _) = generate_test_data(*n);
        // Gamma needs positive y values
        let y_positive = y.mapv(|v| v.max(0.1));

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let params: HashMap<&str, &Array1<f64>> =
                    HashMap::from([("mu", &mu), ("sigma", &sigma)]);
                Gamma.derivatives(black_box(&y_positive), black_box(&params))
            })
        });
    }
    group.finish();
}

fn bench_negative_binomial_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("negative_binomial_derivatives");

    for n in [100, 1_000, 10_000].iter() {
        let (y, mu, _, _) = generate_test_data(*n);
        let sigma: Array1<f64> = Array1::from_elem(*n, 0.5);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let params: HashMap<&str, &Array1<f64>> =
                    HashMap::from([("mu", &mu), ("sigma", &sigma)]);
                NegativeBinomial.derivatives(black_box(&y), black_box(&params))
            })
        });
    }
    group.finish();
}

fn bench_beta_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("beta_derivatives");

    for n in [100, 1_000, 10_000].iter() {
        // Beta needs y in (0, 1)
        let y: Array1<f64> = (0..*n)
            .map(|i| 0.1 + 0.8 * (i as f64 / *n as f64))
            .collect();
        let mu: Array1<f64> = Array1::from_elem(*n, 0.5);
        let phi: Array1<f64> = Array1::from_elem(*n, 10.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let params: HashMap<&str, &Array1<f64>> =
                    HashMap::from([("mu", &mu), ("phi", &phi)]);
                Beta.derivatives(black_box(&y), black_box(&params))
            })
        });
    }
    group.finish();
}

fn bench_special_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("special_functions");

    for n in [100, 1_000, 10_000].iter() {
        // Typical values for nu (degrees of freedom) in StudentT
        let x: Array1<f64> = (0..*n).map(|i| 1.0 + (i as f64 % 50.0)).collect();

        // Benchmark scalar digamma in a loop (old approach)
        group.bench_with_input(BenchmarkId::new("digamma_scalar_loop", n), n, |b, _| {
            b.iter(|| {
                let result: Vec<f64> = x.iter().map(|&xi| statrs_digamma(xi)).collect();
                black_box(result)
            })
        });

        // Benchmark batch digamma (new approach using mapv)
        group.bench_with_input(BenchmarkId::new("digamma_batch", n), n, |b, _| {
            b.iter(|| {
                let result = x.mapv(statrs_digamma);
                black_box(result)
            })
        });
    }
    group.finish();
}

fn bench_full_model_fit(c: &mut Criterion) {
    use gamlss_rs::{GamlssModel, Term};
    use polars::prelude::*;

    let mut group = c.benchmark_group("full_model_fit");
    group.sample_size(20); // Fewer samples for slower benchmarks

    for n in [100, 500].iter() {
        // Generate synthetic data
        let x: Vec<f64> = (0..*n).map(|i| i as f64 / *n as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 5.0 + 2.0 * xi + 0.1 * xi * xi).collect();

        let df = df! {
            "x" => &x,
            "y" => &y,
        }
        .unwrap();

        // Poisson with linear predictor
        group.bench_with_input(BenchmarkId::new("poisson_linear", n), &df, |b, df| {
            b.iter(|| {
                let formula = std::collections::HashMap::from([(
                    "mu".to_string(),
                    vec![
                        Term::Intercept,
                        Term::Linear {
                            col_name: "x".to_string(),
                        },
                    ],
                )]);
                black_box(GamlssModel::fit(df, "y", &formula, &Poisson).unwrap())
            })
        });

        // Gaussian with two parameters
        group.bench_with_input(BenchmarkId::new("gaussian_linear", n), &df, |b, df| {
            b.iter(|| {
                let formula = std::collections::HashMap::from([
                    (
                        "mu".to_string(),
                        vec![
                            Term::Intercept,
                            Term::Linear {
                                col_name: "x".to_string(),
                            },
                        ],
                    ),
                    ("sigma".to_string(), vec![Term::Intercept]),
                ]);
                black_box(GamlssModel::fit(df, "y", &formula, &Gaussian).unwrap())
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_poisson_derivatives,
    bench_gaussian_derivatives,
    bench_student_t_derivatives,
    bench_gamma_derivatives,
    bench_negative_binomial_derivatives,
    bench_beta_derivatives,
    bench_special_functions,
    bench_full_model_fit,
);
criterion_main!(benches);
