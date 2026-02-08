# gamlss_rs

A Rust implementation of Generalized Additive Models for Location, Scale, and Shape (GAMLSS).

GAMLSS extends traditional regression by modeling not just the mean, but also variance and other distribution parameters as functions of predictors. This enables flexible modeling of heteroskedastic data, heavy-tailed distributions, and other complex data structures.

## Features

- **Multiple distribution parameters**: Model mean, variance, and shape parameters simultaneously
- **Flexible terms**: Intercept, linear effects, P-splines, tensor products, and random effects
- **Automatic smoothing**: Smoothing parameters selected via GCV optimization
- **Dual backends**: OpenBLAS (default, max performance) or pure Rust via faer (no system deps)
- **WASM support**: Fit models and predict directly in the browser via wasm-bindgen
- **Type-safe API**: `DataSet`, `Formula`, and newtype wrappers prevent misuse

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
gamlss_rs = { git = "https://github.com/real-limoges/gamlss_rs" }
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `openblas` | OpenBLAS backend (ndarray-linalg) — max performance | yes |
| `pure-rust` | Faer backend — no system dependencies, WASM-compatible | no |
| `serialization` | Serde support for model serialization | no |
| `wasm` | WASM fitting + prediction API (implies `pure-rust` + `serialization`) | no |
| `parallel` | Rayon parallelism for large datasets | yes |

### Requirements

- Rust 2021 edition
- OpenBLAS (only with default `openblas` feature)

On macOS:
```bash
brew install openblas
```

On Ubuntu/Debian:
```bash
sudo apt-get install libopenblas-dev
```

For pure Rust or WASM builds, no system dependencies are needed.

## Quick Start

```rust
use gamlss_rs::{GamlssModel, DataSet, Formula, Term};
use gamlss_rs::distributions::Gaussian;
use ndarray::Array1;

let y = Array1::from_vec(vec![2.1, 4.0, 5.9, 8.1, 10.0]);

let mut data = DataSet::new();
data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new()).unwrap();

println!("Converged: {}", model.converged());
let mu_coeffs = &model.models["mu"].coefficients;
println!("Intercept: {}, Slope: {}", mu_coeffs[0], mu_coeffs[1]);
```

## Distributions

| Distribution | Parameters | Default Links | Use Case |
|--------------|------------|---------------|----------|
| `Poisson` | mu | log | Count data |
| `Binomial` | mu | logit | Binary/count with known trials |
| `Gaussian` | mu, sigma | identity, log | Continuous data |
| `StudentT` | mu, sigma, nu | identity, log, log | Heavy-tailed continuous |
| `Gamma` | mu, sigma | log, log | Positive continuous |
| `NegativeBinomial` | mu, sigma | log, log | Overdispersed counts |
| `Beta` | mu, phi | logit, log | Proportions (0, 1) |

### Usage

```rust
use gamlss_rs::distributions::{Poisson, Binomial, Gaussian, StudentT, Gamma, NegativeBinomial, Beta};

let poisson = Poisson::new();             // Count data
let binomial = Binomial::new(10);         // Binary/count with 10 trials
let gaussian = Gaussian::new();           // Continuous data
let student_t = StudentT::new();          // Heavy-tailed continuous data
let gamma = Gamma::new();                 // Positive continuous (e.g., durations)
let neg_bin = NegativeBinomial::new();    // Overdispersed counts
let beta = Beta::new();                   // Proportions/rates in (0, 1)
```

## Term Types

### Intercept

A constant term (bias).

```rust
Term::Intercept
```

### Linear

A linear effect for a single predictor.

```rust
Term::Linear { col_name: "x".to_string() }
```

### P-Spline (1D Smooth)

A penalized B-spline smooth for nonlinear effects.

```rust
Term::Smooth(Smooth::PSpline1D {
    col_name: "x".to_string(),
    n_splines: 10,      // Number of basis functions
    degree: 3,          // Spline degree (3 = cubic)
    penalty_order: 2,   // Penalty on 2nd derivatives
})
```

### Tensor Product (2D Smooth)

Interaction smooth for two predictors.

```rust
Term::Smooth(Smooth::TensorProduct {
    col_name_1: "x1".to_string(),
    n_splines_1: 5,
    penalty_order_1: 2,
    col_name_2: "x2".to_string(),
    n_splines_2: 5,
    penalty_order_2: 2,
    degree: 3,
})
```

### Random Effect

Group-level random intercepts.

```rust
Term::Smooth(Smooth::RandomEffect {
    col_name: "group".to_string(),
})
```

## Configuration

```rust
use gamlss_rs::FitConfig;

let config = FitConfig {
    max_iterations: 100,
    tolerance: 1e-4,
};

let model = GamlssModel::fit_with_config(
    &y, &data, &formula, &Gaussian::new(), config
)?;
```

## Accessing Results

```rust
let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new())?;

// Convergence diagnostics
println!("Converged: {}", model.diagnostics.converged);
println!("Iterations: {}", model.diagnostics.iterations);

// Per-parameter results
let fitted_mu = &model.models["mu"];
fitted_mu.coefficients     // Coefficients newtype (Deref to Array1<f64>)
fitted_mu.covariance       // CovarianceMatrix newtype (Deref to Array2<f64>)
fitted_mu.fitted_values    // Fitted values on response scale
fitted_mu.eta              // Linear predictor (X * beta)
fitted_mu.edf              // Effective degrees of freedom
fitted_mu.lambdas          // Smoothing parameters
fitted_mu.terms            // Formula terms
```

## Prediction

```rust
let mut new_data = DataSet::new();
new_data.insert_column("x", Array1::from_vec(vec![1.5, 2.5, 3.5]));

let family = Gaussian::new();

// Point predictions (fitted values on response scale)
let predictions = model.predict(&new_data, &family)?;
let mu_pred = &predictions["mu"];

// Predictions with standard errors
let results = model.predict_with_se(&new_data, &family)?;
let mu_result = &results["mu"];
println!("Fitted: {:?}", mu_result.fitted);
println!("SE(eta): {:?}", mu_result.se_eta);

// Posterior samples for uncertainty quantification
let samples = model.predict_samples(&new_data, &family, 1000)?;
let mu_samples = &samples["mu"];  // Vec<Array1<f64>> with 1000 samples
```

## Model Diagnostics

```rust
use gamlss_rs::diagnostics::{
    pearson_residuals_gaussian, response_residuals,
    loglik_gaussian, compute_aic, compute_bic, total_edf,
};

let mu = &model.models["mu"].fitted_values;
let sigma = &model.models["sigma"].fitted_values;

let pearson_resid = pearson_residuals_gaussian(&y, mu, sigma);
let ll = loglik_gaussian(&y, mu, sigma);
let edf = total_edf(&model.models);
let aic = compute_aic(ll, edf);
let bic = compute_bic(ll, edf, y.len());
```

Distribution-specific residual and log-likelihood functions are available for all supported distributions (e.g., `pearson_residuals_poisson`, `loglik_gamma`, etc.).

## Examples

### Heteroskedastic Regression

Model where both mean and variance depend on x:

```rust
let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ])
    .with_terms("sigma", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ]);

let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new())?;
```

### Nonlinear Smooth

```rust
let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Smooth(Smooth::PSpline1D {
            col_name: "x".to_string(),
            n_splines: 15,
            degree: 3,
            penalty_order: 2,
        }),
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new())?;
```

### Count Data with Poisson

```rust
use gamlss_rs::distributions::Poisson;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "predictor".to_string() },
    ]);

let model = GamlssModel::fit(&counts, &data, &formula, &Poisson::new())?;
```

### Binary/Binomial Data

```rust
use gamlss_rs::distributions::Binomial;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ]);

// Fixed number of trials
let model = GamlssModel::fit(&successes, &data, &formula, &Binomial::new(20))?;

// Or varying trials per observation
let trials = Array1::from_vec(vec![10.0, 15.0, 20.0, 25.0]);
let model = GamlssModel::fit(&successes, &data, &formula, &Binomial::with_trials(trials))?;
```

### Heavy-Tailed Data with Student-t

```rust
use gamlss_rs::distributions::StudentT;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ])
    .with_terms("sigma", vec![Term::Intercept])
    .with_terms("nu", vec![Term::Intercept]);

let model = GamlssModel::fit(&y, &data, &formula, &StudentT::new())?;
```

### Mixed Effects Model

```rust
let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
        Term::Smooth(Smooth::RandomEffect {
            col_name: "subject_id".to_string(),
        }),
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&y, &data, &formula, &Gaussian::new())?;
```

### Overdispersed Count Data

```rust
use gamlss_rs::distributions::NegativeBinomial;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&counts, &data, &formula, &NegativeBinomial::new())?;
```

### Proportion/Rate Data

```rust
use gamlss_rs::distributions::Beta;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "x".to_string() },
    ])
    .with_terms("phi", vec![Term::Intercept]);

let model = GamlssModel::fit(&proportions, &data, &formula, &Beta::new())?;
```

### Duration/Positive Continuous Data

```rust
use gamlss_rs::distributions::Gamma;

let formula = Formula::new()
    .with_terms("mu", vec![
        Term::Intercept,
        Term::Linear { col_name: "age".to_string() },
    ])
    .with_terms("sigma", vec![Term::Intercept]);

let model = GamlssModel::fit(&durations, &data, &formula, &Gamma::new())?;
```

## Error Handling

The library uses `GamlssError` for error handling:

```rust
use gamlss_rs::GamlssError;

match GamlssModel::fit(&y, &data, &formula, &Gaussian::new()) {
    Ok(model) => {
        // Use the fitted model
    }
    Err(GamlssError::Input(msg)) => {
        eprintln!("Input error: {}", msg);
    }
    Err(GamlssError::Convergence(iters)) => {
        eprintln!("Failed to converge after {} iterations", iters);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `Input` | Invalid input data or formula |
| `MissingVariable` | Required variable not found in data |
| `MissingFormula` | Formula missing for a distribution parameter |
| `NonFiniteValues` | Variable contains NaN or Inf values (includes count) |
| `EmptyData` | No observations provided |
| `Convergence` | Algorithm failed to converge after N iterations |
| `Optimization` | Smoothing parameter optimization failed |
| `Linalg` | Linear algebra computation failed |
| `UnknownParameter` | Unknown parameter for the given distribution |
| `Shape` | Array shape mismatch |
| `Internal` | Internal computation error |

## Serialization & WASM

Models can be serialized to JSON for transfer to browsers or other systems. Enable with the `serialization` feature:

```toml
[dependencies]
gamlss_rs = { git = "...", features = ["serialization"] }
```

```rust
// Serialize a fitted model (native side)
let json = model.to_json(&Gaussian::new())?;

// Deserialize (returns model + distribution name)
let (model, dist_name) = GamlssModel::from_json(&json)?;
```

For browser-based fitting and prediction, build with the `wasm` feature:

```bash
wasm-pack build --no-default-features --features wasm --target web
```

### Fitting in the Browser

```js
import { WasmGamlssModel } from './pkg/gamlss_rs.js';

const y = JSON.stringify([2.1, 4.0, 5.9, 8.1, 10.0]);
const data = JSON.stringify({ x: [1.0, 2.0, 3.0, 4.0, 5.0] });
const formula = JSON.stringify({
  mu: [{ Intercept: null }, { Linear: { col_name: "x" } }],
  sigma: [{ Intercept: null }],
});

const model = WasmGamlssModel.fit(y, data, formula, "Gaussian");
console.log("Converged:", model.converged());

// With custom configuration
const config = JSON.stringify({ max_iterations: 200, tolerance: 0.001 });
const model2 = WasmGamlssModel.fitWithConfig(y, data, formula, "Gaussian", config);
```

Supported distributions: `Gaussian`, `Poisson`, `StudentT`, `Gamma`, `NegativeBinomial`, `Beta`.

### Loading Pre-fitted Models

```js
const model = WasmGamlssModel.fromJson(modelJson);
```

### Prediction

```js
const predictions = JSON.parse(model.predict('{"x": [1, 2, 3]}'));

// With standard errors
const results = JSON.parse(model.predictWithSe('{"x": [1, 2, 3]}'));

// Access fitted values and coefficients directly
const mu_fitted = model.fittedValues("mu");
const mu_coeffs = model.coefficients("mu");

// Diagnostics
const diagnostics = JSON.parse(model.diagnosticsJson());
```

## Dependencies

- [ndarray](https://crates.io/crates/ndarray) - N-dimensional arrays
- [ndarray-linalg](https://crates.io/crates/ndarray-linalg) - Linear algebra via OpenBLAS (optional: `openblas` feature)
- [faer](https://crates.io/crates/faer) - Pure Rust linear algebra (optional: `pure-rust` feature)
- [argmin](https://crates.io/crates/argmin) - L-BFGS optimization
- [statrs](https://crates.io/crates/statrs) - Statistical functions
- [rayon](https://crates.io/crates/rayon) - Parallel computation (optional: `parallel` feature)
- [serde](https://crates.io/crates/serde) / [serde_json](https://crates.io/crates/serde_json) - Serialization (optional: `serialization` feature)
- [wasm-bindgen](https://crates.io/crates/wasm-bindgen) - JavaScript interop (optional: `wasm` feature)

## Project Structure

This repository is a Cargo workspace:

- **`gamlss_rs`** (root) — the core library
- **`benchmark/`** (`gamlss_benchmark`) — comparison framework against R/mgcv

## Algorithm

GAMLSS fitting uses a penalized quasi-likelihood approach (Rigby-Stasinopoulos algorithm):

1. **Initialization**: Set starting values for all distribution parameters
2. **Outer loop**: Cycle through distribution parameters
3. **Inner loop**: For each parameter, compute working response and weights from derivatives, then fit a penalized weighted least squares model
4. **Smoothing selection**: Optimize smoothing parameters via GCV using L-BFGS
5. **Convergence**: Check if coefficient changes are below tolerance

## Performance

The library includes several optimizations for large datasets:

- **Batched derivatives**: Distribution derivatives are computed for all observations at once, enabling SIMD vectorization
- **Parallel computation**: Special functions (digamma, trigamma) use Rayon parallel iterators for n >= 10,000
- **Warm-starting**: L-BFGS optimization reuses previous smoothing parameters for faster convergence
- **Efficient matrix operations**: Uses sqrt-weighted approach to avoid O(n²) memory allocation

## Benchmark (Comparison with R)

The `benchmark/` directory contains a comparison framework that validates gamlss_rs against R's mgcv and gamlss packages across 15 scenarios (linear, smooth, heteroskedastic) and all supported distributions.

### Quick Start

```bash
# Build the Rust comparison binary
cargo build -p gamlss_benchmark --release

# Run a single scenario
cargo run -p gamlss_benchmark --release --bin compare_fit -- \
    --data benchmark/output/data_gaussian_linear.parquet \
    --scenario gaussian_linear \
    --output result.json

# Run the full comparison suite (requires Python with numpy/polars and R with mgcv)
./benchmark/run_comparison.sh
```

The orchestrator (`benchmark/orchestrate.py`) generates synthetic data with known parameters, fits models in both Rust and R, and produces a comparison summary with fitted value correlations, coefficient recovery, and timing.

## License

MIT
