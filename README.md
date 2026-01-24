# gamlss_rs

A Rust implementation of Generalized Additive Models for Location, Scale, and Shape (GAMLSS).

GAMLSS extends traditional regression by modeling not just the mean, but also variance and other distribution parameters as functions of predictors. This enables flexible modeling of heteroskedastic data, heavy-tailed distributions, and other complex data structures.

## Features

- **Multiple distribution parameters**: Model mean, variance, and shape parameters simultaneously
- **Flexible terms**: Intercept, linear effects, P-splines, tensor products, and random effects
- **Automatic smoothing**: Smoothing parameters selected via optimization
- **Built on Polars**: Efficient data handling with Polars DataFrames
- **Type-safe API**: Rust's type system ensures correct usage

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
gamlss_rs = { git = "https://github.com/your-repo/gamlss_rs" }
```

### Requirements

- Rust 2024 edition
- OpenBLAS (for linear algebra operations)

On macOS:
```bash
brew install openblas
```

On Ubuntu/Debian:
```bash
sudo apt-get install libopenblas-dev
```

## Quick Start

```rust
use gamlss_rs::{GamlssModel, Term, Smooth};
use gamlss_rs::distributions::Gaussian;
use polars::prelude::*;
use std::collections::HashMap;

// Load your data into a Polars DataFrame
let df = df!(
    "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
    "y" => [2.1, 4.0, 5.9, 8.1, 10.0]
).unwrap();

// Define formulas for each distribution parameter
let mut formula = HashMap::new();
formula.insert("mu".to_string(), vec![
    Term::Intercept,
    Term::Linear { col_name: "x".to_string() },
]);
formula.insert("sigma".to_string(), vec![
    Term::Intercept,
]);

// Fit the model
let model = GamlssModel::fit(&df, "y", &formula, &Gaussian::new()).unwrap();

// Check convergence
println!("Converged: {}", model.converged());

// Access results
let mu_coeffs = &model.models["mu"].coefficients;
println!("Intercept: {}, Slope: {}", mu_coeffs[0], mu_coeffs[1]);
```

## Distributions

| Distribution | Parameters | Default Links | Use Case |
|--------------|------------|---------------|----------|
| `Poisson` | mu | log | Count data |
| `Gaussian` | mu, sigma | identity, log | Continuous data |
| `StudentT` | mu, sigma, nu | identity, log, log | Heavy-tailed continuous |
| `Gamma` | mu, sigma | log, log | Positive continuous |
| `NegativeBinomial` | mu, sigma | log, log | Overdispersed counts |
| `Beta` | mu, phi | logit, log | Proportions (0, 1) |

### Usage

```rust
use gamlss_rs::distributions::{Poisson, Gaussian, StudentT, Gamma, NegativeBinomial, Beta};

let poisson = Poisson::new();             // Count data
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

Customize fitting behavior with `FitConfig`:

```rust
use gamlss_rs::FitConfig;

let config = FitConfig {
    max_iterations: 50,  // Maximum outer iterations (default: 20)
    tolerance: 1e-8,     // Convergence tolerance (default: 1e-6)
};

let model = GamlssModel::fit_with_config(
    &df, "y", &formula, &Gaussian::new(), config
)?;
```

## Accessing Results

```rust
let model = GamlssModel::fit(&df, "y", &formula, &Gaussian::new())?;

// Check convergence diagnostics
println!("Converged: {}", model.diagnostics.converged);
println!("Iterations: {}", model.diagnostics.iterations);
println!("Final change: {}", model.diagnostics.final_change);

// For each parameter (mu, sigma, etc.)
let fitted_mu = &model.models["mu"];

fitted_mu.coefficients    // Coefficient estimates (Coefficients wrapper)
fitted_mu.covariance      // Covariance matrix (CovarianceMatrix wrapper)
fitted_mu.fitted_values   // Fitted values on response scale
fitted_mu.eta             // Linear predictor values
fitted_mu.edf             // Effective degrees of freedom
fitted_mu.lambdas         // Smoothing parameters
fitted_mu.terms           // The terms used in the formula
```

## Prediction

Make predictions on new data:

```rust
// Predict on new data
let new_df = df!(
    "x" => [1.5, 2.5, 3.5]
).unwrap();

// Point predictions (returns fitted values for each parameter)
let predictions = model.predict(&new_df)?;
let mu_pred = &predictions["mu"];

// Predictions with standard errors
let (pred, se) = model.predict_with_se(&new_df)?;

// Generate posterior samples for uncertainty quantification
let samples = model.posterior_samples(&new_df, 1000)?;
```

## Model Diagnostics

Access diagnostic information:

```rust
use gamlss_rs::diagnostics::{residuals, ResidualType};

// Get residuals
let response_resid = residuals(&model, &df, "y", ResidualType::Response)?;
let pearson_resid = residuals(&model, &df, "y", ResidualType::Pearson)?;

// Information criteria
let aic = model.aic(&df, "y")?;
let bic = model.bic(&df, "y")?;

// Log-likelihood
let loglik = model.log_likelihood(&df, "y")?;

// Total effective degrees of freedom
let total_edf = model.total_edf();
```

## Examples

### Heteroskedastic Regression

Model where both mean and variance depend on x:

```rust
let mut formula = HashMap::new();

// Mean increases with x
formula.insert("mu".to_string(), vec![
    Term::Intercept,
    Term::Linear { col_name: "x".to_string() },
]);

// Variance also increases with x
formula.insert("sigma".to_string(), vec![
    Term::Intercept,
    Term::Linear { col_name: "x".to_string() },
]);

let model = GamlssModel::fit(&df, "y", &formula, &Gaussian::new())?;
```

### Nonlinear Smooth

Fit a flexible nonparametric curve:

```rust
let mut formula = HashMap::new();

formula.insert("mu".to_string(), vec![
    Term::Smooth(Smooth::PSpline1D {
        col_name: "x".to_string(),
        n_splines: 15,
        degree: 3,
        penalty_order: 2,
    }),
]);
formula.insert("sigma".to_string(), vec![Term::Intercept]);

let model = GamlssModel::fit(&df, "y", &formula, &Gaussian::new())?;
```

### Count Data with Poisson

Model count outcomes:

```rust
use gamlss_rs::distributions::Poisson;

let mut formula = HashMap::new();
formula.insert("mu".to_string(), vec![
    Term::Intercept,
    Term::Linear { col_name: "predictor".to_string() },
]);

let model = GamlssModel::fit(&df, "count", &formula, &Poisson::new())?;
```

### Heavy-Tailed Data with Student-t

Model data with potential outliers:

```rust
use gamlss_rs::distributions::StudentT;

let mut formula = HashMap::new();
formula.insert("mu".to_string(), vec![
    Term::Intercept,
    Term::Linear { col_name: "x".to_string() },
]);
formula.insert("sigma".to_string(), vec![Term::Intercept]);
formula.insert("nu".to_string(), vec![Term::Intercept]); // Degrees of freedom

let model = GamlssModel::fit(&df, "y", &formula, &StudentT::new())?;
```

### Mixed Effects Model

Include random intercepts for grouped data:

```rust
let mut formula = HashMap::new();

formula.insert("mu".to_string(), vec![
    Term::Intercept,
    Term::Linear { col_name: "x".to_string() },
    Term::Smooth(Smooth::RandomEffect {
        col_name: "subject_id".to_string(),
    }),
]);
formula.insert("sigma".to_string(), vec![Term::Intercept]);

let model = GamlssModel::fit(&df, "y", &formula, &Gaussian::new())?;
```

### Overdispersed Count Data

Model count data with more variance than Poisson allows:

```rust
use gamlss_rs::distributions::NegativeBinomial;

let mut formula = HashMap::new();
formula.insert("mu".to_string(), vec![
    Term::Intercept,
    Term::Linear { col_name: "x".to_string() },
]);
formula.insert("sigma".to_string(), vec![Term::Intercept]); // Overdispersion

let model = GamlssModel::fit(&df, "count", &formula, &NegativeBinomial::new())?;
```

### Proportion/Rate Data

Model outcomes bounded between 0 and 1:

```rust
use gamlss_rs::distributions::Beta;

let mut formula = HashMap::new();
formula.insert("mu".to_string(), vec![
    Term::Intercept,
    Term::Linear { col_name: "x".to_string() },
]);
formula.insert("phi".to_string(), vec![Term::Intercept]); // Precision

let model = GamlssModel::fit(&df, "proportion", &formula, &Beta::new())?;
```

### Duration/Positive Continuous Data

Model positive continuous outcomes like survival times:

```rust
use gamlss_rs::distributions::Gamma;

let mut formula = HashMap::new();
formula.insert("mu".to_string(), vec![
    Term::Intercept,
    Term::Linear { col_name: "age".to_string() },
]);
formula.insert("sigma".to_string(), vec![Term::Intercept]); // Coefficient of variation

let model = GamlssModel::fit(&df, "duration", &formula, &Gamma::new())?;
```

## Error Handling

The library uses `GamlssError` for error handling:

```rust
use gamlss_rs::GamlssError;

match GamlssModel::fit(&df, "y", &formula, &Gaussian::new()) {
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
| `MissingColumn` | Required column not in DataFrame |
| `MissingFormula` | Formula missing for a distribution parameter |
| `NonFiniteValues` | NaN or Inf values in data |
| `EmptyData` | DataFrame has no rows |
| `Convergence` | Algorithm failed to converge |
| `Optimization` | Smoothing parameter optimization failed |
| `Linalg` | Linear algebra computation failed |

## Dependencies

- [polars](https://crates.io/crates/polars) - DataFrame operations
- [ndarray](https://crates.io/crates/ndarray) - N-dimensional arrays
- [ndarray-linalg](https://crates.io/crates/ndarray-linalg) - Linear algebra (requires OpenBLAS)
- [argmin](https://crates.io/crates/argmin) - Optimization algorithms
- [statrs](https://crates.io/crates/statrs) - Statistical functions
- [rayon](https://crates.io/crates/rayon) - Parallel computation for large datasets

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

## License

MIT
