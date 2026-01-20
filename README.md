# gamlss_rs

A Rust implementation of Generalized Additive Models for Location, Scale, and Shape (GAMLSS).

GAMLSS extends traditional regression by modeling not just the mean, but also variance and other distribution parameters as functions of predictors.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
gamlss_rs = { git = "https://github.com/your-repo/gamlss_rs" }
```

Requires OpenBLAS for linear algebra operations.

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

// Access results
let mu_coeffs = &model.models["mu"].coefficients;
println!("Intercept: {}, Slope: {}", mu_coeffs[0], mu_coeffs[1]);
```

## Distributions

| Distribution | Parameters | Default Links |
|--------------|------------|---------------|
| `Poisson` | mu | log |
| `Gaussian` | mu, sigma | identity, log |
| `StudentT` | mu, sigma, nu | identity, log, log |

## Term Types

### Intercept
```rust
Term::Intercept
```

### Linear
```rust
Term::Linear { col_name: "x".to_string() }
```

### P-Spline (1D smooth)
```rust
Term::Smooth(Smooth::PSpline1D {
    col_name: "x".to_string(),
    n_splines: 10,
    degree: 3,
    penalty_order: 2,
})
```

### Tensor Product (2D smooth)
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
```rust
Term::Smooth(Smooth::RandomEffect {
    col_name: "group".to_string(),
})
```

## Accessing Results

```rust
let model = GamlssModel::fit(&df, "y", &formula, &Gaussian::new())?;

// For each parameter (mu, sigma, etc.)
let fitted_param = &model.models["mu"];

fitted_param.coefficients   // Coefficient estimates
fitted_param.covariance     // Covariance matrix
fitted_param.fitted_values  // Fitted values on response scale
fitted_param.eta            // Linear predictor
fitted_param.edf            // Effective degrees of freedom
fitted_param.lambdas        // Smoothing parameters
```

## Example: Heteroskedastic Regression

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

## Example: Nonlinear Smooth

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

## License

MIT
