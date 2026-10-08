# The Rust API

A reference for the typed Rust surface: terms and formulas, fit configuration, reading results, prediction, diagnostics, model selection, and errors.
For worked examples start with `cookbook-quickstart.md`; for each family's constructor and parameters see `cookbook-families.md`.

## Terms

| Term | Builder | Formula String |
|------|---------|----------------|
| Intercept | `Term::Intercept` | `1` (implicit; `0` or `-1` suppresses it) |
| Linear | `Term::linear("x")` | `x` |
| Factor | `Term::factor("g")`, `Term::factor_with("g", Contrast::SumToZero)` | `factor(g)`, `factor(g, sum)` |
| Interaction | `Term::interaction(Term::linear("a"), Term::factor("b"))` | `a:b`; `a*b` expands to `a + b + a:b` |
| Offset | `Term::offset("log_e")` | `offset(log_e)` |
| P-spline | `Term::smooth(Smooth::ps("x").n_splines(20))` | `s(x)`, `s(x, k=20)` |
| Cubic regression spline | `Term::smooth(Smooth::cr("x").k(10))` | `s(x, bs="cr")` |
| Tensor product | `Term::smooth(Smooth::tensor("x", "z"))` | `te(x, z)` |
| Random intercept | `Term::smooth(Smooth::re("g"))` | `s(g, bs="re")` |

Factors take numeric level codes; levels resolve from the training column at fit time and replay at predict time.
`Smooth::ps` defaults to `n_splines = 10`, `degree = 3`, `penalty_order = 2`.
`Smooth::cr` defaults to `k = 6` with knots placed from the data, and `.pc(0.0)` pins `f(0) = 0`.

`Formula::from_strings` builds several parameters from strings, and `parse_formula_string` parses one:

```rust
let formula = Formula::from_strings([("mu", "y ~ s(x) + factor(g)"), ("sigma", "~ x")])?;
```

The response name left of `~` is ignored, since `y` is passed to `fit` separately.

## Configuration

```rust
use glissando::{FitConfig, GamlssModel, NaAction, Param, SmoothingCriterion};

let config = FitConfig {
    max_iterations: 200,
    tolerance: 1e-3,
    criterion: SmoothingCriterion::Reml,  // also: Gcv, FellnerSchall
    ..FitConfig::default()
}
.with_link(Param::Mu, "probit")
.with_na_action(NaAction::Fail);

// The third argument is optional per-observation prior weights.
let model = GamlssModel::fit_with_config(&data, &y, None, &formula, &Gaussian::new(), config)?;
```

| Field | Default | Meaning |
|-------|---------|---------|
| `criterion` | `Reml` | Smoothing-parameter selection: `Reml` (Laplace-approximate marginal likelihood, L-BFGS), `Gcv` (L-BFGS), or `FellnerSchall` (fixed-point update, no line search) |
| `tolerance` | `1e-3` | Per-parameter linear-predictor change |
| `gd_tolerance` | `1e-3` | Absolute global-deviance change (the gamlss `c.crit` convention); convergence needs both tolerances |
| `step_halving` | `true` | Line-search each update on the global deviance, so every cycle descends |
| `links` | Empty | Per-parameter link overrides |
| `na_action` | `DropRows` | Drop rows with a non-finite value in `y` or a referenced column, or `Fail` to reject them |

**Links.**
Any parameter can take one of nine links: `identity`, `log`, `logit`, `probit`, `cloglog`, `inverse`, `inverse_square`, `sqrt`, `cauchit`.
`Ocat` parameters and `StudentT`'s `nu` refuse an override.
The fit checks the parameter and link names but not that the link's range suits the parameter, so a logit link on a Poisson mean pins it into `(0, 1)` and still succeeds.

**Prior weights.**
`GamlssModel::fit_weighted(&data, &y, &weights, &formula, &family)` scales each observation's likelihood contribution, matching `mgcv::bam(..., weights = w)`.
Weights must be finite, non-negative, and as long as `y`; a zero weight excludes the row.

## Results

```rust
model.diagnostics.converged;            // also .iterations and .warnings
let mu = &model.models[&Param::Mu];
mu.coefficients;                        // Deref to Array1<f64>
mu.covariance;                          // Deref to Array2<f64>
mu.fitted_values;                       // Response scale
mu.eta;                                 // Linear predictor
mu.edf;                                 // Effective degrees of freedom; mu.term_edf splits it per term
mu.lambdas;                             // Smoothing parameters

// mgcv-style accessors
let x = model.design_matrix(&new_data, Param::Mu)?;     // predict(type = "lpmatrix")
let vcov = model.covariance_matrix(Param::Mu)?;
let index = model.term_index_map(Param::Mu)?;           // (term name, first col, end col exclusive)
```

## Prediction

```rust
let family = Gaussian::new();

let preds = model.predict(&new_data, &family)?;                   // Response scale, keyed by Param
let with_se = model.predict_with_se(&new_data, &family)?;         // .fitted, .eta, .se_eta per Param
let draws = model.predict_samples(&new_data, &family, 1000, Some(42))?;
let beta = model.posterior_samples(Param::Mu, 1000, Some(42))?;   // Raw coefficient draws

let centiles = model.centiles(&new_data, &family, &[10.0, 50.0, 90.0])?;  // Keyed "C10", "C50", ...
let p = Array1::from_elem(new_data.n_obs().unwrap(), 0.95);
let upper = model.quantile_prediction(&new_data, &family, &p)?;           // Row-varying level
```

Predicting with a different family than the one the model was fit with returns `GamlssError::FamilyMismatch`.

## Diagnostics

```rust
let d = model.diagnostics(&family, &y)?;            // ModelDiagnostics
d.log_likelihood; d.total_edf; d.aic; d.bic;
d.pearson_residuals; d.response_residuals;

let resid = model.quantile_residuals(&family, &y, Some(42))?;
```

Randomized quantile residuals are N(0, 1) under a correct model for every family, so one Q-Q or worm plot works across distributions.
The seed only matters for discrete families.
The building blocks (`pearson_residuals`, `response_residuals`, `compute_gaic`, `compute_aic`, `compute_bic`, `total_edf`) are public in `glissando::diagnostics`.

## Model Selection

```rust
use glissando::selection::{ic_table, lr_test, step_gaic, Direction, StepScope};

let aic = model.gaic(&family, &y, 2.0)?;                               // k = ln(n) gives BIC
let rows = ic_table(&[("null", &m0), ("with_x", &m1)], &family, &y, 2.0)?;
let lrt = lr_test(&m0, &m1, &family, &y)?;                             // m0 nested in m1

let scope = vec![StepScope { param: Param::Mu, candidates: vec![/* terms */] }];
let result = step_gaic(&data, &y, &family, start_formula, &scope,
                       (y.len() as f64).ln(), Direction::Both, FitConfig::default())?;
// result.model is the selected fit; result.trace records the accepted moves.
```

## Errors

Every fallible call returns `GamlssError`.

| Variant | Meaning |
|---------|---------|
| `Input` | Invalid data, formula, or configuration |
| `MissingVariable` | A referenced column is not in the data |
| `MissingFormula` | The formula has no entry for one of the family's parameters |
| `NonFiniteValues` | A column has NaN or Inf (at fit time `NaAction::DropRows` drops those rows instead) |
| `EmptyData` | No observations |
| `Convergence` | The RS loop did not converge |
| `Optimization` | Smoothing-parameter optimization failed |
| `Linalg` | A factorization or solve failed |
| `PosteriorNotPositiveDefinite` | Posterior sampling could not factor the covariance |
| `UnknownParameter` | The parameter does not belong to the family |
| `FamilyMismatch` | The model was used with a different family than it was fit with |
| `Shape` | Array shapes disagree |
| `Internal` | A bug in glissando |
