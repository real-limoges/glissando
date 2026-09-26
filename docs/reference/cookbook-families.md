# The distribution gallery

**What this is:** every distribution family glissando ships, what it models, its parameters, and how to construct it.
**How to swap:** the fit call is identical across families; only the family value and the set of modeled parameters change. A family with a `nu` parameter (Student-t, the Box-Cox trio) needs a formula entry for `nu`, or the fit is rejected.
**Runnable:** `examples/families.rs` fits a representative family from each group (`cargo run --example families`); `examples/python/families.py` mirrors it.

The parameter names matter: your formula must supply one predictor per parameter the family exposes.
The response-scale meaning of `mu` is the mean for most families, and `sigma` is a dispersion or scale, but the higher parameters (`nu`, `tau`) are shape parameters whose meaning is family-specific.

## Base families at a glance

| Family | Rust | Parameters | Models |
|--------|------|------------|--------|
| Gaussian | `Gaussian::new()` | mu, sigma | Symmetric continuous |
| Student-t | `StudentT::new()` | mu, sigma, nu | Heavy-tailed continuous |
| Gamma | `Gamma::new()` | mu, sigma | Positive continuous (right-skewed) |
| Weibull | `Weibull::new()` | mu, sigma | Positive continuous, survival / time-to-event |
| Beta | `Beta::new()` | mu, sigma | Proportions on the open (0, 1) |
| Poisson | `Poisson::new()` | mu | Counts |
| Negative Binomial | `NegativeBinomial::new()` | mu, sigma | Overdispersed counts |
| Binomial | `Binomial::new(n)` | mu | Successes out of `n` trials |
| Ocat | `Ocat::new(k)` | mu, delta_1 .. delta_{k-2} | Ordered categorical, 2 to 5 levels |
| BCCG | `BCCG::new()` | mu, sigma, nu | Skewed positive (LMS centiles) |
| BCT | `BCT::new()` | mu, sigma, nu, tau | Skew + kurtosis (growth charts) |
| BCPE | `BCPE::new()` | mu, sigma, nu, tau | Skew + power-exponential tails |

Three structural wrappers (`Censored`, `Truncated`, `Hurdle`) and finite mixtures (`fit_mixture`) wrap any base family; they are at the end of this page.

## Continuous, symmetric

**Gaussian** is the default and the one to reach for when the response is real-valued and roughly symmetric.
Modeling `sigma` on covariates is the GAMLSS move a plain regression cannot make.

```rust
let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
```

**Student-t** trades the Gaussian's thin tails for a `nu` degrees-of-freedom parameter, so it absorbs outliers instead of being dragged by them.
It has three parameters, so the formula must name `nu`:

```rust
let formula = Formula::new()
    .with_terms("mu", vec![Term::Intercept, Term::linear("x")])
    .with_terms("sigma", vec![Term::Intercept])
    .with_terms("nu", vec![Term::Intercept]);   // required, or the fit errors
let model = GamlssModel::fit(&data, &y, &formula, &StudentT::new()).unwrap();
```

## Continuous, positive

**Gamma** and **Weibull** both model strictly positive responses (durations, costs, concentrations).
Gamma is the classical right-skewed choice; Weibull is the survival / reliability workhorse.
Their default links keep `mu` and `sigma` positive, and the newly shipped inverse / sqrt links can be set per parameter through `FitConfig::with_link` if you want a non-default scale.

```rust
let model = GamlssModel::fit(&data, &y, &formula, &Weibull::new()).unwrap();  // y > 0
```

## Bounded proportions

**Beta** models a response on the open interval (0, 1): rates, fractions, proportions that are neither exactly 0 nor exactly 1.
For data with exact zeros or ones you want a zero/one-inflated variant (not yet shipped; see `DIST-5` in the roadmap) or the `Hurdle` wrapper below.

## Counts

**Poisson** is the one-parameter count model (`mu` only).
When the variance exceeds the mean (it usually does), **Negative Binomial** adds a `sigma` dispersion parameter and is the safer default for real count data.

```rust
let model = GamlssModel::fit(&data, &counts, &formula, &NegativeBinomial::new()).unwrap();
```

**Binomial** models successes out of a known number of trials.
The trial count is family state, not a covariate: `Binomial::new(1)` is the Bernoulli / binary case, and `Binomial::with_trials(n_array)` sets a different trial count per row.

```rust
use glissando::ndarray::Array1;
let family = Binomial::with_trials(Array1::from_vec(vec![10.0, 8.0, 12.0 /* ... */]));
```

In Python this constructor takes a *list* of per-row trial counts, not a scalar: `glissando.Binomial([10, 8, 12, ...])`.

## Ordered categorical

**Ocat** models an ordinal response with 2 to 5 levels (coded 1..k).
Beyond `mu` it exposes `delta_1 .. delta_{k-2}` threshold-offset parameters, so a 4-level Ocat has `mu`, `delta_1`, `delta_2`.
It has an extra prediction method, `predict_class_probabilities`, returning an `n * k` matrix of per-category probabilities.

```rust
let family = Ocat::new(4);
let formula = Formula::new()
    .with_terms("mu", vec![Term::Intercept, Term::smooth(Smooth::ps("x"))])
    .with_terms("delta_1", vec![Term::Intercept])
    .with_terms("delta_2", vec![Term::Intercept]);
let model = GamlssModel::fit(&data, &y, &formula, &family).unwrap();
let probs = model.predict_class_probabilities(&data, &family).unwrap();  // n x 4
```

## The Box-Cox family (skew and kurtosis)

These are the distributions GAMLSS is known for, and the engine behind LMS growth-chart / centile curves.
**BCCG** (Cole-Green) adds a `nu` skewness parameter to a positive response; **BCT** adds a `tau` kurtosis parameter on top (Box-Cox-t); **BCPE** replaces the t tail with a power-exponential one.
Fit them exactly like any other family, then read percentile curves off the fit.
BCCG has three parameters, so the formula names `nu` (BCT and BCPE add a `tau` entry on top):

```rust
let family = BCCG::new();
let formula = Formula::new()
    .with_terms("mu", vec![Term::Intercept, Term::smooth(Smooth::ps("x"))])
    .with_terms("sigma", vec![Term::Intercept])
    .with_terms("nu", vec![Term::Intercept]);
let model = GamlssModel::fit(&data, &y, &formula, &family).unwrap();
// centile curves at the standard growth-chart percentiles
let curves = model.centiles(&grid, &family, &[3.0, 15.0, 50.0, 85.0, 97.0]).unwrap();
```

`centiles` returns one curve per requested percentile; `quantile_prediction` instead takes a per-row centile level and returns one value per row.

## Structural wrappers

These wrap a boxed base family and carry per-observation response-side state.
Fit and predict work as usual once constructed.

**Censored** handles survival-style responses where some observations are only known to lie above, below, or within an interval:

```rust
use glissando::distributions::{Censored, CensorStatus, Gaussian};
use glissando::ndarray::Array1;
let status = Array1::from_vec(vec![CensorStatus::Event, CensorStatus::Right /* ... */]);
let family = Censored::new(Box::new(Gaussian::new()), status);
```

**Truncated** bounds the support (observations outside `[lower, upper]` could not have been sampled); `+inf` / `-inf` bounds encode one-sided truncation:

```rust
use glissando::distributions::Truncated;
let lower = Array1::from_elem(n, 0.0);
let upper = Array1::from_elem(n, f64::INFINITY);
let family = Truncated::new(Box::new(Gaussian::new()), lower, upper);
```

**Hurdle** is a two-part structure that adds a `xi` parameter governing a point mass, a clean generalization of zero-inflation:

```rust
use glissando::distributions::Hurdle;
let family = Hurdle::new(Box::new(Gaussian::new()));   // gains a "xi" parameter
```

## Finite mixtures

`fit_mixture` fits a `k`-component mixture of a base family by EM, reusing the prior-weighted RS fit as the M-step:

```rust
use glissando::{fit_mixture, FitConfig};
let mixture = fit_mixture(&data, &y, &formula, &Gaussian::new(), 2, &FitConfig::default(), Some(99)).unwrap();
// mixture.components: Vec<GamlssModel>, mixture.weights: Vec<f64>, mixture.log_likelihood
```

## Not yet shipped

Skew/kurtotic families beyond Box-Cox (`DIST-3`), the extra count families (`DIST-4`), zero-inflated / zero-adjusted families (`DIST-5`), and Tweedie (`DIST-7`) are on the roadmap but not implemented.
For semicontinuous data with exact zeros today, reach for `Hurdle` over a positive base family.
