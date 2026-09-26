# Migrating from R (mgcv / gamlss)

**Who this is for:** people who know `gamlss()` or `mgcv::gam()` and want the glissando equivalent.
**The one big difference:** in R's `gamlss`, each distribution parameter gets its own formula argument (`mu.formula`, `sigma.formula`, ...); in glissando they are entries in a single `Formula`, keyed by parameter name. `mgcv::gam` models only the mean, so a `gam(y ~ s(x))` maps to a glissando fit with a `mu` predictor and default (intercept-only) predictors elsewhere.
**Honesty:** glissando is not at parity yet. This guide maps what exists and names what does not, with the roadmap ID so you can check status.

Read `cookbook-quickstart.md` first for the mechanics; this page is about translation.

## The mental model

R `gamlss` and glissando agree on the core idea: model every parameter of the distribution, not just its mean.
The translation is mechanical.

```r
# R gamlss
library(gamlss)
m <- gamlss(y ~ pb(x), sigma.formula = ~ x, family = NO, data = df)
```

```rust
// glissando
let formula = Formula::from_strings([
    ("mu", "y ~ s(x)"),   // pb(x) penalized spline  ->  s(x)
    ("sigma", "~ x"),
]).unwrap();
let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
```

The response `y` is passed as its own argument to `fit`, so the left side of the `~` is decorative (parsed and discarded).

## Family names

R family codes map to glissando family constructors as follows.

| R (`gamlss.dist`) | glissando | Notes |
|-------------------|-----------|-------|
| `NO` | `Gaussian::new()` | |
| `GA` | `Gamma::new()` | |
| `WEI`, `WEI2`, `WEI3` | `Weibull::new()` | glissando uses one Weibull parameterization |
| `BE` | `Beta::new()` | response on (0, 1) |
| `PO` | `Poisson::new()` | |
| `NBI` | `NegativeBinomial::new()` | overdispersed counts |
| `BI` | `Binomial::new(n)` | trials are family state, not a column |
| `TF`, `TF2` | `StudentT::new()` | needs a `nu` formula entry |
| `BCCG`, `BCCGo` | `BCCG::new()` | LMS |
| `BCT` | `BCT::new()` | |
| `BCPE` | `BCPE::new()` | |
| ordinal (`ocat` in mgcv) | `Ocat::new(k)` | 2 to 5 levels |
| `IG`, `LOGNO`, `EXP`, `GG`, `PARETO` | not shipped | `DIST-2` remainder |
| `SN`, `ST`, `SHASH`, `JSU`, `PE` | not shipped | `DIST-3` |
| `GEOM`, `LG`, `BB`, `PIG`, `SICHEL`, `DEL` | not shipped | `DIST-4` |
| `ZIP`, `ZINBI`, `ZAGA`, `ZAIG`, `BEINF` | not shipped | `DIST-5`; use `Hurdle` as a stopgap |
| `TW` (Tweedie) | not shipped | `DIST-7` |

## Smooths and basis codes

glissando's `s(x)` defaults to a P-spline (mgcv's `bs="ps"`), not mgcv's thin-plate default.
Set the count with `k=`; note the default knot count differs from mgcv's.

| R / mgcv | glissando string | glissando builder | Status |
|----------|------------------|-------------------|--------|
| `s(x)`, `s(x, bs="ps")`, `pb(x)` | `s(x)` or `s(x, bs="ps")` | `Smooth::ps("x")` | shipped |
| `s(x, bs="cr")` | `s(x, bs="cr")` | `Smooth::cr("x")` | shipped |
| `s(x, k=20)` | `s(x, k=20)` | `Smooth::ps("x").n_splines(20)` | shipped |
| `te(x, z)` | `te(x, z)` | `Smooth::tensor("x", "z")` | shipped (2D only) |
| `s(g, bs="re")` (random intercept) | `s(g, bs="re")` | `Smooth::re("g")` | shipped |
| `s(x, bs="cc")` (cyclic) | not available | | `SMOOTH-1` |
| `s(x, bs="tp")` (thin-plate) | not available | | `SMOOTH-2` |
| `s(x, by = f)` (varying-coefficient) | not available | | `SMOOTH-3` |
| `s(x, g, bs="re")` (random slope) | not available | | `SMOOTH-4` |
| `ti(...)`, `te(...)` in >2D | not available | | `SMOOTH-6` |

Factors, interactions, and offsets carry over directly: `factor(g)`, `x:z`, `x*z`, and `offset(logexp)` all parse in a glissando formula string, the same as in R.

## Smoothing-parameter selection

mgcv's default is REML; glissando's is also REML (`SmoothingCriterion::Reml`), so the out-of-the-box behavior lines up.
GCV is available too.
Set it through the config struct (only `na_action` and `links` have builder methods; the numeric knobs are plain fields):

```rust
use glissando::{FitConfig, SmoothingCriterion};
let config = FitConfig { criterion: SmoothingCriterion::Gcv, ..FitConfig::default() };
let model = GamlssModel::fit_with_config(&data, &y, None, &formula, &Gaussian::new(), config).unwrap();
```

mgcv's `method = "GCV.Cp"` maps to `Gcv`; `method = "REML"` is the default.
There is no `magic`-style grid selection (`FIT-4`) or the `gamlss` CG algorithm (`FIT-3`) yet.

## Common workflows, translated

**Residual diagnostics.** R's `residuals(m)` on a `gamlss` fit returns randomized quantile residuals by default. The direct equivalent:

```rust
let resid = model.quantile_residuals(&family, &y, Some(seed)).unwrap();
```

The `wp()` worm plot and `qqnorm` Q-Q plot are not built in (`DIAG-1` / `DIAG-2`), but you have the residuals to draw them yourself.

**Model comparison.** R's `GAIC(m1, m2, k=2)` and `LR.test`:

```rust
let aic = model.gaic(&family, &y, 2.0).unwrap();                 // k = 2 -> AIC
let bic = model.gaic(&family, &y, (n as f64).ln()).unwrap();     // k = ln n -> BIC
let test = glissando::selection::lr_test(&small, &big, &family, &y).unwrap();  // {lr_stat, df, p_value}
```

Note `lr_test` and `step_gaic` live under `glissando::selection`, not the crate root.

**Stepwise selection.** R's `stepGAIC`:

```rust
use glissando::selection::{step_gaic, Direction, StepScope};
let scope = vec![StepScope { param: "mu".into(), candidates: vec![Term::linear("x"), Term::linear("z")] }];
let result = step_gaic(&data, &y, &family, start_formula, &scope, 2.0, Direction::Both, FitConfig::default()).unwrap();
// result.model, result.formula, result.trace
```

**Centiles / growth curves.** R's `centiles()` / `centiles.pred()`:

```rust
let curves = model.centiles(&grid, &family, &[3.0, 15.0, 50.0, 85.0, 97.0]).unwrap();
```

## Known gaps versus R

These are the things an R user will reach for and not find yet.

- **Per-smooth significance / p-values** (mgcv `summary.gam` Tr/Wald table): not implemented (`INFER-5`). The whole-model likelihood-ratio test (`lr_test`) is the nearest thing.
- **Confidence intervals** for curves and coefficients: partial (`INFER-6`). The coefficient covariance (`covariance_matrix`) and the per-observation standard error of the linear predictor (`predict_with_se` -> `se_eta`) are computed and exposed, so you can build a band yourself; there is no function that returns bounds directly.
- **Worm plots, Q-Q plots, deviance residuals, goodness-of-fit statistics** (`DIAG-1` .. `DIAG-4`): not built in.
- **Term-effect / partial-effect plots** (`plot.gam`): the primitives exist (`design_matrix`, `term_index_map`, `covariance_matrix`), but nothing assembles the per-term curve with a band (`DIAG-5`).
- The families and smooths marked "not shipped" in the tables above.
