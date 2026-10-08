# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking): distribution parameters are a `Param` enum, not strings

Parameter names (`"mu"`, `"sigma"`, `"nu"`, `"tau"`, `"phi"`, `"xi"`, `"delta_1"`..`"delta_4"`) were strings everywhere, so a typo or a parameter from the wrong family was only noticed when something looked it up, and sometimes never.
They are now the closed enum `glissando::Param` (`Mu`, `Sigma`, `Nu`, `Tau`, `Phi`, `Xi`, `Delta1`..`Delta4`), and a string is parsed into one exactly once, where it enters the crate.

- Rust API: `Formula::with_terms` / `add_terms` / `parse` take a `Param`; `GamlssModel.models`, `FitDiagnostics.param_diagnostics` and `FitConfig.links` are keyed by `Param`; `design_matrix`, `covariance_matrix`, `term_index_map` and `posterior_samples` take a `Param`; `predict`, `predict_with_se` and `predict_samples` return `Param`-keyed maps; `StepScope.param` is a `Param`.
- `Distribution` trait: `parameters()` returns `&[Param]`, the per-parameter methods take `param: Param`, the params map is `HashMap<Param, &Array1<f64>>`, `DerivativeMap` / `CdfMap` are keyed by `Param`, and `LinkContext` is built from `(Param, &dyn Link, &Array1<f64>)` entries.
- `Formula::from_strings` still takes string pairs and parses the names.
- New error `GamlssError::InvalidParamName { name }` for a string that is no parameter at all; `UnknownParameter` remains for a real parameter the family lacks.
- **New validation**: a formula that gives terms to a parameter the family does not have now fails with `UnknownParameter` instead of being silently ignored, and so does a `StepScope` for such a parameter.
- JSON, WASM and Python keep their string names, and every wire format is byte-identical: `Param` serializes as its name. An invalid name in a JSON formula or config, or in a Python formula, scope, or `links` dict, now fails up front (Python raises `ValueError`).
- **No numerical change**: every derivative and regression snapshot is byte-identical.

### Changed (breaking): derivative maps carry their scale in the type

`Distribution::theta_derivatives`, `eta_derivatives` and `cdf_theta_derivatives` used to return the same untyped `HashMap<String, (Array1<f64>, Array1<f64>)>` whether the numbers were on the natural scale θ or the linear-predictor scale η.
Reading one as the other, or chaining twice, compiled and ran and only moved the estimates.
They now return scale-tagged types, so that mix-up is a compile error.

- New in `glissando::distributions`: the markers `Natural` and `Eta`, `ScoreInfo<S>` (fields `score`, `info`), `CdfGrad<S>` (fields `d1`, `d2`), and the maps `DerivativeMap<S>` and `CdfMap<S>`.
- `theta_derivatives` returns `DerivativeMap<Natural>`, `eta_derivatives` returns `DerivativeMap<Eta>`, and `cdf_theta_derivatives` returns `CdfMap<Natural>`.
- `chain_to_eta` takes a `DerivativeMap<Natural>` and returns a `DerivativeMap<Eta>`; it is the generic way from one scale to the other.
- Natural-scale values are built with `ScoreInfo::new` / `CdfGrad::new`.
  A hand-written `eta_derivatives` builds η-scale values with `ScoreInfo::computed_on_eta`, a deliberately explicit name.
- Removed the aliases `DerivativesResult`, `CdfEtaMap`, `CdfEtaResult`, `CdfThetaMap` and `CdfThetaResult`.
- **No numerical change**: every derivative and regression snapshot is byte-identical.


### Changed: replaced the `argmin` L-BFGS with an in-house optimizer

**No public API change, and single-smooth fits are unchanged.**
The smoothing-parameter search (REML/GCV) now uses a hand-rolled L-BFGS with a strong-Wolfe line search rather than the `argmin` crate, which removes `argmin` and `argmin-math` from the dependency tree.
That drops the duplicate `ndarray 0.16` / `ndarray-linalg 0.17` / `lax` / `lapack-sys` chain `argmin-math` pinned, and the `getrandom 0.3` wasm shim that `argmin`'s `rand 0.9` required.

- The optimizer is deterministic across the `openblas` and `pure-rust` backends and returns the best point it visits, so a failed line search never regresses the result.
- REML fits also screen a cold-start Fellner-Schall candidate on multi-penalty (anisotropic tensor) terms.
  This repairs a corner-basin case where the previous optimizer could settle on an oversmoothed local optimum.
- One characterization snapshot moved: a Gaussian P-spline that collapses to its null space under GCV now reports a different smoothing parameter on the flat high-λ shelf (`1.5e7` to `8.0e5`); its fitted values, EDF and log-likelihood are unchanged.

### Fixed: non-default links now fit correctly

**Refit any model fitted with a non-default link.**
Its coefficients were estimated under the wrong working weights and the wrong chain rule.
Standard errors, EDF and smoothing-parameter selection are affected too.
Default-link models are unchanged, bit for bit, and were never affected.
`predict` uses only `inv_link`, so predictions already stored from such a model are internally consistent with its (wrong) coefficients.
The serialization format does not change, since `FittedParameter.link` already existed, so a saved model reloads and can be refit directly.

- **`Distribution::derivatives` returned score and information "on the linear-predictor scale", but every family hardcoded the chain rule for *its own default link*.**
  `FitConfig::with_link` honored the caller's choice for `η → μ` only, so probit, cloglog, inverse, inverse-square, sqrt and cauchit fits returned wrong estimates.
  Measured against an independent maximum-likelihood optimizer, the fitted log-likelihood fell short of the true maximum by 45.5 for gamma/inverse, 8.5e-3 for binomial/probit, 1.0e-4 for poisson/sqrt and 6.4e-5 for binomial/cloglog.
  All four gaps are now zero, and the four cases run as acceptance gates.

  `derivatives` now returns natural-scale `(∂l/∂θ, i_θ)`, and a new required `eta_derivatives` applies `u = mu_eta·∂l/∂θ`, `w = mu_eta²·i_θ` once, generically, from the resolved link.
  The structural wrappers chain their CDF gradients with the second-order rule `∂²F/∂η² = mu_eta²·∂²F/∂θ² + mu_eta2·∂F/∂θ`, using a new `Link::mu_eta2`.

- **`FitConfig::links` now rejects an override it cannot honor**, instead of accepting it and computing the wrong thing.
  Two parameters refuse: every `Ocat` parameter (its `params["mu"]` holds η, and its threshold Jacobian is `exp(η_k)` only under the log link) and `StudentT`'s `nu` (its ν-floor projection is written against `FlooredLogLink`).
  Families declare this through a new `Distribution::allows_link_override`, which defaults to `true`.

- **An override naming a parameter the family does not have is now an error.**
  It used to be a no-op: `FitConfig::with_link("sigma", "log")` on `Beta`, whose second parameter is `phi`, fit under the default links and reported success.
  No *domain* checking is done: a logit link on a Poisson μ is still accepted and produces nonsense.

- **`weight_floor_hits` now counts correctly.**
  `MIN_WEIGHT` used to be applied at two layers, and ~20 families pre-floored internally to exactly `MIN_WEIGHT`, which the diagnostic's strict `<` test then never counted, so it under-reported for the families most likely to hit the floor.
  The floor is now applied in one place, the scoring loop, after the chain rule (`max(mu_eta²·i_θ, F) ≠ mu_eta²·max(i_θ, F)`, so the order is not a rounding difference).

**Breaking, for anyone implementing `Distribution` outside this crate.**
`eta_derivatives` is required and has no default body on purpose: an external implementor whose `derivatives` returns η-scale values would otherwise keep compiling against a defaulted adapter and double-chain to `mu_eta⁴ · i_θ`.
Requiring the method turns that into a compile error.
`cdf_eta_derivatives` is likewise renamed to `cdf_theta_derivatives` and its contract inverted to the natural scale.

### Fixed: solver convergence & mgcv/gamlss parity

- **Working-response clipping no longer inverts the Fisher step.**
  The per-element `u/w` clip was tightened from a robustness device (±20) to an anti-overflow guard (±1e6).
  At ±20, whenever many rows clipped, the update direction was decided by the count of positive vs negative rows instead of the score-weighted aggregate.
  On Student-t fits this showed up as an unbounded ν → ∞ runaway that could never converge.
  Overshoot control is left to the deviance-guarded step-halving.
- **Step-halving rejects uphill steps at the backtracking floor** instead of accepting a micro-step.
  This restores the exact monotone-descent guarantee and removes slow one-way drift.
- **The global-deviance convergence test is now absolute** (gamlss `c.crit` semantics, default 0.001 deviance units).
  The previous relative test scaled its slack with |GD| and declared convergence while large-deviance fits were still improving several units per cycle.
- **Outer-loop convergence is measured in fit space (max |Δη|)** instead of coefficient space, so λ jitter along fit-equivalent ridges (flat REML valleys) can no longer block convergence (fixed a permanent 2-cycle on tensor smooths).
- **REML λ optimization finishes with a deterministic Fellner–Schall pass**, fixing L-BFGS stalls at warm-start-dependent non-stationary points (a weighted 5-smooth fit went from 200 cycles/67 s without convergence to 7 cycles/5 s).
- **StudentT**: expected information for ν corrected to the Lange–Little–Taylor / gamlss `TF()` formula (was ~50× too large at ν = 5, freezing ν near its seed).
  The μ working weight now uses the expected information `(ν+1)/((ν+3)σ²)` like gamlss `TF()`.
  The ν ≥ 2 floor now uses a KKT-style aggregate projection (frozen at the boundary when the summed score points outward) instead of drifting η indefinitely.
  A Student-t oracle case now converges in 13 cycles to the gamlss optimum, with the deviance matching to 4 decimals.
- **Gamma**: σ Fisher weight corrected to `(4/σ⁴)ψ′(1/σ²) − 4/σ²` (was `−2/σ²`, ~5× inflated); derivatives clamp y to the support.
- **NegativeBinomial**: σ working weight switched to gamlss `NBI`'s squared-score convention; σ seeded from the method-of-moments overdispersion estimate instead of `sd(y)`.
- **Beta**: φ information formula sign corrected (was rescued by `abs()`).
- **Binomial**: initial μ pooled as `Σy/Σn` under per-observation trials.
- **`te()` with an intercept can now represent main effects.**
  The tensor basis gets ONE sum-to-zero constraint on the full k₁k₂ basis (mgcv `te()` semantics, k₁k₂ − 1 coefficients, both penalties transformed with the same Z).
  Previously each marginal was centered before the Kronecker product, which excluded all `f(x1)` and `g(x2)` main effects and made `te()` a pure-interaction (`ti()`-style) smooth.
- **P-spline / tensor knot grids and random-effect level maps are resolved at fit time and stored on the term** (`PSpline1D::range`, `TensorProduct::range_1/2`, `RandomEffect::levels`: all `serde(default)`, so previously serialized models still load).
  Prediction replays the training basis; it used to re-derive knots/levels from the *prediction* data, corrupting grid/subset/reordered-group predictions.
  Unseen random-effect levels at predict time are now an error (mgcv factor semantics).
- **`create_penalty_matrix` supports any difference order** (alternating binomial stencil, identical to R `diff(diag(k), differences = d)`); before, `order ≥ 3` produced a truncated order-2 penalty.
- **CR splines with a point constraint (`pc`) no longer carry a zero-design, zero-penalty coefficient direction** (the system was structurally singular).
  The direction is removed with the same Householder transform used for centering, preserving `f(pc) = 0`.
- **Rejected block updates keep their previous λ/covariance/EDF**: the outer loop previously installed the rejected proposal's values alongside the reverted coefficients, so SEs/GAIC described a state the model was not in.
- **Basin probes for the λ collapse/bound guard include per-coordinate seeds**, rescuing anisotropic tensor corner traps (one margin pinned at the ceiling while the true LAML optimum has it interior).
- **The tensor (multi-penalty) basin probe fires only on the cheap collapse/bound triggers again**, reverting a change that ran it *unconditionally on every outer cycle*.
  Each firing runs a `7^k` grid plus several L-BFGS/Fellner–Schall solves (an eigendecomposition per penalty block).
  Per-cycle probing cost ≈30 s (OpenBLAS) to >2 min (pure-rust) on a default 10×10 tensor fit in an unoptimized build and hung the debug test suites, all to guard a large-interior-λ corner that only the `#[ignore]`d `benchmark/run_comparison.sh` mgcv sweep checks.
  Ceiling/floor corners are still caught by the bound trigger; re-run the mgcv comparison before relying on tensor EDF parity for a new seed.
- Prediction on a model whose stored coefficients no longer match the rebuilt design (old serialized `te()`/`pc` bases) returns a typed `GamlssError::Shape` with a migration hint instead of panicking.
- Benchmark harness:
  - mgcv `gammals` σ extraction now goes through the family's `logb` linkinv (the previous `exp(η₂/2)` read produced σ̂ ≈ 5–20 for a true CV of 0.2–0.7 and a false parity failure).
  - `gaussian_heteroskedastic` is now compared against mgcv `gaulss` (it had no oracle).
  - R scripts fall back to the dependency-free `nanoparquet` reader when `arrow` is missing.
  - `orchestrate.py` has a per-fit 600 s timeout.

### Added

- **`Weibull` distribution** (gamlss `WEI`: scale `μ`, shape `σ`, both log-linked) in `src/distributions/weibull.rs`.
  It has analytic natural-scale score and Fisher information (`theta_derivatives`), a closed-form `cdf` (`1 − exp(−(y/μ)^σ)`) and a closed-form `quantile` (`μ·(−ln(1−p))^(1/σ)`).
  It is selectable by name (`"Weibull"`) through `from_name` on the native and JSON/WASM surfaces, and as the Python `Weibull()` class, bringing the catalog to 12 families.
  The full derivation is in `docs/reference/mathematics.md` `[WEIBULL]`.
- **Tag-based cross-references in `docs/reference/mathematics.md`**: named subsections are now cited by stable bracketed tags (e.g. `[WEIBULL]`, `[CDF-TRIO]`, `[PWLS-CHOLESKY]`) instead of section numbers, so inserting a family no longer renumbers the document.
  Chapter-level references keep the `§N` form.
- **Box-Cox family (`BCCG`, `BCT`, `BCPE`)**: the gamlss Box-Cox Cole-Green, Box-Cox t and Box-Cox power exponential distributions for positive, skewed responses, in `src/distributions/{boxcox,bccg,bct,bcpe}.rs`.
  All three are selectable by name.
- **`Ocat` ordered-categorical family** (proportional-odds / cumulative logit) for 2 to 5 categories, in `src/distributions/ocat.rs`.
  It carries its category count as state, so it is constructed explicitly (`Ocat::new(k)` in Rust, `Ocat(n_categories=k)` in Python) rather than by name.
  `GamlssModel::predict_class_probabilities` returns the per-category probabilities.
- **Cubic regression splines**: `Smooth::cr` (mgcv `bs="cr"`), with an optional point constraint `pc`.
- **Model selection** in `glissando::selection`: `ic_table` (GAIC comparison across models), `lr_test` (likelihood-ratio test for nested models) and `step_gaic` (stepwise term selection, `Direction::{Forward, Backward, Both}`), plus `GamlssModel::gaic`.
- **Centiles and quantile residuals**: `GamlssModel::centiles` and `GamlssModel::quantile_residuals` (randomized for discrete families).
- **`glissando::json` facade** (feature `serialization`): JSON parse and serialize plus `fit` / `predict` / `load` and the diagnostics and selection helpers, for embedders behind a custom FFI.
  `src/wasm.rs` is now a thin shim over it.
- **Per-parameter link overrides**: `FitConfig::with_link(param, link)` (or the `links` field, `{"links": {...}}` in JSON) selects any of the 9 named links: identity, log, logit, probit, cloglog, inverse, inverse_square, sqrt and cauchit.
- **R-style formula strings**: `parse_formula_string` and `Formula::from_strings` parse specs such as `"y ~ s(x, bs=\"cr\") + te(a, b) + factor(g) + a:b + offset(e)"` into the same terms the builder API produces.
  New `Term::Offset`, `Term::Factor` (with `Contrast::Treatment` or sum-to-zero contrasts) and `Term::Interaction` variants back them.
- **Missing-data handling**: `NaAction::DropRows` (the default, R's `na.omit`) or `NaAction::Fail`, set with `FitConfig::with_na_action`.
- **Prior weights**: `GamlssModel::fit_weighted`, and a `weights` argument on `fit_with_config` and `json::fit`.
- **Structural likelihoods**: `Censored`, `Truncated`, and `Hurdle` wrapper distributions (+ the `CensorStatus` enum) over any base family, in `src/distributions/{censored,truncated,hurdle}.rs`.
  Censoring swaps the density for a survival/interval probability built from the base `cdf`; truncation renormalizes by the in-support mass; hurdle composes a logit-linked zero atom (`xi`) with a zero-truncated base.
- **Finite mixtures**: `MixtureModel` and `fit_mixture` in `src/fitting/mixture.rs` fit a `K`-component mixture by EM, reusing the prior-weighted RS fit as the M-step.
  Both are re-exported at the crate root.
- **`Distribution::cdf_theta_derivatives`**: a new trait hook returning analytic natural-scale `(∂F/∂θ, ∂²F/∂θ²)` per parameter.
  It is implemented for Gaussian μ and σ, Student-t μ and σ, and Gamma μ; every other parameter uses a central-difference fallback.
  The shared helper in `src/distributions/structural.rs` chains the result to the η scale once and drives the censoring and truncation score and observed-information weight.
  (It shipped under the name `cdf_eta_derivatives` with an η-scale contract; see the rename under "Fixed: non-default links" above.)
- **Family serialization**: a `FamilyDescriptor` enum (`src/distributions/descriptor.rs`) and a `Distribution::descriptor` hook.
  `Binomial`, `Ocat`, and the structural wrappers now round-trip through `to_json` / `from_json`, as the stateless families already did.
  `MixtureModel` has its own `to_json` / `from_json`.

- `impl Display for GamlssModel` produces an R-style summary block (convergence status, iteration count, per-parameter EDF + λ values, truncated coefficients head).
- `Clone` derive on `fitting::FittedParameter` so callers can duplicate fitted parameters without a serde round-trip.
- `ParamDiagnostic` now carries `weight_floor_hits: usize` and `step_cap_hits: usize`, counts of observations whose IRLS weight or step was clamped in the final iteration.
  Non-zero values signal a degenerate fit.
- Python bindings reach parity with WASM: new methods `fit_with_config`, `predict_with_se`, `predict_samples`, `fitted_values(param)`, `coefficients(param)` on `glissando.GamlssModel`.
- New `GamlssError::PosteriorNotPositiveDefinite` variant returned when the posterior covariance fails Cholesky factorization.
- `scripts/audit-duplicates.sh` (+ CI job + pre-push hook) enforces an allowlist of duplicate transitive dependencies, failing if a new duplicate appears.
- New test suites: `tests/correctness.rs` (B.1/B.2/B.3 + D.4 coverage), `tests/analytic.rs` (closed-form OLS anchor), `tests/terms.rs` (`Term::Smooth(RandomEffect)` end-to-end).
- A `proptest` in `src/preprocessing` exercises NaN/∞ rejection in `validate_inputs`.
- New Poisson cases for `predict_with_se` and `predict_samples` integration tests covering the non-identity-link path.
- `FlooredLogLink`: a log link with a lower bound on the response-scale value (`μ = max(exp(η), floor)`).
  StudentT's ν uses it with `floor = 2` so the variance `σ²ν/(ν−2)` stays finite as the optimizer explores the heavy-tail region.
- Student-t parity is now validated against R **gamlss `TF()`**, the like-for-like Rigby–Stasinopoulos oracle (same μ/σ/ν parameterization), via the new `benchmark/fit_gamlss.R`, wired into `orchestrate.py` / `run_comparison.sh`.
  This gates μ, σ, ν, EDF, SE and the (unweighted) log-likelihood; mgcv `scat()` is retained only as a loose μ-only cross-method sanity check.

### Changed

- **Breaking:** `GamlssModel::from_json` now returns `(Self, FamilyDescriptor)` instead of `(Self, String)`, and `to_json` embeds a structured `FamilyDescriptor` in place of a bare distribution name.
  Reconstruct the family with `descriptor.build()` (the `json::load` helper does this for you, returning the model plus a boxed distribution as before).

- StudentT initialization is now robust to heavy tails: `μ` seeds from the sample median, `σ` from `1.4826·MAD(y)` (instead of mean / sample SD, which the tails bias), and `ν` from a fixed seed of 5 (a sample-kurtosis seed is biased for regression data and could tip the multi-smooth weighted fit into a degenerate over-smoothed basin).
- StudentT's `ν` now uses a floored log link (`ν ≥ 2`) instead of a plain log link, guaranteeing finite variance; the floor does not bind when the true `ν` is well above 2.
  The reported `nu` default link changes from `log` to `floored-log`.
- `tests/mgcv_reference.rs` validates StudentT against gamlss `TF()` with tight, measured tolerances.
  This replaces the previous mgcv `scat()` comparison, whose parameterization mismatch had forced a 40% `fitted_mu` tolerance and skipped σ/ν/EDF/SE entirely.

- **Breaking.** `GamlssModel.models` is now `IndexMap<String, FittedParameter>` instead of `HashMap<String, FittedParameter>`.
  Iteration over `models`, `predict_samples` output, and `posterior_samples` is now deterministic (`family.parameters()` order).
  Downstream code that uses `model.models["mu"]` works unchanged.
- **Breaking.** `fitting::sample_posterior` returns `Result<Vec<Array1<f64>>, GamlssError>` (was `Vec<Array1<f64>>`).
  Failure now returns `PosteriorNotPositiveDefinite` instead of an empty vector.
- **Breaking.** `GamlssModel::posterior_samples` returns `Result<Vec<Coefficients>, GamlssError>` (was `Vec<Coefficients>`), and also returns `UnknownParameter` if the parameter name is not in the model.
- **Breaking.** `Formula`'s inner `HashMap` field is now `pub(crate)` (was `pub`).
  External callers should use `Formula::new()`, `with_terms`, `add_terms`, or `Deref` for read access.
- **Breaking.** `FitDiagnostics::param_diagnostics` is now `IndexMap<String, ParamDiagnostic>` (was `HashMap`), with the same deterministic iteration order as `GamlssModel.models`.
- `GamlssError::Linalg` payload is now `String` on both the `openblas` and `pure-rust` backends (previously it was typed `ndarray_linalg::error::LinalgError` under openblas), so downstream pattern matching is backend-independent.
- `src/linalg.rs` emits a `compile_error!` with an actionable message when both `openblas` and `pure-rust` features activate (e.g. through Cargo feature unification in workspace builds) instead of a cryptic `E0428: name backend defined multiple times`.

### Fixed

- Smoothing-parameter **bistability**: a P-spline smooth could, rarely and nondeterministically, collapse onto its penalty null space (a straight line, EDF → null dimension) instead of recovering the true curve.
  The $\lambda$-objective is unimodal but has a flat high-$\lambda$ shelf where the smooth sits in its null space.
  Under OpenBLAS the nondeterministic floating-point reduction order occasionally tipped the optimizer onto that shelf, where the gradient vanishes (and Fellner-Schall's multiplicative ratio explodes), trapping it.
  `fitting::scoring::step` now applies a **collapse-guarded restart**: a collapsed smooth re-runs the optimizer from a low-$\lambda$ seed (below the shelf) and keeps whichever $\lambda$ has the better objective (`solver::restart_seed`, `solver::lambda_cost`).
  Comparing objectives preserves a fit whose optimum is in the null space (e.g. a linear truth under an order-2 penalty) while repairing a spuriously collapsed one.
  The fix is criterion-agnostic (REML/GCV/Fellner-Schall) and fires only on collapse, so well-behaved fits are unchanged.
  Investigation notes: because the objective is unimodal, the collapse was never a competing optimum.
  Fellner-Schall, despite having no line search, is *not* immune (it still consumes reduction-order-dependent traces and has a one-way denominator-floor trap), so the fix targets the mechanism rather than swapping the default criterion.

### Performance

- IRLS step caches `μ = link⁻¹(η)` on `FittingParameter` so each Fisher-scoring step no longer re-runs `inv_link` over every parameter's full η vector.
  Saves K length-n allocations per step (K = number of distribution parameters).
- IRLS step builds `z` (working response) and `w` (floored weights) in a single `Zip::for_each` pass instead of four chained `mapv` allocations.
- Pure-rust `linalg::from_dmatrix` uses a single `from_shape_vec` on the transposed nalgebra slice instead of a per-element nested loop.
- `sample_from_cholesky` reuses one length-p Gaussian-draw buffer across all `n_samples` iterations.
- `math::par_zip3_map` uses an iterator chain (`par_iter().zip().zip()`) instead of indexed access, dropping per-element bounds checks on the parallel path.

### Internal

- `src/types.rs` (614 lines) split into `src/types/{newtypes,dataset,formula,mod}.rs`.
- `src/distributions/mod.rs` extracted `links.rs` (Link trait + 3 link types + their tests).
- `src/splines.rs` items (`sum_to_zero_basis`, `kronecker_product`, `row_kronecker_into`, `create_basis_matrix`, `create_penalty_matrix`) narrowed from `pub` to `pub(crate)`.
  They were already in a private module, so the `pub` keyword was misleading.
- The `argmin-math 0.5.1` dependency duplication (`ndarray 0.16` + `ndarray-linalg 0.17` alongside our `0.17`/`0.18`) is documented in `Cargo.toml`.
