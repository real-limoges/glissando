# glissando cookbook

**Scope:** worked, copy-pasteable examples for fitting GAMLSS models with glissando, one page per topic.
**Surfaces:** every recipe is shown in Rust; the quickstart and the family gallery also give the WASM/JS and Python equivalents, since the three surfaces share a wire format and differ only in ergonomics.
**Runnable code:** the snippets here are excerpted from the runnable programs under `examples/` at the repo root (`examples/*.rs` for Rust, `examples/python/` and `examples/wasm/` for the bindings). Run the Rust ones with `cargo run --example <name>`.

This is the practical companion to `mathematics.md`, which derives the algorithm.
The cookbook is for people *using* it.

## The pages

- `cookbook-quickstart.md` fits one model end to end (fit, predict, diagnose) in all three surfaces. Start here.
- `cookbook-families.md` is the distribution gallery: what each family models, how to construct it, and a minimal fit for each.
- `cookbook-mgcv-migration.md` maps R `mgcv` / `gamlss` idioms onto glissando, with the common workflows translated and the current gaps named.

## The shape of every fit

Every model, on every surface, is the same four moving parts.

1. A **response** `y` (a length-`n` vector of `f64`).
2. A **dataset** of predictor columns, all length `n`.
3. A **formula**: one additive predictor per distribution parameter (`mu`, `sigma`, and so on), so a GAMLSS models the *whole* distribution as a function of covariates, not just its mean.
4. A **family** (the distribution), which fixes how many parameters there are and what each one means.

You hand those to `fit`, get back a fitted model, and then `predict`, take `centiles`, pull `quantile_residuals`, or compare models with `gaic` / `lr_test` / `step_gaic`.

## Conventions in these pages

- Rust snippets assume `use glissando::{...};` and build arrays through `glissando::ndarray` (re-exported) to avoid an `ndarray` version mismatch.
- The response `y` is passed separately from the data, so a formula's left-hand side (the `y ~` part of a string formula) is parsed and then ignored.
- A few surface-specific traps (WASM argument order, Python's Binomial taking a per-row list) are called out inline where they bite.
