# glissando Documentation

Reference material for glissando, a Rust GAMLSS engine (Rigby-Stasinopoulos backfitting) shipped as a crate, a WASM/npm package, and a Python extension.

Start with the cookbook for copy-pasteable examples; use `rust-api.md` to look up a specific call, and `mathematics.md` for the exact likelihood, score, or expected information behind a family.

| File | What It Is |
|---|---|
| [`cookbook-overview.md`](cookbook-overview.md) | Landing page: scope, the three surfaces (Rust / WASM-JS / Python), and the core API. |
| [`cookbook-quickstart.md`](cookbook-quickstart.md) | One model end to end (fit, predict, diagnose) on all three surfaces. |
| [`cookbook-families.md`](cookbook-families.md) | Distribution gallery: how to construct and fit each of the 12 base families, the structural wrappers, and mixtures. |
| [`cookbook-mgcv-migration.md`](cookbook-mgcv-migration.md) | R `mgcv` / `gamlss` to glissando migration: family and basis mapping, common workflows, and the current gaps. |
| [`rust-api.md`](rust-api.md) | Rust API reference: terms and formula strings, `FitConfig`, results, prediction, diagnostics, model selection, and errors. |
| [`bindings.md`](bindings.md) | Model serialization, the `glissando::json` facade for embedding behind another FFI, and the Python API. |
| [`mathematics.md`](mathematics.md) | Algorithm derivations: per-family log-likelihoods, score functions, expected information, CDF/quantile, and the backfitting and penalty machinery. Built to `mathematics.pdf` by `build-math-pdf.sh`. |
