# glissando documentation

Reference material for using glissando, a Rust GAMLSS engine (Rigby-Stasinopoulos backfitting) shipped as a crate, a WASM/npm package, and a Python extension.

Start with the cookbook for copy-pasteable examples; use `mathematics.md` when you need the exact likelihood, score, or expected information behind a family.

| file | what it is |
|---|---|
| [`cookbook-overview.md`](cookbook-overview.md) | Landing page: scope, the three surfaces (Rust / WASM-JS / Python), and the core API. |
| [`cookbook-quickstart.md`](cookbook-quickstart.md) | One model end to end (fit, predict, diagnose) on all three surfaces. |
| [`cookbook-families.md`](cookbook-families.md) | Distribution gallery: how to construct and fit each of the 12 base families, the structural wrappers, and mixtures. |
| [`cookbook-mgcv-migration.md`](cookbook-mgcv-migration.md) | R `mgcv` / `gamlss` to glissando migration: family and basis mapping, common workflows, and the current gaps. |
| [`smooth-basis-implementation.md`](smooth-basis-implementation.md) | Implementation guide for the `SmoothBasis` trait refactor (one struct and trait impl per smooth type); the prerequisite for the GEO guides. Planned, not yet built. |
| [`geo1-mrf-implementation.md`](geo1-mrf-implementation.md) | Implementation guide for the areal Markov random field smooth (`s(region, bs="mrf")`), decision 0009 GEO-1. Planned, not yet built. |
| [`geo2-tp-implementation.md`](geo2-tp-implementation.md) | Implementation guide for the 2D thin-plate regression spline (`s(x, y, bs="tp")`), decision 0009 GEO-2. Planned, not yet built. |
| [`geo3-parity-implementation.md`](geo3-parity-implementation.md) | Implementation guide for the three spatial mgcv parity scenarios, decision 0009 GEO-3. Planned, not yet built. |
| [`mathematics.md`](mathematics.md) | Algorithm derivations: per-family log-likelihoods, score functions, expected information, CDF/quantile, and the backfitting and penalty machinery. Built to `mathematics.pdf` by `build-math-pdf.sh`. |
