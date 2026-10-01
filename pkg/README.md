# glissando

GAMLSS (Generalized Additive Models for Location, Scale, and Shape) for JavaScript, compiled from the [glissando](https://github.com/real-limoges/glissando) Rust crate to WebAssembly.

GAMLSS gives every distribution parameter its own regression: the mean, the variance, the skew, and the tail weight can each depend on your predictors through linear terms, factors, and penalized smooths.
Smoothing parameters are chosen automatically (REML by default).

This package is built for the default wasm-pack (bundler) target, so Vite, webpack, and Rollup consume it directly.
It is not yet published to the npm registry; depend on the repository's `pkg/` directory instead.

## Quick start

Everything crosses the boundary as a JSON string: inputs are `JSON.stringify`'d and outputs are `JSON.parse`'d.

```js
import { WasmGamlssModel } from "glissando";

const y = JSON.stringify([2.1, 4.0, 5.9, 8.1, 10.0]);
const data = JSON.stringify({ x: [1, 2, 3, 4, 5] });
const formula = JSON.stringify({ mu: "y ~ s(x, k=4)", sigma: "~ 1" });

// Note the argument order: y comes before data.
const model = WasmGamlssModel.fit(y, data, formula, "Gaussian");
console.log("converged:", model.converged());

const newData = JSON.stringify({ x: [6, 7] });
const preds = JSON.parse(model.predict(newData));          // { mu: [...], sigma: [...] }
const withSe = JSON.parse(model.predictWithSe(newData));   // { mu: { fitted, eta, se_eta }, ... }
const draws = JSON.parse(model.predictSamples(newData, 500, 42n));  // seeds are BigInt
```

## Configuration

```js
const counts = JSON.stringify([0, 2, 3, 5, 9]);
const countFormula = JSON.stringify({ mu: "y ~ x" });
const config = JSON.stringify({
  max_iterations: 200,
  tolerance: 0.001,
  criterion: "reml",            // or "gcv", "fellner_schall"
  links: { mu: "sqrt" },        // per-parameter link override
});
const model = WasmGamlssModel.fitWithConfig(counts, data, countFormula, "Poisson", config);
```

Both `fit` and `fitWithConfig` accept an optional trailing JSON array of per-observation prior weights.

## Families

`Gaussian`, `Poisson`, `StudentT`, `Gamma`, `NegativeBinomial`, `Beta`, `Weibull`, `BCCG`, `BCT`, and `BCPE`, selected by name.
`Binomial`, `Ocat`, and the censored / truncated / hurdle wrappers carry state a name cannot express, so they are available from Rust and Python but not from this package.

## Formulas

Each parameter takes an R/mgcv-style formula string: `x` (linear), `s(x)` (P-spline), `s(x, bs="cr")` (cubic regression spline), `s(g, bs="re")` (random effect), `te(x, z)` (tensor product), `factor(g)`, `offset(e)`, and `a:b` / `a*b` interactions.

## Other methods

- `toJson()` / `WasmGamlssModel.fromJson(json)`: persist and reload a fitted model.
- `fittedValues(param)`, `coefficients(param)`, `designMatrix(data, param)`, `covarianceMatrix(param)`, `termIndexMap(param)`, `diagnosticsJson()`.
- `quantileResiduals(y, seed)`, `centiles(data, percentiles)`, `quantilePrediction(data, p)`.
- `gaic(y, k)` (returns `{"gaic": value}`), `lrTest(biggerModelJson, y)`, and the static `WasmGamlssModel.stepGaic(...)`.

## License

MIT
