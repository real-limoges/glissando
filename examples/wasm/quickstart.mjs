// Quickstart in JavaScript via the WASM bindings: fit, predict, diagnose.
//
// See ./README.md for setup. In short, the committed pkg/ is a bundler-target
// build; for plain Node, rebuild for the nodejs target and point the import at
// that output (this file imports the bare package name "glissando").
//
//   node examples/wasm/quickstart.mjs

import { WasmGamlssModel } from "glissando";

const n = 120;
const x = Array.from({ length: n }, (_, i) => i * 0.1);
const y = x.map((v) => Math.sin(v) + 0.1 * v);

// One additive predictor per parameter.
const formula = { mu: "y ~ s(x)", sigma: "~ x" };

// NOTE: the static fit takes y FIRST, then the data (the reverse of the Rust
// and Python surfaces). Every argument and return value is a JSON string.
const model = WasmGamlssModel.fit(
  JSON.stringify(y),
  JSON.stringify({ x }),
  JSON.stringify(formula),
  "Gaussian",
);

console.log("converged:", model.converged());

const preds = JSON.parse(model.predict(JSON.stringify({ x })));
console.log("mu[0..3]    =", preds.mu.slice(0, 3));
console.log("sigma[0..3] =", preds.sigma.slice(0, 3));

// quantileResiduals takes a JSON y and an optional BigInt seed.
const resid = JSON.parse(model.quantileResiduals(JSON.stringify(y), 42n));
const meanResid = resid.reduce((a, b) => a + b, 0) / resid.length;
console.log("mean quantile residual (~0):", meanResid.toFixed(4));

// k = 2 is AIC, k = ln(n) is BIC.
console.log("AIC =", model.gaic(JSON.stringify(y), 2.0).toFixed(2));
console.log("BIC =", model.gaic(JSON.stringify(y), Math.log(n)).toFixed(2));
