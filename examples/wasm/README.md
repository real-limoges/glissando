# WASM / JavaScript examples

Runnable JavaScript version of the cookbook quickstart (`docs/cookbook/`), against the `glissando` npm package (the committed `pkg/`).

## The target-build wrinkle

`pkg/` is built for the default (bundler) target, which a bundler (Vite, webpack, Rollup) consumes directly but plain Node cannot `import`.
To run these scripts under Node, rebuild for the nodejs target:

```bash
wasm-pack build --no-default-features --features wasm --target nodejs --out-dir pkg-node
```

Then either import from that output directly, or link it as the `glissando` package so the bare `import { WasmGamlssModel } from "glissando"` resolves:

```bash
cd pkg-node && npm link && cd ..
npm link glissando            # in this examples/wasm directory, or your project
```

Under a bundler instead, install the published package (`npm install glissando`) and the same import works with no rebuild.

## Run

```bash
node examples/wasm/quickstart.mjs
```

## The shape of the API

Everything crosses the boundary as a JSON string: inputs are `JSON.stringify`'d, outputs are `JSON.parse`'d.

- `data` is `{ "colname": [numbers] }`; `y` is a plain `[numbers]`.
- `formula` is `{ "mu": "y ~ s(x)", "sigma": "~ 1" }` (a formula string per parameter).
- `predict(dataJson)` returns `{ "mu": [...], "sigma": [...] }`.

Two traps specific to this surface:

1. The static `fit` / `fitWithConfig` take `y` **before** `data`, the reverse of the Rust and Python surfaces.
2. Seed arguments are BigInt (`42n`), not `Number`.
