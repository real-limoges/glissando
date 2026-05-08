/* tslint:disable */
/* eslint-disable */

/**
 * WASM wrapper for GAMLSS models.
 *
 * Supports both fitting models in the browser and loading pre-fitted models
 * serialized via `GamlssModel::to_json()`.
 */
export class WasmGamlssModel {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    coefficients(param: string): Float64Array;
    converged(): boolean;
    diagnosticsJson(): string;
    /**
     * Fit a GAMLSS model in the browser.
     *
     * - `y_json`: Response variable as a JSON array, e.g. `[1.0, 2.0, 3.0]`
     * - `data_json`: Predictor data as JSON object, e.g. `{"x": [1.0, 2.0], "z": [3.0, 4.0]}`
     * - `formula_json`: Formula mapping parameter names to terms, e.g.
     *   `{"mu": [{"Intercept": null}, {"Linear": {"col_name": "x"}}]}`
     * - `distribution`: Distribution name (Gaussian, Poisson, StudentT, Gamma, NegativeBinomial, Beta)
     */
    static fit(y_json: string, data_json: string, formula_json: string, distribution: string): WasmGamlssModel;
    /**
     * Fit a GAMLSS model with custom configuration.
     *
     * `config_json` is a JSON object with optional fields:
     * `{"max_iterations": 200, "tolerance": 0.001}`
     */
    static fitWithConfig(y_json: string, data_json: string, formula_json: string, distribution: string, config_json: string): WasmGamlssModel;
    fittedValues(param: string): Float64Array;
    static fromJson(json: string): WasmGamlssModel;
    /**
     * Input/output are JSON: `{"col": [values]}` → `{"param": [predictions]}`.
     */
    predict(data_json: string): string;
    /**
     * Generate prediction samples by sampling from the posterior distribution of coefficients.
     *
     * Returns JSON: `{"mu": [[s1_v1, s1_v2, ...], [s2_v1, ...], ...], "sigma": [...]}`
     * where each inner array is one sample's predictions across all observations.
     */
    predictSamples(data_json: string, n_samples: number): string;
    predictWithSe(data_json: string): string;
    toJson(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmgamlssmodel_free: (a: number, b: number) => void;
    readonly wasmgamlssmodel_coefficients: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmgamlssmodel_converged: (a: number) => number;
    readonly wasmgamlssmodel_diagnosticsJson: (a: number) => [number, number, number, number];
    readonly wasmgamlssmodel_fit: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly wasmgamlssmodel_fitWithConfig: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number];
    readonly wasmgamlssmodel_fittedValues: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmgamlssmodel_fromJson: (a: number, b: number) => [number, number, number];
    readonly wasmgamlssmodel_predict: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmgamlssmodel_predictSamples: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly wasmgamlssmodel_predictWithSe: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmgamlssmodel_toJson: (a: number) => [number, number, number, number];
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
