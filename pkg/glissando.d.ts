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
    /**
     * Returns the `p × p` posterior covariance matrix for `param`
     * as a JSON array of rows.
     */
    covarianceMatrix(param: string): string;
    /**
     * Returns the linear-predictor design matrix X for `data_json` and `param`
     * as a JSON array of rows (`[[col0, col1, ...], ...]`).
     *
     * Equivalent to mgcv's `predict(type="lpmatrix")`.
     */
    designMatrix(data_json: string, param: string): string;
    diagnosticsJson(): string;
    /**
     * Fit a GAMLSS model. Wire formats are documented on [`crate::json`].
     *
     * `weights_json` is an optional JSON array of per-observation prior weights,
     * e.g. `"[0.5, 1.0, 1.5]"`.  Pass `null` / `undefined` for unweighted fitting.
     */
    static fit(y_json: string, data_json: string, formula_json: string, distribution: string, weights_json?: string | null): WasmGamlssModel;
    static fitWithConfig(y_json: string, data_json: string, formula_json: string, distribution: string, config_json: string, weights_json?: string | null): WasmGamlssModel;
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
     *
     * Pass an integer `seed` for reproducible output; omit or pass `null`/`undefined`
     * for non-deterministic sampling.
     */
    predictSamples(data_json: string, n_samples: number, seed?: bigint | null): string;
    predictWithSe(data_json: string): string;
    /**
     * Returns the term → coefficient column block map for `param` as a JSON
     * object `{"term_name": [first, last], ...}`.
     */
    termIndexMap(param: string): string;
    toJson(): string;
}
