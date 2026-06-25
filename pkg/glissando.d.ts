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
    /**
     * Response-scale centile curves for `data_json`, as `{"C<pct>": [..], ...}`.
     * `percentiles_json` is a JSON array of centile levels in percent.
     */
    centiles(data_json: string, percentiles_json: string): string;
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
     * Generalized AIC at penalty `k` against the response `y` this model was fit
     * on. Returns `{"gaic": value}`. `k = 2` ≡ AIC, `k = ln(n)` ≡ BIC.
     */
    gaic(y_json: string, k: number): string;
    /**
     * Likelihood-ratio test treating `self` as the nested (small) model and the
     * model serialized in `bigger_model_json` as the alternative (big) model.
     * Returns `{lr_stat, df, p_value}`. Errors if the pair is mis-ordered.
     */
    lrTest(bigger_model_json: string, y_json: string): string;
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
     * Per-observation quantile prediction; `p_json` is a JSON array of per-row
     * centile levels in `(0,1)`. Returns a JSON array of predicted responses.
     */
    quantilePrediction(data_json: string, p_json: string): string;
    /**
     * Randomized normalized quantile residuals against the response `y` the
     * model was fit on, as a JSON array. `seed` makes the discrete-family
     * randomization reproducible; pass `null`/`undefined` for an unseeded draw.
     */
    quantileResiduals(y_json: string, seed?: bigint | null): string;
    /**
     * Stepwise term selection by GAIC(`k`). Returns `{trace, formula, model}`
     * where `model` is in the same wire form `fromJson` accepts. `scope_json` is
     * a list of `{"param": "...", "candidates": [<term>, ...]}`; `direction` is
     * `"forward"`, `"backward"`, or `"both"`.
     */
    static stepGaic(y_json: string, data_json: string, distribution: string, start_formula_json: string, scope_json: string, k: number, direction: string, config_json?: string | null): string;
    /**
     * Returns the term → coefficient column block map for `param` as a JSON
     * object `{"term_name": [first, last], ...}`.
     */
    termIndexMap(param: string): string;
    toJson(): string;
}
