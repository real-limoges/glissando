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
     *   `{"mu": [{"Intercept": null}, {"Linear": {"col_name": "x"}}]}`.
     *   For a natural cubic regression spline (`bs="cr"`):
     *   `{"mu": [{"Smooth": {"CrSpline1D": {"col_name": "x", "k": 6, "pc": null, "knots": []}}}]}`
     *   (leave `knots` empty — resolved from training data at fit time).
     * - `distribution`: Distribution name (Gaussian, Poisson, StudentT, Gamma, NegativeBinomial, Beta)
     */
    static fit(y_json: string, data_json: string, formula_json: string, distribution: string): WasmGamlssModel;
    /**
     * Fit a GAMLSS model with custom configuration.
     *
     * `config_json` is a JSON object with optional fields:
     * `{"max_iterations": 200, "tolerance": 0.001, "criterion": "reml"}`.
     * `criterion` accepts `"reml"` (default), `"gcv"`, or `"fellner_schall"`.
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
