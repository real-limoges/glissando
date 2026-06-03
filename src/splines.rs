//! B-spline basis construction and penalty matrices for P-spline smoothing,
//! plus natural cubic regression spline (mgcv `bs="cr"`) basis and penalty.
//!
//! Provides routines for building B-spline / natural-cubic-spline basis matrices,
//! penalty matrices, and Kronecker product utilities used by smooth terms.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayViewMut1};

/// Householder-built orthonormal basis (k × (k−1)) of the null-space of `1_k`.
///
/// Used as a sum-to-zero reparameterization for smooth bases that exhibit the
/// partition-of-unity property (P-spline: `B · 1_k = 1_n`). When an [`Intercept`]
/// term sits alongside such a smooth on the same parameter, the design matrix
/// `[1 | B]` is rank-deficient because `1_n ∈ col(B)`; replacing `B` with `B · Z`
/// and the penalty `S` with `Z' · S · Z` removes the constant direction from the
/// smooth and restores identifiability.
///
/// `Z` depends only on `k`, not on training data, so prediction reapplies the
/// same constraint deterministically.
///
/// [`Intercept`]: crate::Term::Intercept
pub(crate) fn sum_to_zero_basis(k: usize) -> Array2<f64> {
    assert!(k >= 2, "sum-to-zero basis requires k >= 2 (got {})", k);

    // Householder reflector that maps 1_k onto −√k · e_1:
    //   v = 1_k + √k · e_1   (sign chosen so v_0 = 1 + √k never vanishes),
    //   H = I − 2·v·v'/(v'·v).
    // The remaining columns H[:, 1..] span the orthogonal complement of 1_k and
    // are orthonormal because H is orthogonal.
    let sqrt_k = (k as f64).sqrt();
    let mut v = Array1::ones(k);
    v[0] += sqrt_k;
    let v_norm_sq: f64 = v.iter().map(|x: &f64| x * x).sum();
    let factor = 2.0 / v_norm_sq;

    let mut z = Array2::<f64>::zeros((k, k - 1));
    for j in 1..k {
        for i in 0..k {
            let kron = if i == j { 1.0 } else { 0.0 };
            z[[i, j - 1]] = kron - factor * v[i] * v[j];
        }
    }
    z
}

/// Compute the Kronecker product of two matrices: C = A ⊗ B.
///
/// If A is (m × n) and B is (p × q), the result is (mp × nq).
/// Used for constructing tensor product basis matrices.
pub(crate) fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();

    let mut c = Array2::<f64>::zeros((m * p, n * q));

    for i in 0..m {
        for j in 0..n {
            let a_scalar = a[[i, j]];
            let mut block = c.slice_mut(s![i * p..(i + 1) * p, j * q..(j + 1) * q]);
            block.assign(&(b * a_scalar));
        }
    }
    c
}

/// Compute row-wise Kronecker product into a pre-allocated output buffer.
///
/// Given vectors a (length m) and b (length n), computes their Kronecker product
/// (length m*n) in-place. Used for efficient tensor product basis evaluation.
#[inline]
pub(crate) fn row_kronecker_into(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    mut out: ArrayViewMut1<f64>,
) {
    let len_b = b.len();
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i * len_b + j] = ai * bj;
        }
    }
}

/// Create a B-spline basis matrix for the given data.
///
/// Constructs an (n_obs × n_splines) matrix where each row contains the
/// B-spline basis function values evaluated at that observation's x value.
/// Uses clamped knots with interior knots placed at data quantiles.
///
/// # Arguments
/// * `x` - Covariate values (n_obs length)
/// * `n_splines` - Number of basis functions (typically 10-20)
/// * `degree` - Polynomial degree (typically 3 for cubic splines)
pub(crate) fn create_basis_matrix(x: &Array1<f64>, n_splines: usize, degree: usize) -> Array2<f64> {
    let n_obs = x.len();

    if n_splines <= degree {
        return Array2::<f64>::zeros((n_obs, n_splines));
    }

    let knots = select_knots(x, n_splines, degree);
    let mut basis_matrix = Array2::<f64>::zeros((n_obs, n_splines));

    let mut basis_buf = vec![0.0; degree + 1];
    let mut left_buf = vec![0.0; degree + 1];
    let mut right_buf = vec![0.0; degree + 1];

    for (row_idx, &x_i) in x.iter().enumerate() {
        let span_index = find_knot_span(x_i, degree, n_splines, &knots);
        evaluate_basis_functions_into(
            x_i,
            span_index,
            degree,
            &knots,
            &mut basis_buf,
            &mut left_buf,
            &mut right_buf,
        );

        if span_index >= degree {
            for (j, &val) in basis_buf.iter().enumerate() {
                let col_idx = span_index - degree + j;
                if col_idx < n_splines {
                    basis_matrix[[row_idx, col_idx]] = val;
                }
            }
        }
    }
    basis_matrix
}

/// Equally-spaced knots extended `degree` beyond the data range (the Eilers–Marx
/// P-spline layout).
///
/// A difference penalty (`create_penalty_matrix`) only approximates a roughness
/// penalty on the fitted function when the knots are **equally spaced** — so the
/// P-spline penalty and the basis must share that assumption. We place
/// `safe_n_splines + degree + 1` uniform knots with spacing
/// `dx = (max − min) / (safe_n_splines − degree)` such that `t[degree] = min` and
/// `t[safe_n_splines] = max`, leaving `degree` knots beyond each end. Over the
/// data range the B-spline basis is then full-support (partition-of-unity holds),
/// which keeps the sum-to-zero reparameterization valid. This matches mgcv's
/// `bs="ps"` construction.
///
/// Knots depend only on the data range (`min`, `max`) and `(n_splines, degree)`,
/// so prediction rebuilds an identical basis deterministically.
fn select_knots(x: &Array1<f64>, n_splines: usize, degree: usize) -> Vec<f64> {
    let min_val = x
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    let max_val = x
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    let safe_n_splines = n_splines.max(degree + 1);
    let num_total_knots = safe_n_splines + degree + 1;

    // `safe_n_splines > degree` always holds, so the denominator is ≥ 1.
    // Degenerate constant-x (or non-finite) input falls back to unit spacing so
    // the basis stays finite rather than dividing by a zero range.
    let range = max_val - min_val;
    let (origin, dx) = if range.is_finite() && range > 0.0 {
        (min_val, range / (safe_n_splines - degree) as f64)
    } else {
        let origin = if min_val.is_finite() { min_val } else { 0.0 };
        (origin, 1.0)
    };

    (0..num_total_knots)
        .map(|j| origin + (j as f64 - degree as f64) * dx)
        .collect()
}

/// Knot span for `x` on a uniform knot vector: the index `i` with
/// `t[i] ≤ x < t[i+1]`, clamped to `[degree, n_splines-1]` so the `degree+1`
/// non-zero basis functions map to valid columns `[i-degree, i]`.
///
/// `x` below the data range clamps to `degree`; at or above the top clamps to
/// `n_splines-1` (the basis extrapolates flat at the edges).
fn find_knot_span(x: f64, degree: usize, n_splines: usize, knots: &[f64]) -> usize {
    if knots.len() <= degree + 1 || n_splines == 0 {
        return degree.min(n_splines.saturating_sub(1));
    }

    let dx = knots[degree + 1] - knots[degree];
    if !dx.is_finite() || dx <= 0.0 {
        return degree.min(n_splines - 1);
    }

    // Span index in f64 to avoid usize underflow for x below the origin.
    let raw = degree as f64 + ((x - knots[degree]) / dx).floor();
    let lo = degree as f64;
    let hi = (n_splines - 1) as f64;
    raw.clamp(lo, hi) as usize
}

/// De Boor-Cox recursion for B-spline basis evaluation.
fn evaluate_basis_functions_into(
    x: f64,
    i: usize,
    degree: usize,
    knots: &[f64],
    basis: &mut [f64],
    left: &mut [f64],
    right: &mut [f64],
) {
    basis.iter_mut().for_each(|v| *v = 0.0);
    basis[0] = 1.0;

    for j in 1..=degree {
        let mut saved = 0.0;
        for r in 0..j {
            let left_idx = (i + 1).saturating_sub(j).saturating_add(r);
            let right_idx = i + r + 1;

            if right_idx < knots.len() {
                left[j] = x - knots[left_idx];
                right[j] = knots[right_idx] - x;

                let denom = right[r + 1] + left[j - r];
                if denom.abs() > 1e-12 {
                    let term = basis[r] / denom;
                    basis[r] = saved + right[r + 1] * term;
                    saved = left[j - r] * term;
                } else {
                    basis[r] = saved;
                    saved = 0.0;
                }
            }
        }
        basis[j] = saved;
    }
}

/// P-spline penalty S = D'D.  Order 2 approximates the integral of the squared
/// second derivative (exact for the natural cubic spline, approximate for B-splines).
pub(crate) fn create_penalty_matrix(n_splines: usize, order: usize) -> Array2<f64> {
    let n_rows_d = n_splines.saturating_sub(order);
    if n_rows_d == 0 {
        return Array2::<f64>::zeros((n_splines, n_splines));
    }

    let mut d_matrix = Array2::<f64>::zeros((n_rows_d, n_splines));
    match order {
        1 => {
            for i in 0..n_rows_d {
                d_matrix[[i, i]] = 1.0;
                d_matrix[[i, i + 1]] = -1.0;
            }
        }
        _ => {
            for i in 0..n_rows_d {
                d_matrix[[i, i]] = 1.0;
                d_matrix[[i, i + 1]] = -2.0;
                d_matrix[[i, i + 2]] = 1.0;
            }
        }
    }
    d_matrix.t().dot(&d_matrix)
}

// ──────────────────────────────────────────────────────────────────────────
//  Natural Cubic Regression Splines  (mgcv `bs = "cr"`)
// ──────────────────────────────────────────────────────────────────────────

/// Quantile knot placement for natural cubic regression splines.
///
/// Matches mgcv's default: `quantile(unique(x), seq(0, 1, length = k))` using
/// type-7 linear interpolation over the sorted unique finite values.  The first
/// and last knots are always the observed minimum and maximum.
///
/// # Panics
/// Panics if `k < 2`.
pub(crate) fn cr_knots(x: &Array1<f64>, k: usize) -> Vec<f64> {
    assert!(k >= 2, "cr_knots requires k ≥ 2 (got {})", k);
    let mut sorted: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();
    let n = sorted.len();

    if n < 2 {
        // Degenerate: constant x or no finite values — space k knots 1 unit apart.
        let v = if sorted.is_empty() { 0.0 } else { sorted[0] };
        return (0..k).map(|i| v + i as f64).collect();
    }

    (0..k)
        .map(|i| {
            let p = i as f64 / (k - 1) as f64; // fraction in [0, 1]
            let pos = p * (n - 1) as f64; // virtual 0-based index
            let j = pos.floor() as usize;
            let g = pos - j as f64; // fractional part
            if j + 1 >= n {
                sorted[n - 1]
            } else if g < f64::EPSILON {
                sorted[j]
            } else {
                (1.0 - g) * sorted[j] + g * sorted[j + 1]
            }
        })
        .collect()
}

/// Thomas (tridiagonal) algorithm for a **symmetric** tridiagonal system `E x = rhs`.
///
/// * `diag`    — main diagonal, length n (modified in place internally).
/// * `off`     — super-diagonal (= sub-diagonal by symmetry), length n-1.
/// * `rhs`     — right-hand side, length n (modified in place).
///
/// Returns the solution vector.  No pivoting; numerically stable when E is
/// symmetric positive-definite (which is always the case for our E matrix).
fn solve_tridiagonal_sym(diag: &[f64], off: &[f64], rhs: &mut [f64]) -> Vec<f64> {
    let n = diag.len();
    debug_assert_eq!(off.len() + 1, n);
    debug_assert_eq!(rhs.len(), n);
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![rhs[0] / diag[0]];
    }

    let mut d = diag.to_vec();

    // Forward elimination
    for i in 1..n {
        let factor = off[i - 1] / d[i - 1];
        d[i] -= factor * off[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = rhs[n - 1] / d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = (rhs[i] - off[i] * x[i + 1]) / d[i];
    }
    x
}

/// Compute the second-derivative map `G = E⁻¹ D`, where
///
/// * `E` is the `(k-2) × (k-2)` symmetric tridiagonal system for interior
///   second derivatives (natural boundary conditions: `γ₀ = γ_{k-1} = 0`).
/// * `D` is the `(k-2) × k` second-difference operator that maps knot values
///   to the right-hand sides of the second-derivative system.
///
/// Returns a `(k-2) × k` matrix.  `G[r, m]` is the interior second derivative
/// at knot `r+1` when the cardinal basis function for knot `m` equals 1.
fn compute_deriv_map(h: &[f64]) -> Array2<f64> {
    // h has length k-1; km2 = k-2 interior knots
    let km2 = h.len().saturating_sub(1);
    let k = h.len() + 1;
    if km2 == 0 {
        return Array2::zeros((0, k));
    }

    // E diag:    E[r,r]   = (h[r] + h[r+1]) / 3
    // E off-diag: E[r,r+1] = h[r+1] / 6  (symmetric)
    let diag_e: Vec<f64> = (0..km2).map(|r| (h[r] + h[r + 1]) / 3.0).collect();
    let off_e: Vec<f64> = (0..km2.saturating_sub(1)).map(|r| h[r + 1] / 6.0).collect();

    // D[r, r]   =  1/h[r]
    // D[r, r+1] = -(1/h[r] + 1/h[r+1])
    // D[r, r+2] =  1/h[r+1]
    // Solve E G[:, m] = D[:, m] for each column m.
    let mut g = Array2::zeros((km2, k));
    for m in 0..k {
        let mut rhs: Vec<f64> = (0..km2)
            .map(|r| {
                let mut v = 0.0;
                if m == r {
                    v += 1.0 / h[r];
                }
                if m == r + 1 {
                    v -= 1.0 / h[r] + 1.0 / h[r + 1];
                }
                if m == r + 2 {
                    v += 1.0 / h[r + 1];
                }
                v
            })
            .collect();
        let sol = solve_tridiagonal_sym(&diag_e, &off_e, &mut rhs);
        for r in 0..km2 {
            g[[r, m]] = sol[r];
        }
    }
    g
}

/// Evaluate the CR basis at a single point `x`.
///
/// Returns a length-k row vector: entry `m` is the `m`-th cardinal natural-cubic-
/// spline basis function evaluated at `x`.  Points outside `[knots[0], knots[k-1]]`
/// are linearly extrapolated (zero second derivative, so the curve stays straight).
///
/// # Arguments
/// * `knots` — knot locations, length k ≥ 2
/// * `g`     — second-derivative map `E⁻¹ D`, shape `(k-2) × k`
pub(crate) fn eval_cr_row(x: f64, knots: &[f64], g: &Array2<f64>) -> Array1<f64> {
    let k = knots.len();
    debug_assert!(k >= 2);
    let h: Vec<f64> = knots.windows(2).map(|w| w[1] - w[0]).collect();
    let mut row = Array1::zeros(k);

    if x <= knots[0] {
        // Cardinal value at left boundary: f_m(knots[0]) = δ_{m,0}.
        row[0] = 1.0;
        if x < knots[0] {
            // Linear extension: f_m(x) = f_m(knots[0]) + (x − knots[0]) · slope_m.
            // slope_m = (δ_{m,1} − δ_{m,0})/h[0] − h[0]/6 · G[0, m]
            // (G[0,m] = γ₁ for cardinal m; zero when k = 2, no interior knots.)
            let dist = x - knots[0]; // < 0
            let h0 = h[0];
            for m in 0..k {
                let g1m = if k > 2 { g[[0, m]] } else { 0.0 };
                let slope = if m == 0 {
                    -1.0 / h0 - h0 / 6.0 * g1m
                } else if m == 1 {
                    1.0 / h0 - h0 / 6.0 * g1m
                } else {
                    -h0 / 6.0 * g1m
                };
                row[m] += dist * slope;
            }
        }
    } else if x >= knots[k - 1] {
        // Cardinal value at right boundary: f_m(knots[k-1]) = δ_{m,k-1}.
        row[k - 1] = 1.0;
        if x > knots[k - 1] {
            // slope_m = (δ_{m,k-1} − δ_{m,k-2})/h[k-2] + h[k-2]/6 · G[k-3, m]
            let dist = x - knots[k - 1]; // > 0
            let hkm2 = h[k - 2];
            for m in 0..k {
                let gkm2m = if k > 2 { g[[k - 3, m]] } else { 0.0 };
                let slope = if m == k - 2 {
                    -1.0 / hkm2 + hkm2 / 6.0 * gkm2m
                } else if m == k - 1 {
                    1.0 / hkm2 + hkm2 / 6.0 * gkm2m
                } else {
                    hkm2 / 6.0 * gkm2m
                };
                row[m] += dist * slope;
            }
        }
    } else {
        // Interior: find segment j s.t. knots[j] ≤ x < knots[j+1].
        let j = knots
            .partition_point(|&kj| kj <= x)
            .saturating_sub(1)
            .min(k - 2);
        let hj = h[j];
        let t = ((x - knots[j]) / hj).clamp(0.0, 1.0);

        // Linear interpolation: (1−t)·δ_j + t·δ_{j+1}
        row[j] = 1.0 - t;
        row[j + 1] = t;

        // Hermite second-derivative corrections (Wood 2017, §4.1.2).
        //   c(t) = h²/6 · (−2t + 3t² − t³)   for γ_j  (= 0 at boundaries: j=0)
        //   d(t) = h²/6 · (t³ − t)             for γ_{j+1} (= 0 at boundaries: j=k-2)
        let ct = hj * hj / 6.0 * (-2.0 * t + 3.0 * t * t - t * t * t);
        let dt = hj * hj / 6.0 * (t * t * t - t);

        if j > 0 {
            // γ_j = G[j-1, :] (interior knot index j maps to G row j-1)
            for m in 0..k {
                row[m] += ct * g[[j - 1, m]];
            }
        }
        if j < k - 2 {
            // γ_{j+1} = G[j, :] (interior knot index j+1 maps to G row j)
            for m in 0..k {
                row[m] += dt * g[[j, m]];
            }
        }
    }
    row
}

/// Natural cubic regression spline basis matrix (mgcv `bs = "cr"`).
///
/// Constructs an `(n_obs × k)` matrix where column `m` is the `m`-th cardinal
/// natural-cubic-spline basis function: equals 1 at knot `m`, 0 at all other
/// knots, with zero second derivative outside `[knots[0], knots[k-1]]`
/// (linear extrapolation beyond the data range).
///
/// # Arguments
/// * `x`     — covariate values, length n_obs
/// * `knots` — knot locations pre-computed by [`cr_knots`], length k ≥ 2
pub(crate) fn create_cr_basis_matrix(x: &Array1<f64>, knots: &[f64]) -> Array2<f64> {
    let k = knots.len();
    let n_obs = x.len();
    assert!(k >= 2, "create_cr_basis_matrix requires k ≥ 2 (got {})", k);

    let h: Vec<f64> = knots.windows(2).map(|w| w[1] - w[0]).collect();
    let g = compute_deriv_map(&h);

    let mut basis = Array2::zeros((n_obs, k));
    for (i, &xi) in x.iter().enumerate() {
        let row = eval_cr_row(xi, knots, &g);
        basis.row_mut(i).assign(&row);
    }
    basis
}

/// Apply a point constraint to a CR basis matrix, pinning `f(pc_val) = 0`.
///
/// Subtracts the basis row evaluated at `pc_val` from every column of `basis`.
/// Any linear predictor `basis · β` then satisfies `f(pc_val) = 0` identically
/// for all `β`, making `pc_val` the reference point of the smooth.
/// The penalty matrix is unchanged — the constraint is absorbed into the basis.
pub(crate) fn apply_cr_pc_constraint(basis: &mut Array2<f64>, knots: &[f64], pc_val: f64) {
    let h: Vec<f64> = knots.windows(2).map(|w| w[1] - w[0]).collect();
    let g = compute_deriv_map(&h);
    let b_pc = eval_cr_row(pc_val, knots, &g);
    let k = knots.len();
    for j in 0..k {
        let pc_j = b_pc[j];
        basis.column_mut(j).map_inplace(|v| *v -= pc_j);
    }
}

/// Natural cubic spline penalty: exact integrated squared second derivative.
///
/// Returns the `k × k` positive semi-definite matrix
///
/// ```text
/// S = Dᵀ · E⁻¹ · D
/// ```
///
/// where `E` is the `(k-2) × (k-2)` tridiagonal second-derivative system and
/// `D` is the `(k-2) × k` second-difference operator.  `S` has rank `k-2`
/// (null space: constants and linear functions), matching mgcv's `bs="cr"` penalty.
///
/// # Arguments
/// * `knots` — knot positions pre-computed by [`cr_knots`], length k ≥ 2
pub(crate) fn create_cr_penalty_matrix(knots: &[f64]) -> Array2<f64> {
    let k = knots.len();
    assert!(
        k >= 2,
        "create_cr_penalty_matrix requires k ≥ 2 (got {})",
        k
    );
    let km2 = k.saturating_sub(2);

    let h: Vec<f64> = knots.windows(2).map(|w| w[1] - w[0]).collect();

    if km2 == 0 {
        // k = 2: natural spline is a straight line; second derivative is identically
        // zero, so the penalty is the zero matrix.
        return Array2::zeros((k, k));
    }

    // G = E⁻¹ D, shape (k-2) × k.
    let g = compute_deriv_map(&h);

    // S = Dᵀ G  where D is (k-2)×k sparse (3 non-zeros per row).
    // S[m, l] = Σ_r  D[r, m] · G[r, l]
    let mut s = Array2::<f64>::zeros((k, k));
    for r in 0..km2 {
        let entries = [
            (r, 1.0 / h[r]),
            (r + 1, -(1.0 / h[r] + 1.0 / h[r + 1])),
            (r + 2, 1.0 / h[r + 1]),
        ];
        for (m, d_rm) in entries {
            for l in 0..k {
                s[[m, l]] += d_rm * g[[r, l]];
            }
        }
    }

    // Symmetrize to eliminate floating-point asymmetry.
    let st = s.t().to_owned();
    (s + st) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Array2};
    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;

    // --- sum_to_zero_basis ---

    #[test]
    fn sum_to_zero_basis_has_correct_shape() {
        let z = sum_to_zero_basis(5);
        assert_eq!(z.dim(), (5, 4));
    }

    #[test]
    fn sum_to_zero_basis_is_orthogonal_to_ones() {
        // Z' · 1_k = 0 — every column sums to zero.
        for k in [2usize, 3, 5, 8, 12] {
            let z = sum_to_zero_basis(k);
            for j in 0..z.ncols() {
                let s: f64 = z.column(j).sum();
                assert!(s.abs() < 1e-12, "k={} col {} sum = {}", k, j, s);
            }
        }
    }

    #[test]
    fn sum_to_zero_basis_columns_are_orthonormal() {
        // Z' · Z = I_{k-1}.
        for k in [2usize, 3, 5, 8] {
            let z = sum_to_zero_basis(k);
            let zt_z = z.t().dot(&z);
            let identity = Array2::<f64>::eye(k - 1);
            for i in 0..(k - 1) {
                for j in 0..(k - 1) {
                    assert!(
                        (zt_z[[i, j]] - identity[[i, j]]).abs() < 1e-10,
                        "k={} Z'Z[{},{}]={} (expected {})",
                        k,
                        i,
                        j,
                        zt_z[[i, j]],
                        identity[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn sum_to_zero_constraint_removes_constant_direction() {
        // After centering, B·Z·γ should not be a non-zero constant for any γ ≠ 0.
        // Equivalent statement: 1_k'·Z·γ = 0 always (Z'·1_k = 0), so β = Z·γ is
        // never parallel to 1_k for non-zero γ.
        let k = 6;
        let z = sum_to_zero_basis(k);
        let gamma = Array1::from_vec(vec![0.3, -1.2, 2.1, 0.7, -0.5]);
        let beta = z.dot(&gamma);
        let dot_with_ones: f64 = beta.iter().sum();
        assert!(dot_with_ones.abs() < 1e-12);
    }

    // --- kronecker_product ---

    #[test]
    fn kronecker_product_dimensions() {
        let a = Array2::<f64>::zeros((2, 3));
        let b = Array2::<f64>::zeros((4, 5));
        let c = kronecker_product(&a, &b);
        assert_eq!(c.dim(), (8, 15));
    }

    #[test]
    fn kronecker_product_with_identities_is_identity() {
        let i2 = Array2::<f64>::eye(2);
        let i3 = Array2::<f64>::eye(3);
        let k = kronecker_product(&i2, &i3);
        let i6 = Array2::<f64>::eye(6);
        assert_eq!(k, i6);
    }

    #[test]
    fn kronecker_product_block_structure() {
        let a = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = ndarray::arr2(&[[0.0, 5.0], [6.0, 7.0]]);
        let c = kronecker_product(&a, &b);
        // Top-left 2x2 block = a[0,0] * b
        assert_eq!(c[[0, 0]], 0.0);
        assert_eq!(c[[0, 1]], 5.0);
        assert_eq!(c[[1, 0]], 6.0);
        assert_eq!(c[[1, 1]], 7.0);
        // Top-right 2x2 block = a[0,1] * b
        assert_eq!(c[[0, 2]], 0.0);
        assert_eq!(c[[0, 3]], 10.0);
        // Bottom-right 2x2 block = a[1,1] * b
        assert_eq!(c[[2, 2]], 0.0);
        assert_eq!(c[[3, 3]], 28.0);
    }

    // --- row_kronecker_into ---

    #[test]
    fn row_kronecker_into_matches_naive() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![10.0, 20.0]);
        let mut out = Array1::<f64>::zeros(6);
        row_kronecker_into(a.view(), b.view(), out.view_mut());
        assert_eq!(out.to_vec(), vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn row_kronecker_into_matches_outer_product_flatten(
            a_vec in proptest::collection::vec(-10.0f64..10.0, 1..6),
            b_vec in proptest::collection::vec(-10.0f64..10.0, 1..6),
        ) {
            let a = Array1::from_vec(a_vec.clone());
            let b = Array1::from_vec(b_vec.clone());
            let mut out = Array1::<f64>::zeros(a.len() * b.len());
            row_kronecker_into(a.view(), b.view(), out.view_mut());
            for i in 0..a.len() {
                for j in 0..b.len() {
                    let expected = a_vec[i] * b_vec[j];
                    let actual = out[i * b.len() + j];
                    prop_assert!((expected - actual).abs() < 1e-12);
                }
            }
        }
    }

    // --- create_basis_matrix ---

    #[test]
    fn basis_matrix_shape() {
        let x = Array1::linspace(0.0, 1.0, 50);
        let b = create_basis_matrix(&x, 10, 3);
        assert_eq!(b.dim(), (50, 10));
    }

    #[test]
    fn knots_are_uniform_and_bracket_range() {
        // P-spline layout: equally-spaced knots with `t[degree] = min`,
        // `t[n_splines] = max`, and `degree` extra knots beyond each end. Equal
        // spacing is what makes the difference penalty a valid roughness penalty.
        let x = Array1::linspace(0.0, 4.0, 200);
        let (n_splines, degree) = (15usize, 3usize);
        let knots = select_knots(&x, n_splines, degree);

        assert_eq!(knots.len(), n_splines + degree + 1);
        assert!(
            (knots[degree] - 0.0).abs() < 1e-9,
            "t[degree] should be min"
        );
        assert!(
            (knots[n_splines] - 4.0).abs() < 1e-9,
            "t[n_splines] should be max"
        );

        let dx = knots[1] - knots[0];
        for w in knots.windows(2) {
            assert!(
                (w[1] - w[0] - dx).abs() < 1e-9,
                "knots must be equally spaced (got {} vs {})",
                w[1] - w[0],
                dx
            );
        }
        // `degree` knots extend strictly below min and above max.
        assert!(knots[degree - 1] < 0.0);
        assert!(knots[n_splines + 1] > 4.0);
    }

    #[test]
    fn knots_handle_constant_x_without_nan() {
        let x = Array1::from_elem(10, 2.5);
        let knots = select_knots(&x, 8, 3);
        assert!(knots.iter().all(|k| k.is_finite()));
    }

    #[test]
    fn basis_matrix_non_negative() {
        let x = Array1::linspace(0.0, 1.0, 50);
        let b = create_basis_matrix(&x, 10, 3);
        for &v in b.iter() {
            assert!(
                v >= -1e-12,
                "basis value {} negative beyond fp tolerance",
                v
            );
        }
    }

    #[test]
    fn basis_matrix_partition_of_unity_in_interior() {
        // B-spline basis sums to 1 in the interior of the knot range.
        // Use evenly spaced x and skip the first/last few rows where boundary effects apply.
        let x = Array1::linspace(0.0, 1.0, 100);
        let b = create_basis_matrix(&x, 12, 3);
        for row in b.outer_iter().skip(20).take(60) {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "interior row sum {} not ≈ 1", s);
        }
    }

    #[test]
    fn basis_matrix_degenerate_returns_zeros() {
        // n_splines <= degree → returns zeros.
        let x = Array1::linspace(0.0, 1.0, 5);
        let b = create_basis_matrix(&x, 3, 3);
        assert!(b.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn basis_matrix_handles_constant_x() {
        // All-equal x is degenerate but must not panic or produce NaN.
        let x = Array1::from_elem(10, 0.5);
        let b = create_basis_matrix(&x, 8, 3);
        assert_eq!(b.dim(), (10, 8));
        assert!(b.iter().all(|v| v.is_finite()));
    }

    // --- create_penalty_matrix ---

    fn is_symmetric(m: &Array2<f64>, eps: f64) -> bool {
        let (r, c) = m.dim();
        if r != c {
            return false;
        }
        for i in 0..r {
            for j in 0..r {
                if (m[[i, j]] - m[[j, i]]).abs() > eps {
                    return false;
                }
            }
        }
        true
    }

    fn is_psd(m: &Array2<f64>) -> bool {
        // A matrix M = D'D is automatically PSD; verify by checking x'Mx >= 0 for random x.
        // Cheap stand-in for an eigenvalue computation that doesn't need a linalg backend.
        let n = m.dim().0;
        let mut rng = StdRngStub::new(42);
        for _ in 0..20 {
            let v: Array1<f64> = Array::from_shape_fn(n, |_| rng.next() - 0.5);
            let q = v.dot(&m.dot(&v));
            if q < -1e-9 {
                return false;
            }
        }
        true
    }

    /// Tiny LCG for test-only deterministic numbers (no rand dep needed in the test scope).
    struct StdRngStub {
        state: u64,
    }
    impl StdRngStub {
        fn new(seed: u64) -> Self {
            Self { state: seed.max(1) }
        }
        fn next(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.state >> 33) as f64) / (1u64 << 31) as f64
        }
    }

    #[test]
    fn penalty_matrix_order2_shape_and_symmetry() {
        let p = create_penalty_matrix(10, 2);
        assert_eq!(p.dim(), (10, 10));
        assert!(is_symmetric(&p, 1e-12));
    }

    #[test]
    fn penalty_matrix_order1_shape_and_symmetry() {
        let p = create_penalty_matrix(8, 1);
        assert_eq!(p.dim(), (8, 8));
        assert!(is_symmetric(&p, 1e-12));
    }

    #[test]
    fn penalty_matrix_psd_order2() {
        let p = create_penalty_matrix(10, 2);
        assert!(is_psd(&p));
    }

    #[test]
    fn penalty_matrix_psd_order1() {
        let p = create_penalty_matrix(10, 1);
        assert!(is_psd(&p));
    }

    #[test]
    fn penalty_matrix_order1_constant_in_null_space() {
        // First-order difference penalty has constant vectors in its null space.
        let p = create_penalty_matrix(8, 1);
        let ones = Array1::from_elem(8, 1.0);
        let q = ones.dot(&p.dot(&ones));
        assert!(
            q.abs() < 1e-10,
            "constant should lie in null space, got {}",
            q
        );
    }

    #[test]
    fn penalty_matrix_order2_linear_in_null_space() {
        // Second-order difference penalty has linear vectors in its null space.
        let p = create_penalty_matrix(8, 2);
        let lin: Array1<f64> = Array1::from_iter((0..8).map(|i| i as f64));
        let q = lin.dot(&p.dot(&lin));
        assert!(
            q.abs() < 1e-10,
            "linear should lie in null space, got {}",
            q
        );
    }

    #[test]
    fn penalty_matrix_degenerate_when_order_ge_n_splines() {
        let p = create_penalty_matrix(2, 3);
        assert_eq!(p.dim(), (2, 2));
        assert!(p.iter().all(|&v| v == 0.0));
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn penalty_matrix_always_symmetric(
            n_splines in 4usize..20,
            order in 1usize..3,
        ) {
            let p = create_penalty_matrix(n_splines, order);
            prop_assert_eq!(p.dim(), (n_splines, n_splines));
            for i in 0..n_splines {
                for j in 0..n_splines {
                    prop_assert!((p[[i, j]] - p[[j, i]]).abs() < 1e-12);
                }
            }
        }
    }

    // --- cr_knots ---

    #[test]
    fn cr_knots_endpoints_are_min_and_max() {
        let x = Array1::from_vec(vec![3.0, 1.0, 5.0, 2.0, 8.0]);
        let knots = cr_knots(&x, 4);
        assert_eq!(knots.len(), 4);
        assert!((knots[0] - 1.0).abs() < 1e-12, "first knot should be min");
        assert!((knots[3] - 8.0).abs() < 1e-12, "last knot should be max");
    }

    #[test]
    fn cr_knots_monotone() {
        let x = Array1::linspace(0.0, 10.0, 50);
        for k in [3usize, 5, 8, 12] {
            let knots = cr_knots(&x, k);
            assert_eq!(knots.len(), k);
            for w in knots.windows(2) {
                assert!(w[1] >= w[0], "knots must be non-decreasing");
            }
        }
    }

    #[test]
    fn cr_knots_matches_mgcv_quantiles() {
        // R: quantile(unique(c(1,2,3,5,8,10,12,15)), seq(0,1,length=5))
        // gives: 1.00, 2.75, 6.50, 10.50, 15.00
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
        let knots = cr_knots(&x, 5);
        let expected = [1.0, 2.75, 6.5, 10.5, 15.0];
        for (i, (&got, &exp)) in knots.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "knot[{}]: expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    #[test]
    fn cr_knots_constant_x_no_nan() {
        let x = Array1::from_elem(10, 3.0);
        let knots = cr_knots(&x, 4);
        assert_eq!(knots.len(), 4);
        assert!(knots.iter().all(|v| v.is_finite()));
    }

    // --- create_cr_basis_matrix ---

    #[test]
    fn cr_basis_matrix_shape() {
        let x = Array1::linspace(1.0, 10.0, 30);
        let knots = cr_knots(&x, 6);
        let b = create_cr_basis_matrix(&x, &knots);
        assert_eq!(b.dim(), (30, 6));
    }

    #[test]
    fn cr_basis_matrix_interpolates_at_knots() {
        // Cardinal property: at knot m, column m = 1, all others = 0.
        let x_raw = vec![1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0];
        let k = 5;
        let x = Array1::from_vec(x_raw);
        let knots = cr_knots(&x, k);

        // Evaluate basis at the knot locations themselves.
        let b = create_cr_basis_matrix(&Array1::from_vec(knots.clone()), &knots);
        for m in 0..k {
            for m2 in 0..k {
                let expected = if m == m2 { 1.0 } else { 0.0 };
                assert!(
                    (b[[m, m2]] - expected).abs() < 1e-10,
                    "B[knot {}, col {}] = {} (expected {})",
                    m,
                    m2,
                    b[[m, m2]],
                    expected
                );
            }
        }
    }

    /// Reference values from R/mgcv:
    /// ```r
    /// x <- c(1,2,3,5,8,10,12,15)
    /// sc <- smoothCon(s(x, bs="cr", k=5), data=data.frame(x=x), absorb.cons=FALSE)[[1]]
    /// sc$X  # basis (8×5)   sc$S[[1]]  # penalty (5×5)
    /// ```
    #[test]
    fn cr_basis_matches_mgcv_reference() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
        let knots = cr_knots(&x, 5);
        let b = create_cr_basis_matrix(&x, &knots);

        // mgcv sc$X rounded to 15 decimal places (rows = observations, cols = basis fns)
        #[rustfmt::skip]
        let expected: [[f64; 5]; 8] = [
            [ 1.000000000000000,  0.000000000000000,  0.000000000000000,  0.000000000000000,  0.000000000000000],
            [ 0.361453301592438,  0.677936225452809, -0.048732675403117,  0.010925057074547, -0.001581908716677],
            [-0.082647539952183,  1.047388728101450,  0.043551878416733, -0.009697183656539,  0.001404117090540],
            [-0.189872188966187,  0.602393140268565,  0.675593431483579, -0.103033220083050,  0.014918837297093],
            [ 0.071590357547322, -0.202998436067519,  0.794072645337863,  0.385521855731723, -0.048186422549390],
            [ 0.017553793884485, -0.049774757770228,  0.136795213602617,  0.933606300595613, -0.038180550312487],
            [-0.025885779000162,  0.073400564453792, -0.198845476280357,  0.922984564405855,  0.228346126420872],
            [ 0.000000000000000,  0.000000000000000,  0.000000000000000,  0.000000000000000,  1.000000000000000],
        ];

        for (i, row_ref) in expected.iter().enumerate() {
            for (j, &ref_val) in row_ref.iter().enumerate() {
                let got = b[[i, j]];
                assert!(
                    (got - ref_val).abs() < 1e-6,
                    "B[{}, {}]: expected {:.9}, got {:.9}",
                    i,
                    j,
                    ref_val,
                    got
                );
            }
        }
    }

    #[test]
    fn cr_basis_linear_extrapolation_outside_knots() {
        // Reference: PredictMat for x=c(-2,0,17,20) with the same k=5 smooth.
        let x_train = Array1::from_vec(vec![1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
        let knots = cr_knots(&x_train, 5);
        let x_pred = Array1::from_vec(vec![-2.0, 0.0, 17.0, 20.0]);
        let b = create_cr_basis_matrix(&x_pred, &knots);

        #[rustfmt::skip]
        let expected: [[f64; 5]; 4] = [
            [ 3.013266461737583, -2.188728900393679,  0.217081917704794, -0.048666163332075,  0.007046684283377],
            [ 1.671088820579194, -0.729576300131227,  0.072360639234931, -0.016222054444025,  0.002348894761126],
            [ 0.031062934800194, -0.088080677344550,  0.238614571536429, -0.752025921731471,  1.570429092739398],
            [ 0.077657337000485, -0.220201693361376,  0.596536428841072, -1.880064804328677,  2.426072731848496],
        ];

        for (i, row_ref) in expected.iter().enumerate() {
            for (j, &ref_val) in row_ref.iter().enumerate() {
                let got = b[[i, j]];
                assert!(
                    (got - ref_val).abs() < 1e-6,
                    "B_pred[{}, {}]: expected {:.9}, got {:.9}",
                    i,
                    j,
                    ref_val,
                    got
                );
            }
        }
    }

    // --- create_cr_penalty_matrix ---

    #[test]
    fn cr_penalty_matches_mgcv_reference() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
        let knots = cr_knots(&x, 5);
        let s = create_cr_penalty_matrix(&knots);

        // mgcv sc$S[[1]] with scale.penalty=FALSE (5×5, unscaled = ∫[f'']² dx)
        // Note: mgcv's default smoothCon applies a data-dependent rescaling
        // (||X||_inf² / norm(S,"1")) that depends on the basis matrix X, not just
        // knots. Our penalty is the mathematically correct unscaled integral; the
        // smoothing parameter λ adapts freely during REML/GCV fitting regardless.
        #[rustfmt::skip]
        let expected: [[f64; 5]; 5] = [
            [ 0.195252733029792, -0.309840448070508,  0.141767782990886, -0.031781984216865,  0.004601916266695],
            [-0.309840448070508,  0.520982502274850, -0.288212646880823,  0.090119581912711, -0.013048989236230],
            [ 0.141767782990886, -0.288212646880823,  0.261482613984313, -0.150388056988661,  0.035350306894286],
            [-0.031781984216865,  0.090119581912711, -0.150388056988661,  0.137618085557560, -0.045567626264745],
            [ 0.004601916266695, -0.013048989236230,  0.035350306894286, -0.045567626264745,  0.018664392339993],
        ];

        for i in 0..5 {
            for j in 0..5 {
                let got = s[[i, j]];
                assert!(
                    (got - expected[i][j]).abs() < 1e-6,
                    "S[{}, {}]: expected {:.9}, got {:.9}",
                    i,
                    j,
                    expected[i][j],
                    got
                );
            }
        }
    }

    #[test]
    fn cr_penalty_shape_and_symmetry() {
        let x = Array1::linspace(0.0, 10.0, 20);
        let knots = cr_knots(&x, 7);
        let s = create_cr_penalty_matrix(&knots);
        assert_eq!(s.dim(), (7, 7));
        assert!(is_symmetric(&s, 1e-10));
    }

    #[test]
    fn cr_penalty_psd() {
        let x = Array1::linspace(0.0, 5.0, 30);
        let knots = cr_knots(&x, 6);
        let s = create_cr_penalty_matrix(&knots);
        assert!(is_psd(&s));
    }

    #[test]
    fn cr_penalty_constants_in_null_space() {
        // S · 1_k ≈ 0: constant vector lies in null space.
        let x = Array1::linspace(0.0, 10.0, 20);
        let knots = cr_knots(&x, 6);
        let s = create_cr_penalty_matrix(&knots);
        let ones = Array1::from_elem(6, 1.0);
        let q = ones.dot(&s.dot(&ones));
        assert!(q.abs() < 1e-8, "constant in null space, q = {}", q);
    }

    #[test]
    fn cr_penalty_linear_in_null_space() {
        // S · x ≈ 0: linear function lies in null space.
        let knots_vec = vec![0.0, 2.5, 6.5, 10.5, 15.0];
        let s = create_cr_penalty_matrix(&knots_vec);
        let lin = Array1::from_vec(knots_vec.clone());
        let q = lin.dot(&s.dot(&lin));
        assert!(q.abs() < 1e-8, "linear in null space, q = {}", q);
    }

    #[test]
    fn cr_penalty_k2_is_zero() {
        // With only 2 knots the natural spline is linear; penalty is zero.
        let knots = vec![0.0, 1.0];
        let s = create_cr_penalty_matrix(&knots);
        assert_eq!(s.dim(), (2, 2));
        assert!(s.iter().all(|&v| v.abs() < 1e-12));
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest! {
        #[test]
        fn cr_penalty_always_symmetric_and_psd(
            n_knots in 3usize..12,
        ) {
            let x = Array1::linspace(0.0, 10.0, 50);
            let knots = cr_knots(&x, n_knots);
            let s = create_cr_penalty_matrix(&knots);
            prop_assert_eq!(s.dim(), (n_knots, n_knots));
            for i in 0..n_knots {
                for j in 0..n_knots {
                    prop_assert!((s[[i, j]] - s[[j, i]]).abs() < 1e-10);
                }
            }
            // PSD check
            let n = s.dim().0;
            let mut rng = StdRngStub::new(99);
            for _ in 0..10 {
                let v: Array1<f64> = Array::from_shape_fn(n, |_| rng.next() - 0.5);
                let q = v.dot(&s.dot(&v));
                prop_assert!(q >= -1e-8, "PSD violated: q = {}", q);
            }
        }
    }
}
