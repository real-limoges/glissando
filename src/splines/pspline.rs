//! B-spline basis construction for P-spline smoothing (Eilers–Marx layout).

use ndarray::{Array1, Array2};

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

    // Standard de Boor–Cox recurrence (Piegl & Tiller, "The NURBS Book", Algorithm A2.2).
    //
    // At the start of outer iteration j, left[1..=j] and right[1..=j] must all be
    // populated before the inner r-loop reads left[j-r] and right[r+1].  The fix is
    // to compute left[j] and right[j] *once* here — outside the r-loop — so that
    // slots written in previous j iterations remain valid when the inner loop reads them.
    //
    // Index safety (find_knot_span clamps i to [degree, n_splines-1]):
    //   knots[i+1-j]: j in 1..=degree → index ≥ i+1-degree ≥ 1 ≥ 0.
    //   knots[i+j]:   j in 1..=degree → index ≤ i+degree ≤ n_splines+degree-1 < knots.len().
    for j in 1..=degree {
        left[j] = x - knots[i + 1 - j];
        right[j] = knots[i + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
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
        basis[j] = saved;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Golden-value test for the cubic P-spline basis against the analytic de Boor result.
    ///
    /// Setup: n_splines=6, degree=3, x ∈ {0, 0.25, 0.5, 0.75, 1}.
    /// Uniform knots: t = [-1, -2/3, -1/3, 0, 1/3, 2/3, 1, 4/3, 5/3, 2].
    ///
    /// Reference values were derived by hand from the textbook de Boor–Cox recurrence
    /// (Piegl & Tiller Algorithm A2.2) and independently cross-checked against
    /// scipy.interpolate.BSpline.basis_element output:
    ///
    /// ```python
    /// from scipy.interpolate import BSpline
    /// import numpy as np
    /// # 6 cubic B-splines on uniform knots over [0,1]
    /// knots = np.linspace(-1, 2, 10)      # same as select_knots with n=6, d=3
    /// for j in range(6):
    ///     for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
    ///         print(f"B[{j}]({x}) =", BSpline.basis_element(knots[j:j+5])(x))
    /// ```
    ///
    /// The previous (buggy) implementation gave wrong interior basis values, e.g. at
    /// x=0.5 it returned [0, 1/48, **2/3, 7/24**, 1/48, 0] instead of the correct
    /// [0, 1/48, **23/48, 23/48**, 1/48, 0].  Both sum to 1, so partition-of-unity
    /// tests did not catch the error.
    #[test]
    fn bspline_basis_golden_values_degree3() {
        let x = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let b = create_basis_matrix(&x, 6, 3);
        assert_eq!(b.dim(), (5, 6));

        // Each inner array is one row of the basis matrix (one observation).
        // Derived from the de Boor recurrence on knots t = [-1, -2/3, -1/3, 0, 1/3, 2/3, 1, 4/3, 5/3, 2].
        #[rustfmt::skip]
        let expected: [[f64; 6]; 5] = [
            // x=0.00, span=3: B[0..3] = [1/6, 2/3, 1/6, 0]
            [1.0/6.0,    2.0/3.0,    1.0/6.0,    0.0,          0.0,          0.0       ],
            // x=0.25, span=3: B[0..3] = [1/384, 121/384, 235/384, 27/384]
            [1.0/384.0,  121.0/384.0, 235.0/384.0, 27.0/384.0, 0.0,          0.0       ],
            // x=0.50, span=4: B[1..4] = [1/48, 23/48, 23/48, 1/48]
            [0.0,        1.0/48.0,   23.0/48.0,  23.0/48.0,   1.0/48.0,     0.0       ],
            // x=0.75, span=5: B[2..5] = [27/384, 235/384, 121/384, 1/384]
            [0.0,        0.0,        27.0/384.0, 235.0/384.0, 121.0/384.0,  1.0/384.0 ],
            // x=1.00, span=5: B[3..5] = [0, 1/6, 2/3, 1/6]
            [0.0,        0.0,        0.0,        1.0/6.0,     2.0/3.0,      1.0/6.0   ],
        ];

        for (i, row_ref) in expected.iter().enumerate() {
            for (j, &ref_val) in row_ref.iter().enumerate() {
                let got = b[[i, j]];
                assert!(
                    (got - ref_val).abs() < 1e-9,
                    "B[{}, {}]: expected {:.12}, got {:.12}",
                    i,
                    j,
                    ref_val,
                    got
                );
            }
        }
    }
}
