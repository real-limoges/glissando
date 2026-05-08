//! B-spline basis construction and penalty matrices for P-spline smoothing.
//!
//! Provides routines for building B-spline basis matrices, difference-based penalty
//! matrices, and Kronecker product utilities used by tensor product smooth terms.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayViewMut1};

/// Compute the Kronecker product of two matrices: C = A ⊗ B.
///
/// If A is (m × n) and B is (p × q), the result is (mp × nq).
/// Used for constructing tensor product basis matrices.
pub fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
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
pub fn row_kronecker_into(a: ArrayView1<f64>, b: ArrayView1<f64>, mut out: ArrayViewMut1<f64>) {
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
pub fn create_basis_matrix(x: &Array1<f64>, n_splines: usize, degree: usize) -> Array2<f64> {
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

/// Clamped knots with interior knots at data quantiles.
fn select_knots(x: &Array1<f64>, n_splines: usize, degree: usize) -> Vec<f64> {
    let min_val = x.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let safe_n_splines = n_splines.max(degree + 1);
    let num_total_knots = safe_n_splines + degree + 1;

    let num_interior_knots = num_total_knots.saturating_sub(2 * (degree + 1));

    let mut knots = Vec::with_capacity(num_total_knots);

    for _ in 0..=degree {
        knots.push(min_val);
    }

    if num_interior_knots > 0 {
        let mut sorted_x = x.to_vec();
        sorted_x.retain(|v| v.is_finite());
        sorted_x.sort_unstable_by(|a, b| a.total_cmp(b));

        if !sorted_x.is_empty() {
            for i in 1..=num_interior_knots {
                let quantile = i as f64 / (num_interior_knots + 1) as f64;
                let idx = ((quantile * (sorted_x.len() - 1) as f64).round() as usize)
                    .min(sorted_x.len() - 1);
                knots.push(sorted_x[idx]);
            }
        } else {
            for _ in 0..=num_interior_knots {
                knots.push(min_val);
            }
        }
    }

    for _ in 0..=degree {
        knots.push(max_val);
    }

    while knots.len() < num_total_knots {
        knots.push(max_val);
    }
    knots
}

fn find_knot_span(x: f64, degree: usize, n_splines: usize, knots: &[f64]) -> usize {
    if knots.is_empty() {
        return 0;
    }

    if degree >= knots.len() {
        return 0;
    }

    let max_idx = (knots.len() - 1).min(n_splines);
    if x >= knots[max_idx] {
        return if n_splines > 0 { n_splines - 1 } else { 0 };
    }
    if x < knots[degree] {
        return degree;
    }
    let idx = knots.partition_point(|&k| k <= x);
    let safe_idx = idx.saturating_sub(1);
    safe_idx.max(degree).min(n_splines - 1)
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

/// P-spline penalty S = D'D. Order 2 approximates integral of squared second derivative.
pub fn create_penalty_matrix(n_splines: usize, order: usize) -> Array2<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, Array2};
    #[cfg(not(target_arch = "wasm32"))]
    use proptest::prelude::*;

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
}
