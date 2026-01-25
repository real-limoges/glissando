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
    let min_val = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

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
        sorted_x.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if !sorted_x.is_empty() {
            for i in 1..=num_interior_knots {
                let quantile = i as f64 / (num_interior_knots + 1) as f64;
                let idx = (quantile * (sorted_x.len() - 1) as f64).round() as usize;
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
