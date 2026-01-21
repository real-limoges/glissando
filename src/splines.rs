use ndarray::{Array1, Array2, s};

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

/// Creates the B-spline basis matrix using the de Boor recursion algorithm.
/// Each row corresponds to an observation, each column to a basis function.
/// Uses "clamped" knots (repeated at boundaries) so the spline passes through endpoints.
pub fn create_basis_matrix(x: &Array1<f64>, n_splines: usize, degree: usize) -> Array2<f64> {
    let n_obs = x.len();

    if n_splines <= degree {
        return Array2::<f64>::zeros((n_obs, n_splines));
    }

    let knots = select_knots(x, n_splines, degree);
    let mut basis_matrix = Array2::<f64>::zeros((n_obs, n_splines));

    for (row_idx, &x_i) in x.iter().enumerate() {
        let span_index = find_knot_span(x_i, degree, n_splines, &knots);
        let basis_values = evaluate_basis_functions(x_i, span_index, degree, &knots);

        for (j, val) in basis_values.iter().enumerate() {
            // Safety check to prevent index out of bounds
            if span_index >= degree {
                let col_idx = span_index - degree + j;
                if col_idx < n_splines {
                    basis_matrix[[row_idx, col_idx]] = *val;
                }
            }
        }
    }
    basis_matrix
}

/// Selects knot positions for B-spline basis.
/// Uses "clamped" knots: (degree+1) repeated knots at each boundary ensure the
/// spline interpolates the endpoints. Interior knots are placed at data quantiles
/// for adaptive resolution (more knots where data is dense).
fn select_knots(x: &Array1<f64>, n_splines: usize, degree: usize) -> Vec<f64> {
    let min_val = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let safe_n_splines = n_splines.max(degree + 1);
    // Total knots = n_splines + degree + 1 (standard B-spline identity)
    let num_total_knots = safe_n_splines + degree + 1;

    let num_interior_knots = num_total_knots.saturating_sub(2 * (degree + 1));

    let mut knots = Vec::with_capacity(num_total_knots);

    // Clamped boundary: repeat min_val (degree+1) times
    for _ in 0..=degree {
        knots.push(min_val);
    }

    // Interior knots at data quantiles for adaptive spacing
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

    // Clamped boundary: repeat max_val (degree+1) times
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

/// Evaluates B-spline basis functions at point x using the de Boor-Cox recursion.
/// This is the standard algorithm from "A Practical Guide to Splines" (de Boor 1978).
///
/// The recursion builds up from degree 0 (indicator functions) to the desired degree.
/// At each stage j, only (j+1) basis functions are non-zero, stored in basis[0..=j].
/// The "left" and "right" arrays store x - t[i-j+r] and t[i+r+1] - x for the recursion.
fn evaluate_basis_functions(x: f64, i: usize, degree: usize, knots: &[f64]) -> Vec<f64> {
    let mut basis = vec![0.0; degree + 1];
    basis[0] = 1.0;

    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];

    for j in 1..=degree {
        let mut saved = 0.0;
        for r in 0..j {
            let left_idx = (i + 1).saturating_sub(j).saturating_add(r);
            let right_idx = i + r + 1;

            if right_idx < knots.len() {
                left[j] = x - knots[left_idx];
                right[j] = knots[right_idx] - x;

                // Standard B-spline recursion: B_{i,j}(x) = w1*B_{i,j-1}(x) + w2*B_{i+1,j-1}(x)
                // Guard against division by zero for coincident knots.
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
    basis
}

/// Creates the P-spline penalty matrix S = D'D where D is the difference matrix.
/// Order 1: penalizes first differences (piecewise constant fit)
/// Order 2: penalizes second differences (piecewise linear fit)
/// The penalty lambda * beta'*S*beta approximates lambda * integral(f''(x)^2 dx)
/// for order 2, encouraging smooth curves. See Eilers & Marx (1996).
pub fn create_penalty_matrix(n_splines: usize, order: usize) -> Array2<f64> {
    let n_rows_d = n_splines.saturating_sub(order);
    if n_rows_d == 0 {
        return Array2::<f64>::zeros((n_splines, n_splines));
    }

    let mut d_matrix = Array2::<f64>::zeros((n_rows_d, n_splines));
    match order {
        // First differences: [1, -1, 0, ...], [0, 1, -1, 0, ...], ...
        1 => {
            for i in 0..n_rows_d {
                d_matrix[[i, i]] = 1.0;
                d_matrix[[i, i + 1]] = -1.0;
            }
        }
        // Second differences: [1, -2, 1, 0, ...], [0, 1, -2, 1, 0, ...], ...
        _ => {
            for i in 0..n_rows_d {
                d_matrix[[i, i]] = 1.0;
                d_matrix[[i, i + 1]] = -2.0;
                d_matrix[[i, i + 2]] = 1.0;
            }
        }
    }
    // S = D'D is positive semi-definite with nullspace = polynomials of degree < order
    d_matrix.t().dot(&d_matrix)
}
