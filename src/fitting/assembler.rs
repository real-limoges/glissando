//! Design matrix and penalty matrix assembly from formula terms.
//!
//! Converts a [`Formula`] into numeric [`ModelMatrix`] and [`PenaltyMatrix`] structures
//! that feed into the penalized weighted least squares solver.

use super::{GamlssError, PenaltyMatrix, Smooth, Term};
use crate::splines::{
    apply_cr_pc_constraint, cr_knots, create_basis_matrix_with_range, create_cr_basis_matrix,
    create_cr_penalty_matrix, create_penalty_matrix, kronecker_product, row_kronecker_into,
    sum_to_zero_basis,
};
use crate::types::{DataSet, ModelMatrix};
use ndarray::concatenate;
use ndarray::{s, Array1, Array2, Axis};
use std::collections::HashMap;

fn get_col<'a>(data: &'a DataSet, name: &str) -> Result<&'a Array1<f64>, GamlssError> {
    data.get(name).ok_or_else(|| GamlssError::MissingVariable {
        name: name.to_string(),
    })
}

/// Per-term layout metadata produced alongside the design matrix: how many
/// coefficient columns the term occupies and, for penalized smooths, the
/// dimension of its penalty null space — the EDF floor the term decays to as
/// `λ → ∞`. This lets the fitter attribute effective degrees of freedom back to
/// individual terms and flag a smooth that has collapsed onto its null space
/// (e.g. an over-penalized smooth that has degenerated to a straight line).
#[derive(Debug, Clone)]
pub(crate) struct TermLayout {
    /// Number of coefficient columns this term contributes to the design matrix.
    pub(crate) n_coeffs: usize,
    /// Dimension of the term's penalty null space (0 for unpenalized terms).
    pub(crate) null_dim: usize,
    /// Whether this term is a penalized smooth.
    pub(crate) is_smooth: bool,
}

/// Penalty null-space dimension for a smooth term, accounting for the
/// sum-to-zero centering applied when an `Intercept` shares the parameter
/// (centering removes the constant direction). This is the minimum EDF the term
/// can reach under heavy penalization.
fn smooth_null_dim(smooth: &Smooth, centered: bool) -> usize {
    // A difference penalty of order `d` has a null space of polynomials of
    // degree `< d`, dimension `d`; centering removes the constant (one
    // direction) when the basis is actually reparameterized (`n_splines >= 2`).
    let margin = |penalty_order: usize, n_splines: usize| -> usize {
        let base = penalty_order.min(n_splines);
        if centered && n_splines >= 2 {
            base.saturating_sub(1)
        } else {
            base
        }
    };
    match smooth {
        Smooth::PSpline1D {
            n_splines,
            penalty_order,
            ..
        } => margin(*penalty_order, *n_splines),
        Smooth::TensorProduct {
            n_splines_1,
            penalty_order_1,
            n_splines_2,
            penalty_order_2,
            ..
        } => {
            // null(S₁⊗I + I⊗S₂) = null(S₁) ⊗ null(S₂); the dimensions multiply.
            // The tensor is centered with ONE sum-to-zero constraint on the full
            // basis (mgcv te() semantics), which removes exactly the constant
            // direction — a member of the product null space — so subtract 1
            // (not one per margin as the `margin` helper would).
            let d1 = (*penalty_order_1).min(*n_splines_1);
            let d2 = (*penalty_order_2).min(*n_splines_2);
            let base = d1 * d2;
            if centered && *n_splines_1 * *n_splines_2 >= 2 {
                base.saturating_sub(1)
            } else {
                base
            }
        }
        // CR penalty null space = constants + linear = dim 2.
        // When `pc` is set it replaces centering (one constraint → dim 1).
        // When `centered` (sum-to-zero) it removes the constant → dim 1.
        Smooth::CrSpline1D { k, pc, .. } => {
            if *k < 2 {
                0
            } else if pc.is_some() || centered {
                1
            } else {
                2
            }
        }
        // Ridge penalty (identity) is full rank.
        Smooth::RandomEffect { .. } => 0,
    }
}

/// Resolve every term's data-dependent state from the training data once,
/// returning an owned `Vec<Term>`: CR-spline quantile knots, P-spline and
/// tensor knot ranges, and random-effect level lists.
///
/// Terms whose state is already populated are cloned unchanged, so the
/// function is a no-op at predict time (stored terms carry their state).
///
/// This must be called before [`assemble_model_matrices`] so the resolved
/// state is embedded in `FittedParameter::terms` and replayed verbatim at
/// predict time, guaranteeing that the fit and predict bases are identical.
pub(crate) fn resolve_terms(terms: &[Term], data: &DataSet) -> Result<Vec<Term>, GamlssError> {
    // Finite (min, max) of a column, for anchoring P-spline knot grids.
    let finite_range = |col: &str| -> Result<(f64, f64), GamlssError> {
        let x = get_col(data, col)?;
        let (lo, hi) = crate::splines::finite_range(x);
        // No finite values at all: storing (inf, -inf) on the term would fit a
        // garbage unit-spaced knot grid silently and cannot round-trip through
        // JSON. (The public fit path already rejects non-finite columns in
        // validate_inputs; this guards internal callers.)
        if !lo.is_finite() || !hi.is_finite() {
            return Err(GamlssError::Input(format!(
                "column '{col}' has no finite values; cannot anchor a spline basis"
            )));
        }
        Ok((lo, hi))
    };
    // Sorted distinct levels of a grouping column (sorted for determinism —
    // first-occurrence order would make the coefficient layout depend on row
    // order of the training data).
    let sorted_levels = |col: &str| -> Result<Vec<String>, GamlssError> {
        let x = get_col(data, col)?;
        let mut levels: Vec<String> = x.iter().map(|v| v.to_string()).collect();
        levels.sort();
        levels.dedup();
        Ok(levels)
    };

    terms
        .iter()
        .map(|term| match term {
            Term::Smooth(Smooth::CrSpline1D {
                col_name,
                k,
                pc,
                knots,
            }) if knots.is_empty() => {
                let x = get_col(data, col_name)?;
                Ok(Term::Smooth(Smooth::CrSpline1D {
                    col_name: col_name.clone(),
                    k: *k,
                    pc: *pc,
                    knots: cr_knots(x, *k),
                }))
            }
            Term::Smooth(Smooth::PSpline1D {
                col_name,
                n_splines,
                degree,
                penalty_order,
                range: None,
            }) => Ok(Term::Smooth(Smooth::PSpline1D {
                col_name: col_name.clone(),
                n_splines: *n_splines,
                degree: *degree,
                penalty_order: *penalty_order,
                range: Some(finite_range(col_name)?),
            })),
            Term::Smooth(Smooth::TensorProduct {
                col_name_1,
                n_splines_1,
                penalty_order_1,
                col_name_2,
                n_splines_2,
                penalty_order_2,
                degree,
                range_1,
                range_2,
            }) if range_1.is_none() || range_2.is_none() => {
                Ok(Term::Smooth(Smooth::TensorProduct {
                    col_name_1: col_name_1.clone(),
                    n_splines_1: *n_splines_1,
                    penalty_order_1: *penalty_order_1,
                    col_name_2: col_name_2.clone(),
                    n_splines_2: *n_splines_2,
                    penalty_order_2: *penalty_order_2,
                    degree: *degree,
                    range_1: Some(match range_1 {
                        Some(r) => *r,
                        None => finite_range(col_name_1)?,
                    }),
                    range_2: Some(match range_2 {
                        Some(r) => *r,
                        None => finite_range(col_name_2)?,
                    }),
                }))
            }
            Term::Smooth(Smooth::RandomEffect { col_name, levels }) if levels.is_empty() => {
                Ok(Term::Smooth(Smooth::RandomEffect {
                    col_name: col_name.clone(),
                    levels: sorted_levels(col_name)?,
                }))
            }
            other => Ok(other.clone()),
        })
        .collect()
}

fn assemble_smooth(
    data: &DataSet,
    n_obs: usize,
    smooth: &Smooth,
    apply_constraint: bool,
) -> Result<(Array2<f64>, Vec<PenaltyMatrix>), GamlssError> {
    match smooth {
        Smooth::PSpline1D {
            col_name,
            n_splines,
            degree,
            penalty_order,
            range,
        } => {
            let x_col = get_col(data, col_name)?;
            let basis = create_basis_matrix_with_range(x_col, *n_splines, *degree, *range);
            let penalty = create_penalty_matrix(*n_splines, *penalty_order);

            if apply_constraint && *n_splines >= 2 {
                let z = sum_to_zero_basis(*n_splines);
                let basis_c = basis.dot(&z);
                let penalty_c = z.t().dot(&penalty).dot(&z);
                Ok((basis_c, vec![PenaltyMatrix(penalty_c)]))
            } else {
                Ok((basis, vec![PenaltyMatrix(penalty)]))
            }
        }

        Smooth::CrSpline1D {
            col_name,
            k,
            pc,
            knots,
        } => {
            if knots.is_empty() {
                return Err(GamlssError::Internal(format!(
                    "CrSpline1D knots for '{}' are unresolved; \
                     call resolve_terms before assemble_model_matrices",
                    col_name
                )));
            }
            let x_col = get_col(data, col_name)?;
            let mut basis = create_cr_basis_matrix(x_col, knots);
            let penalty = create_cr_penalty_matrix(knots);

            if let Some(pc_val) = pc {
                // pc replaces centering: pin f(pc_val) = 0, skip sum_to_zero.
                apply_cr_pc_constraint(&mut basis, knots, *pc_val);
                // The pc-shifted basis maps the coefficient direction 1_k to the
                // zero function (B_pc·1 = 1·(1 − Σb_j(pc)) = 0) while S·1 = 0 too,
                // so keeping all k columns leaves a zero-design, zero-penalty
                // direction and X'WX + λS is singular. Drop that direction with
                // the same Householder null-space transform used for centering;
                // f(pc) = 0 is preserved for every β in the reduced space.
                if *k >= 2 {
                    let z = sum_to_zero_basis(*k);
                    let basis_c = basis.dot(&z);
                    let penalty_c = z.t().dot(&penalty).dot(&z);
                    Ok((basis_c, vec![PenaltyMatrix(penalty_c)]))
                } else {
                    Ok((basis, vec![PenaltyMatrix(penalty)]))
                }
            } else if apply_constraint && *k >= 2 {
                // Same sum-to-zero reparameterization as PSpline1D.
                let z = sum_to_zero_basis(*k);
                let basis_c = basis.dot(&z);
                let penalty_c = z.t().dot(&penalty).dot(&z);
                Ok((basis_c, vec![PenaltyMatrix(penalty_c)]))
            } else {
                Ok((basis, vec![PenaltyMatrix(penalty)]))
            }
        }

        Smooth::TensorProduct {
            col_name_1,
            n_splines_1,
            penalty_order_1,
            col_name_2,
            n_splines_2,
            penalty_order_2,
            degree,
            range_1,
            range_2,
        } => {
            let x1 = get_col(data, col_name_1)?;
            let b1 = create_basis_matrix_with_range(x1, *n_splines_1, *degree, *range_1);
            let s1 = create_penalty_matrix(*n_splines_1, *penalty_order_1);

            let x2 = get_col(data, col_name_2)?;
            let b2 = create_basis_matrix_with_range(x2, *n_splines_2, *degree, *range_2);
            let s2 = create_penalty_matrix(*n_splines_2, *penalty_order_2);

            let (k1, k2) = (*n_splines_1, *n_splines_2);
            let n_full = k1 * k2;
            let mut basis = Array2::<f64>::zeros((n_obs, n_full));
            for i in 0..n_obs {
                row_kronecker_into(b1.row(i), b2.row(i), basis.row_mut(i));
            }

            // Anisotropic penalties: S1⊗I2 for x1 direction, I1⊗S2 for x2 direction.
            let penalty_1 = kronecker_product(&s1, &Array2::<f64>::eye(k2));
            let penalty_2 = kronecker_product(&Array2::<f64>::eye(k1), &s2);

            // When an Intercept shares the parameter, apply ONE sum-to-zero
            // constraint to the FULL tensor basis (removing only the overall
            // constant), transforming both penalties with the same Z — exactly
            // mgcv's te() treatment (k1·k2 − 1 coefficients). Centering each
            // *marginal* before the Kronecker (the previous behaviour) removes
            // every function of the form f(x1)·1 and 1·g(x2), i.e. both main
            // effects — silently reducing te() to a ti()-style pure interaction
            // that cannot represent additive structure.
            if apply_constraint && n_full >= 2 {
                let z = sum_to_zero_basis(n_full);
                let basis_c = basis.dot(&z);
                let p1_c = z.t().dot(&penalty_1).dot(&z);
                let p2_c = z.t().dot(&penalty_2).dot(&z);
                Ok((basis_c, vec![PenaltyMatrix(p1_c), PenaltyMatrix(p2_c)]))
            } else {
                Ok((
                    basis,
                    vec![PenaltyMatrix(penalty_1), PenaltyMatrix(penalty_2)],
                ))
            }
        }

        Smooth::RandomEffect { col_name, levels } => {
            // Ridge-penalized indicators: equivalent to alpha ~ N(0, 1/lambda).
            let group_var = get_col(data, col_name)?;

            // Column layout comes from the levels resolved at FIT time (stored on
            // the term), so prediction maps each group to the coefficient it was
            // fitted with; a map rebuilt from the incoming data would silently
            // misalign columns whenever prediction rows present groups in a
            // different first-occurrence order or omit a group. Legacy models
            // (empty `levels`, pre-dating the field) fall back to
            // first-occurrence order to reproduce their fitted layout.
            let group_to_id: HashMap<String, usize> = if levels.is_empty() {
                let mut m = HashMap::new();
                for val in group_var.iter() {
                    let key: String = val.to_string();
                    let next_id = m.len();
                    m.entry(key).or_insert(next_id);
                }
                m
            } else {
                levels
                    .iter()
                    .enumerate()
                    .map(|(i, l)| (l.clone(), i))
                    .collect()
            };

            let n_groups = group_to_id.len();
            let mut basis = Array2::<f64>::zeros((n_obs, n_groups));

            for (i, val) in group_var.iter().enumerate() {
                let key: String = val.to_string();
                match group_to_id.get(&key) {
                    Some(&group_id) => basis[[i, group_id]] = 1.0,
                    None => {
                        return Err(GamlssError::Input(format!(
                            "random-effect column '{col_name}' has level '{key}' that was \
                             not present in the training data"
                        )))
                    }
                }
            }

            // Indicator basis is partition-of-unity (each row sums to 1), same
            // rank-deficiency story as P-splines.
            if apply_constraint && n_groups >= 2 {
                let z = sum_to_zero_basis(n_groups);
                let basis_c = basis.dot(&z);
                // Z'·I·Z = Z'·Z = I_{k-1} since Z has orthonormal columns.
                let penalty_c = Array2::<f64>::eye(n_groups - 1);
                Ok((basis_c, vec![PenaltyMatrix(penalty_c)]))
            } else {
                let penalty = Array2::<f64>::eye(n_groups);
                Ok((basis, vec![PenaltyMatrix(penalty)]))
            }
        }
    }
}

/// Assemble the design matrix X and penalty matrices S_j from formula terms.
///
/// Horizontally concatenates basis matrices for each term (intercept, linear, smooth)
/// and embeds penalty blocks at the correct offsets in the full coefficient space.
pub(crate) fn assemble_model_matrices(
    data: &DataSet,
    n_obs: usize,
    terms: &[Term],
) -> Result<(ModelMatrix, Vec<PenaltyMatrix>, usize, Vec<TermLayout>), GamlssError> {
    // Smooth bases on this codebase (P-spline, tensor-product, random-effect indicator)
    // are all partition-of-unity, so `1_n ∈ col(B)`. When an `Intercept` term is also
    // present the design matrix is rank-deficient. Apply a sum-to-zero
    // reparameterization to the smooths in that case to restore identifiability.
    let has_intercept = terms.iter().any(|t| matches!(t, Term::Intercept));

    let mut model_matrix_parts = Vec::with_capacity(terms.len());
    let mut penalty_blocks: Vec<(usize, PenaltyMatrix)> = Vec::new();
    let mut term_layouts = Vec::with_capacity(terms.len());
    let mut total_coeffs = 0;

    for term in terms {
        match term {
            Term::Intercept => {
                let part = Array1::ones(n_obs).insert_axis(Axis(1));
                model_matrix_parts.push(part);
                term_layouts.push(TermLayout {
                    n_coeffs: 1,
                    null_dim: 0,
                    is_smooth: false,
                });
                total_coeffs += 1;
            }
            Term::Linear { col_name } => {
                let x_col_vec = get_col(data, col_name)?;
                let part: Array2<f64> = x_col_vec.to_owned().insert_axis(Axis(1));
                model_matrix_parts.push(part);
                term_layouts.push(TermLayout {
                    n_coeffs: 1,
                    null_dim: 0,
                    is_smooth: false,
                });
                total_coeffs += 1;
            }
            Term::Smooth(smooth) => {
                let (basis, penalties) = assemble_smooth(data, n_obs, smooth, has_intercept)?;
                let n_coeffs = basis.ncols();
                model_matrix_parts.push(basis);
                term_layouts.push(TermLayout {
                    n_coeffs,
                    null_dim: smooth_null_dim(smooth, has_intercept),
                    is_smooth: true,
                });

                for penalty_block in penalties {
                    penalty_blocks.push((total_coeffs, penalty_block));
                }
                total_coeffs += n_coeffs;
            }
        }
    }

    let x_model = ModelMatrix(concatenate(
        Axis(1),
        &model_matrix_parts
            .iter()
            .map(|m| m.view())
            .collect::<Vec<_>>(),
    )?);

    let penalty_matrices = penalty_blocks
        .into_iter()
        .map(|(start_index, block)| {
            let mut s_j = PenaltyMatrix(Array2::<f64>::zeros((total_coeffs, total_coeffs)));
            let n = block.ncols();
            s_j.slice_mut(s![
                start_index..start_index + n,
                start_index..start_index + n
            ])
            .assign(&block);
            s_j
        })
        .collect::<Vec<_>>();

    Ok((x_model, penalty_matrices, total_coeffs, term_layouts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::Smooth;

    #[test]
    fn pspline_basis_is_partition_of_unity() {
        // P-spline basis rows should sum to 1 — the property that makes them
        // compatible with an intercept column (and triggers sum-to-zero
        // reparameterization in `assemble_smooth`).
        let mut data = DataSet::new();
        let n_obs = 100;
        data.insert_column("x", Array1::linspace(0.0, 1.0, n_obs));

        let term = Term::Smooth(Smooth::PSpline1D {
            col_name: "x".into(),
            n_splines: 10,
            degree: 3,
            penalty_order: 2,
            range: None,
        });

        let (mm, _, _, _) = assemble_model_matrices(&data, n_obs, &[term]).unwrap();
        for row in mm.0.rows() {
            let row_sum: f64 = row.sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "spline basis row sums to {} (expected 1.0)",
                row_sum
            );
        }
    }
}
