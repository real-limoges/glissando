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
use crate::terms::Contrast;
use crate::types::{DataSet, ModelMatrix};
use ndarray::concatenate;
use ndarray::{s, Array1, Array2, Axis};
use std::collections::HashMap;

/// The numeric realization of a [`Formula`](crate::Formula)'s term list for one
/// distribution parameter: the design matrix, its penalty blocks, the total
/// coefficient count, per-term layout, and the fixed per-row `offset` that enters
/// the linear predictor as `η = X·β + offset` (zeros when no `Term::Offset` is
/// present).
pub(crate) struct AssembledDesign {
    pub(crate) x: ModelMatrix,
    pub(crate) penalties: Vec<PenaltyMatrix>,
    pub(crate) n_coeffs: usize,
    pub(crate) layouts: Vec<TermLayout>,
    pub(crate) offset: Array1<f64>,
}

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
            // direction (a member of the product null space), so subtract 1
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
    terms.iter().map(|term| resolve_term(term, data)).collect()
}

/// Resolve a single term's data-dependent state from the training columns:
/// `CrSpline1D` knots and `Factor` levels. Recurses into `Interaction` operands.
/// All other terms are cloned unchanged.
fn resolve_term(term: &Term, data: &DataSet) -> Result<Term, GamlssError> {
    match term {
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
            range: Some(finite_range(get_col(data, col_name)?, col_name)?),
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
        }) if range_1.is_none() || range_2.is_none() => Ok(Term::Smooth(Smooth::TensorProduct {
            col_name_1: col_name_1.clone(),
            n_splines_1: *n_splines_1,
            penalty_order_1: *penalty_order_1,
            col_name_2: col_name_2.clone(),
            n_splines_2: *n_splines_2,
            penalty_order_2: *penalty_order_2,
            degree: *degree,
            range_1: Some(match range_1 {
                Some(r) => *r,
                None => finite_range(get_col(data, col_name_1)?, col_name_1)?,
            }),
            range_2: Some(match range_2 {
                Some(r) => *r,
                None => finite_range(get_col(data, col_name_2)?, col_name_2)?,
            }),
        })),
        Term::Smooth(Smooth::RandomEffect { col_name, levels }) if levels.is_empty() => {
            Ok(Term::Smooth(Smooth::RandomEffect {
                col_name: col_name.clone(),
                levels: sorted_levels(get_col(data, col_name)?),
            }))
        }
        Term::Factor {
            col_name,
            contrast,
            levels,
            labels,
        } if levels.is_empty() => {
            let x = get_col(data, col_name)?;
            Ok(Term::Factor {
                col_name: col_name.clone(),
                contrast: *contrast,
                levels: distinct_levels(x),
                labels: labels.clone(),
            })
        }
        Term::Interaction(left, right) => Ok(Term::Interaction(
            Box::new(resolve_term(left, data)?),
            Box::new(resolve_term(right, data)?),
        )),
        other => Ok(other.clone()),
    }
}

/// Finite `(min, max)` of a column, for anchoring a P-spline's uniform knot
/// grid to the training-data range so the fit and predict bases coincide.
///
/// Errors when the column has no finite values: storing `(inf, -inf)` on the
/// term would silently fit a garbage unit-spaced knot grid and cannot round-trip
/// through JSON. (The public fit path already rejects non-finite columns in
/// `validate_inputs`; this guards internal callers.)
fn finite_range(x: &Array1<f64>, col: &str) -> Result<(f64, f64), GamlssError> {
    let (lo, hi) = crate::splines::finite_range(x);
    if !lo.is_finite() || !hi.is_finite() {
        return Err(GamlssError::Input(format!(
            "column '{col}' has no finite values; cannot anchor a spline basis"
        )));
    }
    Ok((lo, hi))
}

/// Sorted distinct string levels of a grouping column, sorted numerically (not
/// lexicographically) via [`distinct_levels`] for determinism: first-occurrence
/// order would make the coefficient layout depend on the row order of the
/// training data, and string sorting would order "10" before "2".
fn sorted_levels(x: &Array1<f64>) -> Vec<String> {
    distinct_levels(x)
        .into_iter()
        .map(|v| v.to_string())
        .collect()
}

/// Sorted distinct numeric level codes of a categorical column, in ascending
/// numeric order so the treatment baseline (`levels[0]`) and contrast columns
/// are reproducible. Shared by `Factor` and (via [`sorted_levels`]) `RandomEffect`.
fn distinct_levels(x: &Array1<f64>) -> Vec<f64> {
    let mut seen: Vec<f64> = Vec::new();
    for &v in x.iter() {
        if !seen.contains(&v) {
            seen.push(v);
        }
    }
    seen.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    seen
}

/// Dummy-coded design block for a `Factor` with `L` levels under `contrast`,
/// producing `L − 1` columns. `levels` are the sorted distinct codes (resolved at
/// fit time and replayed at predict time so a level absent from new data still
/// maps to the right column). An observation whose code is not among `levels`
/// (an unseen level at predict time) contributes a zero row — the honest
/// "no information" encoding.
///
/// - `Treatment`: column `j` indicates `levels[j + 1]`; `levels[0]` is the
///   baseline (R's `contr.treatment`).
/// - `SumToZero`: `contr.sum` — column `j` is `+1` for `levels[j]`, `−1` for the
///   last level, `0` otherwise.
fn factor_columns(
    data: &DataSet,
    n_obs: usize,
    col_name: &str,
    contrast: Contrast,
    levels: &[f64],
) -> Result<Array2<f64>, GamlssError> {
    let x = get_col(data, col_name)?;
    // Fall back to discovering levels from the column when unresolved (e.g. the
    // assembler is exercised directly in a unit test without `resolve_terms`).
    let owned;
    let levels: &[f64] = if levels.is_empty() {
        owned = distinct_levels(x);
        &owned
    } else {
        levels
    };
    let n_levels = levels.len();
    let n_cols = n_levels.saturating_sub(1);
    let mut block = Array2::<f64>::zeros((n_obs, n_cols));
    if n_cols == 0 {
        return Ok(block); // single-level (or empty) factor contributes nothing
    }
    let level_index = |v: f64| levels.iter().position(|&u| u == v);
    for (i, &v) in x.iter().enumerate() {
        let Some(idx) = level_index(v) else { continue }; // unseen level → zero row
        match contrast {
            Contrast::Treatment => {
                if idx >= 1 {
                    block[[i, idx - 1]] = 1.0;
                }
            }
            Contrast::SumToZero => {
                if idx == n_levels - 1 {
                    for j in 0..n_cols {
                        block[[i, j]] = -1.0;
                    }
                } else {
                    block[[i, idx]] = 1.0;
                }
            }
        }
    }
    Ok(block)
}

/// Design columns for a single term used as an operand of an `Interaction`
/// (`Intercept`, `Linear`, `Factor`, or a nested `Interaction`). Smooth and
/// offset operands are rejected — smooth-by-factor interactions are SMOOTH-3
/// territory, and an offset has no design column to multiply.
fn term_columns(data: &DataSet, n_obs: usize, term: &Term) -> Result<Array2<f64>, GamlssError> {
    match term {
        Term::Intercept => Ok(Array1::ones(n_obs).insert_axis(Axis(1))),
        Term::Linear { col_name } => Ok(get_col(data, col_name)?.to_owned().insert_axis(Axis(1))),
        Term::Factor {
            col_name,
            contrast,
            levels,
            ..
        } => factor_columns(data, n_obs, col_name, *contrast, levels),
        Term::Interaction(left, right) => {
            let lc = term_columns(data, n_obs, left)?;
            let rc = term_columns(data, n_obs, right)?;
            Ok(row_kronecker_block(&lc, &rc, n_obs))
        }
        Term::Smooth(_) => Err(GamlssError::Input(
            "smooth terms cannot appear inside an interaction (see SMOOTH-3, \
             by-factor smooths)"
                .to_string(),
        )),
        Term::Offset { .. } => Err(GamlssError::Input(
            "an offset cannot appear inside an interaction".to_string(),
        )),
    }
}

/// Row-wise Kronecker product of two design blocks: an `n × p` and an `n × q`
/// block combine into an `n × (p·q)` block whose row `i` is the Kronecker product
/// of the two operand rows — the same primitive tensor smooths use.
fn row_kronecker_block(left: &Array2<f64>, right: &Array2<f64>, n_obs: usize) -> Array2<f64> {
    let n_cols = left.ncols() * right.ncols();
    let mut out = Array2::<f64>::zeros((n_obs, n_cols));
    for i in 0..n_obs {
        row_kronecker_into(left.row(i), right.row(i), out.row_mut(i));
    }
    out
}

/// Sum-to-zero (Householder null-space) reparameterization shared by every
/// smooth type that centers against an Intercept sharing its parameter:
/// `Z = sum_to_zero_basis(k)` projects `basis` and each of `penalties` onto
/// the sum-to-zero subspace, removing the direction collinear with the
/// intercept (or, for a CR spline's `pc` constraint, the direction collinear
/// with `f(pc) = 0`).
fn apply_sum_to_zero(
    basis: &Array2<f64>,
    penalties: &[&Array2<f64>],
    k: usize,
) -> (Array2<f64>, Vec<Array2<f64>>) {
    let z = sum_to_zero_basis(k);
    let basis_c = basis.dot(&z);
    let penalties_c = penalties.iter().map(|p| z.t().dot(*p).dot(&z)).collect();
    (basis_c, penalties_c)
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
                let (basis_c, penalties_c) = apply_sum_to_zero(&basis, &[&penalty], *n_splines);
                Ok((
                    basis_c,
                    penalties_c.into_iter().map(PenaltyMatrix).collect(),
                ))
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

            // pc replaces centering: pin f(pc_val) = 0. The pc-shifted basis
            // maps the coefficient direction 1_k to the zero function
            // (B_pc·1 = 1·(1 − Σb_j(pc)) = 0) while S·1 = 0 too, so keeping all
            // k columns leaves a zero-design, zero-penalty direction and
            // X'WX + λS is singular: it needs the same Householder null-space
            // transform as centering (below) whenever k ≥ 2, regardless of
            // `apply_constraint`; f(pc) = 0 is preserved for every β in the
            // reduced space.
            let needs_zero_sum = pc.is_some() || apply_constraint;
            if let Some(pc_val) = pc {
                apply_cr_pc_constraint(&mut basis, knots, *pc_val);
            }
            if needs_zero_sum && *k >= 2 {
                let (basis_c, penalties_c) = apply_sum_to_zero(&basis, &[&penalty], *k);
                Ok((
                    basis_c,
                    penalties_c.into_iter().map(PenaltyMatrix).collect(),
                ))
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
            // constant), transforming both penalties with the same Z: exactly
            // mgcv's te() treatment (k1·k2 − 1 coefficients). Centering each
            // *marginal* before the Kronecker (the previous behavior) removes
            // every function of the form f(x1)·1 and 1·g(x2), i.e. both main
            // effects, silently reducing te() to a ti()-style pure interaction
            // that cannot represent additive structure.
            if apply_constraint && n_full >= 2 {
                let (basis_c, penalties_c) =
                    apply_sum_to_zero(&basis, &[&penalty_1, &penalty_2], n_full);
                Ok((
                    basis_c,
                    penalties_c.into_iter().map(PenaltyMatrix).collect(),
                ))
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
) -> Result<AssembledDesign, GamlssError> {
    // Smooth bases on this codebase (P-spline, tensor-product, random-effect indicator)
    // are all partition-of-unity, so `1_n ∈ col(B)`. When an `Intercept` term is also
    // present the design matrix is rank-deficient. Apply a sum-to-zero
    // reparameterization to the smooths in that case to restore identifiability.
    let has_intercept = terms.iter().any(|t| matches!(t, Term::Intercept));

    let mut model_matrix_parts = Vec::with_capacity(terms.len());
    let mut penalty_blocks: Vec<(usize, PenaltyMatrix)> = Vec::new();
    let mut term_layouts = Vec::with_capacity(terms.len());
    let mut total_coeffs = 0;
    // Fixed per-row carry into η; stays zero unless a `Term::Offset` is present.
    let mut offset = Array1::<f64>::zeros(n_obs);

    // Push a parametric (unpenalized) design block and record its layout.
    let push_parametric =
        |parts: &mut Vec<Array2<f64>>, layouts: &mut Vec<TermLayout>, block: Array2<f64>| {
            let n_coeffs = block.ncols();
            parts.push(block);
            layouts.push(TermLayout {
                n_coeffs,
                null_dim: 0,
                is_smooth: false,
            });
            n_coeffs
        };

    for term in terms {
        match term {
            Term::Intercept | Term::Linear { .. } | Term::Factor { .. } | Term::Interaction(..) => {
                let block = term_columns(data, n_obs, term)?;
                total_coeffs += push_parametric(&mut model_matrix_parts, &mut term_layouts, block);
            }
            Term::Offset { col_name } => {
                // Enters η additively with a fixed coefficient of 1; contributes
                // to the offset vector, not to β. Record a zero-width layout so
                // term/EDF bookkeeping stays aligned with `terms`.
                offset += get_col(data, col_name)?;
                term_layouts.push(TermLayout {
                    n_coeffs: 0,
                    null_dim: 0,
                    is_smooth: false,
                });
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

    // A formula of nothing but offsets has no design column; emit an explicit
    // `n × 0` matrix rather than tripping `concatenate`'s empty-input error.
    let x_model = if model_matrix_parts.is_empty() {
        ModelMatrix(Array2::<f64>::zeros((n_obs, 0)))
    } else {
        ModelMatrix(concatenate(
            Axis(1),
            &model_matrix_parts
                .iter()
                .map(|m| m.view())
                .collect::<Vec<_>>(),
        )?)
    };

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

    Ok(AssembledDesign {
        x: x_model,
        penalties: penalty_matrices,
        n_coeffs: total_coeffs,
        layouts: term_layouts,
        offset,
    })
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

        let design = assemble_model_matrices(&data, n_obs, &[term]).unwrap();
        for row in design.x.0.rows() {
            let row_sum: f64 = row.sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "spline basis row sums to {} (expected 1.0)",
                row_sum
            );
        }
    }

    fn data_with(name: &str, values: Vec<f64>) -> DataSet {
        let mut d = DataSet::new();
        d.insert_column(name, Array1::from_vec(values));
        d
    }

    /// Treatment (dummy) coding matches R `contr.treatment`: `L − 1` columns,
    /// `levels[0]` the baseline, column `j` indicating `levels[j + 1]`.
    #[test]
    fn factor_treatment_contrast_matches_r() {
        // Levels {0,1,2}; one observation per level, in scrambled order.
        let data = data_with("g", vec![2.0, 0.0, 1.0, 2.0]);
        let cols = factor_columns(&data, 4, "g", Contrast::Treatment, &[]).unwrap();
        // Rows: level 2 → [0,1], level 0 → [0,0], level 1 → [1,0], level 2 → [0,1].
        let expected = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert_eq!(cols.dim(), (4, 2));
        for (i, row) in expected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert_eq!(cols[[i, j]], v, "treatment[{i},{j}]");
            }
        }
    }

    /// Sum-to-zero coding matches R `contr.sum`: the last level is `−1` across
    /// all columns, every other level `i` is `+1` in column `i`.
    #[test]
    fn factor_sum_to_zero_contrast_matches_r() {
        let data = data_with("g", vec![0.0, 1.0, 2.0]);
        let cols = factor_columns(&data, 3, "g", Contrast::SumToZero, &[]).unwrap();
        // contr.sum(3): level0 → [1,0], level1 → [0,1], level2 (last) → [-1,-1].
        let expected = [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]];
        for (i, row) in expected.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert_eq!(cols[[i, j]], v, "contr.sum[{i},{j}]");
            }
        }
    }

    /// A level absent from the resolved `levels` (unseen at predict time) yields
    /// a zero row rather than a panic or a misaligned column.
    #[test]
    fn factor_unseen_level_is_zero_row() {
        let data = data_with("g", vec![5.0]); // 5 not among the fitted levels
        let cols = factor_columns(&data, 1, "g", Contrast::Treatment, &[0.0, 1.0, 2.0]).unwrap();
        assert_eq!(cols.dim(), (1, 2));
        assert_eq!(cols[[0, 0]], 0.0);
        assert_eq!(cols[[0, 1]], 0.0);
    }

    /// Interaction is the row-wise product of operand columns. factor(2 levels) ×
    /// continuous collapses to one column: the non-baseline dummy times `x`.
    #[test]
    fn interaction_factor_times_continuous_is_product() {
        let mut data = DataSet::new();
        data.insert_column("g", Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]));
        data.insert_column("x", Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]));
        let term = Term::interaction(Term::factor("g"), Term::linear("x"));
        let cols = term_columns(&data, 4, &term).unwrap();
        assert_eq!(cols.dim(), (4, 1));
        // g dummy (level 1) is [0,1,1,0]; times x [2,3,4,5] → [0,3,4,0].
        let expected = [0.0, 3.0, 4.0, 0.0];
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(cols[[i, 0]], v, "interaction[{i}]");
        }
    }

    /// continuous × continuous interaction is the elementwise product `x·z`.
    #[test]
    fn interaction_continuous_times_continuous() {
        let mut data = DataSet::new();
        data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0]));
        data.insert_column("z", Array1::from_vec(vec![4.0, 5.0, 6.0]));
        let term = Term::interaction(Term::linear("x"), Term::linear("z"));
        let cols = term_columns(&data, 3, &term).unwrap();
        let expected = [4.0, 10.0, 18.0];
        for (i, &v) in expected.iter().enumerate() {
            assert_eq!(cols[[i, 0]], v);
        }
    }

    /// factor × factor: a 2-level × 3-level interaction yields `1 × 2 = 2`
    /// columns (the product of the two contrast blocks).
    #[test]
    fn interaction_factor_times_factor_column_count() {
        let mut data = DataSet::new();
        data.insert_column("g", Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]));
        data.insert_column("h", Array1::from_vec(vec![0.0, 1.0, 2.0, 0.0]));
        let term = Term::interaction(Term::factor("g"), Term::factor("h"));
        let cols = term_columns(&data, 4, &term).unwrap();
        assert_eq!(cols.dim(), (4, 2));
    }

    /// A smooth operand inside an interaction is rejected with a clear message
    /// (smooth-by-factor is SMOOTH-3, out of scope here).
    #[test]
    fn interaction_rejects_smooth_operand() {
        let mut data = DataSet::new();
        data.insert_column("g", Array1::from_vec(vec![0.0, 1.0]));
        data.insert_column("x", Array1::from_vec(vec![0.0, 1.0]));
        let term = Term::interaction(Term::factor("g"), Term::smooth(Smooth::ps("x")));
        assert!(term_columns(&data, 2, &term).is_err());
    }

    /// `resolve_terms` fills a bare factor's levels from the data (sorted) and
    /// stores them on the term, so predict replays the identical coding.
    #[test]
    fn resolve_terms_fills_factor_levels() {
        let data = data_with("g", vec![2.0, 0.0, 1.0, 0.0]);
        let resolved = resolve_terms(&[Term::factor("g")], &data).unwrap();
        match &resolved[0] {
            Term::Factor { levels, .. } => assert_eq!(levels, &vec![0.0, 1.0, 2.0]),
            other => panic!("expected resolved Factor, got {other:?}"),
        }
    }

    /// The assembled offset vector is the sum of every `Term::Offset` column and
    /// the offset terms contribute no design columns.
    #[test]
    fn offset_terms_accumulate_into_offset_vector() {
        let mut data = DataSet::new();
        data.insert_column("a", Array1::from_vec(vec![1.0, 2.0, 3.0]));
        data.insert_column("b", Array1::from_vec(vec![10.0, 20.0, 30.0]));
        let terms = vec![Term::Intercept, Term::offset("a"), Term::offset("b")];
        let design = assemble_model_matrices(&data, 3, &terms).unwrap();
        assert_eq!(design.n_coeffs, 1); // intercept only
        assert_eq!(design.offset.to_vec(), vec![11.0, 22.0, 33.0]);
    }
}
