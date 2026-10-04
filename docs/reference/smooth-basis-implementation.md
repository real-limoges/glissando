# SmoothBasis Implementation Guide: One Trait per Smooth Type

This is a step-by-step walkthrough of backlog item Altitude #5 in `docs/private-notes/code-quality-backlog.md`: give each smooth type one home for its behavior, so adding a basis means adding a type rather than threading a new arm through six functions.
It is sequenced **before** GEO-1 (`geo1-mrf-implementation.md`) and GEO-2 (`geo2-tp-implementation.md`), which both build against the trait defined here.

Today `Smooth` is an enum with four struct variants (`PSpline1D`, `TensorProduct`, `CrSpline1D`, `RandomEffect`), and what each one *does* is spread across two files:

| Behavior | Where it lives today |
|---|---|
| columns read | `Smooth::column_names`, `src/terms.rs:430` |
| label | `Smooth::term_name`, `src/terms.rs:379` |
| formula spelling | `Smooth::formula_repr`, `src/terms.rs:400` |
| penalty null-space dimension | `smooth_null_dim`, `src/fitting/assembler.rs:59` |
| fit-time state | `resolve_term`, `src/fitting/assembler.rs:132` |
| basis and penalties | `assemble_smooth`, `src/fitting/assembler.rs:374` |

Five of those six match every variant with no wildcard, so the compiler at least lists them.
The sixth, `resolve_term`, ends in `other => Ok(other.clone())` (line 212), so a new variant with fit-time state compiles without an arm and then silently never resolves.
Both GEO guides had to call that trap out.
After this refactor it cannot happen: `resolve` is a required trait method.

**Done criterion**: each smooth type is its own struct implementing a `pub(crate) trait SmoothBasis`, and `Smooth::basis()` is the only `match` over the variants.
`smooth_null_dim` and `assemble_smooth` are gone from the assembler.
Every fit, snapshot and prediction is unchanged, and the serialized JSON of every term is byte-identical to before the refactor.

**Scope**: a reorganization with no numeric change.
Code moves; it is not rewritten.

    src/terms.rs                     <- Smooth becomes newtype variants; basis() dispatch; builders updated; py_parse updated
    src/terms/smooth.rs              <- NEW: the SmoothBasis trait and shared construction helpers
    src/terms/smooth/pspline.rs      <- NEW: PSpline1D struct + impl (moved from terms.rs and assembler.rs)
    src/terms/smooth/tensor.rs       <- NEW: TensorProduct
    src/terms/smooth/cr.rs           <- NEW: CrSpline1D
    src/terms/smooth/random_effect.rs <- NEW: RandomEffect
    src/fitting/assembler.rs         <- three dispatch call sites; smooth_null_dim and assemble_smooth deleted
    src/lib.rs                       <- re-export the four structs
    src/types/parse.rs               <- tests only: patterns updated
    tests/smooth_wire_format.rs      <- NEW, written FIRST: freezes the term JSON
    tests/*, benchmark/src/bin/*     <- 29 struct-literal and pattern sites updated mechanically

Every code block below is a **planned target shape**, labeled with the file it will live in.
Line numbers cited for existing code were verified on 2026-09-30.

------------------------------------------------------------------------

## 1. Layout

Before: the enum owns the data, and two files own the behavior.

    terms.rs       enum Smooth { PSpline1D {..}, TensorProduct {..}, CrSpline1D {..}, RandomEffect {..} }
                   term_name / formula_repr / column_names       3 matches
    assembler.rs   smooth_null_dim / resolve_term / assemble_smooth   3 matches (one with a fall-through)

After: each type owns its data and its behavior; the enum is a closed list of types plus one dispatch.

    terms.rs              enum Smooth { PSpline1D(PSpline1D), TensorProduct(TensorProduct), ... }
                          fn basis(&self) -> &dyn SmoothBasis          the ONE match
    terms/smooth.rs       trait SmoothBasis { column_names, term_name, formula_repr, null_dim, resolve, assemble }
                          shared helpers: column, finite_range, sorted_levels, distinct_levels, apply_sum_to_zero
    terms/smooth/*.rs     struct + impl SmoothBasis, one file per type
    splines/*.rs          the math, unchanged
    assembler.rs          smooth.basis().resolve(..) / .assemble(..) / .null_dim(..)

The dependency direction stays as it is: `terms` calls into `splines` for math, `fitting::assembler` calls into `terms`.
Nothing in `terms` imports from `fitting`.

------------------------------------------------------------------------

## 2. Why newtype variants and a trait object

Three designs were considered.

- **A trait implemented on per-type structs, held in newtype variants** (this guide).
  Each type is a plain struct with its own fields and its own `impl`.
  `Smooth` stays an enum, so it stays a closed, serializable, `Clone`-able value that a `Term` can hold, and one `match` in `basis()` turns it into a `&dyn SmoothBasis`.
- **`Box<dyn SmoothBasis>` inside `Term`.**
  Open to extension, but serde cannot derive through a trait object, `Clone` needs a `dyn-clone` shim, and the closed set of types is a feature here: `FamilyDescriptor`-style rebuilding from JSON needs to know every type anyway.
  Rejected.
- **Keep the struct variants and add a trait over borrowed views.**
  Avoids touching callers, but every type is then defined twice (the variant's fields and a view struct mirroring them).
  Rejected as the less durable of the two.

The cost of the chosen design is a public API change: a variant becomes `Smooth::PSpline1D(PSpline1D { .. })` instead of `Smooth::PSpline1D { .. }`.
The constructors (`Smooth::ps`, `cr`, `re`, `tensor`) and builders do not change, so code that builds smooths through them is untouched.
Code that writes a struct literal or destructures fields needs one extra layer of wrapping; there are 29 such sites, all in `tests/` and `benchmark/`.
Nothing is published (v0.1.0), so there are no external callers to break.

**The wire format does not change.**
serde's default external tagging writes a struct variant as `{"PSpline1D": {"col_name": ..}}` and a newtype variant wrapping a struct as `{"PSpline1D": {"col_name": ..}}`: the same bytes.
Saved models keep loading.
Section 3 freezes that claim in a test before any code moves.

------------------------------------------------------------------------

## 3. Phase 0: freeze the wire format first

Write this test against today's code, run it, and accept the snapshots **before** touching `src/`.
It is a characterization test, in the sense of `tests/derivative_golden.rs`: it records what the code does, so the refactor can be held to it.

**`tests/smooth_wire_format.rs`** (new file)

```rust
// Characterization test for the SmoothBasis refactor (backlog Altitude #5).
// It freezes the JSON every smooth type serializes to, both as built and after
// fit-time resolution. The refactor changes Rust types, not the wire format, so
// these snapshots must not move. A diff here is a serialization break.
#![cfg(all(
    feature = "serialization",
    not(feature = "python"),
    not(target_arch = "wasm32")
))]

mod common;

use common::Generator;
use glissando::distributions::Gaussian;
use glissando::{DataSet, Formula, GamlssModel, Smooth, Term};
use ndarray::Array1;

/// One of every smooth shape, as a user would build it.
fn built_smooths() -> Vec<Smooth> {
    vec![
        Smooth::ps("x"),
        Smooth::ps("x").n_splines(12).degree(2).penalty_order(1),
        Smooth::cr("x"),
        Smooth::cr("x").k(8).pc(0.5),
        Smooth::re("g"),
        Smooth::tensor("x", "z"),
    ]
}

fn data() -> (Array1<f64>, DataSet) {
    let mut rng = Generator::new(77);
    let (y, mut data) = rng.sinusoidal_gaussian(120, 0.3);
    let n = y.len();
    data.insert_column("z", Array1::linspace(0.0, 1.0, n));
    data.insert_column("g", Array1::from_iter((0..n).map(|i| (i % 4) as f64)));
    (y, data)
}

/// The term JSON as built (unresolved: empty knots, `None` ranges, no levels).
#[test]
fn built_terms_serialize_unchanged() {
    let terms: Vec<Term> = built_smooths().into_iter().map(Term::smooth).collect();
    insta::assert_snapshot!(serde_json::to_string_pretty(&terms).unwrap());
}

/// The term JSON after a fit: resolved knots, ranges and levels included. One
/// fit per smooth, each with an intercept so the centered path is exercised.
#[test]
fn resolved_terms_serialize_unchanged() {
    let (y, data) = data();
    let resolved: Vec<Term> = built_smooths()
        .into_iter()
        .map(|smooth| {
            let formula = Formula::new()
                .with_terms("mu", vec![Term::Intercept, Term::smooth(smooth)])
                .with_terms("sigma", vec![Term::Intercept]);
            let model = GamlssModel::fit(&data, &y, &formula, &Gaussian::new()).unwrap();
            model.models["mu"].terms[1].clone()
        })
        .collect();
    insta::assert_snapshot!(serde_json::to_string_pretty(&resolved).unwrap());
}

/// JSON written by hand in today's format, with the fit-time fields omitted,
/// still loads and fits. This is the backward-compatibility half: old saved
/// formulas must keep parsing after the types move.
#[test]
fn hand_written_term_json_still_loads() {
    let json = r#"[
        {"Smooth": {"PSpline1D": {"col_name": "x", "n_splines": 10, "degree": 3, "penalty_order": 2}}},
        {"Smooth": {"CrSpline1D": {"col_name": "x", "k": 6}}},
        {"Smooth": {"RandomEffect": {"col_name": "g"}}},
        {"Smooth": {"TensorProduct": {"col_name_1": "x", "n_splines_1": 5, "penalty_order_1": 2,
                                      "col_name_2": "z", "n_splines_2": 5, "penalty_order_2": 2,
                                      "degree": 3}}}
    ]"#;
    let terms: Vec<Term> = serde_json::from_str(json).unwrap();
    let rendered: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
    assert_eq!(rendered, ["s(x)", "s(x, bs=\"cr\")", "s(g, bs=\"re\")", "te(x,z)"]);
}
```

The resolved snapshot holds only term state (knots from quantiles, ranges from min/max, level strings), none of which touches the linear-algebra backend, so it is identical under `openblas` and `pure-rust`.
It deliberately excludes coefficients, which are backend-sensitive at the last digit and are already frozen by `tests/regression_families.rs`.

Run it and accept the two new snapshots by hand (`cargo-insta` is not installed; see `CLAUDE.md`):

```bash
INSTA_UPDATE=auto cargo test --features serialization --test smooth_wire_format
# inspect tests/snapshots/smooth_wire_format__*.snap.new, then rename each to .snap
```

From here on, these two snapshots plus `derivative_golden`, `regression_families`, and the full suite are the gate for every later phase.

------------------------------------------------------------------------

## 4. Phase 1: the trait

**`src/terms/smooth.rs`** (new file)

```rust
//! The smooth-term contract. Each smooth type is a struct in `smooth/` that
//! implements [`SmoothBasis`]; [`Smooth::basis`](super::Smooth::basis) is the only
//! place that matches over the variants.
//!
//! Adding a smooth type is: a struct and its `impl SmoothBasis` in a new file
//! here, its math in `crate::splines`, one variant on `Smooth`, one arm in
//! `Smooth::basis`, and a constructor.

mod cr;
mod pspline;
mod random_effect;
mod tensor;

pub use cr::CrSpline1D;
pub use pspline::PSpline1D;
pub use random_effect::RandomEffect;
pub use tensor::TensorProduct;

use super::Smooth;
use crate::error::GamlssError;
use crate::splines::sum_to_zero_basis;
use crate::types::DataSet;
use ndarray::{Array1, Array2};

/// Everything the fitting machinery needs from a smooth term.
pub(crate) trait SmoothBasis {
    /// Data columns this smooth reads.
    fn column_names(&self) -> Vec<&str>;

    /// mgcv-style label; the key in `FittedParameter::term_blocks`.
    fn term_name(&self) -> String;

    /// The R/mgcv formula spelling the string parser reads back.
    fn formula_repr(&self) -> String;

    /// Dimension of the penalty null space: the EDF floor the term decays to as
    /// lambda grows. `centered` is true when an Intercept shares the parameter.
    fn null_dim(&self, centered: bool) -> usize;

    /// Resolve data-dependent state (knots, ranges, levels) from the training
    /// data. Must return the smooth unchanged when its state is already
    /// resolved, which is what makes resolution a no-op at predict time.
    fn resolve(&self, data: &DataSet) -> Result<Smooth, GamlssError>;

    /// The design block at the rows of `data` and the penalty block(s) over its
    /// columns. `centered` is true when an Intercept shares the parameter, and
    /// the basis must then be reparameterized to drop the constant direction.
    fn assemble(
        &self,
        data: &DataSet,
        n_obs: usize,
        centered: bool,
    ) -> Result<(Array2<f64>, Vec<Array2<f64>>), GamlssError>;
}
```

`resolve` returns a `Smooth`, not `Self`, so the trait stays object-safe and `basis()` can hand back `&dyn SmoothBasis`.

### Shared construction helpers

Four helpers the smooth arms use today live privately in `assembler.rs`.
They move here, `pub(crate)`, unchanged except for the one rename noted:

| Helper | From | Note |
|---|---|---|
| `column(data, name)` | `get_col`, `assembler.rs:33` | renamed so call sites read `column(data, col_name)?` |
| `finite_range(x, col)` | `assembler.rs:223` | the erroring wrapper, not the raw one in `splines` |
| `sorted_levels(x)` and `distinct_levels(x)` | `assembler.rs:237`, `:247` | `distinct_levels` is also used by `Factor` |
| `apply_sum_to_zero(basis, penalties, k)` | `assembler.rs:363` | |

The assembler keeps using `column` and `distinct_levels` for the parametric terms, now imported from `crate::terms::smooth`.
Moving them rather than making the assembler's copies `pub(crate)` keeps `terms` free of any dependency on `fitting`.

------------------------------------------------------------------------

## 5. Phase 1: one struct per type

Each type's struct carries the fields and doc comments its variant has today (`src/terms.rs:195-269`), with the `serde(default)` attributes moved onto the struct fields.
Field order is preserved exactly, because serde writes fields in declaration order and the wire-format snapshot will catch any reordering.
Fields are `pub`, as the variant fields are today, so struct literals keep working.

The P-spline is shown in full as the worked example.
The other three follow the same recipe and are summarized after it.

**`src/terms/smooth/pspline.rs`** (new file)

```rust
//! 1D P-spline smooth (Eilers-Marx; mgcv `bs = "ps"`).

use super::{apply_sum_to_zero, column, finite_range, SmoothBasis};
use crate::error::GamlssError;
use crate::splines::{create_basis_matrix_with_range, create_penalty_matrix};
use crate::terms::Smooth;
use crate::types::DataSet;
use ndarray::Array2;

/// 1D P-spline smooth.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PSpline1D {
    pub col_name: String,
    /// Typical: 5-50.
    pub n_splines: usize,
    /// Typical: 2-3.
    pub degree: usize,
    /// 1 = linear trends, 2 = constant second differences.
    pub penalty_order: usize,
    /// Training-data range `(min, max)` the uniform knot grid is anchored to.
    /// **Leave `None` when building a formula.** (Doc comment moved verbatim
    /// from `src/terms.rs:204-209`.)
    #[cfg_attr(feature = "serde", serde(default))]
    pub range: Option<(f64, f64)>,
}

impl SmoothBasis for PSpline1D {
    fn column_names(&self) -> Vec<&str> {
        vec![self.col_name.as_str()]
    }

    fn term_name(&self) -> String {
        format!("s({})", self.col_name)
    }

    fn formula_repr(&self) -> String {
        if self.n_splines == Smooth::DEFAULT_N_SPLINES {
            format!("s({})", self.col_name)
        } else {
            format!("s({}, k={})", self.col_name, self.n_splines)
        }
    }

    // A difference penalty of order d has a null space of polynomials of degree
    // < d, so dimension d. Centering removes the constant, but only when the
    // basis is actually reparameterized (n_splines >= 2).
    fn null_dim(&self, centered: bool) -> usize {
        let base = self.penalty_order.min(self.n_splines);
        if centered && self.n_splines >= 2 {
            base.saturating_sub(1)
        } else {
            base
        }
    }

    fn resolve(&self, data: &DataSet) -> Result<Smooth, GamlssError> {
        let range = match self.range {
            Some(r) => r,
            None => finite_range(column(data, &self.col_name)?, &self.col_name)?,
        };
        Ok(Smooth::PSpline1D(PSpline1D {
            range: Some(range),
            ..self.clone()
        }))
    }

    fn assemble(
        &self,
        data: &DataSet,
        _n_obs: usize,
        centered: bool,
    ) -> Result<(Array2<f64>, Vec<Array2<f64>>), GamlssError> {
        let x = column(data, &self.col_name)?;
        let basis = create_basis_matrix_with_range(x, self.n_splines, self.degree, self.range);
        let penalty = create_penalty_matrix(self.n_splines, self.penalty_order);
        if centered && self.n_splines >= 2 {
            Ok(apply_sum_to_zero(&basis, &[&penalty], self.n_splines))
        } else {
            Ok((basis, vec![penalty]))
        }
    }
}
```

Each method body is the corresponding arm from today's code with `self.` in front of the field names:
`null_dim` from the `margin` closure and the `PSpline1D` arm (`assembler.rs:63-76`), `resolve` from `assembler.rs:148-160`, `assemble` from `assembler.rs:381-398`.
`apply_constraint` is renamed `centered` throughout, which is the word the null-space code already uses for the same flag.

The other three, with today's source for each method:

| Type | `null_dim` | `resolve` (resolves when) | `assemble` |
|---|---|---|---|
| `TensorProduct` | `assembler.rs:77-97` | `range_1` or `range_2` is `None`; `:161-187` | `:437-482` (two penalties, one sum-to-zero over the full basis) |
| `CrSpline1D` | `:101-109` | `knots` is empty; `:134-147` | `:400-435` (the `pc` constraint forces the transform even uncentered) |
| `RandomEffect` | `:111` (always 0) | `levels` is empty; `:188-193` | `:484-539` (legacy first-occurrence fallback kept; centered penalty is `eye(k-1)`) |

Every `resolve` has the same shape: if the state is already present, return `Smooth::X(self.clone())`; otherwise compute it and return the filled-in struct.
Keep the comments that travel with each arm.
The tensor's comment about centering the full basis rather than each margin, and the CR's comment about `pc` making `X'WX + lambda S` singular, record real past bugs and are the most valuable lines in the file.

------------------------------------------------------------------------

## 6. Phase 1: the enum and its single dispatch

**`src/terms.rs`** (replacing the enum at lines 192-270)

```rust
mod smooth;

pub use smooth::{CrSpline1D, PSpline1D, RandomEffect, TensorProduct};
pub(crate) use smooth::{column, distinct_levels, SmoothBasis};

/// A penalized smooth term. Each variant wraps the struct that owns that
/// type's fields and behavior (see `terms/smooth/`). (The existing doc comment
/// at lines 174-191 stays.)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Smooth {
    /// 1D P-spline smooth.
    PSpline1D(PSpline1D),
    /// 2D tensor product of two P-spline bases: f(x1, x2).
    TensorProduct(TensorProduct),
    /// 1D natural cubic regression spline (mgcv `bs = "cr"`).
    CrSpline1D(CrSpline1D),
    /// Random intercept indexed by a grouping variable.
    RandomEffect(RandomEffect),
}

impl Smooth {
    /// The one place that matches over the smooth types. Everything else asks
    /// the basis.
    pub(crate) fn basis(&self) -> &dyn SmoothBasis {
        match self {
            Smooth::PSpline1D(s) => s,
            Smooth::TensorProduct(s) => s,
            Smooth::CrSpline1D(s) => s,
            Smooth::RandomEffect(s) => s,
        }
    }

    /// mgcv-style label for this smooth term.
    pub fn term_name(&self) -> String {
        self.basis().term_name()
    }

    /// The R/mgcv formula spelling the string parser accepts.
    pub fn formula_repr(&self) -> String {
        self.basis().formula_repr()
    }

    /// The column names this smooth reads.
    pub fn column_names(&self) -> Vec<&str> {
        self.basis().column_names()
    }
}
```

The three public methods keep their names and signatures, so `Term::term_name`, `Term::column_names` and `Display for Term` (`src/terms.rs:110-172`) do not change.

The constructors build the struct inside the variant:

```rust
    pub fn ps(col_name: impl Into<String>) -> Self {
        Smooth::PSpline1D(PSpline1D {
            col_name: col_name.into(),
            n_splines: Self::DEFAULT_N_SPLINES,
            degree: Self::DEFAULT_DEGREE,
            penalty_order: Self::DEFAULT_PENALTY_ORDER,
            range: None,
        })
    }
```

and the builders destructure one layer deeper:

```rust
    /// Builder: set the B-spline degree (applies to P-splines and tensor products).
    pub fn degree(mut self, d: usize) -> Self {
        match &mut self {
            Smooth::PSpline1D(s) => s.degree = d,
            Smooth::TensorProduct(s) => s.degree = d,
            _ => {}
        }
        self
    }
```

The builders keep their `_ => {}` no-op arms; a builder that does not apply to a type is documented as a no-op, and that does not change.

**`src/lib.rs`** (line 84)

```rust
pub use terms::{Contrast, CrSpline1D, PSpline1D, RandomEffect, Smooth, TensorProduct, Term};
```

The structs have to be nameable outside the crate, because a caller that destructures a variant now names its struct.

------------------------------------------------------------------------

## 7. Phase 1: the assembler shrinks to three calls

**`src/fitting/assembler.rs`**

```rust
// resolve_term (line 132): the four smooth arms become one.
        Term::Smooth(smooth) => Ok(Term::Smooth(smooth.basis().resolve(data)?)),

// assemble_model_matrices (line 596): the smooth arm.
            Term::Smooth(smooth) => {
                let basis_impl = smooth.basis();
                let (basis, penalties) = basis_impl.assemble(data, n_obs, has_intercept)?;
                let n_coeffs = basis.ncols();
                model_matrix_parts.push(basis);
                term_layouts.push(TermLayout {
                    n_coeffs,
                    null_dim: basis_impl.null_dim(has_intercept),
                    is_smooth: true,
                });
                // ...penalty offsets unchanged...
            }
```

Delete `smooth_null_dim` (lines 55-113), `assemble_smooth` (lines 374-541), `apply_sum_to_zero`, `sorted_levels`, `finite_range`, and the smooth arms of `resolve_term`.
The `Factor` and `Interaction` arms of `resolve_term` and its `other => Ok(other.clone())` fall-through stay: the fall-through now covers only `Intercept`, `Linear` and `Offset`, which have no state.
Trim the `crate::splines` import at line 9 to what the parametric terms still use (nothing, after the move); the compiler's unused-import warnings list it exactly.

The assembler's existing unit tests (`mod tests` at line 640) exercise `assemble_model_matrices` end to end and keep passing unchanged, except where one constructs a variant by struct literal (section 8).
`tensor_product_penalties_share_offset` and `second_smooth_penalty_offset_equals_first_smooth_coeff_count` are the ones that guard the penalty-offset bookkeeping this phase touches.

------------------------------------------------------------------------

## 8. Phase 1: migrate the call sites

The compiler finds every site; there is nothing to search for by hand.
For the record, as of 2026-09-30:

- `src/`: `py_parse` in `src/terms.rs` (lines 602, 645, 662), the `src/terms.rs` unit tests (lines 450-542), `src/types/parse.rs` tests (lines 338-354), and one test in `src/fitting/solver.rs` (line 1554).
- `tests/`: 12 sites across `common/mod.rs`, `ocat.rs`, `validation.rs`, `lambda_bistability.rs`, `comprehensive.rs`, `regression_families.rs`, `scale_smooth_recovery.rs`, `prediction.rs`.
- `benchmark/`: 17 sites in `src/bin/compare_fit.rs`, `src/bin/fit_ocat.rs` and `src/bin/spike_ocat.rs`, counting one prose mention in `benchmark/CLAUDE.md`.

Two rewrites cover all of them.

```rust
// A struct literal gains a wrapping layer:
Term::Smooth(Smooth::PSpline1D { col_name: "x".to_string(), n_splines: 20, degree: 3, penalty_order: 2, range: None })
// becomes
Term::Smooth(Smooth::PSpline1D(PSpline1D { col_name: "x".to_string(), n_splines: 20, degree: 3, penalty_order: 2, range: None }))

// A pattern destructures through the struct:
Smooth::CrSpline1D { k, pc, .. } => { /* ... */ }
// becomes
Smooth::CrSpline1D(CrSpline1D { k, pc, .. }) => { /* ... */ }
```

Prefer a constructor plus builders where one already says the same thing: `Smooth::ps("x").n_splines(20)` is shorter than either literal and is what the struct-literal sites mostly spell out by hand.
Where a site needs a field the builders do not cover (the tensor's per-margin sizes in `tests/common/mod.rs:49`), keep the literal.

`py_parse` gets the same treatment, and it is a natural moment to switch its three literals to the constructors (`Smooth::re(col_name)`, `Smooth::cr(col_name).k(k)`, and so on), so the Python path stops restating defaults.
`pc` has no builder that takes an `Option`, so the CR arm keeps `if let Some(pc) = pc { smooth = smooth.pc(pc) }`.

------------------------------------------------------------------------

## 9. Tests

This phase adds no behavior, so it adds almost no tests.
The gate is everything that already exists, plus the Phase 0 snapshots:

- `tests/smooth_wire_format.rs`: both snapshots byte-identical, and the hand-written JSON still loads.
- `tests/regression_families.rs` and `tests/derivative_golden.rs`: zero diff.
  A move-only refactor must not move a fit; if one does, a method body was changed in transit.
- `tests/basis_stability.rs`: the fit-time resolution contract (subset prediction, reordered levels, unseen levels) for every type.
- The full suite on both backends, the `python` build, and the `wasm` build.

One new unit test per type is worth adding while the impls are fresh, because the trait now makes it cheap: call `null_dim`, `term_name` and `formula_repr` directly on a struct, with no `DataSet` and no fit.
For example, in `src/terms/smooth/cr.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_dim_is_two_minus_one_per_constraint() {
        let cr = CrSpline1D { col_name: "x".into(), k: 6, pc: None, knots: vec![] };
        assert_eq!(cr.null_dim(false), 2);
        assert_eq!(cr.null_dim(true), 1);
        let pinned = CrSpline1D { pc: Some(0.0), ..cr };
        assert_eq!(pinned.null_dim(false), 1);
    }
}
```

These replace nothing; `smooth_null_dim` had no direct tests, because it was private to the assembler and only reachable through a fit.

------------------------------------------------------------------------

## 10. Verification commands

```bash
cargo test                                                   # default: openblas + parallel
cargo test --features serialization --test smooth_wire_format
cargo test --no-default-features --features pure-rust,parallel
cargo build --features python                                # or maturin build, as CI does
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
cargo clippy --all-targets && cargo fmt --check
cargo build -p glissando_benchmark --release                 # the benchmark binaries compile
```

`pkg/` does not need regenerating: the serialized `Smooth` shape is unchanged, which is the point of Phase 0.
Regenerating it anyway is a cheap confirmation; the generated `.d.ts` and `.js` should show no diff (then `git checkout -- pkg/README.md pkg/.gitignore`).

------------------------------------------------------------------------

## 11. Decisions and gotchas

### Freeze before you move

The wire-format snapshots must be accepted against the **old** code.
Written after the refactor, they would freeze whatever the refactor produced and prove nothing.

### Field order is wire format

serde serializes struct fields in declaration order.
Moving a field while moving the struct changes the bytes of every saved model's term list, which the resolved-terms snapshot catches.
Load compatibility would survive a reorder (JSON objects are unordered on read), but the snapshot would not, and it is right not to: byte-stable output is what keeps saved-model diffs readable.

### `resolve` must be a no-op on resolved state

At predict time `resolve_terms` runs again over the stored terms.
Every `resolve` must check its own state and return itself unchanged when that state is present, exactly as the guards on today's arms do (`if knots.is_empty()`, `range: None`, ...).
A `resolve` that recomputes unconditionally would rebuild knots from the *prediction* data, which is the bug `tests/basis_stability.rs` exists to catch.

### The trait removes the fall-through trap, not the guard

Before, forgetting a `resolve_term` arm compiled.
After, forgetting `resolve` does not compile.
But a `resolve` that forgets its "already resolved" guard still compiles, so `basis_stability.rs` remains the safety net.
Each new type should add its own subset-prediction test there (GEO-1 and GEO-2 both do).

### `Term` identity is still a display string

Backlog Altitude #6 (terms keyed by `term_name`, which collapses `ps`, `cr` and `re` on one column to the same `s(x)`) is not addressed here.
The trait makes it easier to fix later, because `term_name` now has one implementation per type.
It does mean a new type should pick a `term_name` that does not collide with the existing ones; GEO-1 does.

### The assembler no longer knows what smooths exist

That is the goal, and it has one consequence for reading the code: to see how a P-spline builds its basis, open `terms/smooth/pspline.rs`, not `assembler.rs`.
Update the "Source structure" section of `CLAUDE.md` to say so.

------------------------------------------------------------------------

## 12. Exit checklist

- [ ] Phase 0: `tests/smooth_wire_format.rs` written and its snapshots accepted **against the unmodified code**.
- [ ] `SmoothBasis` and the shared helpers live in `src/terms/smooth.rs`; one file per type under `src/terms/smooth/`.
- [ ] `Smooth` has newtype variants; `Smooth::basis()` is the only match over them; `term_name`, `formula_repr`, `column_names` delegate.
- [ ] Constructors and builders build and edit the inner structs; the four structs are re-exported from `src/lib.rs`.
- [ ] `assembler.rs` calls `basis().resolve`, `.assemble`, `.null_dim`; `smooth_null_dim` and `assemble_smooth` are deleted.
- [ ] Every call site in `src/`, `tests/` and `benchmark/` compiles; `py_parse` uses the constructors.
- [ ] Wire-format snapshots byte-identical; `derivative_golden` and `regression_families` zero diff.
- [ ] Default, `pure-rust`, `python`, `wasm`, and benchmark builds pass; clippy and fmt clean.
- [ ] `CLAUDE.md` "Source structure" describes `terms/smooth/`; `benchmark/CLAUDE.md`'s `Smooth::PSpline1D { .. }` mention is updated.
- [ ] Backlog Altitude #5 moved to "Already applied" in `code-quality-backlog.md`, with the pass record.

When these hold, adding a smooth is one new file plus one arm.
Next is **GEO-1** (`geo1-mrf-implementation.md`), the first type built against the trait.
