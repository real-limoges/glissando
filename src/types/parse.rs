//! DATA-5 · string formula parser.
//!
//! Parses an R/mgcv-style formula string such as
//! `"y ~ s(x) + region + age:sex + offset(log_e)"` into the same `Vec<Term>` the
//! builder API produces, so nothing downstream changes. The parser is a pure
//! front-end over [`Term`]/[`Formula`](crate::Formula).
//!
//! Grammar (each `+`-separated piece is one term, parentheses respected):
//! - `1`: explicit intercept; `0` or `-1` suppresses the (otherwise implicit) intercept
//! - `s(x)`: P-spline smooth; `k=` sets the basis size, `bs="cr"` a cubic
//!   regression spline, `bs="re"` a random effect
//! - `te(x, z)`: tensor-product smooth
//! - `offset(e)`: an [`Term::Offset`]
//! - `factor(f)` / `factor(f, sum)`: a [`Term::Factor`] (treatment / sum-to-zero)
//! - bare name: a [`Term::Linear`]
//! - `a:b`: interaction (product); `a*b` expands to `a + b + a:b`
//!
//! The response (left of `~`) is returned to the caller but not stored in a
//! `Formula`; glissando takes the response array separately at fit time. A formula
//! with no `~` is treated as all right-hand side (the `~ s(x)` spelling gamlss
//! uses for scale/shape parameters).

use crate::error::GamlssError;
use crate::terms::{Contrast, Smooth, Term};

/// Parse a formula string into its optional response name and term list.
///
/// `"y ~ s(x) + z"` → `(Some("y"), [Intercept, s(x), z])`;
/// `"~ s(x)"` → `(None, [Intercept, s(x)])`.
pub fn parse_formula_string(formula: &str) -> Result<(Option<String>, Vec<Term>), GamlssError> {
    let (response, rhs) = match formula.split_once('~') {
        Some((lhs, rhs)) => {
            let lhs = lhs.trim();
            let response = if lhs.is_empty() {
                None
            } else {
                Some(validate_ident(lhs)?.to_string())
            };
            (response, rhs)
        }
        None => (None, formula),
    };

    let terms = parse_rhs(rhs)?;
    Ok((response, terms))
}

/// Parse just the right-hand side (term list) of a formula.
fn parse_rhs(rhs: &str) -> Result<Vec<Term>, GamlssError> {
    let mut terms: Vec<Term> = Vec::new();
    let mut suppress_intercept = false;
    let mut saw_any_piece = false;

    for piece in split_top_level(rhs, '+') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        saw_any_piece = true;
        match piece {
            "1" => {} // explicit intercept; the implicit one is added below anyway
            "0" | "-1" => suppress_intercept = true,
            _ => parse_term_piece(piece, &mut terms)?,
        }
    }

    // A formula needs at least one piece. `y ~ ` (empty right-hand side) is an
    // error; `y ~ 1` is a valid intercept-only model.
    if !saw_any_piece {
        return Err(GamlssError::Input(
            "formula has an empty right-hand side".to_string(),
        ));
    }

    // R semantics: an intercept is present unless explicitly removed (`0`/`-1`).
    if !suppress_intercept {
        terms.insert(0, Term::Intercept);
    }
    if terms.is_empty() {
        return Err(GamlssError::Input(
            "formula has no terms after removing the intercept".to_string(),
        ));
    }
    Ok(terms)
}

/// Parse one `+`-separated piece, which may itself carry `*` (crossing) or `:`
/// (interaction), pushing the resulting term(s) onto `out`.
fn parse_term_piece(piece: &str, out: &mut Vec<Term>) -> Result<(), GamlssError> {
    // `*` crossing binds looser than `:`. `a*b` ⇒ a + b + a:b. For more than two
    // operands we emit each main effect plus the single full interaction
    // (`a*b*c` ⇒ a + b + c + a:b:c). R's full factorial of pairwise terms is not
    // expanded yet, so write the pairwise terms out with `:` if you need them.
    let crossed = split_top_level(piece, '*');
    if crossed.len() > 1 {
        let mut atoms = Vec::with_capacity(crossed.len());
        for operand in &crossed {
            let term = parse_interaction(operand.trim())?;
            out.push(term.clone());
            atoms.push(term);
        }
        out.push(fold_interaction(atoms));
        return Ok(());
    }
    out.push(parse_interaction(piece)?);
    Ok(())
}

/// Parse a (possibly `:`-joined) interaction into a single term.
fn parse_interaction(s: &str) -> Result<Term, GamlssError> {
    let parts = split_top_level(s, ':');
    let mut atoms = Vec::with_capacity(parts.len());
    for part in &parts {
        atoms.push(parse_atom(part.trim())?);
    }
    Ok(fold_interaction(atoms))
}

/// Left-fold a non-empty list of terms into a chain of `Term::Interaction`s; a
/// single term passes through unchanged.
fn fold_interaction(mut atoms: Vec<Term>) -> Term {
    let mut acc = atoms.remove(0);
    for next in atoms {
        acc = Term::interaction(acc, next);
    }
    acc
}

/// Parse a single atomic term: a function call (`s`, `te`, `offset`, `factor`) or
/// a bare column name (→ `Linear`).
fn parse_atom(s: &str) -> Result<Term, GamlssError> {
    if let Some(args) = call_args(s, "s") {
        return parse_smooth(&args);
    }
    if let Some(args) = call_args(s, "te") {
        let cols = split_top_level(&args, ',');
        if cols.len() != 2 {
            return Err(GamlssError::Input(format!(
                "te(...) takes exactly two columns, got `{s}`"
            )));
        }
        return Ok(Term::smooth(Smooth::tensor(
            validate_ident(cols[0].trim())?,
            validate_ident(cols[1].trim())?,
        )));
    }
    if let Some(args) = call_args(s, "offset") {
        return Ok(Term::offset(validate_ident(args.trim())?));
    }
    if let Some(args) = call_args(s, "factor") {
        return parse_factor(&args);
    }
    // Bare identifier → linear term.
    Ok(Term::linear(validate_ident(s)?))
}

/// Parse `s(col)` / `s(col, k=10)` / `s(col, bs="cr")` / `s(col, bs="re")`.
fn parse_smooth(args: &str) -> Result<Term, GamlssError> {
    let parts = split_top_level(args, ',');
    let col = validate_ident(parts[0].trim())?;
    let mut k: Option<usize> = None;
    let mut bs = "ps".to_string();
    for kv in parts.iter().skip(1) {
        let (key, val) = kv.split_once('=').ok_or_else(|| {
            GamlssError::Input(format!("expected key=value in smooth args, got `{kv}`"))
        })?;
        match key.trim() {
            "k" => {
                k = Some(val.trim().parse().map_err(|_| {
                    GamlssError::Input(format!("k must be a positive integer, got `{val}`"))
                })?)
            }
            "bs" => bs = unquote(val.trim()).to_string(),
            other => {
                return Err(GamlssError::Input(format!(
                    "unknown smooth argument `{other}` (expected `k` or `bs`)"
                )))
            }
        }
    }
    let smooth = match bs.as_str() {
        "ps" => {
            let mut s = Smooth::ps(col);
            if let Some(k) = k {
                s = s.n_splines(k);
            }
            s
        }
        "cr" => {
            let mut s = Smooth::cr(col);
            if let Some(k) = k {
                s = s.k(k);
            }
            s
        }
        "re" => Smooth::re(col),
        other => {
            return Err(GamlssError::Input(format!(
                "unknown basis `bs=\"{other}\"` (expected `ps`, `cr`, or `re`)"
            )))
        }
    };
    Ok(Term::smooth(smooth))
}

/// Parse `factor(col)` / `factor(col, sum)` / `factor(col, treatment)`.
fn parse_factor(args: &str) -> Result<Term, GamlssError> {
    let parts = split_top_level(args, ',');
    let col = validate_ident(parts[0].trim())?;
    let contrast = match parts.get(1).map(|s| unquote(s.trim())) {
        None | Some("treatment") => Contrast::Treatment,
        Some("sum") | Some("sum_to_zero") => Contrast::SumToZero,
        Some(other) => {
            return Err(GamlssError::Input(format!(
                "unknown contrast `{other}` (expected `treatment` or `sum`)"
            )))
        }
    };
    Ok(Term::factor_with(col, contrast))
}

/// If `s` is a call `name(...)`, return the argument string between the outer
/// parentheses; otherwise `None`.
fn call_args(s: &str, name: &str) -> Option<String> {
    let rest = s.strip_prefix(name)?;
    let rest = rest.trim_start();
    let inner = rest.strip_prefix('(')?.strip_suffix(')')?;
    Some(inner.to_string())
}

/// Split `s` on `delim`, but only at parenthesis depth zero (so `s(x, k=3)` is
/// one piece when splitting on `+` or `:`).
fn split_top_level(s: &str, delim: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                current.push(c);
            }
            ')' => {
                depth -= 1;
                current.push(c);
            }
            c if c == delim && depth == 0 => {
                out.push(std::mem::take(&mut current));
            }
            c => current.push(c),
        }
    }
    out.push(current);
    out
}

/// Strip surrounding single or double quotes from a token.
fn unquote(s: &str) -> &str {
    s.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| s.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(s)
}

/// Validate that `s` is a plausible column identifier (non-empty, no whitespace
/// or operators), returning it trimmed.
fn validate_ident(s: &str) -> Result<&str, GamlssError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(GamlssError::Input(
            "empty identifier in formula".to_string(),
        ));
    }
    if s.chars()
        .any(|c| c.is_whitespace() || "+~*:()=".contains(c))
    {
        return Err(GamlssError::Input(format!(
            "`{s}` is not a valid column identifier"
        )));
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms_of(s: &str) -> Vec<Term> {
        parse_formula_string(s).unwrap().1
    }

    #[test]
    fn parses_response_and_implicit_intercept() {
        let (resp, terms) = parse_formula_string("y ~ x").unwrap();
        assert_eq!(resp.as_deref(), Some("y"));
        assert!(matches!(terms[0], Term::Intercept));
        assert!(matches!(&terms[1], Term::Linear { col_name } if col_name == "x"));
    }

    #[test]
    fn no_response_is_allowed() {
        let (resp, terms) = parse_formula_string("~ s(x)").unwrap();
        assert_eq!(resp, None);
        assert_eq!(terms.len(), 2); // intercept + smooth
    }

    #[test]
    fn suppresses_intercept_with_zero_or_minus_one() {
        // Our grammar reads `0` / `-1` as their own `+`-joined pieces (not the R
        // `x - 1` infix spelling), so suppression is written `0 + x` / `-1 + x`.
        let terms = terms_of("y ~ 0 + x");
        assert!(!terms.iter().any(|t| matches!(t, Term::Intercept)));
        let terms = terms_of("y ~ -1 + x");
        assert!(!terms.iter().any(|t| matches!(t, Term::Intercept)));
    }

    #[test]
    fn explicit_one_is_intercept_only() {
        let terms = terms_of("y ~ 1");
        assert_eq!(terms.len(), 1);
        assert!(matches!(terms[0], Term::Intercept));
    }

    #[test]
    fn parses_smooth_variants() {
        assert!(matches!(
            terms_of("y ~ s(x)")[1],
            Term::Smooth(Smooth::PSpline1D { .. })
        ));
        match &terms_of("y ~ s(x, k=20)")[1] {
            Term::Smooth(Smooth::PSpline1D { n_splines, .. }) => assert_eq!(*n_splines, 20),
            other => panic!("expected PSpline1D, got {other:?}"),
        }
        assert!(matches!(
            terms_of("y ~ s(x, bs=\"cr\")")[1],
            Term::Smooth(Smooth::CrSpline1D { .. })
        ));
        assert!(matches!(
            terms_of("y ~ s(g, bs=\"re\")")[1],
            Term::Smooth(Smooth::RandomEffect { .. })
        ));
        assert!(matches!(
            terms_of("y ~ te(x, z)")[1],
            Term::Smooth(Smooth::TensorProduct { .. })
        ));
    }

    #[test]
    fn parses_offset_and_factor() {
        assert!(
            matches!(&terms_of("y ~ offset(e)")[1], Term::Offset { col_name } if col_name == "e")
        );
        assert!(matches!(
            terms_of("y ~ factor(g)")[1],
            Term::Factor {
                contrast: Contrast::Treatment,
                ..
            }
        ));
        assert!(matches!(
            terms_of("y ~ factor(g, sum)")[1],
            Term::Factor {
                contrast: Contrast::SumToZero,
                ..
            }
        ));
    }

    #[test]
    fn parses_interaction_and_crossing() {
        // `a:b` → one interaction term.
        let terms = terms_of("y ~ a:b");
        assert!(matches!(terms[1], Term::Interaction(..)));

        // `a*b` → a + b + a:b (3 terms after the intercept).
        let terms = terms_of("y ~ a*b");
        assert_eq!(terms.len(), 4);
        assert!(matches!(terms[1], Term::Linear { .. }));
        assert!(matches!(terms[2], Term::Linear { .. }));
        assert!(matches!(terms[3], Term::Interaction(..)));
    }

    #[test]
    fn rejects_malformed() {
        assert!(parse_formula_string("y ~ s(x").is_err()); // unbalanced, not a call
        assert!(parse_formula_string("y ~ ").is_err()); // no terms
        assert!(parse_formula_string("y ~ s(x, foo=3)").is_err()); // unknown smooth arg
        assert!(parse_formula_string("y ~ factor(g, bogus)").is_err()); // unknown contrast
    }

    #[test]
    fn display_round_trips_through_parser() {
        // Render every term kind, reparse, and confirm the term list is stable.
        for spec in [
            "y ~ x",
            "y ~ s(x)",
            "y ~ s(x, k=15)",
            "y ~ s(x, bs=\"cr\")",
            "y ~ s(g, bs=\"re\")",
            "y ~ te(x, z)",
            "y ~ offset(e)",
            "y ~ factor(g)",
            "y ~ factor(g, sum)",
            "y ~ a:b",
            "y ~ 0 + x",
        ] {
            let terms = terms_of(spec);
            let rendered: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
            // Intercept presence is a formula property, not a term, so a faithful
            // string emits `0` to suppress the otherwise-implicit intercept.
            let has_intercept = terms.iter().any(|t| matches!(t, Term::Intercept));
            let mut rhs = rendered.clone();
            if !has_intercept {
                rhs.insert(0, "0".to_string());
            }
            let reparsed = terms_of(&format!("~ {}", rhs.join(" + ")));
            let rerendered: Vec<String> = reparsed.iter().map(|t| t.to_string()).collect();
            assert_eq!(rendered, rerendered, "round-trip drifted for `{spec}`");
        }
    }
}
