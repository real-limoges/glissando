//! A representative fit from each distribution group: symmetric continuous
//! (Gaussian), overdispersed counts (Negative Binomial), positive continuous
//! (Weibull), and a skew/kurtotic Box-Cox fit (BCT) with centile curves.
//!
//! Run with: `cargo run --example families`

use glissando::distributions::{Gaussian, NegativeBinomial, Weibull, BCCG};
use glissando::ndarray::Array1;
use glissando::{DataSet, Formula, GamlssError, GamlssModel, Smooth, Term};

// A tiny deterministic LCG in (0, 1), so the example is reproducible without
// pulling `rand` into the snippet.
fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f64) / (u32::MAX as f64)
}

fn main() -> Result<(), GamlssError> {
    let n = 200;
    let mut s = 42u64;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * 6.0).collect();

    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x.clone()));

    // Gaussian: symmetric continuous, mean varying smoothly with x.
    let y = Array1::from_vec(
        x.iter()
            .map(|&v| v.sin() + 0.3 * (unit(&mut s) - 0.5))
            .collect(),
    );
    let f = Formula::from_strings([("mu", "y ~ s(x)"), ("sigma", "~ 1")])?;
    let m = GamlssModel::fit(&data, &y, &f, &Gaussian::new())?;
    println!(
        "Gaussian:    converged={}  AIC={:.1}",
        m.converged(),
        m.gaic(&Gaussian::new(), &y, 2.0)?
    );

    // Negative Binomial: overdispersed counts (a non-negative integer response).
    let counts = Array1::from_vec(
        x.iter()
            .map(|&v| ((0.4 + 0.35 * v).exp() * (0.6 + 0.8 * unit(&mut s))).round())
            .collect(),
    );
    let f = Formula::from_strings([("mu", "y ~ x"), ("sigma", "~ 1")])?;
    let m = GamlssModel::fit(&data, &counts, &f, &NegativeBinomial::new())?;
    println!("NegBinomial: converged={}", m.converged());

    // Weibull: strictly positive continuous (durations, reliability).
    let pos = Array1::from_vec(
        x.iter()
            .map(|&v| (0.3 + 0.5 * v) * (0.5 + unit(&mut s)) + 0.05)
            .collect(),
    );
    let f = Formula::from_strings([("mu", "y ~ x"), ("sigma", "~ 1")])?;
    let m = GamlssModel::fit(&data, &pos, &f, &Weibull::new())?;
    println!("Weibull:     converged={}", m.converged());

    // BCCG (Cole-Green): the LMS centile family; three parameters (mu, sigma,
    // nu), so the formula names nu. Then read percentile curves off the fit.
    let f = Formula::new()
        .with_terms("mu", vec![Term::Intercept, Term::smooth(Smooth::ps("x"))])
        .with_terms("sigma", vec![Term::Intercept])
        .with_terms("nu", vec![Term::Intercept]);
    let m = GamlssModel::fit(&data, &pos, &f, &BCCG::new())?;
    let curves = m.centiles(&data, &BCCG::new(), &[3.0, 50.0, 97.0])?;
    println!(
        "BCCG:        converged={}  {} centile curves",
        m.converged(),
        curves.len()
    );

    Ok(())
}
