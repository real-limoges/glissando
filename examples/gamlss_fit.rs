use gamlss_rs::distributions::StudentT;
use gamlss_rs::{DataSet, Formula, GamlssError, GamlssModel, Smooth, Term};
use ndarray::Array1;
use rand::RngExt;

fn main() -> Result<(), GamlssError> {
    // Generate Synthetic Data

    let mut rng = rand::rng();
    let n = 200;

    let x_vals: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();

    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| {
            let mu = x.sin();
            let sigma = 0.5 + 0.1 * x;

            let noise: f64 = rng.random_range(-1.0..1.0);
            mu + sigma * noise
        })
        .collect();

    let y = Array1::from_vec(y_vals);
    let mut data = DataSet::new();
    data.insert_column("x", Array1::from_vec(x_vals));

    // the formula
    let mut formulas = Formula::new();

    // Mu: Smooth P-Spline
    formulas.add_terms(
        "mu".to_string(),
        vec![
            Term::Intercept,
            Term::Smooth(Smooth::PSpline1D {
                col_name: "x".to_string(),
                n_splines: 20,
                degree: 3,
                penalty_order: 2,
            }),
        ],
    );

    // Sigma: Linear
    formulas.add_terms(
        "sigma".to_string(),
        vec![
            Term::Intercept,
            Term::Linear {
                col_name: "x".to_string(),
            },
        ],
    );

    // Nu: Constant
    formulas.add_terms("nu".to_string(), vec![Term::Intercept]);

    // Fit
    println!("Fitting GAMLSS model...");
    let model = GamlssModel::fit(&y, &data, &formulas, &StudentT::new())?;
    println!("Successfully Trained GAMLSS Model!");

    // 5. Inspect Results
    let mu_model = &model.models["mu"];
    let sigma_model = &model.models["sigma"];
    let nu_model = &model.models["nu"];

    println!("--- Results ---");
    println!("Mu coefficients count: {}", mu_model.coefficients.len());
    println!("Sigma coefficients: {:?}", sigma_model.coefficients);
    println!("Nu coefficients: {:?}", nu_model.coefficients);

    Ok(())
}
