use polars::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Poisson};

pub struct Generator {
    pub rng: StdRng,
}

impl Generator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn poisson_data(&mut self, n: usize, intercept: f64, slope: f64) -> DataFrame {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 4.0).collect();
        let y: Vec<f64> = x.iter().map(|&x_val| {
            let mu = (intercept + slope * x_val).exp();
            let dist = Poisson::new(mu).unwrap();
            dist.sample(&mut self.rng)
        }).collect();

        df!("x" => x, "y" => y).unwrap()
    }

    pub fn heteroskedastic_gaussian(&mut self, n: usize) -> DataFrame {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 3.0).collect();
        let y: Vec<f64> = x.iter().map(|&x_val| {
            let mu = 10.0 + 2.0 * x_val;
            let sigma = (-1.0 + 0.5 * x_val).exp();
            let dist = Normal::new(mu, sigma).unwrap();
            dist.sample(&mut self.rng)
        }).collect();

        df!("x" => x, "y" => y).unwrap()
    }

    pub fn tensor_surface(&mut self, n: usize) -> DataFrame {
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        let mut y = Vec::new();

        for _ in 0..n {
            // Updated to .random() and .random_range() for Rust 2024 compatibility
            let v1: f64 = self.rng.random();
            let v2: f64 = self.rng.random();

            let dist_sq = (v1 - 0.5).powi(2) + (v2 - 0.5).powi(2);
            let mu = (-dist_sq * 5.0).exp();

            let noise = self.rng.random_range(-0.1..0.1);

            x1.push(v1);
            x2.push(v2);
            y.push(mu + noise);
        }

        df!("x1" => x1, "x2" => x2, "y" => y).unwrap()
    }

    pub fn linear_gaussian(&mut self, n: usize, slope: f64, intercept: f64, sigma: f64) -> DataFrame {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&x_val| {
            let mu = intercept + slope * x_val;
            let dist = Normal::new(mu, sigma).unwrap();
            dist.sample(&mut self.rng)
        }).collect();

        df!("x" => x, "y" => y).unwrap()
    }
}