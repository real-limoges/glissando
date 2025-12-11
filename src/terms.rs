#[derive(Debug, Clone)]
pub enum Term {
    // 3 types of Terms. A constant (Intercept), a Linear, and a Smooth
    Intercept,
    Linear { col_name: String },
    Smooth(Smooth),
}

#[derive(Debug, Clone)]
pub enum Smooth {
    // 3 tyeps of smooths implemented right now
    PSpline1D {
        col_name: String,
        n_splines: usize,
        degree: usize,
        penalty_order: usize,
    },
    TensorProduct {
        col_name_1: String,
        n_splines_1: usize,
        penalty_order_1: usize,

        col_name_2: String,
        n_splines_2: usize,
        penalty_order_2: usize,

        // I'm just forcing them to have the same degree
        degree: usize,
    },
    RandomEffect {
        col_name: String,
    },
}
