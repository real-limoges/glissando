#[derive(Debug, Clone)]
pub enum Term {
    // 3 types of Terms. A constant (Intercept), a Linear, and a Smooth
    Intercept,
    Linear { col_name: String },
    Smooth(Smooth),
}

impl Term {
    /// Returns the column names referenced by this term.
    pub fn column_names(&self) -> Vec<&str> {
        match self {
            Term::Intercept => vec![],
            Term::Linear { col_name } => vec![col_name.as_str()],
            Term::Smooth(smooth) => smooth.column_names(),
        }
    }
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

impl Smooth {
    /// Returns the column names referenced by this smooth term.
    pub fn column_names(&self) -> Vec<&str> {
        match self {
            Smooth::PSpline1D { col_name, .. } => vec![col_name.as_str()],
            Smooth::TensorProduct {
                col_name_1,
                col_name_2,
                ..
            } => vec![col_name_1.as_str(), col_name_2.as_str()],
            Smooth::RandomEffect { col_name } => vec![col_name.as_str()],
        }
    }
}
