//! `DataSet` — a named collection of equal-length numeric columns. The
//! length-invariant is enforced at insertion time so downstream code can rely
//! on `n_obs()` reflecting every column uniformly.

use ndarray::Array1;
use std::collections::HashMap;
use std::ops::Deref;

use crate::error::GamlssError;

/// A dataset of named columns. All columns are guaranteed to have the same length;
/// this invariant is enforced by [`DataSet::insert_column`] and the [`TryFrom`] impl.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        try_from = "HashMap<String, Array1<f64>>",
        into = "HashMap<String, Array1<f64>>"
    )
)]
pub struct DataSet(HashMap<String, Array1<f64>>);

impl DataSet {
    /// Creates an empty dataset.
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Returns the column with the given name, if present.
    pub fn column(&self, name: &str) -> Option<&Array1<f64>> {
        self.0.get(name)
    }

    /// Returns the number of observations (rows), or `None` if the dataset is empty.
    pub fn n_obs(&self) -> Option<usize> {
        self.0.values().next().map(|v| v.len())
    }

    /// Returns the number of columns in the dataset.
    pub fn n_columns(&self) -> usize {
        self.0.len()
    }

    /// Inserts or replaces a named column.
    ///
    /// # Panics
    ///
    /// Panics if `values` has a different length than existing columns. Mismatched
    /// column lengths are a programmer error; use [`DataSet::try_insert_column`] for
    /// fallible insertion of runtime-unsafe data.
    ///
    /// # Examples
    ///
    /// ```
    /// use glissando::DataSet;
    /// use ndarray::Array1;
    ///
    /// let mut data = DataSet::new();
    /// data.insert_column("x", Array1::from_vec(vec![1.0, 2.0, 3.0]));
    /// assert_eq!(data.n_obs(), Some(3));
    /// ```
    pub fn insert_column(&mut self, name: impl Into<String>, values: Array1<f64>) {
        if let Some(existing) = self.n_obs() {
            assert_eq!(
                values.len(),
                existing,
                "DataSet column length mismatch: tried to insert {} rows into a dataset of {} rows",
                values.len(),
                existing,
            );
        }
        self.0.insert(name.into(), values);
    }

    /// Inserts or replaces a named column, returning an error if the length disagrees
    /// with existing columns.
    pub fn try_insert_column(
        &mut self,
        name: impl Into<String>,
        values: Array1<f64>,
    ) -> Result<(), GamlssError> {
        if let Some(existing) = self.n_obs() {
            if values.len() != existing {
                return Err(GamlssError::Input(format!(
                    "DataSet column length mismatch: tried to insert {} rows into a dataset of {} rows",
                    values.len(),
                    existing,
                )));
            }
        }
        self.0.insert(name.into(), values);
        Ok(())
    }

    /// Creates a `DataSet` from a `HashMap<String, Vec<f64>>`, converting each to `Array1<f64>`.
    ///
    /// # Errors
    ///
    /// Returns `GamlssError::Input` if columns have differing lengths.
    pub fn from_vecs(data: HashMap<String, Vec<f64>>) -> Result<Self, GamlssError> {
        let mut ds = Self::new();
        for (name, values) in data {
            ds.try_insert_column(name, Array1::from_vec(values))?;
        }
        Ok(ds)
    }
}

impl Deref for DataSet {
    type Target = HashMap<String, Array1<f64>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TryFrom<HashMap<String, Array1<f64>>> for DataSet {
    type Error = GamlssError;

    fn try_from(map: HashMap<String, Array1<f64>>) -> Result<Self, Self::Error> {
        let mut ds = Self::new();
        for (name, values) in map {
            ds.try_insert_column(name, values)?;
        }
        Ok(ds)
    }
}

impl From<DataSet> for HashMap<String, Array1<f64>> {
    fn from(ds: DataSet) -> Self {
        ds.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn dataset_insert_and_retrieve() {
        let mut d = DataSet::new();
        d.insert_column("x", array![1.0, 2.0, 3.0]);
        assert_eq!(d.column("x").unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
        assert!(d.column("missing").is_none());
    }

    #[test]
    fn dataset_n_obs_and_n_columns() {
        let mut d = DataSet::new();
        assert_eq!(d.n_obs(), None);
        assert_eq!(d.n_columns(), 0);
        d.insert_column("x", array![1.0, 2.0]);
        d.insert_column("z", array![3.0, 4.0]);
        assert_eq!(d.n_obs(), Some(2));
        assert_eq!(d.n_columns(), 2);
    }

    #[test]
    fn dataset_from_vecs_round_trip() {
        let mut m: HashMap<String, Vec<f64>> = HashMap::new();
        m.insert("x".into(), vec![1.0, 2.0]);
        m.insert("y".into(), vec![3.0, 4.0]);
        let d = DataSet::from_vecs(m).unwrap();
        assert_eq!(d.n_columns(), 2);
        assert_eq!(d.column("x").unwrap().to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn dataset_from_vecs_rejects_mismatched_lengths() {
        let mut m: HashMap<String, Vec<f64>> = HashMap::new();
        m.insert("x".into(), vec![1.0, 2.0]);
        m.insert("y".into(), vec![3.0, 4.0, 5.0]);
        let err = DataSet::from_vecs(m).unwrap_err();
        assert!(matches!(err, GamlssError::Input(_)));
    }

    #[test]
    fn dataset_try_from_hashmap_rejects_mismatched_lengths() {
        let mut m: HashMap<String, Array1<f64>> = HashMap::new();
        m.insert("x".into(), array![1.0, 2.0]);
        m.insert("y".into(), array![3.0, 4.0, 5.0]);
        let err = DataSet::try_from(m).unwrap_err();
        assert!(matches!(err, GamlssError::Input(_)));
    }

    #[test]
    fn dataset_try_from_hashmap_accepts_equal_lengths() {
        let mut m: HashMap<String, Array1<f64>> = HashMap::new();
        m.insert("a".into(), array![5.0, 6.0]);
        m.insert("b".into(), array![7.0, 8.0]);
        let d = DataSet::try_from(m).unwrap();
        assert_eq!(d.n_columns(), 2);
    }

    #[test]
    #[should_panic(expected = "DataSet column length mismatch")]
    fn dataset_insert_column_panics_on_mismatched_length() {
        let mut d = DataSet::new();
        d.insert_column("x", array![1.0, 2.0, 3.0]);
        d.insert_column("y", array![1.0, 2.0]);
    }

    #[test]
    fn dataset_try_insert_column_returns_error_on_mismatch() {
        let mut d = DataSet::new();
        d.try_insert_column("x", array![1.0, 2.0, 3.0]).unwrap();
        let err = d.try_insert_column("y", array![1.0, 2.0]).unwrap_err();
        assert!(matches!(err, GamlssError::Input(_)));
    }

    #[test]
    fn dataset_default_is_empty() {
        let d = DataSet::default();
        assert_eq!(d.n_columns(), 0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn dataset_json_round_trip() {
        let mut d = DataSet::new();
        d.insert_column("x", array![1.0, 2.0]);
        let s = serde_json::to_string(&d).unwrap();
        let back: DataSet = serde_json::from_str(&s).unwrap();
        assert_eq!(back.column("x").unwrap().to_vec(), vec![1.0, 2.0]);
    }
}
