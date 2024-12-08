//! # Custom addons module to Candle
//!
//! #### Features
//! - `SequentialT`: a version of `Sequential` that is `ModuleT`
use candle_core::{ModuleT, Result, Tensor};

/// A sequential layer combining multiple other layers.
pub struct SequentialT {
    layers: Vec<Box<dyn ModuleT>>,
}

/// Creates a new empty sequential layer.
pub fn seqt() -> SequentialT {
    SequentialT { layers: vec![] }
}

impl SequentialT {
    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl ModuleT for SequentialT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward_t(&xs, train)?
        }
        Ok(xs)
    }
}

impl SequentialT {
    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: ModuleT + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) -> Result<Tensor> + Send + Sync,
    {
        self.add(candle_nn::func(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all(&self, xs: &Tensor, train: bool) -> Result<Vec<Tensor>> {
        let mut vec = Vec::with_capacity(self.layers.len());
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward_t(&xs, train)?;
            vec.push(xs.clone())
        }
        Ok(vec)
    }
}
