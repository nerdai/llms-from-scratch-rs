//! # Custom addons module to Candle
//!
//! #### Features
//! - `SequentialT`: a version of `Sequential` that is `ModuleT`
use candle_core::{Device, IndexOp, ModuleT, Result, Tensor};

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

/// Trait for returning top-k elements of a Tensor
pub trait TopK {
    /// Returns a `Tensor`'s top-k elements and its positions along dim 0
    fn topk_last_dim0(&self, top_k: usize) -> Result<(Tensor, Tensor)>;

    /// Returns a `Tensor`'s top-k elements and its positions along dim 1
    fn topk_last_dim1(&self, top_k: usize) -> Result<(Tensor, Tensor)>;
}

impl TopK for Tensor {
    fn topk_last_dim0(&self, top_k: usize) -> Result<(Tensor, Tensor)> {
        let top_pos = self.arg_sort_last_dim(false)?;
        let top_pos = top_pos.i(..top_k)?;
        let top_els = self.i(top_pos.to_vec1::<u32>()?)?;
        Ok((top_els, top_pos))
    }

    fn topk_last_dim1(&self, top_k: usize) -> Result<(Tensor, Tensor)> {
        // get CUDA error sometimes when using `.arg_sort_last_dim`
        // moving to CPU to carry out the op
        let top_pos = self.to_device(&Device::Cpu)?.arg_sort_last_dim(false)?;
        let top_pos = top_pos.to_device(&Device::cuda_if_available(0)?)?;
        let (batch_size, vocab_size) = top_pos.dims2()?;
        let top_pos = top_pos.i((.., ..top_k))?.flatten_all()?;

        // get appropriate sum starting index
        let aux = Tensor::arange(0u32, batch_size as u32, self.device())?;
        let aux = (vocab_size as f64 * aux.broadcast_left(top_k)?.t()?.flatten_all()?)?;
        let top_pos = (top_pos + &aux)?;
        let top_els = self.flatten_all()?.i(top_pos.to_vec1::<u32>()?)?;

        // reshape
        let top_els = top_els.reshape((batch_size, top_k))?;
        let top_pos = (top_pos - &aux)?;
        let top_pos = top_pos.reshape((batch_size, top_k))?;
        Ok((top_els, top_pos))
    }
}
