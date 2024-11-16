use candle_core::{Module, Result, Tensor};
use candle_nn::{Dropout, Embedding, Linear, Sequential};

/// Listing 4.1
/// DummyGPTModel
#[allow(dead_code)]
pub struct DummyGPTModel {
    tok_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    trf_blocks: Sequential, // of transformer blocks
    final_norm: DummyLayerNorm,
    out_head: Linear,
}

impl DummyGPTModel {
    pub fn new() -> Result<Self> {
        todo!()
    }
}

impl Module for DummyGPTModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.to_owned())
    }
}

/// Listing 4.1 auxiliary
/// DummyLayerNorm
pub struct DummyLayerNorm {}

impl DummyLayerNorm {
    pub fn new(_emb_dim: usize) -> Result<Self> {
        Ok(Self {})
    }
}

impl Module for DummyLayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.to_owned())
    }
}

/// Listing 4.1 auxiliary
///DummyTransformerBlock
pub struct DummyTransformerBlock {}

impl DummyTransformerBlock {
    pub fn new(_emb_dim: usize) -> Result<Self> {
        Ok(Self {})
    }
}

impl Module for DummyTransformerBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use rstest::*;

    #[fixture]
    pub fn vb() -> VarBuilder<'static> {
        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, &dev)
    }

    #[rstest]
    fn test_dummy_gpt_model_init(vb: VarBuilder<'_>) {
        println!("{:?}", vb.device());
        let (_d_in, _d_out) = (3_usize, 5_usize);
        let _dummy_gpt = DummyGPTModel::new().unwrap();
        assert!(true);
    }
}
