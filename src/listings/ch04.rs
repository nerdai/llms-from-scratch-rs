use candle_core::{Module, Result, Tensor};
use candle_nn::{embedding, linear_b, seq, Dropout, Embedding, Linear, Sequential, VarBuilder};

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f32,
    pub qkv_bias: bool,
}

impl Config {
    #[allow(dead_code)]
    pub fn gpt2_124m() -> Self {
        Self {
            vocab_size: 50_257,
            context_length: 1_024,
            emb_dim: 768,
            n_heads: 12,
            n_layers: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }
}

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
    pub fn new(cfg: Config, vb: VarBuilder<'_>) -> Result<Self> {
        let tok_emb = embedding(cfg.vocab_size, cfg.emb_dim, vb.pp("tok_emb"))?;
        let pos_emb = embedding(cfg.context_length, cfg.emb_dim, vb.pp("pos_emb"))?;
        let drop_emb = Dropout::new(cfg.drop_rate);
        let mut trf_blocks = seq();
        for _ in 0..cfg.n_layers {
            trf_blocks = trf_blocks.add(DummyTransformerBlock::new(cfg).unwrap());
        }
        let final_norm = DummyLayerNorm::new(cfg.emb_dim)?;
        let out_head = linear_b(cfg.emb_dim, cfg.vocab_size, false, vb.pp("out_head"))?;
        Ok(Self {
            tok_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        })
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
    #[allow(unused_variables)]
    pub fn new(emb_dim: usize) -> Result<Self> {
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
    #[allow(unused_variables)]
    pub fn new(cfg: Config) -> Result<Self> {
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
        let (_d_in, _d_out) = (3_usize, 5_usize);
        let _dummy_gpt = DummyGPTModel::new(Config::gpt2_124m(), vb).unwrap();
        assert!(true);
    }
}
