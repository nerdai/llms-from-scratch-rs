use candle_core::{Module, Result, Tensor, D};
use candle_nn::{embedding, linear_b, seq, Dropout, Embedding, Linear, Sequential, VarBuilder};

const EPS: f32 = 1e-5;

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

    #[allow(dead_code)]
    pub fn gpt_sm_test() -> Self {
        Self {
            vocab_size: 500,
            context_length: 10,
            emb_dim: 12,
            n_heads: 3,
            n_layers: 2,
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
        let (_batch_size, seq_len) = xs.dims2()?;
        let tok_embeds = self.tok_emb.forward(xs)?;
        let pos_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let pos_embeds = self.pos_emb.embeddings().index_select(&pos_ids, 0)?;

        let mut x = tok_embeds.broadcast_add(&pos_embeds)?;
        x = self.drop_emb.forward(&x, true)?;
        x = self.trf_blocks.forward(&x)?;
        x = self.final_norm.forward(&x)?;

        let logits = self.out_head.forward(&x)?;
        Ok(logits)
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

/// Listing 4.2
#[allow(dead_code)]
pub struct LayerNorm {
    eps: f32,
    scale: Tensor,
    shift: Tensor,
}

impl LayerNorm {
    pub fn new(emb_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let scale = vb.get_with_hints(emb_dim, "scale", candle_nn::Init::Const(1.))?;
        let shift = vb.get_with_hints(emb_dim, "shift", candle_nn::Init::Const(0.))?;
        Ok(Self {
            eps: EPS,
            scale,
            shift,
        })
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = xs.mean_keepdim(D::Minus1)?;
        let var = xs.var_keepdim(D::Minus1)?;
        let norm_xs = xs.broadcast_sub(&mean)?.broadcast_div(&var.sqrt()?)?;
        let out_norm = norm_xs
            .broadcast_mul(&self.scale)?
            .broadcast_add(&self.shift)?;
        Ok(out_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use rstest::*;

    #[fixture]
    pub fn vb() -> VarBuilder<'static> {
        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, &dev)
    }

    #[fixture]
    pub fn batch_token_ids() -> Tensor {
        let dev = Device::cuda_if_available(0).unwrap();
        Tensor::new(&[[101_u32, 366, 100, 345], [101, 110, 322, 57]], &dev).unwrap()
    }

    #[rstest]
    fn test_dummy_gpt_model_init(vb: VarBuilder<'_>) {
        let cfg = Config::gpt_sm_test();
        let model = DummyGPTModel::new(cfg, vb).unwrap();

        assert_eq!(model.pos_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.tok_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.trf_blocks.len() as usize, cfg.n_layers);
        assert_eq!(
            model.out_head.weight().dims(),
            &[cfg.vocab_size, cfg.emb_dim]
        );
    }

    #[rstest]
    fn test_dummy_gpt_model_forward(vb: VarBuilder<'_>, batch_token_ids: Tensor) {
        let (batch_size, seq_len) = batch_token_ids.dims2().unwrap();

        let cfg = Config::gpt_sm_test();
        let model = DummyGPTModel::new(cfg, vb).unwrap();

        let logits = model.forward(&batch_token_ids).unwrap();

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);
    }

    #[rstest]
    fn test_layer_norm_init(vb: VarBuilder<'_>) {
        let cfg = Config::gpt_sm_test();
        let layer_norm = LayerNorm::new(cfg.emb_dim, vb).unwrap();
        assert_eq!(layer_norm.eps, EPS);
        assert_eq!(layer_norm.scale.dims(), &[cfg.emb_dim]);
        assert_eq!(layer_norm.shift.dims(), &[cfg.emb_dim]);
        assert_eq!(
            layer_norm.scale.i(..=1).unwrap().to_vec1::<f32>().unwrap(),
            &[1., 1.]
        );
        assert_eq!(
            layer_norm.shift.i(..=1).unwrap().to_vec1::<f32>().unwrap(),
            &[0., 0.]
        );
    }

    #[rstest]
    fn test_layer_norm_forward(vb: VarBuilder<'_>) {
        let cfg = Config::gpt_sm_test();
        let batch_size = 2_usize;
        let batch_example =
            Tensor::rand(0f32, 1f32, (batch_size, cfg.emb_dim), vb.device()).unwrap();
        let layer_norm = LayerNorm::new(cfg.emb_dim, vb.pp("layer_norm")).unwrap();

        let out_norm = layer_norm.forward(&batch_example).unwrap();
        let mean = out_norm.mean_keepdim(D::Minus1).unwrap();
        let var = out_norm.var_keepdim(D::Minus1).unwrap();

        let mean_minus_zero = mean
            .broadcast_sub(&mean.zeros_like().unwrap())
            .unwrap()
            .abs()
            .unwrap();
        let var_minus_one = var
            .broadcast_sub(&var.ones_like().unwrap())
            .unwrap()
            .abs()
            .unwrap();
        let tol_val: f64 = 1e-5;
        let tol = (mean.ones_like().unwrap() * tol_val).unwrap();

        assert_eq!(out_norm.dims(), &[batch_size, cfg.emb_dim]);
        assert_eq!(
            mean_minus_zero
                .lt(&tol)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<u8>()
                .unwrap(),
            batch_size as u8
        );
        assert_eq!(
            var_minus_one
                .lt(&tol)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<u8>()
                .unwrap(),
            batch_size as u8
        );
    }
}
