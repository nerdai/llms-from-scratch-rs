use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{linear_b, Dropout, Linear, VarBuilder};

/// Listing 3.1
/// `SelfAttentionV1` is a simple implementation of a self-attention layer.
/// It follows a similar interface to other candle `Module`s.
pub struct SelfAttentionV1 {
    pub w_query: Tensor,
    pub w_key: Tensor,
    pub w_value: Tensor,
    pub scaling: f64,
}

impl SelfAttentionV1 {
    pub fn new(d_in: usize, d_out: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let init = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let w_query = vb.get_with_hints((d_in, d_out), "query", init)?;
        let w_key = vb.get_with_hints((d_in, d_out), "key", init)?;
        let w_value = vb.get_with_hints((d_in, d_out), "value", init)?;
        let scaling = 1. / (w_key.dims()[1] as f64).sqrt();

        Ok(Self {
            w_query,
            w_key,
            w_value,
            scaling,
        })
    }

    pub fn w_query(&self) -> &Tensor {
        &self.w_query
    }

    pub fn w_key(&self) -> &Tensor {
        &self.w_key
    }

    pub fn w_value(&self) -> &Tensor {
        &self.w_value
    }
}

impl Module for SelfAttentionV1 {
    /// Computes the context vector for `xs`
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let queries = xs.matmul(&self.w_query)?;
        let keys = xs.matmul(&self.w_key)?;
        let values = xs.matmul(&self.w_value)?;

        let attn_scores = queries.matmul(&keys.t()?)?;
        let attn_weights = candle_nn::ops::softmax(&(attn_scores * self.scaling)?, 1)?;
        attn_weights.matmul(&values)
    }
}

/// Listing 3.2
/// Note: `candle_nn::linear` takes in dimensions in reverse.  
pub struct SelfAttentionV2 {
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    scaling: f64,
}

impl SelfAttentionV2 {
    pub fn new(d_in: usize, d_out: usize, qkv_bias: bool, vb: VarBuilder<'_>) -> Result<Self> {
        let w_query = linear_b(d_in, d_out, qkv_bias, vb.pp("query"))?;
        let w_key = linear_b(d_in, d_out, qkv_bias, vb.pp("key"))?;
        let w_value = linear_b(d_in, d_out, qkv_bias, vb.pp("value"))?;
        let scaling = 1. / (w_key.weight().dims()[0] as f64).sqrt();

        Ok(Self {
            w_query,
            w_key,
            w_value,
            scaling,
        })
    }

    pub fn w_query(&self) -> &Linear {
        &self.w_query
    }

    pub fn w_key(&self) -> &Linear {
        &self.w_key
    }

    pub fn w_value(&self) -> &Linear {
        &self.w_value
    }
}

impl Module for SelfAttentionV2 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let queries = self.w_query.forward(xs)?;
        let keys = self.w_key.forward(xs)?;
        let values = self.w_value.forward(xs)?;

        let attn_scores = queries.matmul(&keys.t()?)?;
        let attn_weights = candle_nn::ops::softmax(&(attn_scores * self.scaling)?, D::Minus1)?;
        attn_weights.matmul(&values)
    }
}

/// Listing 3.1
pub struct CausalAttention {
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    scaling: f64,
    dropout: Dropout,
    drop_p: f32,
}

impl CausalAttention {
    pub fn new(
        d_in: usize,
        d_out: usize,
        drop_p: f32,
        qkv_bias: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let w_query = linear_b(d_in, d_out, qkv_bias, vb.pp("query"))?;
        let w_key = linear_b(d_in, d_out, qkv_bias, vb.pp("key"))?;
        let w_value = linear_b(d_in, d_out, qkv_bias, vb.pp("value"))?;
        let scaling = 1. / (w_key.weight().dims()[0] as f64).sqrt();
        let dropout = Dropout::new(drop_p);

        Ok(Self {
            w_query,
            w_key,
            w_value,
            scaling,
            dropout,
            drop_p, // a private field in Dropout
        })
    }

    fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<_> = (0..size)
            .flat_map(|i| (0..size).map(move |j| u32::from(j > i)))
            .collect();
        Tensor::from_slice(&mask, (size, size), device)
    }

    fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
        let shape = mask.shape();
        let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
        let m = mask.where_cond(&on_true, on_false)?;
        Ok(m)
    }

    pub fn w_query(&self) -> &Linear {
        &self.w_query
    }

    pub fn w_key(&self) -> &Linear {
        &self.w_key
    }

    pub fn w_value(&self) -> &Linear {
        &self.w_value
    }

    pub fn drop_p(&self) -> f32 {
        self.drop_p
    }
}

impl Module for CausalAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, num_tokens, _d_in) = xs.dims3()?;
        let queries = self.w_query.forward(xs)?;
        let keys = self.w_key.forward(xs)?;
        let values = self.w_value.forward(xs)?;

        let attn_scores = queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)?;
        let mask = Self::get_mask(num_tokens, xs.device())?;
        let masked = Self::masked_fill(&attn_scores, &mask, f32::NEG_INFINITY)?;

        // scale
        let mut attn_weights = softmax(&(masked * self.scaling)?, 1)?;
        // dropout
        attn_weights = self.dropout.forward(&attn_weights, true).unwrap();

        // context vectors
        attn_weights.matmul(&values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use rstest::*;

    #[fixture]
    pub fn device() -> Device {
        Device::cuda_if_available(0).unwrap()
    }

    #[rstest]
    fn test_self_attention_v1_init(device: Device) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn_v1_layer = SelfAttentionV1::new(d_in, d_out, vb.pp("attn")).unwrap();

        assert_eq!(attn_v1_layer.w_query.dims(), &[d_in, d_out]);
        assert_eq!(attn_v1_layer.w_key.dims(), &[d_in, d_out]);
        assert_eq!(attn_v1_layer.w_value.dims(), &[d_in, d_out]);
    }

    #[rstest]
    fn test_self_attention_v1_forward(device: Device) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn_v1_layer = SelfAttentionV1::new(d_in, d_out, vb.pp("attn")).unwrap();

        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), vb.device()).unwrap();
        let context_vectors = attn_v1_layer.forward(&xs).unwrap();

        assert_eq!(context_vectors.dims(), &[input_length, d_out]);
    }

    #[rstest]
    fn test_self_attention_v2_init(device: Device) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn_v2_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn")).unwrap();

        assert_eq!(attn_v2_layer.w_query.weight().dims(), &[d_out, d_in]);
        assert_eq!(attn_v2_layer.w_key.weight().dims(), &[d_out, d_in]);
        assert_eq!(attn_v2_layer.w_value.weight().dims(), &[d_out, d_in]);
    }

    #[rstest]
    fn test_self_attention_v2_forward(device: Device) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn_v2_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn")).unwrap();

        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), vb.device()).unwrap();
        let context_vectors = attn_v2_layer.forward(&xs).unwrap();

        assert_eq!(context_vectors.dims(), &[input_length, d_out]);
    }

    #[rstest]
    fn test_causal_attention_init(device: Device) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let casual_attn = CausalAttention::new(d_in, d_out, 0.5_f32, false, vb.pp("attn")).unwrap();

        assert_eq!(casual_attn.w_query.weight().dims(), &[d_out, d_in]);
        assert_eq!(casual_attn.w_key.weight().dims(), &[d_out, d_in]);
        assert_eq!(casual_attn.w_value.weight().dims(), &[d_out, d_in]);
        assert_eq!(casual_attn.drop_p, 0.5_f32);
    }

    #[rstest]
    fn test_causal_attention_forward(device: Device) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let casual_attn = CausalAttention::new(d_in, d_out, 0.5_f32, false, vb.pp("attn")).unwrap();

        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), &vb.device()).unwrap();
        let context_vectors = casual_attn.forward(&xs).unwrap();

        println!("{:?}", context_vectors);
        assert_eq!(context_vectors.dims(), &[input_length, d_out]);
    }
}
