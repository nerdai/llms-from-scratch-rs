use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_b, Linear, VarBuilder};

/// Listing 3.1
/// `SelfAttentionV1` is a simple implementation of a self-attention layer.
/// It follows a similar interface to other candle `Module`s.
pub struct SelfAttentionV1 {
    w_query: Tensor,
    w_key: Tensor,
    w_value: Tensor,
    scaling: f64,
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

pub struct SelfAttentionV2 {
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    _scaling: f64,
}

impl SelfAttentionV2 {
    pub fn new(d_in: usize, d_out: usize, qkv_bias: bool, vb: VarBuilder<'_>) -> Result<Self> {
        let w_query = linear_b(d_in, d_out, qkv_bias, vb.pp("query"))?;
        let w_key = linear_b(d_in, d_out, qkv_bias, vb.pp("key"))?;
        let w_value = linear_b(d_in, d_out, qkv_bias, vb.pp("value"))?;
        let scaling = 1. / (w_key.weight().dims()[1] as f64).sqrt();

        Ok(Self {
            w_query,
            w_key,
            w_value,
            _scaling: scaling,
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use rstest::*;

    #[rstest]
    fn test_self_attention_v1_init() {
        let (d_in, d_out) = (3_usize, 5_usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let attn_v1_layer = SelfAttentionV1::new(d_in, d_out, vb.pp("attn")).unwrap();

        assert_eq!(attn_v1_layer.w_query.dims(), &[d_in, d_out]);
        assert_eq!(attn_v1_layer.w_key.dims(), &[d_in, d_out]);
        assert_eq!(attn_v1_layer.w_value.dims(), &[d_in, d_out]);
    }

    #[rstest]
    fn test_self_attention_v1_forward() {
        let (d_in, d_out) = (3_usize, 5_usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let attn_v1_layer = SelfAttentionV1::new(d_in, d_out, vb.pp("attn")).unwrap();

        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), &Device::Cpu).unwrap();
        let context_vectors = attn_v1_layer.forward(&xs).unwrap();

        assert_eq!(context_vectors.dims(), &[input_length, d_out]);
    }
}
