use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// Listing 3.1
/// `SelfAttentionV1` is a simple implementation of a self-attention layer.
/// It follows a similar interface to other candle `Module`s.
pub struct SelfAttentionV1 {
    w_query: Tensor,
    w_key: Tensor,
    w_value: Tensor,
}

impl SelfAttentionV1 {
    pub fn new(d_in: usize, d_out: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let init = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let w_query = vb.get_with_hints((d_in, d_out), "query", init)?;
        let w_key = vb.get_with_hints((d_in, d_out), "key", init)?;
        let w_value = vb.get_with_hints((d_in, d_out), "value", init)?;
        Ok(Self {
            w_query,
            w_key,
            w_value,
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
    fn forward(&self, _xs: &Tensor) -> Result<Tensor> {
        todo!()
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
}
