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
        // handles batches now
        let (b, num_tokens, _d_in) = xs.dims3()?;
        let queries = self.w_query.forward(xs)?;
        let keys = self.w_key.forward(xs)?;
        let values = self.w_value.forward(xs)?;

        let attn_scores = queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)?;
        let mask = Self::get_mask(num_tokens, xs.device())?;
        let masked = Self::masked_fill(
            &attn_scores,
            &mask.broadcast_left(b).unwrap(),
            f32::NEG_INFINITY,
        )?;

        // scale
        let mut attn_weights = softmax(&(masked * self.scaling)?, 1)?;
        // dropout
        attn_weights = self.dropout.forward(&attn_weights, true).unwrap();

        // context vectors
        attn_weights.matmul(&values)
    }
}

/// Listing 3.4
pub struct MultiHeadAttentionWrapper {
    heads: Vec<CausalAttention>,
}

impl MultiHeadAttentionWrapper {
    pub fn new(
        num_heads: usize,
        d_in: usize,
        d_out: usize,
        drop_p: f32,
        qkv_bias: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let heads = (0..num_heads)
            .map(|i| {
                CausalAttention::new(d_in, d_out, drop_p, qkv_bias, vb.pp(format!("head-{}", i)))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        Ok(Self { heads })
    }
}

impl Module for MultiHeadAttentionWrapper {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let context_vectors = self
            .heads
            .iter()
            .map(|attn| attn.forward(&xs).unwrap())
            .collect::<Vec<_>>();
        let reduced = context_vectors
            .into_iter()
            .reduce(|acc, e| Tensor::cat(&[&acc, &e], D::Minus1).unwrap())
            .unwrap(); // todo us ok_or to convert Option to Result
        Ok(reduced)
    }
}

/// Listing 3.5
/// An efficient implementation of Multi-Head Attention
pub struct MultiHeadAttention {
    num_heads: usize,
    d_out: usize,
    head_dim: usize,
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    out_proj: Linear,
    scaling: f64,
    dropout: Dropout,
    drop_p: f32,
}

impl MultiHeadAttention {
    pub fn new(
        d_in: usize,
        d_out: usize,
        drop_p: f32,
        num_heads: usize,
        qkv_bias: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        if d_out % num_heads != 0 {
            panic!("`d_out` must be divisible by `num_heads`.")
        }
        let head_dim = d_out / num_heads;

        let w_query = linear_b(d_in, d_out, qkv_bias, vb.pp("query"))?;
        let w_key = linear_b(d_in, d_out, qkv_bias, vb.pp("key"))?;
        let w_value = linear_b(d_in, d_out, qkv_bias, vb.pp("value"))?;
        let out_proj = linear_b(d_out, d_out, true, vb.pp("out_proj"))?;
        let scaling = 1. / (head_dim as f64).sqrt();
        let dropout = Dropout::new(drop_p);

        Ok(Self {
            num_heads,
            d_out,
            head_dim,
            w_query,
            w_key,
            w_value,
            out_proj,
            scaling,
            dropout,
            drop_p,
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

    pub fn dropout(&self) -> &Dropout {
        &self.dropout
    }

    pub fn out_proj(&self) -> &Linear {
        &self.out_proj
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn d_out(&self) -> usize {
        self.d_out
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn drop_p(&self) -> f32 {
        self.drop_p
    }

    pub fn scaling(&self) -> f64 {
        self.scaling
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        println!("{:?}", xs);
        todo!()
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
    fn test_self_attention_v1_init(vb: VarBuilder<'_>) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let attn_v1_layer = SelfAttentionV1::new(d_in, d_out, vb.pp("attn")).unwrap();

        assert_eq!(attn_v1_layer.w_query.dims(), &[d_in, d_out]);
        assert_eq!(attn_v1_layer.w_key.dims(), &[d_in, d_out]);
        assert_eq!(attn_v1_layer.w_value.dims(), &[d_in, d_out]);
    }

    #[rstest]
    fn test_self_attention_v1_forward(vb: VarBuilder<'_>) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let attn_v1_layer = SelfAttentionV1::new(d_in, d_out, vb.pp("attn")).unwrap();

        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), vb.device()).unwrap();
        let context_vectors = attn_v1_layer.forward(&xs).unwrap();

        assert_eq!(context_vectors.dims(), &[input_length, d_out]);
    }

    #[rstest]
    fn test_self_attention_v2_init(vb: VarBuilder<'_>) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let attn_v2_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn")).unwrap();

        assert_eq!(attn_v2_layer.w_query.weight().dims(), &[d_out, d_in]);
        assert_eq!(attn_v2_layer.w_key.weight().dims(), &[d_out, d_in]);
        assert_eq!(attn_v2_layer.w_value.weight().dims(), &[d_out, d_in]);
    }

    #[rstest]
    fn test_self_attention_v2_forward(vb: VarBuilder<'_>) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let attn_v2_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn")).unwrap();

        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), vb.device()).unwrap();
        let context_vectors = attn_v2_layer.forward(&xs).unwrap();

        assert_eq!(context_vectors.dims(), &[input_length, d_out]);
    }

    #[rstest]
    fn test_causal_attention_init(vb: VarBuilder<'_>) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let casual_attn = CausalAttention::new(d_in, d_out, 0.5_f32, false, vb.pp("attn")).unwrap();

        assert_eq!(casual_attn.w_query.weight().dims(), &[d_out, d_in]);
        assert_eq!(casual_attn.w_key.weight().dims(), &[d_out, d_in]);
        assert_eq!(casual_attn.w_value.weight().dims(), &[d_out, d_in]);
        assert_eq!(casual_attn.drop_p, 0.5_f32);
    }

    #[rstest]
    fn test_causal_attention_forward(vb: VarBuilder<'_>) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let casual_attn = CausalAttention::new(d_in, d_out, 0.5_f32, false, vb.pp("attn")).unwrap();

        // create batch
        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), &vb.device()).unwrap();
        let batch = Tensor::stack(&[&xs, &xs], 0).unwrap();
        let context_vectors = casual_attn.forward(&batch).unwrap();

        assert_eq!(context_vectors.dims(), &[2_usize, input_length, d_out]);
    }

    #[rstest]
    fn test_multihead_attention_wrapper_init(vb: VarBuilder<'_>) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let num_heads = 3_usize;
        let multihead_attn = MultiHeadAttentionWrapper::new(
            num_heads,
            d_in,
            d_out,
            0.5_f32,
            false,
            vb.pp("multihead_attn"),
        )
        .unwrap();

        assert_eq!(multihead_attn.heads.len(), num_heads);

        for i in 0..num_heads {
            let causal_attn = &multihead_attn.heads[i];
            assert_eq!(causal_attn.w_query.weight().dims(), &[d_out, d_in]);
            assert_eq!(causal_attn.w_key.weight().dims(), &[d_out, d_in]);
            assert_eq!(causal_attn.w_value.weight().dims(), &[d_out, d_in]);
            assert_eq!(causal_attn.drop_p, 0.5_f32);
        }
    }

    #[rstest]
    fn test_multihead_attention_wrapper_forward(vb: VarBuilder<'_>) {
        let (d_in, d_out) = (3_usize, 5_usize);
        let num_heads = 3_usize;
        let multihead_attn = MultiHeadAttentionWrapper::new(
            num_heads,
            d_in,
            d_out,
            0.5_f32,
            false,
            vb.pp("multihead_attn"),
        )
        .unwrap();

        // create batch
        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), &vb.device()).unwrap();
        let batch = Tensor::stack(&[&xs, &xs], 0).unwrap();
        let context_vectors = multihead_attn.forward(&batch).unwrap();

        assert_eq!(
            context_vectors.dims(),
            &[2_usize, input_length, num_heads * d_out]
        );
    }

    #[rstest]
    fn test_mha_init(vb: VarBuilder<'_>) {
        let (d_in, d_out, num_heads) = (3_usize, 6_usize, 2_usize);
        let mha =
            MultiHeadAttention::new(d_in, d_out, 0.5_f32, num_heads, false, vb.pp("attn")).unwrap();

        assert_eq!(mha.w_query.weight().dims(), &[d_out, d_in]);
        assert_eq!(mha.w_key.weight().dims(), &[d_out, d_in]);
        assert_eq!(mha.w_value.weight().dims(), &[d_out, d_in]);
        assert_eq!(mha.out_proj.weight().dims(), &[d_out, d_out]);
        assert_eq!(mha.head_dim, d_out / num_heads);
        assert_eq!(mha.drop_p, 0.5_f32);
    }

    #[rstest]
    #[should_panic(expected = "`d_out` must be divisible by `num_heads`.")]
    fn test_mha_init_panics_nondivisible_heads(vb: VarBuilder<'_>) {
        let (d_in, d_out, num_heads) = (3_usize, 6_usize, 4_usize);
        let _ =
            MultiHeadAttention::new(d_in, d_out, 0.5_f32, num_heads, false, vb.pp("attn")).unwrap();
    }
}
