use crate::Exercise;

/// 3.1
pub struct X3P1;

impl Exercise for X3P1 {
    fn name(&self) -> String {
        String::from("3.1")
    }

    fn main(&self) {
        use crate::listings::ch03::{SelfAttentionV1, SelfAttentionV2};
        use candle_core::{DType, Device, Module, Tensor};
        use candle_nn::{VarBuilder, VarMap};

        let (d_in, d_out) = (3_usize, 5_usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let attn_v2_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn_v2")).unwrap();
        let attn_v1_layer = SelfAttentionV1 {
            w_query: attn_v2_layer.w_query().weight().t().unwrap(),
            w_key: attn_v2_layer.w_key().weight().t().unwrap(),
            w_value: attn_v2_layer.w_value().weight().t().unwrap(),
            scaling: 1. / (attn_v2_layer.w_key().weight().dims()[0] as f64).sqrt(),
        };

        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), &Device::Cpu).unwrap();
        let context_vectors_from_v1 = attn_v1_layer.forward(&xs).unwrap();
        let context_vectors_from_v2 = attn_v2_layer.forward(&xs).unwrap();

        println!(
            "Context vectors from SelfAttention V1 and V2 are equal when using same weights: {}",
            context_vectors_from_v1.to_vec2::<f32>().unwrap()
                == context_vectors_from_v2.to_vec2::<f32>().unwrap()
        )
    }
}
