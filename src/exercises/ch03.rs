use crate::Exercise;

/// 3.1
pub struct X1;

impl Exercise for X1 {
    fn name(&self) -> String {
        String::from("3.1")
    }

    fn statement(&self) -> String {
        let stmt = "Note that `nn.Linear` in `SelfAttention_v2` uses a \
        different weight initialization scheme as `nn.Parameter(torch.rand(d_in, d_out))` \
        used in `SelfAttention_v1`, which causes both mechanisms to produce \
        different results. To check that both implementations, `SelfAttention_v1` \
        and `SelfAttention_v2`, are otherwise similar, we can transfer the \
        weight matrices from a `SelfAttention_v2` object to a `SelfAttention_v1`, \
        such that both objects then produce the same results. Your task is to \
        correctly assign the weights from an instance of `SelfAttention_v2` to \
        an instance of `SelfAttention_v1`. To do this, you need to understand \
        the relationship between the weights in both versions. (Hint: `nn.Linear` \
        stores the weight matrix in a transposed form.) After the assignment, \
        you should observe that both instances produce the same outputs.";
        stmt.to_string()
    }

    fn main(&self) {
        use crate::listings::ch03::{SelfAttentionV1, SelfAttentionV2};
        use candle_core::{DType, Device, Module, Tensor};
        use candle_nn::{VarBuilder, VarMap};

        let (d_in, d_out) = (3_usize, 5_usize);
        let varmap = VarMap::new();
        let dev = Device::cuda_if_available(0).unwrap();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let attn_v2_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn_v2")).unwrap();
        let attn_v1_layer = SelfAttentionV1 {
            w_query: attn_v2_layer.w_query().weight().t().unwrap(),
            w_key: attn_v2_layer.w_key().weight().t().unwrap(),
            w_value: attn_v2_layer.w_value().weight().t().unwrap(),
            scaling: 1. / (attn_v2_layer.w_key().weight().dims()[0] as f64).sqrt(),
        };

        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), &dev).unwrap();
        let context_vectors_from_v1 = attn_v1_layer.forward(&xs).unwrap();
        let context_vectors_from_v2 = attn_v2_layer.forward(&xs).unwrap();

        println!(
            "Context vectors from SelfAttention V1 and V2 are equal when using same weights: {}",
            context_vectors_from_v1.to_vec2::<f32>().unwrap()
                == context_vectors_from_v2.to_vec2::<f32>().unwrap()
        )
    }
}

/// Exercise 3.2
pub struct X2;

impl Exercise for X2 {
    fn name(&self) -> String {
        String::from("3.2")
    }

    fn statement(&self) -> String {
        let stmt = "Change the input arguments for the \
        `MultiHeadAttentionWrapper(..., num_heads=2)` call such that the output \
        context vectors are two-dimensional instead of four dimensional while \
        keeping the setting `num_heads=2`. Hint: You don’t have to modify the \
        class implementation; you just have to change one of the other input arguments.";
        stmt.to_string()
    }

    fn main(&self) {
        use crate::listings::ch03::MultiHeadAttentionWrapper;
        use candle_core::{DType, Device, Module, Tensor};
        use candle_nn::{VarBuilder, VarMap};

        let (d_in, d_out) = (3_usize, 1_usize); // set d_out to 1 to get desired final dim
        let varmap = VarMap::new();
        let dev = Device::cuda_if_available(0).unwrap();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let num_heads = 2_usize;
        let mha =
            MultiHeadAttentionWrapper::new(num_heads, d_in, d_out, 0.0_f32, false, vb.pp("mha"))
                .unwrap();

        // create random input batch
        let input_length = 6_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), vb.device()).unwrap();
        let batch = Tensor::stack(&[&xs, &xs], 0).unwrap();
        println!("batch shape: {:?}", batch);

        // run forward on mha
        let context_vectors = mha.forward(&batch).unwrap();
        println!("context_vectors.shape: {:?}", context_vectors);
        println!("context_vectors: {:?}", context_vectors.to_vec3::<f32>());
    }
}

/// Exercise 3.3
pub struct X3;

impl Exercise for X3 {
    fn name(&self) -> String {
        String::from("3.3")
    }

    fn statement(&self) -> String {
        let stmt = "Using the `MultiHeadAttention` class, initialize a \
        multi-head attention module that has the same number of attention heads \
        as the smallest GPT-2 model (12 attention heads). Also ensure that you \
        use the respective input and output embedding sizes similar to GPT-2 \
        (768 dimensions). Note that the smallest GPT-2 model supports a context \
        length of 1,024 tokens.";
        stmt.to_string()
    }

    fn main(&self) {
        use crate::listings::ch03::MultiHeadAttention;
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        let (d_in, d_out, num_heads) = (768_usize, 768_usize, 12_usize); // set d_out to 1 to get desired final dim
        let varmap = VarMap::new();
        let dev = Device::cuda_if_available(0).unwrap();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mha =
            MultiHeadAttention::new(d_in, d_out, 0.0_f32, num_heads, false, vb.pp("mha")).unwrap();

        println!("mha.num_heads: {:?}", mha.num_heads());
        println!("mha.head_dim: {:?}", mha.head_dim());
        println!("mha.w_query.shape: {:?}", mha.w_query().weight().dims());
    }
}
