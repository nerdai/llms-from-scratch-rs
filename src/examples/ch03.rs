use crate::Example;
use candle_core::{Device, Result, Tensor};
use candle_nn::Module;

fn get_inputs() -> Tensor {
    let dev = Device::cuda_if_available(0).unwrap();
    Tensor::new(
        &[
            [0.43_f32, 0.15, 0.89], // Your
            [0.55, 0.87, 0.66],     // journey
            [0.57, 0.85, 0.64],     // starts
            [0.22, 0.58, 0.33],     // with
            [0.77, 0.25, 0.10],     // one
            [0.05, 0.80, 0.55],     // step
        ],
        &dev,
    )
    .unwrap()
}

// use for cuda enabled dev
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

/// Example 03.01
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Computing attention scores as a dot product.")
    }

    fn page_source(&self) -> usize {
        57_usize
    }

    fn main(&self) {
        use candle_core::{IndexOp, Tensor};
        use candle_nn::ops::softmax;

        let inputs = get_inputs();
        let dev = inputs.device().to_owned();

        let query = inputs
            .index_select(&Tensor::new(&[1u32], &dev).unwrap(), 0)
            .unwrap();

        // compute attention scores
        let mut optional_attn_scores_2: Option<Tensor> = None;
        for i in 0..inputs.dims()[0] {
            let x_i = inputs
                .index_select(&Tensor::new(&[i as u32], &dev).unwrap(), 0)
                .unwrap();
            let a_i = x_i
                .matmul(&query.t().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap();
            optional_attn_scores_2 = match optional_attn_scores_2 {
                Some(attn_scores_2) => Some(Tensor::cat(&[&attn_scores_2, &a_i], 0).unwrap()),
                None => Some(a_i),
            }
        }

        if let Some(attn_scores_2) = optional_attn_scores_2 {
            // raw attention scores
            println!("Raw attention scores: {:?}", attn_scores_2);

            // basic normalization
            let sum = attn_scores_2.sum_all().unwrap();
            let normalized_attn_scores = (attn_scores_2.broadcast_div(&sum))
                .unwrap()
                .to_vec1::<f32>();
            println!("Normalized attention scores: {:?}", normalized_attn_scores);

            // naive softmax normalization
            let exponentiator = attn_scores_2.exp().unwrap();
            let exponentiator_sum = exponentiator.sum_all().unwrap();
            let naive_softmax_attn_scores =
                exponentiator.broadcast_div(&exponentiator_sum).unwrap();
            println!(
                "Naive Softmax-normalized attention scores: {:?}",
                naive_softmax_attn_scores
            );

            // candle softmax
            let softmax_attn_scores = softmax(&attn_scores_2, 0).unwrap();
            println!(
                "Softmax-normalized attention scores: {:?}",
                softmax_attn_scores
            );

            // compute second context vector
            let mut context_vec_2 = Tensor::zeros_like(&query).unwrap();
            for i in 0..inputs.dims()[0] {
                let x_i = inputs
                    .index_select(&Tensor::new(&[i as u32], &dev).unwrap(), 0)
                    .unwrap();
                context_vec_2 = context_vec_2
                    .add(
                        &x_i.broadcast_mul(&softmax_attn_scores.i(i).unwrap())
                            .unwrap(),
                    )
                    .unwrap();
            }
            println!("Context vector 2: {:?}", context_vec_2.to_vec2::<f32>());
        }
    }
}

/// Example 03.02
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        String::from("Manual computation of multiple context vectors simultaneously.")
    }

    fn page_source(&self) -> usize {
        62_usize
    }

    fn main(&self) {
        use candle_nn::ops::softmax;

        let inputs = get_inputs();

        // matmul to get attn scores
        let attn_scores = inputs.matmul(&inputs.t().unwrap()).unwrap();

        // apply softmax
        let attn_weights = softmax(&attn_scores, 1).unwrap();

        // check sums along rows equal to 1
        let sum = attn_weights.sum(1).unwrap();

        // context vectors
        let all_context_vectors = attn_weights.matmul(&inputs).unwrap();

        println!("Attention Weights: {:?}\n", attn_weights.to_vec2::<f32>());
        println!("All Rows Sum: {:?}\n\n", sum.flatten_all());
        println!(
            "Context Vectors: {:?}",
            all_context_vectors.to_vec2::<f32>()
        );
    }
}

/// Example 03.03
pub struct EG03;

impl Example for EG03 {
    fn description(&self) -> String {
        let desc = "Implementing the self-attention mechanism with \
        trainable weights to compute single context vector.";
        String::from(desc)
    }

    fn page_source(&self) -> usize {
        66_usize
    }

    fn main(&self) {
        use candle_core::DType;
        use candle_nn::init::DEFAULT_KAIMING_NORMAL;
        use candle_nn::ops::softmax;
        use candle_nn::{VarBuilder, VarMap};

        let inputs = get_inputs();
        let dev = inputs.device().to_owned();
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let x_2 = inputs
            .index_select(&Tensor::new(&[1u32], &dev).unwrap(), 0)
            .unwrap();
        let d_in = x_2.dims()[1]; // input embedding dim
        let d_out = 2_usize;

        // projections
        let init = DEFAULT_KAIMING_NORMAL;
        let w_query = vs.get_with_hints((d_in, d_out), "query", init).unwrap();
        let w_key = vs.get_with_hints((d_in, d_out), "key", init).unwrap();
        let w_value = vs.get_with_hints((d_in, d_out), "value", init).unwrap();

        // query, key, value vectors
        let query_2 = x_2.matmul(&w_query).unwrap();
        let key_2 = x_2.matmul(&w_key).unwrap();
        let value_2 = x_2.matmul(&w_value).unwrap();

        println!("Query 2: {:?}", query_2.to_vec2::<f32>());
        println!("Key 2: {:?}", key_2.to_vec2::<f32>());
        println!("Value 2: {:?}", value_2.to_vec2::<f32>());

        // key and value vectors all input elements
        let keys = inputs.matmul(&w_key).unwrap();
        let values = inputs.matmul(&w_value).unwrap();

        println!("Keys shape: {:?}", keys);
        println!("Values shape: {:?}", values);

        // compute attn scores
        let attn_scores = query_2.matmul(&keys.t().unwrap()).unwrap();
        println!("Attn scores: {:?}", attn_scores.to_vec2::<f32>());

        // compute attns weights by first scaling then softmax
        let d_k = Tensor::new(&[f32::powf(keys.dims()[1] as f32, 0.5_f32)], &dev).unwrap();
        let attn_weights = softmax(&attn_scores.broadcast_div(&d_k).unwrap(), 1).unwrap();
        println!("Attn weights: {:?}", attn_weights.to_vec2::<f32>());

        // compute context vector
        let context_vec_2 = attn_weights.matmul(&values).unwrap();
        println!("Context vector 2: {:?}", context_vec_2.to_vec2::<f32>());
    }
}

/// Example 03.04
pub struct EG04;

impl Example for EG04 {
    fn description(&self) -> String {
        String::from(
            "Implement self-attention mechanism to compute context vectors in the input sequence.",
        )
    }

    fn page_source(&self) -> usize {
        71_usize
    }

    fn main(&self) {
        use crate::listings::ch03::SelfAttentionV1;
        use candle_core::{DType, Module};
        use candle_nn::{VarBuilder, VarMap};

        let inputs = get_inputs();
        let d_in = inputs.dims()[1]; // input embedding dim
        let d_out = 2_usize;

        // construct self attention layer
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let attn_v1_layer = SelfAttentionV1::new(d_in, d_out, vb.pp("attn")).unwrap();

        // run a random, embedded input sequence through self-attention
        let context_vectors = attn_v1_layer.forward(&inputs).unwrap();

        println!("context vectors: {:?}", context_vectors.to_vec2::<f32>());
    }
}

/// Example 03.05
pub struct EG05;

impl Example for EG05 {
    fn description(&self) -> String {
        let desc = "Implement self-attention mechanism to compute \
        contextualized vectors, using candle_nn::Linear.";
        String::from(desc)
    }

    fn page_source(&self) -> usize {
        73_usize
    }

    fn main(&self) {
        use crate::listings::ch03::SelfAttentionV2;
        use candle_core::{DType, Module};
        use candle_nn::{VarBuilder, VarMap};

        let inputs = get_inputs();
        let d_in = inputs.dims()[1]; // input embedding dim
        let d_out = 2_usize;

        // construct self attention layer
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let attn_v2_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn")).unwrap();

        // run a random, embedded input sequence through self-attention
        let context_vectors = attn_v2_layer.forward(&inputs).unwrap();

        println!("context vectors: {:?}", context_vectors.to_vec2::<f32>());
    }
}

/// Example 03.06
pub struct EG06;

impl EG06 {
    fn main_with_return(&self) -> Result<Tensor> {
        use crate::listings::ch03::SelfAttentionV2;
        use candle_core::{DType, Module, D};
        use candle_nn::ops::softmax;
        use candle_nn::{VarBuilder, VarMap};

        let inputs = get_inputs();
        let d_in = inputs.dims()[1]; // input embedding dim
        let d_out = 2_usize;

        // construct self attention layer
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, inputs.device());
        let attn_v2_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn"))?;

        // attn scores
        let queries = attn_v2_layer.w_query().forward(&inputs)?;
        let keys = attn_v2_layer.w_key().forward(&inputs)?;
        let attn_scores = queries.matmul(&keys.t().unwrap())?;
        let scaling = 1. / (keys.dims()[1] as f64).sqrt();
        let attn_weights = softmax(&(attn_scores * scaling)?, 1)?;

        // causal mask
        let context_length = inputs.dims()[0];
        let mask_simple: Vec<_> = (0..context_length as u32)
            .flat_map(|i| (0..context_length as u32).map(move |j| f32::from(j <= i)))
            .collect();
        let mask_simple = Tensor::from_slice(
            &mask_simple,
            (context_length, context_length),
            inputs.device(),
        )?;
        let masked_simple = (attn_weights * mask_simple)?;
        println!("masked_simple: {:?}", masked_simple.to_vec2::<f32>());

        // normalize
        let row_sums = masked_simple.sum_keepdim(D::Minus1)?;
        let attn_weights = masked_simple.broadcast_div(&row_sums)?;
        println!("masked_simple_norm: {:?}", attn_weights.to_vec2::<f32>());
        Ok(attn_weights)
    }
}

impl Example for EG06 {
    fn description(&self) -> String {
        String::from("Compute causal attention weights.")
    }

    fn page_source(&self) -> usize {
        75_usize
    }

    fn main(&self) {
        let _ = self.main_with_return();
    }
}

/// Example 03.07
pub struct EG07;

impl EG07 {
    fn main_with_return(&self) -> Result<Tensor> {
        use crate::listings::ch03::SelfAttentionV2;
        use candle_core::{DType, Module};
        use candle_nn::ops::softmax;
        use candle_nn::{VarBuilder, VarMap};

        let inputs = get_inputs();
        let d_in = inputs.dims()[1]; // input embedding dim
        let d_out = 2_usize;

        // construct self attention layer
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, inputs.device());
        let attn_v2_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn"))?;

        // attn scores
        let queries = attn_v2_layer.w_query().forward(&inputs)?;
        let keys = attn_v2_layer.w_key().forward(&inputs)?;
        let attn_scores = queries.matmul(&keys.t()?)?;

        // efficient computation of causal mask
        let context_length = attn_scores.dims()[0];
        let mask: Vec<_> = (0..context_length as u32)
            .flat_map(|i| (0..context_length as u32).map(move |j| u32::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (context_length, context_length), inputs.device())?;
        let masked = masked_fill(&attn_scores, &mask, f32::NEG_INFINITY)?;
        println!("masked: {:?}", masked.to_vec2::<f32>());

        // masked attn weights
        let scaling = 1. / (keys.dims()[1] as f64).sqrt();
        let attn_weights = softmax(&(masked * scaling)?, 1)?;
        println!("attn_weights: {:?}", attn_weights.to_vec2::<f32>());
        Ok(attn_weights)
    }
}

impl Example for EG07 {
    fn description(&self) -> String {
        let desc = "Compute causal attention weights more efficiently \
        using `f32::NEGATIVE_INFINITY` and `masked_fill()`.";
        String::from(desc)
    }

    fn page_source(&self) -> usize {
        77_usize
    }

    fn main(&self) {
        let _ = self.main_with_return();
    }
}

/// Example 03.08
pub struct EG08;

impl Example for EG08 {
    fn description(&self) -> String {
        String::from("Dropout on attention weights.")
    }

    fn page_source(&self) -> usize {
        80_usize
    }

    fn main(&self) {
        use candle_nn::Dropout;

        let eg07 = EG07;
        let attn_weights = eg07.main_with_return().unwrap();
        let dropout = Dropout::new(0.5);

        // could have also just used the candle_nn::ops::dropout directly
        let dropped_out = dropout.forward(&attn_weights, true).unwrap();
        println!("dropped_out: {:?}", dropped_out.to_vec2::<f32>());
    }
}

/// Example 03.09
pub struct EG09;

impl Example for EG09 {
    fn description(&self) -> String {
        String::from("Example usage of `CausalAttention`.")
    }

    fn page_source(&self) -> usize {
        81_usize
    }

    fn main(&self) {
        use crate::listings::ch03::CausalAttention;
        use candle_core::{DType, Tensor};
        use candle_nn::{VarBuilder, VarMap};

        // create batch
        let inputs = get_inputs();
        let d_in = inputs.dims()[1]; // input embedding dim
        let d_out = 2_usize;
        let batch = Tensor::stack(&[&inputs, &inputs], 0usize).unwrap();
        println!("batch shape: {:?}", batch);

        // build causal attn layer
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, inputs.device());
        let causal_attn =
            CausalAttention::new(d_in, d_out, 0.0_f32, false, vb.pp("casual_attn")).unwrap();

        // context vectors
        let context_vectors = causal_attn.forward(&batch).unwrap();
        println!("context_vectors.shape: {:?}", context_vectors);
    }
}
