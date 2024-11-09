use crate::Example;
use candle_core::{Device, Tensor};

fn get_inputs() -> Tensor {
    Tensor::new(
        &[
            [0.43_f32, 0.15, 0.89], // Your
            [0.55, 0.87, 0.66],     // journey
            [0.57, 0.85, 0.64],     // starts
            [0.22, 0.58, 0.33],     // with
            [0.77, 0.25, 0.10],     // one
            [0.05, 0.80, 0.55],     // step
        ],
        &Device::Cpu,
    )
    .unwrap()
}

/// Example 03.01
pub struct EG01 {}

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
pub struct EG02 {}

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
pub struct EG03 {}

impl Example for EG03 {
    fn description(&self) -> String {
        String::from("Implementing the self-attention mechanism with trainable weights.")
    }

    fn page_source(&self) -> usize {
        66_usize
    }

    fn main(&self) {
        use candle_core::DType;
        use candle_nn::init::DEFAULT_KAIMING_NORMAL;
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

        println!("Query 2: {:?}", query_2.to_vec2::<f32>().unwrap());
        println!("Key 2: {:?}", key_2.to_vec2::<f32>().unwrap());
        println!("Value 2: {:?}", value_2.to_vec2::<f32>().unwrap());

        // key and value vectors all input elements
        let keys = inputs.matmul(&w_key).unwrap();
        let values = inputs.matmul(&w_value).unwrap();

        println!("Keys shape: {:?}", keys);
        println!("Values shape: {:?}", values);
    }
}