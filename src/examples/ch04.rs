use crate::{listings::ch04::ExampleDeepNeuralNetwork, Example};
use candle_core::{Device, Module, Tensor};
use tiktoken_rs::get_bpe_from_model;

fn get_batch_for_gpts() -> Tensor {
    let dev = Device::cuda_if_available(0).unwrap();

    // create batch
    let mut batch_tokens: Vec<u32> = Vec::new();
    let tokenizer = get_bpe_from_model("gpt2").unwrap();
    batch_tokens.append(&mut tokenizer.encode_with_special_tokens("Every effort moves you"));
    batch_tokens.append(&mut tokenizer.encode_with_special_tokens("Every day holds a"));

    Tensor::from_vec(batch_tokens, (2_usize, 4_usize), &dev).unwrap()
}

/// Example 04.01
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Getting logits with DummyGPTModel.")
    }

    fn page_source(&self) -> usize {
        97_usize
    }

    fn main(&self) {
        use crate::listings::ch04::{Config, DummyGPTModel};
        use candle_core::{DType, IndexOp, Module};
        use candle_nn::{VarBuilder, VarMap};

        let batch = get_batch_for_gpts();
        println!("batch: {:?}", batch.to_vec2::<u32>());

        // create model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, batch.device());
        let model = DummyGPTModel::new(Config::gpt2_124m(), vb).unwrap();

        // get logits
        let logits = model.forward(&batch).unwrap();
        println!("output shape: {:?}", logits.shape());

        // print first 10 next-token logits for each token of every input sequence
        println!(
            "logits: {:?}",
            logits.i((.., .., 0..10)).unwrap().to_vec3::<f32>()
        );
    }
}

/// Example 04.02
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        String::from("Manual computation of layer normalization.")
    }

    fn page_source(&self) -> usize {
        100_usize
    }

    fn main(&self) {
        use candle_core::{DType, Device, Module, Tensor, D};
        use candle_nn::{linear_b, seq, Activation, VarBuilder, VarMap};

        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        // create batch
        let batch_example = Tensor::rand(0f32, 1f32, (2_usize, 5_usize), vb.device()).unwrap();

        // create layer
        let layer = seq()
            .add(linear_b(5_usize, 6_usize, false, vb.pp("linear")).unwrap())
            .add(Activation::Relu);

        // execute layer on batch
        let out = layer.forward(&batch_example).unwrap();
        println!("out: {:?}", out.to_vec2::<f32>());

        // calculate stats on outputs
        let mean = out.mean_keepdim(D::Minus1).unwrap();
        let var = out.var_keepdim(D::Minus1).unwrap();
        println!("mean: {:?}", mean.to_vec2::<f32>());
        println!("variance: {:?}", var.to_vec2::<f32>());

        // layer normalization
        let out_norm = (out
            .broadcast_sub(&mean)
            .unwrap()
            .broadcast_div(&var.sqrt().unwrap()))
        .unwrap();
        let mean = out_norm.mean_keepdim(D::Minus1).unwrap();
        let var = out_norm.var_keepdim(D::Minus1).unwrap();
        println!("normalized out: {:?}", out_norm.to_vec2::<f32>());
        println!("mean: {:?}", mean.to_vec2::<f32>());
        println!("variance: {:?}", var.to_vec2::<f32>());
    }
}

/// Example 04.03
pub struct EG03;

impl Example for EG03 {
    fn description(&self) -> String {
        String::from("Example usage of `LayerNorm`.")
    }

    fn page_source(&self) -> usize {
        104_usize
    }

    fn main(&self) {
        use crate::listings::ch04::LayerNorm;
        use candle_core::{DType, Device, Module, Tensor, D};
        use candle_nn::{VarBuilder, VarMap};

        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        // create batch
        let batch_example = Tensor::rand(0f32, 1f32, (2_usize, 5_usize), vb.device()).unwrap();

        // construct layer norm layer
        let emb_dim = 5_usize;
        let ln = LayerNorm::new(emb_dim, vb.pp("layer_norm")).unwrap();
        let out_ln = ln.forward(&batch_example).unwrap();

        // compute stats on out_ln
        let mean = out_ln.mean_keepdim(D::Minus1).unwrap();
        let var = out_ln.var_keepdim(D::Minus1).unwrap();
        println!("mean: {:?}", mean.to_vec2::<f32>());
        println!("variance: {:?}", var.to_vec2::<f32>());
    }
}

/// Example 04.04
pub struct EG04;

impl Example for EG04 {
    fn description(&self) -> String {
        String::from("Sample usage of `FeedForward` Module.")
    }

    fn page_source(&self) -> usize {
        108_usize
    }

    fn main(&self) {
        use crate::listings::ch04::{Config, FeedForward};
        use candle_core::{DType, Device, IndexOp, Module, Tensor};
        use candle_nn::{VarBuilder, VarMap};

        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let cfg = Config::gpt2_124m();

        // create batch
        let (batch_size, seq_len) = (2_usize, 3_usize);
        let x = Tensor::rand(0f32, 1f32, (batch_size, seq_len, cfg.emb_dim), vb.device()).unwrap();

        // feedforward
        let ffn = FeedForward::new(cfg, vb.pp("ffn")).unwrap();
        let out = ffn.forward(&x).unwrap();

        println!("{:?}", out);
        // first 10 hidden states of the embedding for 1st sequence, 1st token
        println!("{:?}", out.i((0, 0, 0..10)).unwrap().to_vec1::<f32>());
    }
}

/// Example 04.05
pub struct EG05;

impl EG05 {
    #[allow(unused_variables, dead_code)]
    fn print_gradients(model: ExampleDeepNeuralNetwork, x: &Tensor) {
        use candle_nn::loss::mse;

        let output = model.forward(x).unwrap();
        let target = Tensor::new(&[[0_f32]], x.device()).unwrap();

        let loss = mse(&output, &target).unwrap();
        let grads = loss.backward().unwrap();

        for (ix, tensor_id) in model.tensor_ids.iter().enumerate() {
            let grad_tensor = grads.get_id(tensor_id.to_owned()).unwrap();
            println!(
                "layer.{}.weight has gradient mean of {:?}",
                ix,
                grad_tensor
                    .abs()
                    .unwrap()
                    .mean_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
            );
        }
        println!("\n");
    }
}

impl Example for EG05 {
    fn description(&self) -> String {
        String::from("Comparison of gradients with and without shortcut connections.")
    }

    fn page_source(&self) -> usize {
        111_usize
    }

    fn main(&self) {
        use crate::listings::ch04::ExampleDeepNeuralNetwork;
        use candle_core::{DType, Device, Tensor};
        use candle_nn::{VarBuilder, VarMap};

        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let layer_sizes = &[3_usize, 3, 3, 3, 3, 1];
        let sample_input = Tensor::new(&[[1_f32, 0., -1.]], vb.device()).unwrap();
        let model_without_shortcut =
            ExampleDeepNeuralNetwork::new(layer_sizes, false, vb.pp("model_wout_shortcut"))
                .unwrap();

        let model_with_shortcut =
            ExampleDeepNeuralNetwork::new(layer_sizes, true, vb.pp("model_wout_shortcut")).unwrap();

        println!("model_without_shortcut gradients:");
        EG05::print_gradients(model_without_shortcut, &sample_input);
        println!("model_with_shortcut gradients:");
        EG05::print_gradients(model_with_shortcut, &sample_input);
    }
}

/// Example 04.06
pub struct EG06;

impl Example for EG06 {
    fn description(&self) -> String {
        String::from("Sample usage of `TransformerBlock`.")
    }

    fn page_source(&self) -> usize {
        116_usize
    }

    fn main(&self) {
        use crate::listings::ch04::{Config, TransformerBlock};
        use candle_core::{DType, Device, IndexOp, Tensor};
        use candle_nn::{VarBuilder, VarMap};

        // construct transformer block
        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let cfg = Config::gpt2_124m();
        let block = TransformerBlock::new(cfg, vb.pp("block")).unwrap();

        // create sample input
        let (batch_size, num_tokens) = (2_usize, 4_usize);
        let x = Tensor::rand(
            0f32,
            1f32,
            (batch_size, num_tokens, cfg.emb_dim),
            vb.device(),
        )
        .unwrap();

        // execute forward pass
        let output = block.forward(&x).unwrap();

        println!("Input shape: {:?}", x.shape());
        println!("Output shape: {:?}", output.shape());

        // print the first 10 features of all tokens of the first input
        println!(
            "Output: {:?}",
            output.i((0..1, .., 0..10)).unwrap().to_vec3::<f32>()
        );
    }
}

/// EG 04.07
pub struct EG07;

impl Example for EG07 {
    fn description(&self) -> String {
        String::from("Sample usage of GPTModel.")
    }

    fn page_source(&self) -> usize {
        120_usize
    }

    fn main(&self) {
        use crate::listings::ch04::{Config, GPTModel};
        use candle_core::{DType, IndexOp, Module};
        use candle_nn::{VarBuilder, VarMap};

        let batch = get_batch_for_gpts();
        println!("batch: {:?}", batch.to_vec2::<u32>());

        // create model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, batch.device());
        let model = GPTModel::new(Config::gpt2_124m(), vb).unwrap();

        // get logits
        let logits = model.forward(&batch).unwrap();
        println!("output shape: {:?}", logits.shape());

        // print first 10 next-token logits for each token of every input sequence
        println!(
            "logits: {:?}",
            logits.i((.., .., 0..10)).unwrap().to_vec3::<f32>()
        );

        // get total number of params from the VarMap (todo: turn this into a util)
        let mut total_params = 0_usize;
        for t in varmap.all_vars().iter() {
            let this_tensor_params = match *t.dims() {
                [d1] => d1,
                [d1, d2] => d1 * d2,
                _ => panic!("Variable with more than 2 dimensions."),
            };
            total_params += this_tensor_params;
        }
        println!("Total number of parameters: {}", total_params);

        // Get token embedding and output layer shapes
        let varmap_binding = varmap.data().lock().unwrap();
        let tok_emb_dims = varmap_binding.get("tok_emb.weight").unwrap().dims();
        println!("Token embedding layer shape {:?}", tok_emb_dims);
        let out_head_dims = varmap_binding.get("out_head.weight").unwrap().dims();
        println!("Output layer shape {:?}", out_head_dims);

        // total number of params if weight tying with token emb and output layer shapes
        let total_params_gpt2 = total_params - (out_head_dims[0] * out_head_dims[1]);
        println!(
            "Number of trainable parameters considering weight tying {}",
            total_params_gpt2
        );
    }
}
