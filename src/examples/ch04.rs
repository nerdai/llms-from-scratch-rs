use crate::Example;

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
        use candle_core::{DType, Device, Module, Tensor};
        use candle_nn::{VarBuilder, VarMap};
        use tiktoken_rs::get_bpe_from_model;

        let dev = Device::cuda_if_available(0).unwrap();

        // create batch
        let mut batch_tokens: Vec<u32> = Vec::new();
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        batch_tokens.append(&mut tokenizer.encode_with_special_tokens("Every effort moves you"));
        batch_tokens.append(&mut tokenizer.encode_with_special_tokens("Every day holds a"));

        let batch = Tensor::from_vec(batch_tokens, (2_usize, 4_usize), &dev).unwrap();
        println!("batch: {:?}", batch.to_vec2::<u32>());

        // create model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = DummyGPTModel::new(Config::gpt2_124m(), vb).unwrap();

        // get logits
        let logits = model.forward(&batch).unwrap();
        println!("logits: {:?}", logits.to_vec3::<f32>());
        println!("output shape: {:?}", logits.shape());
    }
}

/// Example 04.02
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        String::from("Manual computation of layern normalization.")
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
    }
}
