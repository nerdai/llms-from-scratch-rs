use crate::Example;

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
