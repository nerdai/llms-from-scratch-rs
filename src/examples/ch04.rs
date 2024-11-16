use crate::Example;
use candle_core::{Device, NdArray, Result, Tensor};
use candle_nn::Module;

pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Getting logits with DummyGPTModel.")
    }

    fn page_source(&self) -> usize {
        97_usize
    }

    fn main(&self) {
        use crate::listings::ch04::DummyGPTModel;
        use candle_core::{DType, Device, Tensor};
        use candle_nn::{VarBuilder, VarMap};
        use tiktoken_rs::get_bpe_from_model;

        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let mut batch_tokens: Vec<Vec<u32>> = Vec::new();
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        batch_tokens.push(tokenizer.encode_with_special_tokens("Every effort moves you"));
        batch_tokens.push(tokenizer.encode_with_special_tokens("Every day holds a"));
        let batch = Tensor::from_vec(batch_tokens, batch_tokens.shape().unwrap(), vb.device());
    }
}
