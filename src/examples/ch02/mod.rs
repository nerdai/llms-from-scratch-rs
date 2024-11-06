use crate::Example;

pub struct EG01 {}

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Use candle to generate an Embedding Layer.")
    }

    fn page_source(&self) -> usize {
        42_usize
    }

    fn main(&self) {
        use candle_core::{DType, Device};
        use candle_nn::{embedding, VarBuilder, VarMap};

        let vocab_size = 6_usize;
        let output_dim = 3_usize;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let emb = embedding(vocab_size, output_dim, vs).unwrap();

        println!("{:?}", emb.embeddings().to_vec2::<f32>());
    }
}
