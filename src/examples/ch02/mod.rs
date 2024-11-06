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
        use candle_core::{DType, Device, Tensor};
        use candle_nn::{embedding, VarBuilder, VarMap};

        let vocab_size = 6_usize;
        let output_dim = 3_usize;
        let varmap = VarMap::new();
        let dev = Device::Cpu;
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let emb = embedding(vocab_size, output_dim, vs).unwrap();

        println!("{:?}", emb.embeddings().to_vec2::<f32>());
        // print specific embedding of a given token id
        let token_ids = Tensor::new(&[3u32], &dev).unwrap();
        println!(
            "{:?}",
            emb.embeddings()
                .index_select(&token_ids, 0)
                .unwrap()
                .to_vec2::<f32>()
        );
    }
}
