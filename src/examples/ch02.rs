use crate::Example;

/// Example 02.01
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

/// Example 02.02
pub struct EG02 {}

impl Example for EG02 {
    fn description(&self) -> String {
        String::from("Create absolute postiional embeddings.")
    }

    fn page_source(&self) -> usize {
        47_usize
    }

    fn main(&self) {
        use crate::listings::ch02::{GPTDatasetIter, GPTDatasetV1};
        use candle_core::{DType, Device, Tensor};
        use candle_datasets::Batcher;
        use candle_nn::{embedding, VarBuilder, VarMap};
        use std::fs;
        use tiktoken_rs::get_bpe_from_model;

        // create data batcher
        let raw_text = fs::read_to_string("data/the-verdict.txt").expect("Unable to read the file");
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        let max_length = 4_usize;
        let stride = max_length;
        let dataset = GPTDatasetV1::new(&raw_text[..], tokenizer, max_length, stride);
        let device = Device::Cpu;
        let iter = GPTDatasetIter::new(&dataset, device, false);
        let batch_size = 8_usize;
        let mut batch_iter = Batcher::new_r2(iter).batch_size(batch_size);

        // get embeddings of first batch inputs
        match batch_iter.next() {
            Some(Ok((inputs, _targets))) => {
                let varmap = VarMap::new();
                let dev = Device::Cpu;
                let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

                let vocab_size = 50_257_usize;
                let output_dim = 256_usize;
                let mut final_dims = inputs.dims().to_vec();
                final_dims.push(output_dim);

                // token embeddings of the current batch inputs
                let token_embedding_layer =
                    embedding(vocab_size, output_dim, vs.pp("tok_emb")).unwrap();
                let token_embeddings = token_embedding_layer
                    .embeddings()
                    .index_select(&inputs.flatten_all().unwrap(), 0)
                    .unwrap();
                let token_embeddings = token_embeddings.reshape(final_dims).unwrap();
                println!("token embeddings dims: {:?}", token_embeddings.dims());

                // position embeddings
                let context_length = max_length;
                let pos_embedding_layer =
                    embedding(context_length, output_dim, vs.pp("pos_emb")).unwrap();
                let pos_ids = Tensor::arange(0u32, context_length as u32, &dev).unwrap();
                let pos_embeddings = pos_embedding_layer
                    .embeddings()
                    .index_select(&pos_ids, 0)
                    .unwrap();
                println!("pos embeddings dims: {:?}", pos_embeddings.dims());

                // incorporate positional embeddings
                let input_embeddings = token_embeddings.broadcast_add(&pos_embeddings).unwrap();
                println!("input embeddings dims: {:?}", input_embeddings.dims());
            }
            Some(Err(err)) => panic!("{}", err),
            None => panic!("None"),
        }
    }
}
