//! Examples from Chapter 2

use crate::Example;
use anyhow::Result;

/// # Example of reading text files into Rust
///
/// #### Id
/// 02.01
///
/// #### Page
/// This example starts on page 22
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 02.01
///
/// # with cuda
/// cargo run --features cuda example 02.01
/// ```
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Example usage of `listings::ch02::sample_read_text`")
    }

    fn page_source(&self) -> usize {
        22_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch02::sample_read_text;
        sample_read_text()?;
        Ok(())
    }
}

/// # Use candle to generate an Embedding Layer
///
/// #### Id
/// 02.02
///
/// #### Page
/// This example starts on page 25
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 02.02
///
/// # with cuda
/// cargo run --features cuda example 02.02
/// ```
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        todo!()
    }

    fn page_source(&self) -> usize {
        todo!()
    }

    fn main(&self) -> Result<()> {
        todo!()
    }
}

/// # Use candle to generate an Embedding Layer
///
/// #### Id
/// 02.03
///
/// #### Page
/// This example starts on page 42
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 02.03
///
/// # with cuda
/// cargo run --features cuda example 02.03
/// ```
pub struct EG03;

impl Example for EG03 {
    fn description(&self) -> String {
        String::from("Use candle to generate an Embedding Layer.")
    }

    fn page_source(&self) -> usize {
        42_usize
    }

    fn main(&self) -> Result<()> {
        use candle_core::{DType, Device, Tensor};
        use candle_nn::{embedding, VarBuilder, VarMap};

        let vocab_size = 6_usize;
        let output_dim = 3_usize;
        let varmap = VarMap::new();
        let dev = Device::cuda_if_available(0)?;
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let emb = embedding(vocab_size, output_dim, vs)?;

        println!("{:?}", emb.embeddings().to_vec2::<f32>());
        // print specific embedding of a given token id
        let token_ids = Tensor::new(&[3u32], &dev)?;
        println!(
            "{:?}",
            emb.embeddings()
                .index_select(&token_ids, 0)?
                .to_vec2::<f32>()
        );
        Ok(())
    }
}

/// # Create absolute positional embeddings
///
/// #### Id
/// 02.04
///
/// #### Page
/// This example starts on page 47
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 02.04
///
/// # with cuda
/// cargo run --features cuda example 02.04
/// ```
pub struct EG04;

impl Example for EG04 {
    fn description(&self) -> String {
        String::from("Create absolute postiional embeddings.")
    }

    fn page_source(&self) -> usize {
        47_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch02::create_dataloader_v1;
        use candle_core::{DType, Tensor};
        use candle_nn::{embedding, VarBuilder, VarMap};
        use std::fs;

        // create data batcher
        let raw_text = fs::read_to_string("data/the-verdict.txt").expect("Unable to read the file");
        let max_length = 4_usize;
        let stride = max_length;
        let shuffle = false;
        let drop_last = false;
        let batch_size = 8_usize;
        let data_loader = create_dataloader_v1(
            &raw_text[..],
            batch_size,
            max_length,
            stride,
            shuffle,
            drop_last,
        );

        let mut batch_iter = data_loader.batcher();

        // get embeddings of first batch inputs
        match batch_iter.next() {
            Some(Ok((inputs, _targets))) => {
                let varmap = VarMap::new();
                let vs = VarBuilder::from_varmap(&varmap, DType::F32, inputs.device());

                let vocab_size = 50_257_usize;
                let output_dim = 256_usize;
                let mut final_dims = inputs.dims().to_vec();
                final_dims.push(output_dim);

                // token embeddings of the current batch inputs
                let token_embedding_layer = embedding(vocab_size, output_dim, vs.pp("tok_emb"))?;
                let token_embeddings = token_embedding_layer
                    .embeddings()
                    .index_select(&inputs.flatten_all()?, 0)?;
                let token_embeddings = token_embeddings.reshape(final_dims)?;
                println!("token embeddings dims: {:?}", token_embeddings.dims());

                // position embeddings
                let context_length = max_length;
                let pos_embedding_layer = embedding(context_length, output_dim, vs.pp("pos_emb"))?;
                let pos_ids = Tensor::arange(0u32, context_length as u32, inputs.device())?;
                let pos_embeddings = pos_embedding_layer.embeddings().index_select(&pos_ids, 0)?;
                println!("pos embeddings dims: {:?}", pos_embeddings.dims());

                // incorporate positional embeddings
                let input_embeddings = token_embeddings.broadcast_add(&pos_embeddings)?;
                println!("input embeddings dims: {:?}", input_embeddings.dims());
            }
            Some(Err(err)) => panic!("{}", err),
            None => panic!("None"),
        }
        Ok(())
    }
}
