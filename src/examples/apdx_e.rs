//! Examples from Appendix E

use crate::Example;
use anyhow::Result;

/// # Example usage of `create_candle_dataloaders`
///
/// #### Id
/// E.01
///
/// #### Page
/// This example starts on page 326
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example E.01
///
/// # with cuda
/// cargo run --features cuda example E.01
/// ```
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        "Example usage of `create_candle_dataloaders`.".to_string()
    }

    fn page_source(&self) -> usize {
        326_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::apdx_e::create_candle_dataloaders;

        let (train_loader, val_loader, test_loader) = create_candle_dataloaders()?;

        // print last batch of train loader
        let (input_batch, target_batch) = train_loader.batcher().last().unwrap()?;
        println!("Input batch dimensions: {:?}", input_batch.shape());
        println!("Label batch dimensions: {:?}", target_batch.shape());

        // print total number of batches in each data loader
        println!("{:?} training batches", train_loader.len());
        println!("{:?} validation batches", val_loader.len());
        println!("{:?} test batches", test_loader.len());

        Ok(())
    }
}

/// # Example usage of `download_and_load_gpt2` and attaching spam classification head
///
/// #### Id
/// E.02
///
/// #### Page
/// This example starts on page 327
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example E.02
///
/// # with cuda
/// cargo run --features cuda example E.02
/// ```
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        "Example usage of `download_and_load_gpt2` and attaching spam classification head."
            .to_string()
    }

    fn page_source(&self) -> usize {
        327_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            apdx_e::{create_candle_dataloaders, download_and_load_gpt2},
            ch04::Config,
            ch05::{generate, text_to_token_ids, token_ids_to_text},
            ch06::{calc_accuracy_loader, modify_out_head_for_classification, HF_GPT2_MODEL_ID},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use rand::{rngs::StdRng, SeedableRng};
        use tiktoken_rs::get_bpe_from_model;

        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;

        // sample setup and load tokenizer
        let tokenizer = get_bpe_from_model("gpt2")?;
        let mut rng = StdRng::seed_from_u64(42_u64);

        // generate next tokens with model
        let text_1 = "Every effort moves you";
        let token_ids = generate(
            &model,
            text_to_token_ids(text_1, &tokenizer, vb.device())?,
            15_usize,
            cfg.context_length,
            None,
            None,
            None,
            &mut rng,
        )?;

        // decode the token ids to print the output text
        println!("{:?}", token_ids_to_text(token_ids, &tokenizer));

        // attach spam classification head
        let num_classes = 2_usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        // calc classification accuracy
        let (train_loader, val_loader, test_loader) = create_candle_dataloaders()?;

        // compute accuracies
        let num_batches = Some(10_usize);
        let train_accuracy =
            calc_accuracy_loader(&train_loader, &model, vb.device(), num_batches, None)?;
        let val_accuracy =
            calc_accuracy_loader(&val_loader, &model, vb.device(), num_batches, None)?;
        let test_accuracy =
            calc_accuracy_loader(&test_loader, &model, vb.device(), num_batches, None)?;

        println!("Training accuracy: {}", train_accuracy);
        println!("Validation accuracy: {}", val_accuracy);
        println!("Test accuracy: {}", test_accuracy);

        Ok(())
    }
}
