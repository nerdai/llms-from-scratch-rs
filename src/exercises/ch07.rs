//! Exercises from Chapter 7

use crate::Exercise;
use anyhow::Result;

/// # Changing prompt styles
///
/// #### Id
/// 7.1
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run exercise 7.1
///
/// # with cuda
/// cargo run --features cuda exercise 7.1
/// ```
pub struct X1;

impl X1 {
    pub fn get_data_loaders(
        &self,
        verbose: bool,
    ) -> Result<(
        crate::listings::ch07::InstructionDataLoader<
            crate::listings::ch07::InstructionDataCollator,
        >,
        crate::listings::ch07::InstructionDataLoader<
            crate::listings::ch07::InstructionDataCollator,
        >,
        crate::listings::ch07::InstructionDataLoader<
            crate::listings::ch07::InstructionDataCollator,
        >,
    )> {
        use crate::listings::ch07::{
            download_and_load_file, partition_data, DataLoader, InstructionDataCollator,
            InstructionDataLoader, InstructionDataset, Phi3PromptFormatter, DATA_DIR,
            INSTRUCTION_DATA_FILENAME, INSTRUCTION_DATA_URL,
        };
        use candle_core::Device;
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2")?;

        // load instruction examples
        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;

        // partition data and create train, val, test datasets
        let (train_data, val_data, test_data) = partition_data(data, 0.85_f32, 0.05_f32)?;
        let prompt_formatter = Phi3PromptFormatter;
        let train_dataset = InstructionDataset::new(train_data, &tokenizer, &prompt_formatter);
        let val_dataset = InstructionDataset::new(val_data, &tokenizer, &prompt_formatter);
        let test_dataset = InstructionDataset::new(test_data, &tokenizer, &prompt_formatter);

        // create loaders
        let batch_size = 5_usize;
        let collator = InstructionDataCollator::new()
            .device(Device::cuda_if_available(0)?)
            .allowed_max_length(Some(1024_usize));
        let train_loader =
            InstructionDataLoader::new(train_dataset, batch_size, true, true, collator.clone());
        let val_loader =
            InstructionDataLoader::new(val_dataset, batch_size, false, false, collator.clone());
        let test_loader =
            InstructionDataLoader::new(test_dataset, batch_size, false, false, collator);

        if verbose {
            println!("Train loader:");
            let mut batcher = train_loader.batcher();
            while let Some(Ok((inputs, targets))) = batcher.next() {
                println!("inputs: {:?} targets: {:?}", inputs, targets);
            }
        }

        Ok((train_loader, val_loader, test_loader))
    }
}

impl Exercise for X1 {
    fn name(&self) -> String {
        "7.1".to_string()
    }

    fn title(&self) -> String {
        "Changing prompt styles".to_string()
    }

    fn statement(&self) -> String {
        let stmt = "After fine-tuning the model with the Alpaca prompt style, \
        try the Phi-3 prompt style shown in figure 7.4 and observe whether it \
        affects the response quality of the model.";
        stmt.to_string()
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch05::plot_losses,
            ch07::{
                download_and_load_gpt2, train_model_simple, Phi3PromptFormatter, PromptFormatter,
                DEFAULT_IGNORE_INDEX,
            },
        };
        use candle_core::{DType, Device};
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
        use ndarray::linspace;
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        // use `download_and_load_gpt2`
        let model_id = "openai-community/gpt2"; // use `gpt2-medium` for med instead
        let mut cfg = Config::gpt2_124m(); // use `gpt2_medium()` for med instead
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, model_id)?;

        // get data loaders that is built on a dataset that used phi3 prompt format style
        let (train_loader, val_loader, _test_loader) = self.get_data_loaders(false)?;

        // invoke training
        let (eval_freq, eval_iter, num_epochs) = (5_usize, 5_usize, 1_usize);
        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.00005,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;
        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = Phi3PromptFormatter;
        let start_context = prompt_formatter.format_input(&val_loader.dataset().data()[0]);
        let (train_losses, val_losses, tokens_seen) = train_model_simple(
            &model,
            &train_loader,
            &val_loader,
            optimizer,
            vb.device(),
            num_epochs,
            eval_freq,
            eval_iter,
            start_context.as_str(),
            &tokenizer,
            Some(DEFAULT_IGNORE_INDEX),
        )?;

        // save model
        println!("Saving weights to `./ift.phi3.checkpoint.safetensors`");
        varmap.save("ift.phi3.checkpoint.safetensors")?;

        // plot loss curves
        println!("Saving plot to `./plot_ift_phi3_loss.html`");
        let epochs_seen = Vec::from_iter(linspace(0_f32, num_epochs as f32, train_losses.len()));
        let tokens_seen = tokens_seen
            .into_iter()
            .map(|el| el as f32)
            .collect::<Vec<_>>();
        let save_path = Path::new("plot_ift_phi3_loss.html").to_path_buf();
        plot_losses(
            epochs_seen,
            tokens_seen,
            train_losses,
            val_losses,
            save_path,
        )?;

        Ok(())
    }
}
