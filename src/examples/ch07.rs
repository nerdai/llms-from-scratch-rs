//! Examples from Chapter 7

use crate::{listings::ch05::DEFAULT_IGNORE_INDEX, Example};
use anyhow::{anyhow, Context, Result};

/// # Example usage of `download_and_load_file`
///
/// #### Id
/// 07.01
///
/// #### Page
/// This example starts on page 207
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.01
///
/// # with cuda
/// cargo run --features cuda example 07.01
/// ```
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        "Example usage of `download_and_load_file`.".to_string()
    }

    fn page_source(&self) -> usize {
        207_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::{
            download_and_load_file, DATA_DIR, INSTRUCTION_DATA_FILENAME, INSTRUCTION_DATA_URL,
        };
        use std::path::Path;

        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;
        println!("Number of entries: {}", data.len());

        // See example at index 50
        println!("Example entry:\n{}\n", data[50]);

        // See another example at index 999
        println!("Another example entry:\n{}", data[999]);

        Ok(())
    }
}

/// # Example usage of `format_input`
///
/// #### Id
/// 07.02
///
/// #### Page
/// This example starts on page 209
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.02
///
/// # with cuda
/// cargo run --features cuda example 07.02
/// ```
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        "Example usage of `format_input`.".to_string()
    }

    fn page_source(&self) -> usize {
        209_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::{
            download_and_load_file, AlpacaPromptFormatter, InstructionExample, PromptFormatter,
            DATA_DIR, INSTRUCTION_DATA_FILENAME, INSTRUCTION_DATA_URL,
        };
        use std::path::Path;

        // load instruction examples
        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;
        let prompt_formatter = AlpacaPromptFormatter;

        // first model input
        let model_input = prompt_formatter.format_input(&data[50]);
        let detailed_response = format!("\n\n### Response:\n{}", data[50].output());
        println!("{}", model_input + &detailed_response);

        // print a separator
        println!("\n---\n");

        // print another model input
        let model_input = prompt_formatter.format_input(&data[999]);
        let detailed_response = format!("\n\n### Response:\n{}", data[999].output());
        println!("{}", model_input + &detailed_response);

        Ok(())
    }
}

/// # Example usage of `partition_data`
///
/// #### Id
/// 07.03
///
/// #### Page
/// This example starts on page 210
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.03
///
/// # with cuda
/// cargo run --features cuda example 07.03
/// ```
pub struct EG03;

impl Example for EG03 {
    fn description(&self) -> String {
        String::from("Example usage of `partition_data`")
    }

    fn page_source(&self) -> usize {
        210_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::{
            download_and_load_file, partition_data, DATA_DIR, INSTRUCTION_DATA_FILENAME,
            INSTRUCTION_DATA_URL,
        };
        use std::path::Path;

        // load instruction examples
        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;

        // partition data
        let (train_data, val_data, test_data) = partition_data(data, 0.85_f32, 0.05_f32)?;

        println!("Training set length: {}", train_data.len());
        println!("Validation set length: {}", val_data.len());
        println!("Test set length: {}", test_data.len());

        Ok(())
    }
}

/// # Example usage of `<|endoftext|>` special token with tiktoken
///
/// #### Id
/// 07.04
///
/// #### Page
/// This example starts on page 214
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.04
///
/// # with cuda
/// cargo run --features cuda example 07.04
/// ```
pub struct EG04;

impl Example for EG04 {
    fn description(&self) -> String {
        "Example usage of `<|endoftext|>` special token with tiktoken.".to_string()
    }

    fn page_source(&self) -> usize {
        214_usize
    }

    fn main(&self) -> Result<()> {
        use std::collections::HashSet;
        use tiktoken_rs::get_bpe_from_model;

        let allowed_special = HashSet::from(["<|endoftext|>"]);
        let tokenizer = get_bpe_from_model("gpt2")?;
        println!("{:?}", tokenizer.encode("<|endoftext|>", allowed_special));

        Ok(())
    }
}

/// # Example usage of `InstructionDataCollator.custom_collate_fn`
///
/// #### Id
/// 07.05
///
/// #### Page
/// This example starts on page 220
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.05
///
/// # with cuda
/// cargo run --features cuda example 07.05
/// ```
pub struct EG05;

impl Example for EG05 {
    fn description(&self) -> String {
        String::from("Example usage of `InstructionDataCollator.custom_collate_fn`.")
    }

    fn page_source(&self) -> usize {
        220_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::InstructionDataCollator;
        use candle_core::{Device, Tensor};

        let device = Device::cuda_if_available(0)?;
        let inputs_1 = Tensor::new(&[0_u32, 1, 2, 3, 4], &device)?;
        let inputs_2 = Tensor::new(&[5_u32, 6], &device)?;
        let inputs_3 = Tensor::new(&[7_u32, 8, 9], &device)?;
        let batch = vec![inputs_1, inputs_2, inputs_3];

        let collator = InstructionDataCollator::new();
        let (inputs, targets) = collator.custom_collate_fn(batch)?;

        println!("inputs:\n{:?}", inputs.to_vec2::<u32>()?);
        println!("targets:\n{:?}", targets.to_vec2::<i64>()?);

        Ok(())
    }
}

/// # An adapted example demonstrating effect of `ignore_index` in `calc_loss_batch`
///
/// #### Id
/// 07.06
///
/// #### Page
/// This example starts on page 221
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.06
///
/// # with cuda
/// cargo run --features cuda example 07.06
/// ```
pub struct EG06;

impl Example for EG06 {
    fn description(&self) -> String {
        "An adapted example demonstrating effect of `ignore_index` in `calc_loss_batch`."
            .to_string()
    }

    fn page_source(&self) -> usize {
        221_usize
    }

    /// In this example, we make a slight modification to the one found in the book
    /// as the `candle_nn::loss::cross_entropy()` method does not allow for `ignore_index`.
    /// So, we opt to implement such logic within `calc_loss_batch`.
    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::{calc_loss_batch, DEFAULT_IGNORE_INDEX},
        };
        use candle_core::{DType, Device, Tensor};
        use candle_nn::{VarBuilder, VarMap};

        // create model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb.pp("model"))?;

        // create sample inputs
        let inputs = Tensor::new(&[[100_u32, 20, 300]], vb.device())?;
        let targets = Tensor::new(&[[1_u32, 2, 3]], vb.device())?;
        let loss = calc_loss_batch(&inputs, &targets, &model, vb.device(), false, None)?;

        println!("Inputs: {:?}", inputs.to_vec2::<u32>()?);
        println!("Targets: {:?}", inputs.to_vec2::<u32>()?);
        println!("Loss: {:?}", loss);

        // Note targets that use ignore_index will now be a Tensor of Dtype::I64
        let inputs_2 = Tensor::new(&[[100_u32, 20, 300], [400, 7, 88]], vb.device())?;
        let targets_2 = Tensor::new(
            &[
                [1_i64, 2, 3],
                [
                    DEFAULT_IGNORE_INDEX,
                    DEFAULT_IGNORE_INDEX,
                    DEFAULT_IGNORE_INDEX,
                ],
            ],
            vb.device(),
        )?;
        let loss_2 = calc_loss_batch(
            &inputs_2,
            &targets_2,
            &model,
            vb.device(),
            false,
            Some(DEFAULT_IGNORE_INDEX),
        )?;

        println!(
            "---\nSimilar inputs but now a second sequence whose target has the ignore index:\n"
        );

        println!("Inputs: {:?}", inputs_2.to_vec2::<u32>()?);
        println!("Targets: {:?}", targets_2.to_vec2::<i64>()?);
        println!("Loss: {:?}", loss_2);

        Ok(())
    }
}

/// # Creating a `InstructionDataLoader` for each of the train, val and test data partitions
///
/// #### Id
/// 07.07
///
/// #### Page
/// This example starts on page 225
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.07
///
/// # with cuda
/// cargo run --features cuda example 07.07
/// ```
pub struct EG07;

impl EG07 {
    pub fn main_with_return(
        &self,
        batch_size: usize,
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
            download_and_load_file, partition_data, AlpacaPromptFormatter, DataLoader,
            InstructionDataCollator, InstructionDataLoader, InstructionDataset, DATA_DIR,
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
        let prompt_formatter = AlpacaPromptFormatter;
        let train_dataset = InstructionDataset::new(train_data, &tokenizer, &prompt_formatter);
        let val_dataset = InstructionDataset::new(val_data, &tokenizer, &prompt_formatter);
        let test_dataset = InstructionDataset::new(test_data, &tokenizer, &prompt_formatter);

        // create loaders
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

impl Example for EG07 {
    fn description(&self) -> String {
        let desc = "Creating a `InstructionDataLoader` for each of the train, \
        val and test data partitions";
        desc.to_string()
    }

    fn page_source(&self) -> usize {
        225_usize
    }

    fn main(&self) -> Result<()> {
        let _ = self.main_with_return(8_usize, true);
        Ok(())
    }
}

/// # Example usage of `download_and_load_gpt2` and sample instruction inference
///
/// #### Id
/// 07.08
///
/// #### Page
/// This example starts on page 227
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.08
///
/// # with cuda
/// cargo run --features cuda example 07.08
/// ```
pub struct EG08;

impl Example for EG08 {
    fn description(&self) -> String {
        "Example usage of `download_and_load_gpt2` and sample instruction inference.".to_string()
    }

    fn page_source(&self) -> usize {
        227_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch05::{generate, text_to_token_ids, token_ids_to_text},
            ch07::{
                download_and_load_file, download_and_load_gpt2, partition_data,
                AlpacaPromptFormatter, PromptFormatter, DATA_DIR, INSTRUCTION_DATA_FILENAME,
                INSTRUCTION_DATA_URL,
            },
        };
        use candle_core::{DType, Device, Tensor};
        use candle_nn::{VarBuilder, VarMap};
        use rand::{rngs::StdRng, SeedableRng};
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        // partition data and create train, val, test datasets
        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;
        let (_train_data, val_data, _test_data) = partition_data(data, 0.85_f32, 0.05_f32)?;

        // use `download_and_load_gpt2` for gpt2-medium
        let mut cfg = Config::gpt2_medium();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let model_id = "openai-community/gpt2-medium";
        let model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, model_id)?;

        // input instructions
        let prompt_formatter = AlpacaPromptFormatter;
        let input_text = prompt_formatter.format_input(&val_data[0]);
        println!("{}", input_text);

        // run inference
        let tokenizer = get_bpe_from_model("gpt2")?;
        let mut rng = StdRng::seed_from_u64(42_u64);
        let token_ids = generate(
            &model,
            text_to_token_ids(input_text.as_str(), &tokenizer, vb.device())?,
            35_usize,
            cfg.context_length,
            None,
            None,
            Some(Tensor::new(&[50_256_u32], vb.device())?),
            &mut rng,
        )?;
        let generated_text = token_ids_to_text(token_ids, &tokenizer)?;
        let response_text = &generated_text[input_text.len()..].trim();

        println!("---generated-text-below---\n{}", response_text);

        Ok(())
    }
}

/// # Example usage of `calc_loss_loader` to compute cross-entropy loss on train, val, test sets
///
/// #### Id
/// 07.09
///
/// #### Page
/// This example starts on page 230
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.09
///
/// # with cuda
/// cargo run --features cuda example 07.09
/// ```
pub struct EG09;

impl Example for EG09 {
    fn description(&self) -> String {
        let desc = "Example usage of `calc_loss_loader` to compute accuracy on \
        train, validation and test instruction datasets.";
        desc.to_string()
    }

    fn page_source(&self) -> usize {
        230_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch07::{calc_loss_loader, download_and_load_gpt2},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        // use `download_and_load_gpt2` for gpt2-medium
        let mut cfg = Config::gpt2_medium();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let model_id = "openai-community/gpt2-medium";
        let model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, model_id)?;

        // re-use eg 07.07
        let eg07 = EG07;
        let (train_loader, val_loader, _test_loader) = eg07.main_with_return(8_usize, false)?;

        // // compute losses
        let num_batches = Some(5_usize);
        let train_loss = calc_loss_loader(
            &train_loader,
            &model,
            vb.device(),
            num_batches,
            Some(DEFAULT_IGNORE_INDEX),
        )?;
        let val_loss = calc_loss_loader(
            &val_loader,
            &model,
            vb.device(),
            num_batches,
            Some(DEFAULT_IGNORE_INDEX),
        )?;

        println!("Training loss: {}", train_loss);
        println!("Validation loss: {}", val_loss);

        Ok(())
    }
}

/// # Example usage of `train_model_simple` and `plot_losses` functions
///
/// NOTE: In the book this material is encapsulated within Listing 7.8. Here,
/// we choose to make it as an example instead.
///
/// #### Id
/// 07.10
///
/// #### Page
/// This example starts on page 231
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.10
///
/// # with cuda
/// cargo run --features cuda example 07.10
/// ```
pub struct EG10;

impl Example for EG10 {
    fn description(&self) -> String {
        "Example usage of `train_model_simple` and `plot_losses` functions".to_string()
    }

    fn page_source(&self) -> usize {
        231_usize
    }

    // TODO: This fails silently if run into OOM issues.
    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch05::plot_losses,
            ch07::{
                download_and_load_gpt2, train_model_simple, AlpacaPromptFormatter, PromptFormatter,
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

        // re-use eg 07.07
        let eg07 = EG07;
        let (train_loader, val_loader, _test_loader) = eg07.main_with_return(8_usize, false)?;

        // invoke training
        let (eval_freq, eval_iter, num_epochs) = (5_usize, 5_usize, 2_usize);
        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.00005,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;
        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;
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
        println!("Saving weights to `./ift.checkpoint.safetensors`");
        varmap.save("ift.checkpoint.safetensors")?;

        // plot loss curves
        println!("Saving plot to `./plot_ift_loss.html`");
        let epochs_seen = Vec::from_iter(linspace(0_f32, num_epochs as f32, train_losses.len()));
        let tokens_seen = tokens_seen
            .into_iter()
            .map(|el| el as f32)
            .collect::<Vec<_>>();
        let save_path = Path::new("plot_ift_loss.html").to_path_buf();
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

/// # Example of extracting model-generated responses and comparing to correct ones
///
/// #### Id
/// 07.11
///
/// #### Page
/// This example starts on page 234
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.11
///
/// # with cuda
/// cargo run --features cuda example 07.11
/// ```
pub struct EG11;

impl Example for EG11 {
    fn description(&self) -> String {
        let desc = "Example of extracting model-generated responses and \
        comparing to correct ones";
        desc.to_string()
    }

    fn page_source(&self) -> usize {
        234_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::{generate, text_to_token_ids, token_ids_to_text},
            ch07::{AlpacaPromptFormatter, InstructionExample, PromptFormatter},
        };
        use candle_core::{DType, Device, Tensor};
        use candle_nn::{VarBuilder, VarMap};
        use rand::{rngs::StdRng, SeedableRng};
        use tiktoken_rs::get_bpe_from_model;

        // setup the gpt2 model
        let mut cfg = Config::gpt2_124m(); // must match model size used in EG10
        cfg.qkv_bias = true;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let model = GPTModel::new(cfg, vb.pp("model"))?;

        // load instructed-finetuned weights
        varmap
            .load("ift.checkpoint.safetensors")
            .with_context(|| "Missing 'ift.checkpoint.safetensors' file. Please run EG 07.10.")?;

        // extract responses
        let eg07 = EG07;
        let (_train_loader, _val_loader, test_loader) = eg07.main_with_return(8_usize, false)?;
        let tokenizer = get_bpe_from_model("gpt2")?;
        let mut rng = StdRng::seed_from_u64(42_u64);
        let prompt_formatter = AlpacaPromptFormatter;

        for entry in &test_loader.dataset().data()[..3] {
            let input_text = prompt_formatter.format_input(entry);
            let token_ids = generate(
                &model,
                text_to_token_ids(&input_text[..], &tokenizer, vb.device())?,
                256_usize,
                cfg.context_length,
                None,
                None,
                Some(Tensor::new(&[50_256_u32], vb.device())?),
                &mut rng,
            )?;
            let generated_text = token_ids_to_text(token_ids, &tokenizer)?;
            let response_text = &generated_text[input_text.len()..].trim();

            // print
            println!("{}", input_text);
            println!("\nCorrect response:\n>>{}", entry.output());
            println!("\nModel response:\n>>{}", response_text.trim());
            println!("-----------------------------------------");
        }

        Ok(())
    }
}

/// # Example usage of `generate_test_set_responses`
///
/// #### Id
/// 07.12
///
/// #### Page
/// This example starts on page 237
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.12
///
/// # with cuda
/// cargo run --features cuda example 07.12
/// ```
pub struct EG12;

impl Example for EG12 {
    fn description(&self) -> String {
        "Example usage of `generate_test_set_responses`.".to_string()
    }

    fn page_source(&self) -> usize {
        237_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch07::{generate_test_set_responses, AlpacaPromptFormatter, DATA_DIR},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use std::path::Path;

        // setup the gpt2 model
        let mut cfg = Config::gpt2_124m(); // must match model size used in EG10
        cfg.qkv_bias = true;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let model = GPTModel::new(cfg, vb.pp("model"))?;

        // load instructed-finetuned weights
        varmap
            .load("ift.checkpoint.safetensors")
            .with_context(|| "Missing 'ift.checkpoint.safetensors' file. Please run EG 07.10.")?;

        // get data loaders
        let eg07 = EG07;
        let (_train_loader, _val_loader, test_loader) = eg07.main_with_return(8_usize, false)?;

        // generate test set responses
        let save_path = Path::new(DATA_DIR).join("instruction_data_with_response.json");
        let prompt_formatter = AlpacaPromptFormatter;
        let mut test_data = test_loader.dataset().data().clone();
        generate_test_set_responses(
            &mut test_data,
            &model,
            cfg.context_length,
            vb.device(),
            save_path,
            &prompt_formatter,
        )?;

        println!("{}", test_data[0]);

        Ok(())
    }
}

/// # An example to check if `ollama` process is running
///
/// #### Id
/// 07.13
///
/// #### Page
/// This example starts on page 241
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.13
///
/// # with cuda
/// cargo run --features cuda example 07.13
/// ```
pub struct EG13;

impl Example for EG13 {
    fn description(&self) -> String {
        "An example to check if `ollama` process is running.".to_string()
    }

    fn page_source(&self) -> usize {
        241_usize
    }

    fn main(&self) -> Result<()> {
        use anyhow::anyhow;
        use sysinfo::System;

        let sys = System::new_all();
        let mut ollama_processes = sys.processes_by_exact_name("ollama".as_ref());
        let _ = ollama_processes.next().ok_or(anyhow!(
            "Ollama not running. Launch ollama before proceeding."
        ))?;

        println!("Ollama running");

        Ok(())
    }
}

/// # Example usage of `query_model`
///
/// #### Id
/// 07.14
///
/// #### Page
/// This example starts on page 243
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.14
///
/// # with cuda
/// cargo run --features cuda example 07.14
/// ```
pub struct EG14;

impl Example for EG14 {
    fn description(&self) -> String {
        "Example usage of `query_model`.".to_string()
    }

    fn page_source(&self) -> usize {
        243_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::{query_model, DEFAULT_OLLAMA_API_URL};

        let model = "llama3";
        let result = query_model("What do Llamas eat?", model, DEFAULT_OLLAMA_API_URL)?;

        println!("{}", result);
        Ok(())
    }
}

/// # Using Llama3.2 (via Ollama) as the LLM judge to evaluate model responses
///
/// #### Id
/// 07.15
///
/// #### Page
/// This example starts on page 244
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.15
///
/// # with cuda
/// cargo run --features cuda example 07.15
/// ```
pub struct EG15;

impl Example for EG15 {
    fn description(&self) -> String {
        let desc = "Using Llama3.2 (via Ollama) as the LLM judge to evaluate model responses.";
        desc.to_string()
    }

    fn page_source(&self) -> usize {
        244_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::{
            load_instruction_data_from_json, query_model, AlpacaPromptFormatter,
            InstructionExample, InstructionResponseExample, PromptFormatter, DATA_DIR,
            DEFAULT_OLLAMA_API_URL,
        };
        use std::path::Path;

        // load test instruction data with response
        let file_path = Path::new(DATA_DIR).join("instruction_data_with_response.json");
        let test_data: Vec<InstructionResponseExample> = load_instruction_data_from_json(file_path)
            .with_context(|| {
                "Missing 'instruction_data_with_response.json' file. Please run EG 07.12."
            })?;

        let model = "llama3";
        let prompt_formatter = AlpacaPromptFormatter;
        for (ix, entry) in test_data.iter().enumerate().take(3_usize) {
            let model_response = entry
                .model_response()
                .as_ref()
                .ok_or_else(|| anyhow!("Entry {ix} is missing a model response."))?;
            let prompt = format!(
                "Given the input `{}` and the correct output `{}`, score the \
                model response `{}` on a scale from 0 to 100, where 100 is the ]
                best score.",
                prompt_formatter.format_input(entry),
                entry.output(),
                model_response
            );

            println!("\nDataset response:");
            println!("\n>>{}", entry.output());
            println!("\nModel response:");
            println!("\n>>{}", model_response);
            println!("\nScore:");
            println!(
                "\n>>{}",
                query_model(prompt.as_str(), model, DEFAULT_OLLAMA_API_URL)?
            );
        }

        Ok(())
    }
}

/// # Example usage of `generate_model_scores`
///
/// #### Id
/// 07.16
///
/// #### Page
/// This example starts on page 246
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.16
///
/// # with cuda
/// cargo run --features cuda example 07.16
/// ```
pub struct EG16;

impl Example for EG16 {
    fn description(&self) -> String {
        "Example usage of `generate_model_scores`.".to_string()
    }

    fn page_source(&self) -> usize {
        246_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::{
            generate_model_scores, load_instruction_data_from_json, AlpacaPromptFormatter,
            DATA_DIR, DEFAULT_OLLAMA_API_URL,
        };
        use std::path::Path;

        // load test instruction data with response
        let file_path = Path::new(DATA_DIR).join("instruction_data_with_response.json");
        let test_data = load_instruction_data_from_json(file_path).with_context(|| {
            "Missing 'instruction_data_with_response.json' file. Please run EG 07.12."
        })?;

        // invoke generate_model_scores
        let model = "llama3";
        let prompt_formatter = AlpacaPromptFormatter;
        let scores =
            generate_model_scores(&test_data, DEFAULT_OLLAMA_API_URL, model, &prompt_formatter)?;

        // print stats
        println!("Number of scores: {} of {}", scores.len(), test_data.len());
        let average_score = scores.iter().sum::<f32>() / scores.len() as f32;
        println!("Average score: {}", average_score);

        Ok(())
    }
}

/// # [Bonus] Usage of `generate_chosen_and_rejected_response` to create preference example
///
/// #### Id
/// 07.17
///
/// #### Page
/// This example is from `04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb`
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.17
///
/// # with cuda
/// cargo run --features cuda example 07.17
/// ```
pub struct EG17;

impl Example for EG17 {
    fn description(&self) -> String {
        let desc = "[Bonus from DPO notebook] Usage of \
        `generate_chosen_and_rejected_response` to create preference example.";
        desc.to_string()
    }

    fn page_source(&self) -> usize {
        0_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::{
            bonus::generate_chosen_and_rejected_response, download_and_load_file,
            AlpacaPromptFormatter, DATA_DIR, DEFAULT_OLLAMA_API_URL, INSTRUCTION_DATA_FILENAME,
            INSTRUCTION_DATA_URL,
        };
        use rand::{rngs::StdRng, SeedableRng};
        use std::path::Path;

        // load instruction examples
        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;

        // invoke generate_chose_and_rejected_response
        let model = "llama3";
        let prompt_formatter = AlpacaPromptFormatter;
        let mut rng = StdRng::seed_from_u64(42_u64);
        let preference_example = generate_chosen_and_rejected_response(
            &data[42],
            DEFAULT_OLLAMA_API_URL,
            model,
            &prompt_formatter,
            &mut rng,
        )?;

        println!("{:#?}", preference_example);

        Ok(())
    }
}

/// # [BONUS] Example usage of `generate_preference_dataset`
///
/// #### Id
/// 07.18
///
/// #### Page
/// This example is adapted from `04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb`
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.18
///
/// # with cuda
/// cargo run --features cuda example 07.18
/// ```
pub struct EG18;

impl Example for EG18 {
    fn description(&self) -> String {
        "[Bonus from DPO notebook] Example usage of `generate_preference_dataset`.".to_string()
    }

    fn page_source(&self) -> usize {
        0_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch07::bonus::generate_preference_dataset,
            ch07::{
                download_and_load_file, AlpacaPromptFormatter, DATA_DIR, DEFAULT_OLLAMA_API_URL,
                INSTRUCTION_DATA_FILENAME, INSTRUCTION_DATA_URL,
            },
        };
        use std::path::Path;

        // load instruction examples
        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;

        // invoke generate_chose_and_rejected_response
        let model = "llama3";
        let prompt_formatter = AlpacaPromptFormatter;
        let save_path = Path::new(DATA_DIR).join("instruction_data_with_preference.json");
        generate_preference_dataset(
            &data,
            DEFAULT_OLLAMA_API_URL,
            model,
            &prompt_formatter,
            save_path,
        )?;

        Ok(())
    }
}

/// # [BONUS] Example usage of `PreferenceDataCollator.custom_collate_fn`
///
/// #### Id
/// 07.19
///
/// #### Page
/// This example is adapted from `04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb`
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.19
///
/// # with cuda
/// cargo run --features cuda example 07.19
/// ```
pub struct EG19;

impl Example for EG19 {
    fn description(&self) -> String {
        let desc = "[Bonus from DPO notebook] Example usage of \
        `PreferenceDataCollator.custom_collate_fn`.";
        desc.to_string()
    }

    fn page_source(&self) -> usize {
        0_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch05::token_ids_to_text,
            ch07::bonus::{
                CustomCollator, EncodedPreferenceExample, PreferenceDataCollator, PreferenceExample,
            },
            ch07::{load_instruction_data_from_json, AlpacaPromptFormatter, DATA_DIR},
        };
        use candle_core::{Device, IndexOp};
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;

        // load preference examples
        let file_path = Path::new(DATA_DIR).join("instruction_data_with_preference.json");
        let preference_data: Vec<PreferenceExample> = load_instruction_data_from_json(file_path)
            .with_context(|| {
                "Missing 'instruction_data_with_preference.json' file. Please run EG 07.18."
            })?;

        // take a small sample and encode
        let sample = preference_data
            .into_iter()
            .take(2_usize)
            .collect::<Vec<_>>();

        println!("Sample Preference Examples:\n{:#?}", sample);

        let batch = sample
            .into_iter()
            .map(|el| EncodedPreferenceExample::from_example(&el, &prompt_formatter, &tokenizer))
            .collect::<Vec<_>>();

        let collator = PreferenceDataCollator::new().device(Device::cuda_if_available(0)?);
        let collated_item = collator.collate(batch)?;

        // print prompts
        println!(
            "\nCollated Batch: Prompt Tokens\n\n{:?}\n",
            collated_item
                .prompt()
                .iter()
                .map(|el| el.to_vec1::<u32>())
                .collect::<Vec<_>>()
        );

        // print chosen
        println!(
            "\nCollated Batch: Chosen Tokens\n\n{:?}\n",
            collated_item.chosen().to_vec2::<u32>()?
        );

        // Decode prompt and print
        let prompt_text = token_ids_to_text(collated_item.prompt()[1].clone(), &tokenizer)?;
        println!("\nCollated Batch Item 1: Prompt Text\n\n{}\n", prompt_text);

        // Decode chosen and print
        let chosen = collated_item.chosen().i((1, ..))?;
        let chosen_text = token_ids_to_text(chosen.clone(), &tokenizer)?;
        println!("\nCollated Batch Item 1: Chosen Text\n\n{}\n", chosen_text);

        // Decode chosen and print
        let rejected = collated_item.rejected().i((1, ..))?;
        let rejected_text = token_ids_to_text(rejected.clone(), &tokenizer)?;
        println!(
            "\nCollated Batch Item 1: Rejected Text\n\n{}\n",
            rejected_text
        );

        // Print masks and their shapes
        let chosen_mask = &collated_item.chosen_mask().i((1, ..))?;
        println!("\nCollated Batch: Masks\n");
        println!("Chosen inputs: {:?}", chosen);
        println!("Chosen mask: {:?}", chosen_mask);

        println!(
            "\nCollated Batch Item 1: Chosen Mask Indexes\n\n{:?}\n",
            chosen_mask.to_vec1::<u32>()?
        );

        // decode chosen mask
        // let chosen_masked_text = token_ids_to_text(
        //     chosen.index_select(chosen_mask.clone(), 0)?,
        //     &tokenizer,
        // )?;
        // println!(
        //     "\nCollated Batch Item 1: Chosen Mask Text\n\n{}\n",
        //     chosen_masked_text
        // );

        // decode rejected mask
        // let rejected_masked_text = token_ids_to_text(
        //     rejected.index_select(&collated_item.rejected_mask()[1], 0)?,
        //     &tokenizer,
        // )?;
        // println!(
        //     "\nCollated Batch Item 1: Rejected Mask Text\n\n{}\n",
        //     rejected_masked_text
        // );

        Ok(())
    }
}

/// # [BONUS] Creating a `PreferenceDataLoader`
///
/// #### Id
/// 07.20
///
/// #### Page
/// This example is adapted from `04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb`
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.20
///
/// # with cuda
/// cargo run --features cuda example 07.20
/// ```
pub struct EG20;

impl Example for EG20 {
    fn description(&self) -> String {
        "[BONUS] Creating a `PreferenceDataLoader`.".to_string()
    }

    fn page_source(&self) -> usize {
        0_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch07::bonus::{
                PreferenceDataCollator, PreferenceDataLoader, PreferenceDataset, PreferenceExample,
            },
            ch07::{
                load_instruction_data_from_json, partition_data, AlpacaPromptFormatter, DataLoader,
                DATA_DIR,
            },
        };
        use candle_core::Device;
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;

        // load preference examples
        let file_path = Path::new(DATA_DIR).join("instruction_data_with_preference.json");
        let preference_data: Vec<PreferenceExample> = load_instruction_data_from_json(file_path)
            .with_context(|| {
                "Missing 'instruction_data_with_preference.json' file. Please run EG 07.18."
            })?;

        // partition data and create train, val, test datasets
        let (train_data, val_data, test_data) =
            partition_data(preference_data, 0.85_f32, 0.05_f32)?;
        let train_dataset = PreferenceDataset::new(train_data, &tokenizer, &prompt_formatter);
        let val_dataset = PreferenceDataset::new(val_data, &tokenizer, &prompt_formatter);
        let test_dataset = PreferenceDataset::new(test_data, &tokenizer, &prompt_formatter);

        // create loaders
        let collator = PreferenceDataCollator::new().device(Device::cuda_if_available(0)?);
        let batch_size = 3_usize;
        let train_loader =
            PreferenceDataLoader::new(train_dataset, batch_size, true, true, collator.clone());
        let val_loader =
            PreferenceDataLoader::new(val_dataset, batch_size, false, false, collator.clone());
        let test_loader =
            PreferenceDataLoader::new(test_dataset, batch_size, false, false, collator);

        println!("Train loader:");
        let mut batcher = train_loader.batcher();
        while let Some(Ok(batch)) = batcher.next() {
            println!("batch: {:#?}", batch);
        }

        println!("Val loader:");
        let mut batcher = val_loader.batcher();
        while let Some(Ok(batch)) = batcher.next() {
            println!("batch: {:#?}", batch);
        }

        println!("Test loader:");
        let mut batcher = test_loader.batcher();
        while let Some(Ok(batch)) = batcher.next() {
            println!("batch: {:#?}", batch);
        }

        Ok(())
    }
}

/// # [BONUS] Example usage of `compute_dpo_loss_batch`
///
/// #### Id
/// 07.21
///
/// #### Page
/// This example is adapted from `04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb`
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.21
///
/// # with cuda
/// cargo run --features cuda example 07.21
/// ```
pub struct EG21;

impl Example for EG21 {
    fn description(&self) -> String {
        "[BONUS] Example usage of `compute_dpo_loss_batch`.".to_string()
    }

    fn page_source(&self) -> usize {
        0_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch07::bonus::{
                compute_dpo_loss_batch, PreferenceDataCollator, PreferenceDataLoader,
                PreferenceDataset, PreferenceExample,
            },
            ch07::{
                load_instruction_data_from_json, partition_data, AlpacaPromptFormatter, DataLoader,
                DATA_DIR,
            },
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;

        // load reference and policy model
        let mut cfg = Config::gpt2_124m(); // must match model size used in EG10
        cfg.qkv_bias = true;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let policy_model = GPTModel::new(cfg, vb.pp("model"))?;

        // load instructed-finetuned weights
        varmap
            .load("ift.checkpoint.safetensors")
            .with_context(|| "Missing 'ift.checkpoint.safetensors' file. Please run EG 07.10.")?;

        let vb2 = vb.clone();
        let reference_model = GPTModel::new(cfg, vb2.pp("model"))?;

        // load preference examples
        let file_path = Path::new(DATA_DIR).join("instruction_data_with_preference.json");
        let preference_data: Vec<PreferenceExample> = load_instruction_data_from_json(file_path)
            .with_context(|| {
                "Missing 'instruction_data_with_preference.json' file. Please run EG 07.18."
            })?;

        // partition data and create train, val, test datasets
        let (train_data, _val_data, _test_data) =
            partition_data(preference_data, 0.85_f32, 0.05_f32)?;
        let train_dataset = PreferenceDataset::new(train_data, &tokenizer, &prompt_formatter);

        // create loaders
        let collator = PreferenceDataCollator::new().device(Device::cuda_if_available(0)?);
        let batch_size = 2_usize;
        let train_loader =
            PreferenceDataLoader::new(train_dataset, batch_size, true, true, collator.clone());

        // get batcher
        let mut batcher = train_loader.batcher();

        if let Some(Ok(batch)) = batcher.next() {
            let (loss, chosen_r, rejected_r) =
                compute_dpo_loss_batch(&batch, &policy_model, &reference_model, 0.1, false)?;
            println!("{:?}", loss);
            println!("{:?}", chosen_r);
            println!("{:?}", rejected_r);
        }

        Ok(())
    }
}

/// # [BONUS] Example usage of `compute_dpo_loss_loader`
///
/// #### Id
/// 07.22
///
/// #### Page
/// This example is adapted from `04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb`
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.22
///
/// # with cuda
/// cargo run --features cuda example 07.22
/// ```
pub struct EG22;

impl Example for EG22 {
    fn description(&self) -> String {
        "[BONUS] Example usage of `compute_dpo_loss_loader`.".to_string()
    }

    fn page_source(&self) -> usize {
        0_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch07::bonus::{
                compute_dpo_loss_loader, PreferenceDataCollator, PreferenceDataLoader,
                PreferenceDataset, PreferenceExample,
            },
            ch07::{
                load_instruction_data_from_json, partition_data, AlpacaPromptFormatter, DATA_DIR,
            },
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;

        // load reference and policy model
        let mut cfg = Config::gpt2_124m(); // must match model size used in EG10
        cfg.qkv_bias = true;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let policy_model = GPTModel::new(cfg, vb.pp("model"))?;

        // load instructed-finetuned weights
        varmap
            .load("ift.checkpoint.safetensors")
            .with_context(|| "Missing 'ift.checkpoint.safetensors' file. Please run EG 07.10.")?;

        let vb2 = vb.clone();
        let reference_model = GPTModel::new(cfg, vb2.pp("model"))?;

        // load preference examples
        let file_path = Path::new(DATA_DIR).join("instruction_data_with_preference.json");
        let preference_data: Vec<PreferenceExample> = load_instruction_data_from_json(file_path)
            .with_context(|| {
                "Missing 'instruction_data_with_preference.json' file. Please run EG 07.18."
            })?;

        // partition data and create train, val, test datasets
        let (train_data, _val_data, _test_data) =
            partition_data(preference_data, 0.85_f32, 0.05_f32)?;
        let train_dataset = PreferenceDataset::new(train_data, &tokenizer, &prompt_formatter);

        // create loaders
        let collator = PreferenceDataCollator::new().device(Device::cuda_if_available(0)?);
        let batch_size = 2_usize;
        let train_loader =
            PreferenceDataLoader::new(train_dataset, batch_size, true, true, collator.clone());

        // compute dpo loss with loader
        let num_batches = Some(5_usize);
        let (train_loss, chosen_rewards, rejected_rewards) = compute_dpo_loss_loader(
            &train_loader,
            &policy_model,
            &reference_model,
            0.1,
            num_batches,
            false,
        )?;

        println!("Training loss: {}", train_loss);
        println!("Chosen rewards: {}", chosen_rewards);
        println!("Rejected rewards: {}", rejected_rewards);

        Ok(())
    }
}

/// # [BONUS] Example usage of `train_model_dpo_simple` and `plot_losses` functions
///
/// #### Id
/// 07.23
///
/// #### Page
/// This example is adapted from `04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb`
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.23
///
/// # with cuda
/// cargo run --features cuda example 07.23
/// ```
pub struct EG23;

impl Example for EG23 {
    fn description(&self) -> String {
        "[BONUS] Example usage of `train_model_dpo_simple` and `plot_losses` functions".to_string()
    }

    fn page_source(&self) -> usize {
        0_usize
    }

    // TODO: This fails silently if run into OOM issues.
    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch07::bonus::{
                train_model_dpo_simple, PreferenceDataCollator, PreferenceDataLoader,
                PreferenceDataset, PreferenceExample,
            },
            ch07::{
                load_instruction_data_from_json, partition_data, AlpacaPromptFormatter,
                PromptFormatter, DATA_DIR,
            },
        };
        use candle_core::{DType, Device};
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;

        // load preference examples
        let file_path = Path::new(DATA_DIR).join("instruction_data_with_preference.json");
        let preference_data: Vec<PreferenceExample> = load_instruction_data_from_json(file_path)
            .with_context(|| {
                "Missing 'instruction_data_with_preference.json' file. Please run EG 07.18."
            })?;

        // partition data and create train, val, test datasets
        let (train_data, val_data, _test_data) =
            partition_data(preference_data, 0.85_f32, 0.05_f32)?;
        let train_dataset = PreferenceDataset::new(train_data, &tokenizer, &prompt_formatter);
        let val_dataset = PreferenceDataset::new(val_data, &tokenizer, &prompt_formatter);

        // create loaders
        let collator = PreferenceDataCollator::new().device(Device::cuda_if_available(0)?);
        let batch_size = 3_usize;
        let train_loader =
            PreferenceDataLoader::new(train_dataset, batch_size, true, true, collator.clone());
        let val_loader =
            PreferenceDataLoader::new(val_dataset, batch_size, false, false, collator.clone());

        println!("Train loader: {}", train_loader.len());
        println!("Val loader: {}", val_loader.len());

        // load reference and policy model
        let mut cfg = Config::gpt2_124m(); // must match model size used in EG10
        cfg.qkv_bias = true;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let policy_model = GPTModel::new(cfg, vb.pp("model"))?;

        // load instructed-finetuned weights
        varmap
            .load("ift.checkpoint.safetensors")
            .with_context(|| "Missing 'ift.checkpoint.safetensors' file. Please run EG 07.10.")?;

        // load reference model
        let mut varmap_reference = VarMap::new();
        let vb_reference = VarBuilder::from_varmap(
            &varmap_reference,
            DType::F32,
            &Device::cuda_if_available(0)?,
        );
        let reference_model = GPTModel::new(cfg, vb_reference.pp("model"))?;

        // load instructed-finetuned weights
        varmap_reference
            .load("ift.checkpoint.safetensors")
            .with_context(|| "Missing 'ift.checkpoint.safetensors' file. Please run EG 07.10.")?;

        // invoke training
        let (eval_freq, eval_iter, num_epochs) = (5_usize, 5_usize, 2_usize);
        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.00005,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;
        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;
        let start_context = prompt_formatter.format_input(&val_loader.dataset().data()[0]);
        let tracking = train_model_dpo_simple(
            &policy_model,
            &reference_model,
            &train_loader,
            &val_loader,
            0.1,
            optimizer,
            vb.device(),
            num_epochs,
            eval_freq,
            eval_iter,
            start_context.as_str(),
            &tokenizer,
        )?;

        println!("{:#?}", tracking);

        // save model
        println!("Saving weights to `./dpo.checkpoint.safetensors`");
        varmap.save("dpo.checkpoint.safetensors")?;

        Ok(())
    }
}
