//! Examples from Chapter 7

use crate::Example;
use anyhow::Result;
use candle_core::Device;

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
            download_and_load_file, format_input, DATA_DIR, INSTRUCTION_DATA_FILENAME,
            INSTRUCTION_DATA_URL,
        };
        use std::path::Path;

        // load instruction examples
        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;

        // first model input
        let model_input = format_input(&data[50]);
        let detailed_response = format!("\n\n### Response:\n{}", data[50].output());
        println!("{}", model_input + &detailed_response);

        // print a separator
        println!("\n---\n");

        // print another model input
        let model_input = format_input(&data[999]);
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
    #[allow(unused_variables)]
    pub fn main_with_return(
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
            download_and_load_file, partition_data, InstructionDataCollator, InstructionDataLoader,
            InstructionDataset, DATA_DIR, INSTRUCTION_DATA_FILENAME, INSTRUCTION_DATA_URL,
        };
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2")?;

        // load instruction examples
        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;

        // partition data and create train, val, test datasets
        let (train_data, val_data, test_data) = partition_data(data, 0.85_f32, 0.05_f32)?;
        let train_dataset = InstructionDataset::new(train_data, &tokenizer);
        let val_dataset = InstructionDataset::new(val_data, &tokenizer);
        let test_dataset = InstructionDataset::new(test_data, &tokenizer);

        // create loaders
        let batch_size = 8_usize;
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
        let _ = self.main_with_return(true);
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
                download_and_load_file, download_and_load_gpt2, format_input, partition_data,
                DATA_DIR, INSTRUCTION_DATA_FILENAME, INSTRUCTION_DATA_URL,
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
        let input_text = format_input(&val_data[0]);
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
