//! Examples from Chapter 7

use crate::Example;
use anyhow::{Ok, Result};

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

/// # Example usage of `custom_collate_fn`
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
        String::from("Example usage of `custom_collate_fn`.")
    }

    fn page_source(&self) -> usize {
        220_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::{
            download_and_load_file, partition_data, InstructDataCollator, InstructionDataBatcher,
            InstructionDataBatcher_, DATA_DIR, INSTRUCTION_DATA_FILENAME, INSTRUCTION_DATA_URL,
        };
        use candle_core::{DType, Device, Tensor};
        use tiktoken_rs::get_bpe_from_model;

        let device = Device::cuda_if_available(0)?;
        let inputs_1 = Tensor::new(&[0_u32, 1, 2, 3, 4], &device)?;
        let inputs_2 = Tensor::new(&[5_u32, 6], &device)?;
        let inputs_3 = Tensor::new(&[7_u32, 8, 9], &device)?;
    }
}
