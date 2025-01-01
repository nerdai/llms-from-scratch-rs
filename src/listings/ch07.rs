//! Listings from Chapter 7

use anyhow::Context;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, NoneAsEmptyString};
use std::{
    fmt::Display,
    fs::{read_to_string, File},
    io,
    path::Path,
};

pub const INSTRUCTION_DATA_FILENAME: &str = "instruction_data.json";
pub const DATA_DIR: &str = "data";
pub const INSTRUCTION_DATA_URL: &str = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch\
/main/ch07/01_main-chapter-code/instruction-data.json";

/// A type for containing an instruction-response pair
#[serde_as]
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct InstructionResponseExample {
    instruction: String,
    #[serde_as(as = "NoneAsEmptyString")]
    input: Option<String>,
    output: String,
}

impl InstructionResponseExample {
    pub fn instruction(&self) -> &String {
        &self.instruction
    }

    pub fn input(&self) -> &Option<String> {
        &self.input
    }

    pub fn output(&self) -> &String {
        &self.output
    }
}

impl Display for InstructionResponseExample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Instruction: {}\nInput: {:?}\nOutput: {}",
            self.instruction, self.input, self.output
        )
    }
}

/// [Listing 7.1] Downloading the dataset
#[allow(unused_variables)]
pub fn download_and_load_file<P: AsRef<Path>>(
    file_path: P,
    url: &str,
    overwrite: bool,
) -> anyhow::Result<Vec<InstructionResponseExample>> {
    if !file_path.as_ref().exists() || overwrite {
        // download json file
        let resp = reqwest::blocking::get(url)?;
        let content: Bytes = resp.bytes()?;
        let mut out = File::create(file_path.as_ref())?;
        io::copy(&mut content.as_ref(), &mut out)?;
    }
    let json_str = read_to_string(file_path.as_ref())
        .with_context(|| format!("Unable to read {}", file_path.as_ref().display()))?;
    let data: Vec<InstructionResponseExample> = serde_json::from_str(&json_str[..])?;

    Ok(data)
}

/// [Listing 7.2] Implementing the prompt formatting function
pub fn format_input(entry: &InstructionResponseExample) -> String {
    let instruction_text = format!(
        "Below is an instruction that describes a task. Write a response that \
        appropriately completes the request.\n\n### Instruction:\n{}",
        entry.instruction
    );
    let input_text = if let Some(inp) = &entry.input {
        format!("\n\n### Input:\n{}", inp)
    } else {
        String::default()
    };
    instruction_text + &input_text
}

/// [Listing 7.3] Partitioning the dataset
#[allow(unused_variables)]
pub fn partition_data(
    data: Vec<InstructionResponseExample>,
    train_frac: f32,
    validation_frac: f32,
) -> anyhow::Result<(
    Vec<InstructionResponseExample>,
    Vec<InstructionResponseExample>,
    Vec<InstructionResponseExample>,
)> {
    let train_portion = (data.len() as f32 * train_frac) as usize;
    let val_portion = (data.len() as f32 * validation_frac) as usize;

    let train_data = &data[..train_portion];
    let val_data = &data[train_portion..train_portion + val_portion];
    let test_data = &data[train_portion + val_portion..];

    Ok((train_data.to_vec(), val_data.to_vec(), test_data.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use rstest::*;
    use tempfile::NamedTempFile;

    #[fixture]
    fn instruction_example() -> InstructionResponseExample {
        let instruction = "Here is a fake instruction.".to_string();
        let input = Some("Here is a fake input.".to_string());
        let output = "here is a fake output.".to_string();
        InstructionResponseExample {
            instruction,
            input,
            output,
        }
    }

    #[rstest]
    fn test_download_and_load_file() -> Result<()> {
        let test_file = NamedTempFile::new().unwrap();
        let file_path = test_file.into_temp_path().keep().unwrap();
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, true)?;
        assert_eq!(data.len(), 1100);
        Ok(())
    }

    #[rstest]
    fn test_format_input_with_some_input(
        mut instruction_example: InstructionResponseExample,
    ) -> Result<()> {
        instruction_example.input = None; // set input to None
        let prompt = format_input(&instruction_example);
        let expected_output = format!(
            "Below is an instruction that describes a task. Write a response that \
            appropriately completes the request.\n\n### Instruction:\n{}",
            instruction_example.instruction,
        );

        assert_eq!(prompt, expected_output);
        Ok(())
    }

    #[rstest]
    fn test_format_input_with_no_input(
        instruction_example: InstructionResponseExample,
    ) -> Result<()> {
        let prompt = format_input(&instruction_example);
        let expected_output = format!(
            "Below is an instruction that describes a task. Write a response that \
            appropriately completes the request.\n\n### Instruction:\n{}\
            \n\n### Input:\n{}",
            instruction_example.instruction,
            instruction_example.input.unwrap()
        );

        assert_eq!(prompt, expected_output);
        Ok(())
    }

    #[rstest]
    fn test_partition_data(instruction_example: InstructionResponseExample) -> Result<()> {
        let data = vec![
            instruction_example.clone(),
            instruction_example.clone(),
            instruction_example.clone(),
            instruction_example.clone(),
            instruction_example,
        ];

        let (train_data, val_data, test_data) = partition_data(data, 0.6, 0.2)?;

        assert_eq!(train_data.len(), 3);
        assert_eq!(val_data.len(), 1);
        assert_eq!(test_data.len(), 1);

        Ok(())
    }
}
