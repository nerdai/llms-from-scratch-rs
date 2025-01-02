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
    rc::Rc,
};
use tiktoken_rs::CoreBPE;

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

#[allow(dead_code)]
pub struct InstructionDataset_ {
    data: Vec<InstructionResponseExample>,
    encoded_texts: Vec<Vec<u32>>,
}

/// [Listing 7.4] Implementing an `InsructionDataset` type
///
/// InsructionDataset is a wrapper for `InstructionDataset_` which is refcounted.
/// Note: pad_token_id is handled via the tokenizer in this example.
#[derive(Clone)]
pub struct InsructionDataset(Rc<InstructionDataset_>);

impl AsRef<InsructionDataset> for InsructionDataset {
    fn as_ref(&self) -> &InsructionDataset {
        self
    }
}

impl std::ops::Deref for InsructionDataset {
    type Target = InstructionDataset_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl InsructionDataset {
    pub fn new(data: Vec<InstructionResponseExample>, tokenizer: &CoreBPE) -> Self {
        let mut encoded_texts = vec![];
        for entry in data.iter() {
            let instruction_plus_input = format_input(entry);
            let response_text = format!("\n\n### Response:\n{}", entry.output());
            let full_text = instruction_plus_input + &response_text;
            let encoded_text = tokenizer.encode_with_special_tokens(&full_text);
            encoded_texts.push(encoded_text);
        }
        let dataset_ = InstructionDataset_ {
            data,
            encoded_texts,
        };
        Self(Rc::new(dataset_))
    }

    /// Gets the number of finetuning examples.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Checks whether the dataset is empty or has no finetuning examples.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the tokenized and formatted instruction entry at specified index
    #[allow(unused_variables)]
    pub fn get_item_at_index(&self, idx: usize) -> anyhow::Result<&Vec<u32>> {
        let encoded = &self.encoded_texts[idx];
        Ok(encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use rstest::*;
    use tempfile::NamedTempFile;
    use tiktoken_rs::get_bpe_from_model;

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

    #[rstest]
    pub fn test_instruction_dataset_init(
        instruction_example: InstructionResponseExample,
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let data = vec![
            instruction_example.clone(),
            instruction_example.clone(),
            instruction_example.clone(),
            instruction_example.clone(),
            instruction_example,
        ];
        let instruction_dataset = InsructionDataset::new(data, &tokenizer);

        assert_eq!(instruction_dataset.len(), 5);

        Ok(())
    }
}
