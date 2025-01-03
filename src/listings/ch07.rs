//! Listings from Chapter 7

use anyhow::Context;
use bytes::Bytes;
use candle_core::{Device, Result, Tensor};
use rand::{seq::SliceRandom, thread_rng};
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
    pub fn new(instruction: &str, input: Option<&str>, output: &str) -> Self {
        Self {
            instruction: instruction.to_string(),
            input: input.map(|inp| inp.to_string()),
            output: output.to_string(),
        }
    }

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
pub struct InstructionDataset(Rc<InstructionDataset_>);

impl AsRef<InstructionDataset> for InstructionDataset {
    fn as_ref(&self) -> &InstructionDataset {
        self
    }
}

impl std::ops::Deref for InstructionDataset {
    type Target = InstructionDataset_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl InstructionDataset {
    /// Creates a new `InstructionDataset`.
    ///
    /// ```rust
    /// use llms_from_scratch_rs::listings::ch07::{
    ///     InstructionDataset, InstructionResponseExample
    /// };
    /// use tiktoken_rs::get_bpe_from_model;
    ///
    /// let entry = InstructionResponseExample::new(
    ///     "Some instruction",
    ///     None,
    ///     "Some output"
    /// );
    /// let data = vec![entry];
    /// let tokenizer = get_bpe_from_model("gpt2").unwrap();
    /// let dataset = InstructionDataset::new(data, &tokenizer);
    /// ```
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
    pub fn get_item_at_index(&self, idx: usize) -> anyhow::Result<&Vec<u32>> {
        let encoded = &self.encoded_texts[idx];
        Ok(encoded)
    }
}

pub struct InstructionDatasetIter {
    dataset: InstructionDataset,
    remaining_indices: Vec<usize>,
}

impl InstructionDatasetIter {
    pub fn new(dataset: InstructionDataset, shuffle: bool) -> Self {
        let mut remaining_indices = (0..dataset.len()).rev().collect::<Vec<_>>();
        if shuffle {
            remaining_indices.shuffle(&mut thread_rng());
        }
        Self {
            dataset,
            remaining_indices,
        }
    }
}

impl Iterator for InstructionDatasetIter {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.remaining_indices.pop() {
            let encoded = self.dataset.get_item_at_index(idx).unwrap();

            // turn into Tensors and return
            let dev = Device::cuda_if_available(0).unwrap();
            let inputs_tensor = Tensor::new(&encoded[..], &dev);
            Some(inputs_tensor)
        } else {
            None
        }
    }
}

pub struct IterResult1<I: Iterator<Item = Result<Tensor>>> {
    inner: I,
}

/// A type alias for candle_datasets::Batcher
///
/// This struct is responsible for getting batches from a type that implements
/// the `Iterator` Trait.
pub struct InstructionDataBatcher<C: CustomCollator> {
    inner: IterResult1<InstructionDatasetIter>,
    batch_size: usize,
    return_last_incomplete_batch: bool,
    collator: C,
}

pub trait CustomCollator {
    fn collate(&self, batch: Vec<Tensor>) -> Result<(Tensor, Tensor)>;
}

impl<C: CustomCollator> InstructionDataBatcher<C> {
    pub fn new(inner: InstructionDatasetIter, collator: C) -> Self {
        Self {
            inner: IterResult1 { inner },
            collator,
            batch_size: 16,
            return_last_incomplete_batch: false,
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn return_last_incomplete_batch(mut self, r: bool) -> Self {
        self.return_last_incomplete_batch = r;
        self
    }

    pub fn collator(mut self, collator: C) -> Self {
        self.collator = collator;
        self
    }
}

impl<C: CustomCollator> Iterator for InstructionDataBatcher<C> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut items = Vec::with_capacity(self.batch_size);
        let mut errs = vec![];
        for _i in 0..self.batch_size {
            // We have two levels of inner here so that we can have two implementations of the
            // Iterator trait that are different for Iter1 and Iter2. If rust gets better
            // specialization at some point we can get rid of this.
            match self.inner.inner.next() {
                Some(Ok(item)) => items.push(item),
                Some(Err(err)) => errs.push(err),
                None => {
                    if self.return_last_incomplete_batch {
                        break;
                    }
                    return None;
                }
            }
        }
        Some(self.collator.collate(items))
    }
}

/// A type for specifying how to collate batches of instruct entries
///
/// NOTE: used for implementing Listing 7.5
pub struct InstructDataCollator {
    pad_token_id: u32,
    ignore_index: i64,
    allowed_max_length: Option<usize>,
    device: Device,
}

const DEFAULT_IGNORE_INDEX: i64 = -100;
const DEFAULT_PAD_TOKEN_ID: u32 = 50_256;

impl Default for InstructDataCollator {
    fn default() -> Self {
        Self {
            pad_token_id: DEFAULT_PAD_TOKEN_ID,
            ignore_index: DEFAULT_IGNORE_INDEX,
            allowed_max_length: None,
            device: Device::Cpu,
        }
    }
}

impl InstructDataCollator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pad_token_id(mut self, pad_token_id: u32) -> Self {
        self.pad_token_id = pad_token_id;
        self
    }

    pub fn ignore_index(mut self, ignore_index: i64) -> Self {
        self.ignore_index = ignore_index;
        self
    }

    pub fn allowed_max_length(mut self, allowed_max_length: Option<usize>) -> Self {
        self.allowed_max_length = allowed_max_length;
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// [Listing 7.5] Implementing a custom batch collate function
    ///
    /// NOTE: this function gets applied via a wrapper on candle_datasets::Batcher
    fn custom_collate_fn(&self, batch: Vec<Tensor>) -> Result<(Tensor, Tensor)> {
        // modify batch
        let batch_max_length = batch
            .iter()
            .map(|el| el.elem_count())
            .collect::<Vec<_>>()
            .into_iter()
            .max()
            .unwrap();
        let mut inputs_lst: Vec<Vec<u32>> = vec![];
        let mut targets_lst: Vec<Vec<i64>> = vec![];

        for item in batch.into_iter() {
            let mut input = item.to_vec1::<u32>()?;
            let mut target = item
                .to_vec1::<u32>()?
                .into_iter()
                .map(|el| el as i64)
                .collect::<Vec<_>>()[1..]
                .to_vec();

            // padding and ignore index
            target.push(self.pad_token_id as i64);
            let num_pad =
                std::cmp::max(0isize, batch_max_length as isize - input.len() as isize) as usize;
            if num_pad > 0 {
                let padding_input = std::iter::repeat(self.pad_token_id)
                    .take(num_pad)
                    .collect::<Vec<u32>>();
                input.extend(padding_input);
            }
            let ignore_index_target = std::iter::repeat(self.ignore_index)
                .take(num_pad)
                .collect::<Vec<i64>>();
            target.extend(ignore_index_target);

            if let Some(a) = self.allowed_max_length {
                input = input[..a].to_vec();
                target = target[..a].to_vec();
            }

            inputs_lst.push(input);
            targets_lst.push(target);
        }

        let inputs_shape = (inputs_lst.len(), inputs_lst[0].len());
        let inputs_tensor = Tensor::from_vec(
            inputs_lst.into_iter().flatten().collect(),
            inputs_shape,
            &self.device,
        );
        let targets_shape = (targets_lst.len(), targets_lst[0].len());
        let targets_tensor = Tensor::from_vec(
            targets_lst.into_iter().flatten().collect(),
            targets_shape,
            &self.device,
        );
        candle_core::error::zip(inputs_tensor, targets_tensor)
    }
}

impl CustomCollator for InstructDataCollator {
    fn collate(&self, batch: Vec<Tensor>) -> Result<(Tensor, Tensor)> {
        self.custom_collate_fn(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use ndarray::iter;
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

    #[fixture]
    fn instruction_data(
        instruction_example: InstructionResponseExample,
    ) -> Vec<InstructionResponseExample> {
        let data = vec![
            instruction_example.clone(),
            instruction_example.clone(),
            instruction_example.clone(),
            instruction_example.clone(),
            instruction_example,
        ];
        data
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
        instruction_data: Vec<InstructionResponseExample>,
        instruction_example: InstructionResponseExample,
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let instruction_dataset = InstructionDataset::new(instruction_data, &tokenizer);

        // test encoded
        let prompt = format_input(&instruction_example);
        let response_text = format!("\n\n### Response:\n{}", instruction_example.output());
        let full_text = prompt + &response_text;
        let encoded = tokenizer.encode_with_special_tokens(&full_text);

        assert_eq!(instruction_dataset.len(), 5);
        assert_eq!(*instruction_dataset.get_item_at_index(0_usize)?, encoded);

        Ok(())
    }

    #[rstest]
    pub fn test_instruction_dataset_iter(
        instruction_data: Vec<InstructionResponseExample>,
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let instruction_dataset = InstructionDataset::new(instruction_data, &tokenizer);
        let mut iter = InstructionDatasetIter::new(instruction_dataset.clone(), false);
        let mut count = 0_usize;

        // user iter to sequentially get next pair checking equality with dataset
        while let Some(Ok(item)) = iter.next() {
            assert!(item.dims()[0] == instruction_dataset.encoded_texts[count].len());
            count += 1;
        }
        assert_eq!(count, instruction_dataset.len());
        Ok(())
    }

    #[rstest]
    pub fn test_instruction_collator(
        instruction_data: Vec<InstructionResponseExample>,
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let instruction_dataset = InstructionDataset::new(instruction_data, &tokenizer);
        let iter = InstructionDatasetIter::new(instruction_dataset.clone(), false);
        let batch_size = 2_usize;
        let collator = InstructDataCollator::new().device(Device::cuda_if_available(0)?);
        let mut instruct_batcher = InstructionDataBatcher::new(iter, collator)
            .batch_size(batch_size)
            .return_last_incomplete_batch(false);
        let mut count = 0_usize;

        while let Some(Ok((inputs_batch, targets_batch))) = instruct_batcher.next() {
            assert!(inputs_batch.dims()[0] == targets_batch.dims()[0]);
            assert!(inputs_batch.dims()[1] == targets_batch.dims()[1]);
            count += 1;
        }

        assert_eq!(count, 2_usize);
        Ok(())
    }
}
