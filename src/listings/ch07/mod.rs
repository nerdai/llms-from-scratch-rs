//! Listings from Chapter 7

use super::{
    ch04::GPT,
    ch05::{generate, text_to_token_ids, token_ids_to_text},
};
use anyhow::Context;
use bytes::Bytes;
use candle_core::{Device, Result, Tensor};
use candle_nn::ModuleT;
use hf_hub::api::sync::Api;
use rand::{rng, seq::SliceRandom};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, NoneAsEmptyString};
use std::{
    fmt::Display,
    fs::{read_to_string, File},
    io,
    io::Write,
    path::Path,
    rc::Rc,
};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};
use tqdm::tqdm;

/// Bonus material
pub mod bonus;

pub const INSTRUCTION_DATA_FILENAME: &str = "instruction_data.json";
pub const DATA_DIR: &str = "data";
pub const INSTRUCTION_DATA_URL: &str = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch\
/main/ch07/01_main-chapter-code/instruction-data.json";

/// A type for containing an instruction-response pair
#[serde_as]
#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct InstructionResponseExample {
    instruction: String,
    #[serde_as(as = "NoneAsEmptyString")]
    input: Option<String>,
    output: String,
    model_response: Option<String>, // added for Listing 7.9
}

impl InstructionResponseExample {
    pub fn new(instruction: &str, input: Option<&str>, output: &str) -> Self {
        Self {
            instruction: instruction.to_string(),
            input: input.map(|inp| inp.to_string()),
            output: output.to_string(),
            model_response: None,
        }
    }

    pub fn model_response(&self) -> &Option<String> {
        &self.model_response
    }

    pub fn set_model_response(&mut self, model_response: &str) {
        self.model_response = Some(model_response.to_string());
    }
}

// Marker trait for instruction example
pub trait InstructionExample {
    fn instruction(&self) -> &String;
    fn input(&self) -> &Option<String>;
    fn output(&self) -> &String;
}

impl InstructionExample for InstructionResponseExample {
    fn instruction(&self) -> &String {
        &self.instruction
    }

    fn input(&self) -> &Option<String> {
        &self.input
    }

    fn output(&self) -> &String {
        &self.output
    }
}

impl Display for InstructionResponseExample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Instruction: {}\nInput: {:?}\nOutput: {}\nModel Response: {:?}",
            self.instruction, self.input, self.output, self.model_response
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
    // load json file
    let data = load_instruction_data_from_json(file_path)?;
    Ok(data)
}

/// Prompt trait introduced for Excercise 7.1 to extend `format_input` to other prompt styles
pub trait PromptFormatter {
    fn format_input<T: InstructionExample>(&self, entry: &T) -> String;
}

/// Alpaca prompt formatter type [used in Listing 7.2]
pub struct AlpacaPromptFormatter;

impl PromptFormatter for AlpacaPromptFormatter {
    /// [Listing 7.2] Implementing the prompt formatting function
    fn format_input<T: InstructionExample>(&self, entry: &T) -> String {
        let instruction_text = format!(
            "Below is an instruction that describes a task. Write a response that \
            appropriately completes the request.\n\n### Instruction:\n{}",
            entry.instruction()
        );
        let input_text = if let Some(inp) = &entry.input() {
            format!("\n\n### Input:\n{}", inp)
        } else {
            String::default()
        };
        instruction_text + &input_text
    }
}

pub struct Phi3PromptFormatter;

impl PromptFormatter for Phi3PromptFormatter {
    fn format_input<T: InstructionExample>(&self, entry: &T) -> String {
        match entry.input() {
            Some(input_str) => format!(
                "<|user|>\n{}\n{}\n\n<|assistant|>\n{}",
                entry.instruction(),
                input_str,
                entry.output()
            ),
            None => format!(
                "<|user|>\n{}\n\n<|assistant|>\n{}",
                entry.instruction(),
                entry.output()
            ),
        }
    }
}

/// [Listing 7.3] Partitioning the dataset
#[allow(unused_variables)]
pub fn partition_data<T: Clone>(
    data: Vec<T>,
    train_frac: f32,
    validation_frac: f32,
) -> anyhow::Result<(Vec<T>, Vec<T>, Vec<T>)> {
    let train_portion = (data.len() as f32 * train_frac) as usize;
    let val_portion = (data.len() as f32 * validation_frac) as usize;

    let train_data = &data[..train_portion];
    let val_data = &data[train_portion..train_portion + val_portion];
    let test_data = &data[train_portion + val_portion..];

    Ok((train_data.to_vec(), val_data.to_vec(), test_data.to_vec()))
}

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
    ///     AlpacaPromptFormatter, InstructionDataset, InstructionResponseExample,
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
    /// let prompt_formatter = AlpacaPromptFormatter;
    /// let dataset = InstructionDataset::new(data, &tokenizer, &prompt_formatter);
    /// ```
    pub fn new<P: PromptFormatter>(
        data: Vec<InstructionResponseExample>,
        tokenizer: &CoreBPE,
        prompt_formatter: &P, // introduced for Exercise 7.1
    ) -> Self {
        let mut encoded_texts = vec![];
        for entry in data.iter() {
            let instruction_plus_input = prompt_formatter.format_input(entry);
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

    pub fn data(&self) -> &Vec<InstructionResponseExample> {
        &self.data
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
            remaining_indices.shuffle(&mut rng());
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

// Taken from `candle_datasets::batcher` to create a similar interface for `InstructionDataBatcher`
pub struct IterResult1<I: Iterator<Item = Result<Tensor>>> {
    inner: I,
}

pub struct IterResult2<I: Iterator<Item = Result<(Tensor, Tensor)>>> {
    inner: I,
}

/// The `InstructionDataBatcher` for batching instruction examples
///
/// NOTE: Had to implement own version of candle_datasets::Batcher since we
/// needed to work with `Vec<Tensor>` in collate function. The former utilizes
/// `Tensor::cat()` which requires all Tensor's to have same rank, but we only
/// get this after collation is performed.
pub struct InstructionDataBatcher<C: CustomCollator, I> {
    inner: I,
    batch_size: usize,
    return_last_incomplete_batch: bool,
    collator: C,
}

/// A trait for collating a Vector of Tensor's into a batch
pub trait CustomCollator {
    type BatchItem;

    fn collate(&self, batch: Vec<Self::BatchItem>) -> Result<(Tensor, Tensor)>;
}

impl<C, I> InstructionDataBatcher<C, IterResult1<I>>
where
    C: CustomCollator<BatchItem = Tensor>,
    I: Iterator<Item = Result<Tensor>>,
{
    pub fn new(inner: I, collator: C) -> Self {
        Self::new_r1(inner, collator)
    }

    pub fn new_r1(inner: I, collator: C) -> Self {
        Self {
            inner: IterResult1 { inner },
            collator,
            batch_size: 16,
            return_last_incomplete_batch: false,
        }
    }
}

// needed for Exercise 7.2
impl<C, I> InstructionDataBatcher<C, IterResult2<I>>
where
    C: CustomCollator<BatchItem = (Tensor, Tensor)>,
    I: Iterator<Item = Result<(Tensor, Tensor)>>,
{
    pub fn new_r2(inner: I, collator: C) -> Self {
        Self {
            inner: IterResult2 { inner },
            collator,
            batch_size: 16,
            return_last_incomplete_batch: false,
        }
    }
}

impl<C: CustomCollator, I> InstructionDataBatcher<C, I> {
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

impl<C, I> Iterator for InstructionDataBatcher<C, IterResult1<I>>
where
    C: CustomCollator<BatchItem = Tensor>,
    I: Iterator<Item = Result<Tensor>>,
{
    type Item = Result<(Tensor, Tensor)>;

    // This closely mirrors logic used in candle_datasets::batcher.
    // However here, the inner iterator has associated item Result<Tensor>
    // and the outer iterator has associated item Result<(Tensor, Tensor)>
    fn next(&mut self) -> Option<Self::Item> {
        let mut items = Vec::with_capacity(self.batch_size);
        let mut errs = vec![];
        for _i in 0..self.batch_size {
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

// needed for Exercise 7.2
impl<C, I> Iterator for InstructionDataBatcher<C, IterResult2<I>>
where
    C: CustomCollator<BatchItem = (Tensor, Tensor)>,
    I: Iterator<Item = Result<(Tensor, Tensor)>>,
{
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut items = Vec::with_capacity(self.batch_size);
        let mut errs = vec![];
        for _i in 0..self.batch_size {
            match self.inner.inner.next() {
                Some(Ok(item)) => items.push(item),
                Some(Err(err)) => errs.push(err),
                None => {
                    if self.return_last_incomplete_batch && !items.is_empty() {
                        break;
                    }
                    return None;
                }
            }
        }
        Some(self.collator.collate(items))
    }
}

pub use crate::listings::ch05::DEFAULT_IGNORE_INDEX;
pub const DEFAULT_PAD_TOKEN_ID: u32 = 50_256;

/// A type for specifying how to collate batches of instruct entries [used for Listing 7.5]
#[derive(Clone)]
pub struct InstructionDataCollator {
    pad_token_id: u32,
    ignore_index: i64,
    allowed_max_length: Option<usize>,
    device: Device,
}

impl Default for InstructionDataCollator {
    fn default() -> Self {
        Self {
            pad_token_id: DEFAULT_PAD_TOKEN_ID,
            ignore_index: DEFAULT_IGNORE_INDEX,
            allowed_max_length: None,
            device: Device::Cpu,
        }
    }
}

impl InstructionDataCollator {
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
    pub fn custom_collate_fn(&self, batch: Vec<Tensor>) -> Result<(Tensor, Tensor)> {
        // modify batch
        let batch_max_length = batch
            .iter()
            .map(|el| el.elem_count())
            .collect::<Vec<_>>()
            .into_iter()
            .max()
            .ok_or_else(|| {
                candle_core::Error::Msg("Unable to get max length for batch.".to_string())
            })?;
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
                input = input[..std::cmp::min(a, batch_max_length)].to_vec();
                target = target[..std::cmp::min(a, batch_max_length)].to_vec();
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

impl CustomCollator for InstructionDataCollator {
    type BatchItem = Tensor;

    fn collate(&self, batch: Vec<Tensor>) -> Result<(Tensor, Tensor)> {
        self.custom_collate_fn(batch)
    }
}

/// [Listing 7.6] Initializing the data loaders (`InstructionDataLoader`)
pub struct InstructionDataLoader<C: CustomCollator<BatchItem = Tensor>> {
    dataset: InstructionDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    collator: C,
}

impl<C: CustomCollator<BatchItem = Tensor> + Clone> DataLoader for InstructionDataLoader<C> {
    type Batcher = InstructionDataBatcher<C, IterResult1<InstructionDatasetIter>>;

    /// Returns a `InstructionDataBatcher` that itself provides batches over the
    /// associated dataset.
    fn batcher(&self) -> InstructionDataBatcher<C, IterResult1<InstructionDatasetIter>> {
        let iter = InstructionDatasetIter::new(self.dataset.clone(), self.shuffle);
        InstructionDataBatcher::new(iter, self.collator.clone())
            .batch_size(self.batch_size)
            .return_last_incomplete_batch(!self.drop_last)
    }
}

impl<C: CustomCollator<BatchItem = Tensor> + Clone> InstructionDataLoader<C> {
    /// Creates a new `InstructionDataLoader`.
    ///
    /// ```rust
    /// use candle_core::Device;
    /// use llms_from_scratch_rs::listings::ch07::{
    ///     AlpacaPromptFormatter, InstructionDataCollator, InstructionDataset,
    ///     InstructionDataLoader, InstructionResponseExample
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
    /// let prompt_formatter = AlpacaPromptFormatter;
    /// let dataset = InstructionDataset::new(data, &tokenizer, &prompt_formatter);
    ///
    /// // create InstructionDataLoader
    /// let batch_size = 2_usize;
    /// let shuffle = false;
    /// let drop_last = false;
    /// let collator = InstructionDataCollator::default();
    /// let data_loader = InstructionDataLoader::new(
    ///     dataset, batch_size, shuffle, drop_last, collator
    /// );
    /// ```
    pub fn new(
        dataset: InstructionDataset,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        collator: C,
    ) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            collator,
        }
    }

    pub fn len(&self) -> usize {
        if self.drop_last {
            self.batcher().count()
        } else {
            // There is a bug in candle_datasets::Batcher, such that if
            // return_last_incomplete_batch is set to true, then the iterator
            // will never return None. This breaks `Iterator.count()` which consumes
            // the iterator until a None is encountered.
            let mut batcher = self.batcher();
            let mut count = 0_usize;
            while let Some(Ok(_el)) = batcher.next() {
                count += 1;
            }
            count
        }
    }

    pub fn is_empty(&self) -> bool {
        (self.dataset.len() < self.batch_size) && (self.drop_last)
    }

    pub fn dataset(&self) -> &InstructionDataset {
        &self.dataset
    }
}

/// [Listing 7.7] Loading a pretrained GPT model
///
/// NOTE: This is merely a re-export from `listings::ch06`.
#[doc(inline)]
pub use crate::listings::ch06::download_and_load_gpt2;

/// Delete previously downloaded model weights from local HF cache.
pub fn delete_hf_cache(model_id: &str) -> Result<()> {
    let api = Api::new().map_err(candle_core::Error::wrap)?;
    let repo = api.model(model_id.to_string());
    let weights = repo
        .get("model.safetensors")
        .map_err(candle_core::Error::wrap)?;
    std::fs::remove_file(weights)?;

    Ok(())
}

/// [Listing 7.8] Instruction fine-tuning the pretrained LLM
///
/// NOTE: This has been modified from the actual listing from the book. Here
/// we merely re-export `listings::ch05::train_model_simple`. The act of actually
/// fine-tuning is left as an example — see EG 07.10.
#[doc(inline)]
pub use crate::listings::ch05::train_model_simple;

// for convenience we also re-export the following
pub use crate::listings::ch02::DataLoader;
pub use crate::listings::ch05::calc_loss_loader;

/// Helper function to write instruction data to a json
pub fn load_instruction_data_from_json<P: AsRef<Path>, S: Serialize + for<'a> Deserialize<'a>>(
    file_path: P,
) -> anyhow::Result<Vec<S>> {
    let json_str = read_to_string(file_path.as_ref())
        .with_context(|| format!("Unable to read {}", file_path.as_ref().display()))?;
    let data: Vec<S> = serde_json::from_str(&json_str[..])?;
    Ok(data)
}

/// Helper function to write instruction data to a json
pub fn write_instruction_data_to_json<P: AsRef<Path>, S: Serialize + for<'a> Deserialize<'a>>(
    instruction_data: &Vec<S>,
    save_path: P,
) -> anyhow::Result<()> {
    let file = File::create(save_path)?;
    let mut writer = io::BufWriter::new(file);
    serde_json::to_writer(&mut writer, instruction_data)?;
    writer.flush()?;
    Ok(())
}

/// [Listing 7.9] Generating test set responses
pub fn generate_test_set_responses<T: AsRef<Path>, P: PromptFormatter, M: GPT + ModuleT>(
    test_data: &mut Vec<InstructionResponseExample>,
    model: &M,
    context_size: usize,
    device: &Device,
    save_path: T,
    prompt_formatter: &P, // introduced for Exercise 7.1
) -> anyhow::Result<()> {
    let tokenizer = get_bpe_from_model("gpt2")?;
    let mut rng = StdRng::seed_from_u64(42_u64);

    for entry in tqdm(test_data.iter_mut()) {
        let input_text = prompt_formatter.format_input(entry);
        let token_ids = generate(
            model,
            text_to_token_ids(&input_text[..], &tokenizer, device)?,
            256_usize,
            context_size,
            None,
            None,
            Some(Tensor::new(&[50_256_u32], device)?),
            &mut rng,
        )?;
        let generated_text = token_ids_to_text(token_ids, &tokenizer)?;
        let mut response_text = generated_text[input_text.len()..].replace("### Response:", "");
        response_text = response_text.trim().to_string();

        // add model response
        entry.set_model_response(&response_text[..]);
    }

    // write to json
    println!(
        "Saving test data with model responses to {:?}",
        save_path.as_ref().to_str()
    );
    write_instruction_data_to_json(test_data, save_path)?;

    Ok(())
}

// Data types for convenience
// NOTE: you could also opt to just more simply use the RequestBuilder reqwest offers
/// OllamaRequestData type to represent payload for sending model query requests
#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct OllamaRequestData {
    model: String,
    messages: Vec<OllamaChatMessage>,
    options: Option<OllamaOptions>,
    stream: bool,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct OllamaChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct OllamaOptions {
    seed: u32,
    temperature: f32,
    num_ctx: u32,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct OllamaChatResponse {
    model: String,
    message: OllamaChatMessage,
}

pub const DEFAULT_OLLAMA_API_URL: &str = "http://127.0.0.1:11434/api/chat";

/// [Listing 7.10] Querying a local Ollama model
pub fn query_model(prompt: &str, model: &str, url: &str) -> anyhow::Result<String> {
    let request_data = OllamaRequestData {
        model: model.to_string(),
        messages: vec![OllamaChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        options: Some(OllamaOptions {
            seed: 123_u32,
            temperature: 0_f32,
            num_ctx: 2048_u32,
        }),
        stream: false,
    };
    let client = reqwest::blocking::Client::new();
    let res = client
        .post(url)
        .json(&request_data)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .send()?;

    if res.status().is_success() {
        let ollama_response = res.json::<OllamaChatResponse>()?;
        Ok(ollama_response.message.content)
    } else if res.status().is_server_error() {
        Err(anyhow::anyhow!("server error!"))
    } else {
        Err(anyhow::anyhow!(
            "Something else happened. Status: {:?}",
            res.status()
        ))
    }
}

/// [Listing 7.11] Evaluating the instruction fine-tuning LLM
pub fn generate_model_scores<P: PromptFormatter>(
    instruction_data: &[InstructionResponseExample],
    url: &str,
    model: &str,
    prompt_formatter: &P, // introduced for Exercise 7.1
) -> anyhow::Result<Vec<f32>> {
    let mut scores: Vec<f32> = vec![];

    for (ix, entry) in tqdm(instruction_data.iter().enumerate()) {
        let prompt = format!(
            "Given the input `{}` and the correct output `{}`, score the \
            model response `{}` on a scale from 0 to 100, where 100 is the ]
            best score. Respond with the integer number only.",
            prompt_formatter.format_input(entry),
            entry.output(),
            entry
                .model_response()
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Entry {ix} is missing a model response."))?
        );

        let score_str = query_model(prompt.as_str(), model, url)?;
        let score = score_str.parse::<f32>();
        match score {
            Ok(val) => scores.push(val),
            Err(_e) => println!("Unable to convert `score_str` to `u32`: {}.", score_str),
        }
    }

    Ok(scores)
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
            model_response: None,
        }
    }

    #[fixture]
    fn another_instruction_example() -> InstructionResponseExample {
        let instruction = "Here is yet another fake instruction.".to_string();
        let output = "here is yet another fake output.".to_string();
        InstructionResponseExample {
            instruction,
            input: None,
            output,
            model_response: None,
        }
    }

    #[fixture]
    fn instruction_data(
        instruction_example: InstructionResponseExample,
        another_instruction_example: InstructionResponseExample,
    ) -> Vec<InstructionResponseExample> {
        let data = vec![
            instruction_example.clone(),
            another_instruction_example.clone(),
            instruction_example.clone(),
            another_instruction_example.clone(),
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
    fn test_alpaca_formatter_with_no_input(
        mut instruction_example: InstructionResponseExample,
    ) -> Result<()> {
        instruction_example.input = None; // set input to None
        let prompt_formatter = AlpacaPromptFormatter;
        let prompt = prompt_formatter.format_input(&instruction_example);
        let expected_output = format!(
            "Below is an instruction that describes a task. Write a response that \
            appropriately completes the request.\n\n### Instruction:\n{}",
            instruction_example.instruction,
        );

        assert_eq!(prompt, expected_output);
        Ok(())
    }

    #[rstest]
    fn test_alpaca_formatter_with_some_input(
        instruction_example: InstructionResponseExample,
    ) -> Result<()> {
        let prompt_formatter = AlpacaPromptFormatter;
        let prompt = prompt_formatter.format_input(&instruction_example);
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
        let prompt_formatter = AlpacaPromptFormatter;
        let instruction_dataset =
            InstructionDataset::new(instruction_data, &tokenizer, &prompt_formatter);

        // test encoded
        let prompt = prompt_formatter.format_input(&instruction_example);
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
        let prompt_formatter = AlpacaPromptFormatter;
        let tokenizer = get_bpe_from_model("gpt2")?;
        let instruction_dataset =
            InstructionDataset::new(instruction_data, &tokenizer, &prompt_formatter);
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
    pub fn test_instruction_collator() -> Result<()> {
        // arrange
        let collator = InstructionDataCollator::new().device(Device::cuda_if_available(0)?);
        let device = Device::cuda_if_available(0)?;
        let inputs_1 = Tensor::new(&[1_u32, 2, 3], &device)?;
        let inputs_2 = Tensor::new(&[4_u32, 5, 6, 7], &device)?;
        let batch = vec![inputs_1, inputs_2];

        // act
        let (inputs, targets) = collator.collate(batch)?;

        // assert
        assert_eq!(inputs.dims(), targets.dims());
        assert_eq!(
            inputs.to_vec2::<u32>()?,
            &[[1_u32, 2, 3, 50256], [4_u32, 5, 6, 7]],
        );
        assert_eq!(
            targets.to_vec2::<i64>()?,
            &[[2_i64, 3, 50256, -100], [5_i64, 6, 7, 50256]]
        );

        Ok(())
    }

    #[rstest]
    pub fn test_instruction_batcher(
        instruction_data: Vec<InstructionResponseExample>,
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;
        let instruction_dataset =
            InstructionDataset::new(instruction_data, &tokenizer, &prompt_formatter);
        let iter = InstructionDatasetIter::new(instruction_dataset.clone(), false);
        let batch_size = 2_usize;
        let collator = InstructionDataCollator::new().device(Device::cuda_if_available(0)?);
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

    #[rstest]
    fn test_instruct_data_loader(instruction_data: Vec<InstructionResponseExample>) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;
        let instruction_dataset =
            InstructionDataset::new(instruction_data, &tokenizer, &prompt_formatter);
        let batch_size = 2_usize;
        let allowed_max_length = 10_usize;
        let collator = InstructionDataCollator::new()
            .device(Device::cuda_if_available(0)?)
            .allowed_max_length(Some(allowed_max_length));
        let shuffle = false;
        let drop_last = false;
        let data_loader = InstructionDataLoader::new(
            instruction_dataset,
            batch_size,
            shuffle,
            drop_last,
            collator,
        );

        let mut batcher = data_loader.batcher();
        let mut count = 0_usize;
        while let Some(Ok((inputs, targets))) = batcher.next() {
            assert!(inputs.dims()[0] <= batch_size);
            assert!(targets.dims()[0] <= batch_size);
            assert_eq!(inputs.dims()[1], allowed_max_length);
            assert_eq!(targets.dims()[1], allowed_max_length);
            count += 1;
        }
        assert_eq!(data_loader.len(), count);
        assert!(!data_loader.is_empty());
        Ok(())
    }

    #[rstest]
    fn test_write_instruction_data_to_json(
        instruction_data: Vec<InstructionResponseExample>,
    ) -> Result<()> {
        let test_file = NamedTempFile::new().unwrap();
        let save_path = test_file.into_temp_path().keep().unwrap();

        // write data
        write_instruction_data_to_json(&instruction_data, save_path.clone())?;

        // load data
        let json_str = read_to_string(AsRef::<Path>::as_ref(&save_path))?;
        let reloaded_data: Vec<InstructionResponseExample> = serde_json::from_str(&json_str[..])?;

        assert_eq!(instruction_data, reloaded_data);
        Ok(())
    }

    #[rstest]
    fn test_phi3_formatter_with_no_input(
        mut instruction_example: InstructionResponseExample,
    ) -> Result<()> {
        instruction_example.input = None; // set input to None
        let prompt_formatter = Phi3PromptFormatter;
        let prompt = prompt_formatter.format_input(&instruction_example);
        let expected_output = "<|user|>\n\
        Here is a fake instruction.\n\n\
        <|assistant|>\n\
        here is a fake output.";

        assert_eq!(prompt, expected_output);
        Ok(())
    }

    #[rstest]
    fn test_phi3_formatter_with_some_input(
        instruction_example: InstructionResponseExample,
    ) -> Result<()> {
        let prompt_formatter = Phi3PromptFormatter;
        let prompt = prompt_formatter.format_input(&instruction_example);
        let expected_output = "<|user|>\n\
        Here is a fake instruction.\n\
        Here is a fake input.\n\n\
        <|assistant|>\n\
        here is a fake output.";

        assert_eq!(prompt, expected_output);
        Ok(())
    }
}
