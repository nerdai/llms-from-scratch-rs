//! Bonus material module for Chapter 7

use super::{
    query_model, write_instruction_data_to_json, InstructionExample, InstructionResponseExample,
    PromptFormatter, DEFAULT_PAD_TOKEN_ID, GPT,
};
use candle_core::{Device, IndexOp, ModuleT, Result, Tensor, D};
use rand::{rngs::StdRng, seq::SliceRandom, thread_rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, NoneAsEmptyString};
use std::{path::Path, rc::Rc};
use tiktoken_rs::CoreBPE;
use tqdm::tqdm;

// for convenience we also re-export the following
pub use crate::listings::ch02::DataLoader;
pub use crate::listings::ch05::calc_loss_loader;

#[serde_as]
#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct PreferenceExample {
    instruction: String,
    #[serde_as(as = "NoneAsEmptyString")]
    input: Option<String>,
    output: String,
    rejected: String,
    chosen: String,
}

impl From<InstructionResponseExample> for PreferenceExample {
    fn from(value: InstructionResponseExample) -> Self {
        Self {
            instruction: value.instruction().to_owned(),
            input: value.input().to_owned(),
            output: value.output().to_owned(),
            ..Default::default()
        }
    }
}

impl PreferenceExample {
    fn rejected(&self) -> &String {
        &self.rejected
    }

    fn chosen(&self) -> &String {
        &self.chosen
    }

    pub fn set_rejected(&mut self, rejected: &str) {
        self.rejected = rejected.to_string();
    }

    pub fn set_chosen(&mut self, chosen: &str) {
        self.chosen = chosen.to_string()
    }
}

impl InstructionExample for PreferenceExample {
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

/// Using Ollama to generate a `chosen` and `rejected` responses for an instruction entry
pub fn generate_chosen_and_rejected_response<P: PromptFormatter>(
    entry: &InstructionResponseExample,
    url: &str,
    model: &str,
    prompt_formatter: &P,
    rng: &mut StdRng,
) -> anyhow::Result<PreferenceExample> {
    let u: f32 = rng.gen_range(0.0..1.0);
    let politeness = if u < 0.5 { "polite" } else { "impolite" };

    let prompt = format!(
        "Given the input `{}` and correct output `{}`, \
        slightly rewrite the output to be more {}
        Keep the modification minimal.
        Only return return the generated response and nothing else.",
        prompt_formatter.format_input(entry),
        entry.output(),
        politeness
    );

    let response = query_model(prompt.as_str(), model, url)?;
    let mut preference_example = PreferenceExample::from(entry.clone());

    if politeness == "polite" {
        preference_example.set_chosen(response.as_str());
        preference_example.set_rejected(entry.output().as_str());
    } else {
        preference_example.set_chosen(entry.output().as_str());
        preference_example.set_rejected(response.as_str());
    }

    Ok(preference_example)
}

/// Create a preference dataset from an instruction dataset and Ollama
pub fn generate_preference_dataset<P: PromptFormatter, T: AsRef<Path>>(
    instruction_data: &[InstructionResponseExample],
    url: &str,
    model: &str,
    prompt_formatter: &P,
    save_path: T,
) -> anyhow::Result<()> {
    let mut dataset = vec![];
    let mut rng = StdRng::seed_from_u64(42_u64);
    for entry in tqdm(instruction_data.iter()) {
        let preference_example =
            generate_chosen_and_rejected_response(entry, url, model, prompt_formatter, &mut rng)?;
        dataset.push(preference_example);
    }

    // write to json
    println!(
        "Saving preference data to {:?}",
        save_path.as_ref().to_str()
    );
    write_instruction_data_to_json(&dataset, save_path)?;

    Ok(())
}
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedPreferenceExample {
    prompt: Vec<u32>,
    chosen: Vec<u32>,
    rejected: Vec<u32>,
}

impl EncodedPreferenceExample {
    #[allow(unused_variables)]
    pub fn from_example<P: PromptFormatter>(
        example: &PreferenceExample,
        prompt_formatter: &P,
        tokenizer: &CoreBPE,
    ) -> Self {
        let prompt = prompt_formatter.format_input(example);
        let rejected_response = example.rejected();
        let chosen_response = example.chosen();

        let prompt_tokens = tokenizer.encode_with_special_tokens(&prompt);
        let chosen_full_text = format!("{prompt}\n\n### Response:\n{chosen_response}");
        let rejected_full_text = format!("{prompt}\n\n### Response:\n{rejected_response}");
        let chosen_full_tokens = tokenizer.encode_with_special_tokens(&chosen_full_text);
        let rejected_full_tokens = tokenizer.encode_with_special_tokens(&rejected_full_text);

        Self {
            prompt: prompt_tokens,
            chosen: chosen_full_tokens,
            rejected: rejected_full_tokens,
        }
    }
}

#[allow(dead_code)]
pub struct PreferenceDataset_ {
    data: Vec<PreferenceExample>,
    encoded_texts: Vec<EncodedPreferenceExample>,
}

/// Implementing a `PreferenceDataset`
#[derive(Clone)]
pub struct PreferenceDataset(Rc<PreferenceDataset_>);

impl AsRef<PreferenceDataset> for PreferenceDataset {
    fn as_ref(&self) -> &PreferenceDataset {
        self
    }
}

impl std::ops::Deref for PreferenceDataset {
    type Target = PreferenceDataset_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl PreferenceDataset {
    pub fn new<P: PromptFormatter>(
        data: Vec<PreferenceExample>,
        tokenizer: &CoreBPE,
        prompt_formatter: &P,
    ) -> Self {
        let mut encoded_examples = vec![];
        for example in data.iter() {
            let encoded_example =
                EncodedPreferenceExample::from_example(example, prompt_formatter, tokenizer);
            encoded_examples.push(encoded_example);
        }

        let dataset_ = PreferenceDataset_ {
            data,
            encoded_texts: encoded_examples,
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
    pub fn get_item_at_index(&self, idx: usize) -> anyhow::Result<&EncodedPreferenceExample> {
        let encoded = &self.encoded_texts[idx];
        Ok(encoded)
    }

    pub fn data(&self) -> &Vec<PreferenceExample> {
        &self.data
    }
}

pub struct PreferenceDatasetIter {
    dataset: PreferenceDataset,
    remaining_indices: Vec<usize>,
}

impl PreferenceDatasetIter {
    pub fn new(dataset: PreferenceDataset, shuffle: bool) -> Self {
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

impl Iterator for PreferenceDatasetIter {
    type Item = Result<EncodedPreferenceExample>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.remaining_indices.pop() {
            let encoded = self.dataset.get_item_at_index(idx).unwrap();

            Some(Ok(encoded.clone()))
        } else {
            None
        }
    }
}

pub struct IterResult1<I: Iterator<Item = Result<EncodedPreferenceExample>>> {
    inner: I,
}

#[derive(Debug, Clone)]
pub struct PreferenceDatasetCollatorItem {
    prompt: Vec<Tensor>,
    chosen: Tensor,
    rejected: Tensor,
    rejected_mask: Vec<Tensor>,
    chosen_mask: Tensor,
}

impl PreferenceDatasetCollatorItem {
    pub fn prompt(&self) -> &Vec<Tensor> {
        &self.prompt
    }

    pub fn chosen(&self) -> &Tensor {
        &self.chosen
    }

    pub fn chosen_mask(&self) -> &Tensor {
        &self.chosen_mask
    }

    pub fn rejected(&self) -> &Tensor {
        &self.rejected
    }

    pub fn rejected_mask(&self) -> &Vec<Tensor> {
        &self.rejected_mask
    }
}

pub trait CustomCollator {
    type BatchItem;

    fn collate(&self, batch: Vec<Self::BatchItem>) -> Result<PreferenceDatasetCollatorItem>;
}

#[allow(dead_code)]
pub struct PreferenceDataBatcher<C: CustomCollator, I> {
    inner: I,
    batch_size: usize,
    return_last_incomplete_batch: bool,
    collator: C,
}

impl<C, I> PreferenceDataBatcher<C, IterResult1<I>>
where
    C: CustomCollator<BatchItem = EncodedPreferenceExample>,
    I: Iterator<Item = Result<EncodedPreferenceExample>>,
{
    pub fn new(inner: I, collator: C) -> Self {
        Self {
            inner: IterResult1 { inner },
            collator,
            batch_size: 16,
            return_last_incomplete_batch: false,
        }
    }
}

impl<C: CustomCollator, I> PreferenceDataBatcher<C, I> {
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

impl<C, I> Iterator for PreferenceDataBatcher<C, IterResult1<I>>
where
    // These items need to match
    C: CustomCollator<BatchItem = EncodedPreferenceExample>,
    I: Iterator<Item = Result<EncodedPreferenceExample>>,
{
    type Item = Result<PreferenceDatasetCollatorItem>;

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

#[derive(Clone)]
pub struct PreferenceDataCollator {
    pad_token_id: u32,
    allowed_max_length: Option<usize>,
    mask_prompt_tokens: bool,
    device: Device,
}

impl Default for PreferenceDataCollator {
    fn default() -> Self {
        Self {
            pad_token_id: DEFAULT_PAD_TOKEN_ID,
            mask_prompt_tokens: true,
            allowed_max_length: Some(1024_usize),
            device: Device::Cpu,
        }
    }
}

impl PreferenceDataCollator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pad_token_id(mut self, pad_token_id: u32) -> Self {
        self.pad_token_id = pad_token_id;
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

    fn _apply_padding_and_get_mask(
        &self,
        elements: &mut Vec<u32>,
        batch_max_length: usize,
        prompt_length: usize,
    ) -> Vec<u32> {
        let elements_length = elements.len();
        // apply padding
        let num_pad =
            std::cmp::max(0isize, batch_max_length as isize - elements_length as isize) as usize;
        if num_pad > 0 {
            let padding_input = std::iter::repeat(self.pad_token_id)
                .take(num_pad)
                .collect::<Vec<u32>>();
            elements.extend(padding_input);
        }

        // mask vec
        let mut mask = (0..batch_max_length as u32)
            .map(|j| u32::from(j < elements_length as u32))
            .collect::<Vec<u32>>();

        if self.mask_prompt_tokens {
            mask[..prompt_length + 2].fill(0_u32);
        }

        mask
    }

    fn _build_stacked_tensor(&self, elements_vec: Vec<Vec<u32>>) -> Result<Tensor> {
        let shape = (elements_vec.len(), elements_vec[0].len());
        Tensor::from_vec(
            elements_vec.into_iter().flatten().collect(),
            shape,
            &self.device,
        )
    }

    fn _build_tensor_from_only_true_values(&self, mask: Vec<u32>) -> Result<Tensor> {
        let reduced_mask = mask
            .iter()
            .enumerate()
            .filter_map(|(ix, el)| (*el > 0).then_some(ix as u32))
            .collect::<Vec<_>>();
        let shape = reduced_mask.len();
        Tensor::from_vec(reduced_mask, shape, &self.device)
    }

    pub fn custom_collate_fn(
        &self,
        batch: Vec<EncodedPreferenceExample>,
    ) -> Result<PreferenceDatasetCollatorItem> {
        let mut prompt_vec = vec![];
        let mut chosen_vec = vec![];
        let mut rejected_vec = vec![];
        let mut rejected_mask_vec = vec![];
        let mut chosen_mask_vec = vec![];

        let batch_max_length = batch
            .iter()
            .map(|el| std::cmp::max(el.chosen.len(), el.rejected.len()))
            .collect::<Vec<_>>()
            .into_iter()
            .max()
            .ok_or_else(|| {
                candle_core::Error::Msg("Unable to get max length for batch.".to_string())
            })?;

        for item in batch.into_iter() {
            let prompt = item.prompt.clone();
            let prompt_tensor = Tensor::from_vec(prompt, item.prompt.len(), &self.device)?;

            let mut chosen = item.chosen.clone();
            let mut chosen_mask =
                self._apply_padding_and_get_mask(&mut chosen, batch_max_length, item.prompt.len());

            let mut rejected = item.rejected.clone();
            let mut rejected_mask = self._apply_padding_and_get_mask(
                &mut rejected,
                batch_max_length,
                item.prompt.len(),
            );

            if let Some(a) = self.allowed_max_length {
                chosen = chosen[..std::cmp::min(a, batch_max_length)].to_vec();
                chosen_mask = chosen_mask[..std::cmp::min(a, batch_max_length)].to_vec();
                rejected = rejected[..std::cmp::min(a, batch_max_length)].to_vec();
                rejected_mask = rejected_mask[..std::cmp::min(a, batch_max_length)].to_vec();
            }

            // get only the indexes of mask that are to keep
            // let chosen_mask_tensor = self._build_tensor_from_only_true_values(chosen_mask)?;
            let rejected_mask_tensor = self._build_tensor_from_only_true_values(rejected_mask)?;

            chosen_vec.push(chosen);
            rejected_vec.push(rejected);
            rejected_mask_vec.push(rejected_mask_tensor);
            chosen_mask_vec.push(chosen_mask);
            prompt_vec.push(prompt_tensor);
        }

        let chosen_tensor = self._build_stacked_tensor(chosen_vec)?;
        let chosen_mask_tensor = self._build_stacked_tensor(chosen_mask_vec)?;
        let rejected_tensor = self._build_stacked_tensor(rejected_vec)?;

        Ok(PreferenceDatasetCollatorItem {
            prompt: prompt_vec,
            chosen: chosen_tensor,
            rejected: rejected_tensor,
            rejected_mask: rejected_mask_vec,
            chosen_mask: chosen_mask_tensor,
        })
    }
}

impl CustomCollator for PreferenceDataCollator {
    type BatchItem = EncodedPreferenceExample;

    fn collate(&self, batch: Vec<Self::BatchItem>) -> Result<PreferenceDatasetCollatorItem> {
        self.custom_collate_fn(batch)
    }
}

pub struct PreferenceDataLoader<C: CustomCollator<BatchItem = EncodedPreferenceExample>> {
    dataset: PreferenceDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    collator: C,
}

impl<C: CustomCollator<BatchItem = EncodedPreferenceExample> + Clone> DataLoader
    for PreferenceDataLoader<C>
{
    type Batcher = PreferenceDataBatcher<C, IterResult1<PreferenceDatasetIter>>;

    /// Returns a `PreferenceDataBatcher` that itself provides batches over the
    /// associated dataset.
    fn batcher(&self) -> PreferenceDataBatcher<C, IterResult1<PreferenceDatasetIter>> {
        let iter = PreferenceDatasetIter::new(self.dataset.clone(), self.shuffle);
        PreferenceDataBatcher::new(iter, self.collator.clone())
            .batch_size(self.batch_size)
            .return_last_incomplete_batch(!self.drop_last)
    }
}

impl<C: CustomCollator<BatchItem = EncodedPreferenceExample> + Clone> PreferenceDataLoader<C> {
    pub fn new(
        dataset: PreferenceDataset,
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

    pub fn dataset(&self) -> &PreferenceDataset {
        &self.dataset
    }
}

pub fn compute_dpo_loss(
    model_chosen_logprobs: &Tensor,
    model_rejected_logprobs: &Tensor,
    reference_chosen_logprobs: &Tensor,
    reference_rejected_logprobs: &Tensor,
    beta: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let model_logratios = (model_chosen_logprobs - model_rejected_logprobs)?;
    let reference_logratios = (reference_chosen_logprobs - reference_rejected_logprobs)?;
    let logits = (model_logratios - reference_logratios)?;

    let mut losses = candle_nn::ops::sigmoid(&(beta * &logits)?)?;
    losses = (-1_f64 * losses.log()?)?;

    // Optional values to track progress during training
    let chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs)?.detach();
    let rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs)?.detach();

    Ok((
        losses.mean_all()?,
        chosen_rewards.mean_all()?,
        rejected_rewards.mean_all()?,
    ))
}

pub fn compute_logprobs(
    logits: &Tensor,
    labels: &Tensor,
    selection_mask: Option<&Tensor>,
) -> Result<Tensor> {
    // Labels are the inputs shifted by one
    let labels = labels.i((.., 1..))?;
    let labels_dims = labels.dims();

    // Truncate logits to match the labels num_tokens
    let logits = logits.i((.., ..labels_dims[1], ..))?;

    let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;

    let selected_log_probs = log_probs
        .gather(&labels.unsqueeze(D::Minus1)?, D::Minus1)?
        .squeeze(D::Minus1)?;

    if let Some(m) = selection_mask {
        // this is brittle, if there is a value < 0, then this breaks...
        let mask = (m - 1_f64)?;
        let selected_log_probs = selected_log_probs.gather(&mask, D::Minus1)?;

        let avg_log_prob = selected_log_probs
            .sum(D::Minus1)?
            .broadcast_div(&mask.sum(D::Minus1)?)?;
        Ok(avg_log_prob)
    } else {
        selected_log_probs.mean(D::Minus1)
    }
}

#[allow(unused_variables)]
pub fn compute_dpo_loss_batch<M: GPT + ModuleT>(
    batch: &PreferenceDatasetCollatorItem,
    policy_model: M,
    reference_model: M,
    beta: f64,
    train: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    // where policy_model(batch["chosen"]) are the logits
    // let policy_chosen_log_probas = compute_logprobs(
    //     &policy_model.forward_t(batch.chosen(), train)?,
    //     batch.chosen(),
    //     batch.chosen_mask(),
    // )?;

    // let policy_chosen_log_probas = compute_logprobs(
    //     &policy_model.forward_t(&batch.chosen)?,
    //     batch.chosen(),
    //     selection_mask=batch["chosen_mask"]
    // )?;

    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::listings::ch07::AlpacaPromptFormatter;
    use anyhow::Result;
    use rstest::*;
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
    fn preference_example() -> PreferenceExample {
        let instruction = "Here is a fake instruction.".to_string();
        let input = Some("Here is a fake input.".to_string());
        let output = "here is a fake output.".to_string();
        let chosen = "Here is a fake chosen.".to_string();
        let rejected = "Here is a fake rejected.".to_string();
        PreferenceExample {
            instruction,
            input,
            output,
            chosen,
            rejected,
        }
    }

    #[fixture]
    fn another_preference_example() -> PreferenceExample {
        let instruction = "Here is yet another fake instruction.".to_string();
        let output = "here is yet another fake output.".to_string();
        let chosen = "Here is yet another fake chosen.".to_string();
        let rejected = "Here is yet another fake rejected.".to_string();
        PreferenceExample {
            instruction,
            input: None,
            output,
            chosen,
            rejected,
        }
    }

    #[fixture]
    fn preference_data(
        preference_example: PreferenceExample,
        another_preference_example: PreferenceExample,
    ) -> Vec<PreferenceExample> {
        let data = vec![
            preference_example.clone(),
            another_preference_example.clone(),
            preference_example.clone(),
            another_preference_example.clone(),
            preference_example,
        ];
        data
    }

    #[rstest]
    fn test_prompt_for_rejection_chosen(
        instruction_example: InstructionResponseExample,
    ) -> Result<()> {
        let politeness = "polite";
        let prompt_formatter = AlpacaPromptFormatter;
        let prompt = format!(
            "Given the input `{}` and correct output `{}`, \
            slightly rewrite the output to be more {}. \
            Keep the modification minimal. \
            Only return return the generated response and nothing else.",
            prompt_formatter.format_input(&instruction_example),
            instruction_example.output(),
            politeness
        );

        let expected = "Given the input `Below is an instruction that \
        describes a task. Write a response that appropriately completes the \
        request.\n\n\
        ### Instruction:\n\
        Here is a fake instruction.\n\n\
        ### Input:\n\
        Here is a fake input.` and correct output `here is a fake output.`, \
        slightly rewrite the output to be more polite. Keep the modification \
        minimal. Only return return the generated response and nothing else.";

        assert_eq!(prompt, expected);
        Ok(())
    }

    #[rstest]
    fn test_encoded_preference_example_from_preference_example(
        preference_example: PreferenceExample,
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;
        let encoded = EncodedPreferenceExample::from_example(
            &preference_example,
            &prompt_formatter,
            &tokenizer,
        );

        let prompt = prompt_formatter.format_input(&preference_example);
        let formatted_rejection =
            format!("{prompt}\n\n### Response:\n{}", preference_example.rejected);
        let formatted_chosen = format!("{prompt}\n\n### Response:\n{}", preference_example.chosen);

        let expected_encoded_prompt = tokenizer.encode_with_special_tokens(&prompt);
        let expected_encoded_rejected = tokenizer.encode_with_special_tokens(&formatted_rejection);
        let expected_encoded_chosen = tokenizer.encode_with_special_tokens(&formatted_chosen);

        assert_eq!(encoded.prompt, expected_encoded_prompt);
        assert_eq!(encoded.rejected, expected_encoded_rejected);
        assert_eq!(encoded.chosen, expected_encoded_chosen);

        Ok(())
    }

    #[rstest]
    pub fn test_preference_dataset_init(
        preference_data: Vec<PreferenceExample>,
        preference_example: PreferenceExample,
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;
        let preference_dataset =
            PreferenceDataset::new(preference_data, &tokenizer, &prompt_formatter);

        // test encoded example
        let encoded_example = EncodedPreferenceExample::from_example(
            &preference_example,
            &prompt_formatter,
            &tokenizer,
        );

        assert_eq!(preference_dataset.len(), 5);
        assert_eq!(
            *preference_dataset.get_item_at_index(0_usize)?,
            encoded_example
        );

        Ok(())
    }

    #[rstest]
    pub fn test_preference_collator(preference_example: PreferenceExample) -> Result<()> {
        // arrange
        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;
        let encoded_example = EncodedPreferenceExample::from_example(
            &preference_example,
            &prompt_formatter,
            &tokenizer,
        );
        let batch = vec![encoded_example];
        let collator = PreferenceDataCollator::new().device(Device::cuda_if_available(0)?);

        // act
        let collated_item = collator.collate(batch)?;

        // assert
        assert_eq!(
            collated_item.chosen_mask().i((0, ..))?.to_vec1::<u32>()?,
            &[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ]
        );
        assert_eq!(
            collated_item.rejected_mask()[0].to_vec1::<u32>()?,
            &[44, 45, 46, 47, 48, 49, 50, 51, 52, 53,]
        );
        assert_eq!(
            collated_item.rejected.elem_count(),
            collated_item.chosen.elem_count()
        );
        assert_eq!(collated_item.prompt.len(), 1);

        Ok(())
    }

    #[rstest]
    fn test_preference_data_loader(preference_data: Vec<PreferenceExample>) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = AlpacaPromptFormatter;
        let preference_dataset =
            PreferenceDataset::new(preference_data, &tokenizer, &prompt_formatter);
        let batch_size = 2_usize;
        let allowed_max_length = 5_usize;
        let collator = PreferenceDataCollator::new()
            .device(Device::cuda_if_available(0)?)
            .allowed_max_length(Some(allowed_max_length));
        let shuffle = false;
        let drop_last = false;
        let data_loader =
            PreferenceDataLoader::new(preference_dataset, batch_size, shuffle, drop_last, collator);

        let mut batcher = data_loader.batcher();
        let mut count = 0_usize;
        while let Some(Ok(collated_item)) = batcher.next() {
            assert_eq!(collated_item.chosen.dims()[1], allowed_max_length);
            assert_eq!(collated_item.rejected.dims()[1], allowed_max_length);
            assert_eq!(collated_item.chosen_mask.dims()[1], allowed_max_length);
            assert!(collated_item.chosen.dims()[0] <= batch_size);
            assert!(collated_item.rejected.dims()[0] <= batch_size);
            assert!(collated_item.rejected_mask.len() <= batch_size);
            assert!(collated_item.prompt.len() <= batch_size);

            let max_length_rejected_mask = collated_item
                .rejected_mask
                .iter()
                .map(|el| el.elem_count())
                .collect::<Vec<_>>()
                .into_iter()
                .max()
                .unwrap();
            assert!(max_length_rejected_mask <= allowed_max_length);

            count += 1;
        }
        assert_eq!(data_loader.len(), count);
        assert!(!data_loader.is_empty());

        Ok(())
    }
}
