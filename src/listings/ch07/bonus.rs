//! Bonus material module for Chapter 7

use super::{
    query_model, write_instruction_data_to_json, InstructionExample, InstructionResponseExample,
    PromptFormatter,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, NoneAsEmptyString};
use std::{path::Path, rc::Rc};
use tiktoken_rs::CoreBPE;
use tqdm::tqdm;

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

#[cfg(test)]
mod tests {
    use crate::listings::ch07::AlpacaPromptFormatter;

    use super::*;
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
    pub fn test_instruction_dataset_init(
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
}
