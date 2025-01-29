//! Bonus material module for Chapter 7

use super::{
    query_model, write_instruction_data_to_json, InstructionExample, InstructionResponseExample,
    PromptFormatter,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, NoneAsEmptyString};
use std::path::Path;
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
) -> anyhow::Result<PreferenceExample> {
    let mut rng = StdRng::seed_from_u64(69420);
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
    for entry in tqdm(instruction_data.iter()) {
        let preference_example =
            generate_chosen_and_rejected_response(entry, url, model, prompt_formatter)?;
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
#[derive(Clone)]
#[allow(dead_code)]
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
    encoded_texts: Vec<Vec<u32>>,
}

/// Implementing a `PreferenceDataset`
#[derive(Clone)]
pub struct PreferenceDataset;

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
        let expected_encoded_prompt = tokenizer.encode_with_special_tokens(&prompt);

        assert_eq!(encoded.prompt, expected_encoded_prompt);

        Ok(())
    }
}
