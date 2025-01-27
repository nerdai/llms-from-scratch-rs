//! Bonus material module for Chapter 7

use super::{query_model, InstructionResponseExample, PromptFormatter};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub struct PreferenceExample {
    instruction: String,
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
    pub fn set_rejected(&mut self, rejected: &str) {
        self.rejected = rejected.to_string();
    }

    pub fn set_chosen(&mut self, chosen: &str) {
        self.chosen = chosen.to_string()
    }
}

/// Using Ollama to generate a `chosen` and `rejected` responses for an instruction entry
#[allow(unused_variables)]
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

#[cfg(test)]
mod tests {
    use crate::listings::ch07::AlpacaPromptFormatter;

    use super::*;
    use anyhow::Result;
    use rstest::*;

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
}
