//! Bonus material module for Chapter 7

use super::{InstructionResponseExample, PromptFormatter};

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
    instruction_data: &[InstructionResponseExample],
    url: &str,
    model: &str,
    prompt_formatter: &P,
) -> anyhow::Result<PreferenceExample> {
    todo!()
}
