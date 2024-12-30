//! Listings from Chapter 7

use std::{fmt::Display, path::Path};

/// A type for containing an instruction-response pair
#[derive(Debug, Default)]
pub struct InstructionResponseExample {
    instruction: String,
    input: Option<String>,
    output: String,
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
) -> anyhow::Result<Vec<InstructionResponseExample>> {
    todo!()
}
