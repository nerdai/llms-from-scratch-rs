//! Listings from Chapter 7

use anyhow::Context;
use bytes::Bytes;
use std::{
    fmt::Display,
    fs::{read_to_string, File},
    io,
    path::Path,
};

#[allow(dead_code)]
const INSTRUCTION_DATA_FILENAME: &str = "instruction_data.json";
#[allow(dead_code)]
const DATA_DIR: &str = "data";

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
    let file_pathbuf = file_path.as_ref().to_path_buf();
    if !file_pathbuf.exists() {
        // download json file
        let resp = reqwest::blocking::get(url)?;
        let content: Bytes = resp.bytes()?;
        let mut out = File::create(file_pathbuf)?;
        io::copy(&mut content.as_ref(), &mut out)?;
    } else {
        let json_str = read_to_string(file_pathbuf)
            .with_context(|| format!("Unable to read {}", file_path.as_ref().display()))?;
    }
    Ok(vec![])
}
