//! Examples from Chapter 7

use crate::Example;
use anyhow::Result;

/// # Example usage of `download_and_load_file`
///
/// #### Id
/// 07.01
///
/// #### Page
/// This example starts on page 207
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 07.01
///
/// # with cuda
/// cargo run --features cuda example 07.01
/// ```
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        "Example usage of `download_and_load_file`.".to_string()
    }

    fn page_source(&self) -> usize {
        207_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::{
            download_and_load_file, DATA_DIR, INSTRUCTION_DATA_FILENAME, INSTRUCTION_DATA_URL,
        };
        use std::path::Path;

        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;
        println!("Number of entries: {}", data.len());

        // See example at index 50
        println!("Example entry:\n{}\n", data[50]);

        // See another example at index 999
        println!("Another example entry:\n{}", data[999]);

        Ok(())
    }
}
