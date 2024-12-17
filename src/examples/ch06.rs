//! Examples from Chapter 6

use crate::Example;
use anyhow::Result;

/// # Example usage of `download_and_unzip_spam_data`
///
/// #### Id
/// 06.01
///
/// #### Page
/// This example starts on page 173
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 06.01
///
/// # with cuda
/// cargo run --features cuda example 06.01
/// ```
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Sample usage of `download_and_unzip_spam_data`")
    }

    fn page_source(&self) -> usize {
        173_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch06::{download_and_unzip_spam_data, EXTRACTED_PATH, URL, ZIP_PATH};
        download_and_unzip_spam_data(URL, ZIP_PATH, EXTRACTED_PATH)?;
        Ok(())
    }
}
