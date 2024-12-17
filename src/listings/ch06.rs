//! Listings from Chapter 6

use candle_core::Result;
use std::fs::File;
use std::io;

pub const URL: &str = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip";
pub const ZIP_PATH: &str = "data/sms_spam_collection.zip";
pub const EXTRACTED_PATH: &str = "sms_spam_collection";
pub const DATA_FILE_PATH: &str = "sms_spam_collection/SMSSPamCollection.tsv";

/// [Listing 6.1] Downloading and unzipping the dataset
#[allow(unused_variables)]
pub fn download_and_unzip_spam_data(
    url: &str,
    zip_path: &str,
    extracted_path: &str,
    datafile_path: &str,
) -> Result<()> {
    let resp = reqwest::blocking::get(url).map_err(candle_core::Error::wrap)?;
    let body = resp.text().map_err(candle_core::Error::wrap)?;
    let mut out = File::create(zip_path)?;
    io::copy(&mut body.as_bytes(), &mut out).map_err(candle_core::Error::wrap)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use rstest::*;

    #[rstest]
    fn test_download_and_unzip_spam_data() -> Result<()> {
        download_and_unzip_spam_data(&URL, &ZIP_PATH, &EXTRACTED_PATH, &DATA_FILE_PATH)?;
        Ok(())
    }
}
