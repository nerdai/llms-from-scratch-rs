//! Listings from Chapter 6

use candle_core::Result;

pub const URL: &'static str =
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip";
pub const ZIP_PATH: &'static str = "sms_spam_collection.zip";
pub const EXTRACTED_PATH: &'static str = "sms_spam_collection";
pub const DATA_FILE_PATH: &'static str = "sms_spam_collection/SMSSPamCollection.tsv";

/// [Listing 6.1] Downloading and unzipping the dataset
pub fn download_and_unzip_spam_data(
    url: &str,
    zip_path: &str,
    extracted_path: &str,
    datafile_path: &str,
) -> Result<()> {
    todo!()
}
