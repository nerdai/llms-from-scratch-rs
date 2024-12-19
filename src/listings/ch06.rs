//! Listings from Chapter 6

use ::zip::ZipArchive;
use anyhow::Error;
use bytes::Bytes;
use polars::prelude::*;
use std::fs::{create_dir_all, remove_file, rename, File};
use std::io;
use std::path::{Path, PathBuf};

pub const URL: &str = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip";
pub const ZIP_PATH: &str = "data/sms_spam_collection.zip";
pub const EXTRACTED_PATH: &str = "sms_spam_collection";
pub const EXTRACTED_FILENAME: &str = "SMSSpamCollection";
pub const PARQUET_URL: &str = "https://huggingface.co/datasets/ucirvine/sms_spam/resolve/main/plain_text/train-00000-of-00001.parquet?download=true";
pub const PARQUET_FILENAME: &str = "train-00000-of-00001.parquet";

/// [Listing 6.1 (parquet)] Download spam parquet dataset from HuggingFace
pub fn download_smsspam_parquet(url: &str) -> anyhow::Result<()> {
    // download parquet file
    let resp = reqwest::blocking::get(url)?;
    let content: Bytes = resp.bytes()?;

    let mut out_path = PathBuf::from("data");
    out_path.push(PARQUET_FILENAME);
    let mut out = File::create(out_path)?;
    io::copy(&mut content.as_ref(), &mut out)?;
    Ok(())
}

/// [Listing 6.1] Downloading and unzipping the dataset
#[allow(unused_variables)]
pub fn download_and_unzip_spam_data(
    url: &str,
    zip_path: &str,
    extracted_path: &str,
) -> anyhow::Result<()> {
    // download zip file
    let resp = reqwest::blocking::get(url)?;
    let content: Bytes = resp.bytes()?;
    let mut out = File::create(zip_path)?;
    io::copy(&mut content.as_ref(), &mut out)?;

    // unzip file
    _unzip_file(zip_path)?;

    // rename file to add .tsv extension
    let mut original_file_path = PathBuf::from("data");
    original_file_path.push(EXTRACTED_FILENAME);
    let mut data_file_path: PathBuf = original_file_path.clone();
    data_file_path.set_extension("tsv");
    rename(original_file_path, data_file_path)?;

    // remove zip file and readme file
    let readme_file: PathBuf = ["data", "readme"].iter().collect();
    remove_file_if_exists(readme_file)?;
    remove_file_if_exists(ZIP_PATH)?;
    Ok(())
}

/// A wrapper for std::fs::remove_file that passes on any ErrorKind::NotFound
fn remove_file_if_exists<P: AsRef<Path>>(fname: P) -> anyhow::Result<()> {
    match remove_file(fname) {
        Ok(()) => Ok(()),
        Err(e) => {
            if e.kind() == io::ErrorKind::NotFound {
                Ok(())
            } else {
                Err(Error::from(e))
            }
        }
    }
}

/// Helper function to unzip file using `zip::ZipArchive`
///
/// NOTE: adapted from https://github.com/zip-rs/zip2/blob/master/examples/extract.rs
fn _unzip_file(filename: &str) -> anyhow::Result<()> {
    let file = File::open(filename)?;

    let mut archive = ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = match file.enclosed_name() {
            Some(path) => {
                let mut retval = PathBuf::from("data");
                retval.push(path);
                retval
            }
            None => continue,
        };

        {
            let comment = file.comment();
            if !comment.is_empty() {
                println!("File {i} comment: {comment}");
            }
        }

        if file.is_dir() {
            println!("File {} extracted to \"{}\"", i, outpath.display());
            create_dir_all(&outpath)?;
        } else {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(())
}

/// [Listing 6.2] Creating a balanced dataset
pub fn create_balanced_dataset(df: DataFrame) -> anyhow::Result<DataFrame> {
    // balance by undersampling
    let mask = df.column("label")?.i64()?.equal(1);
    let spam_subset = df.filter(&mask)?;
    let num_spam = spam_subset.shape().0;

    let mask = df.column("label")?.i64()?.equal(0);
    let ham_subset = df.filter(&mask)?;
    let n = Series::from_iter([num_spam as i32].iter());
    let undersampled_ham_subset = ham_subset.sample_n(&n, false, true, Some(1234_u64))?;

    let balanced_df = concat(
        [
            spam_subset.clone().lazy(),
            undersampled_ham_subset.clone().lazy(),
        ],
        UnionArgs::default(),
    )?
    .collect()?;

    Ok(balanced_df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use rstest::*;

    #[fixture]
    pub fn sms_spam_df() -> (DataFrame, usize) {
        let df = df!(
            "sms"=> &["sms1", "sms2", "sms3", "sms4", "sms5"],
            "label"=> &[0_i64, 0, 1, 1, 0],
        )
        .unwrap();
        (df, 2usize)
    }

    #[rstest]
    pub fn test_create_balanced_dataset(
        #[from(sms_spam_df)] (df, num_spam): (DataFrame, usize),
    ) -> Result<()> {
        let balanced_df = create_balanced_dataset(df)?;

        assert_eq!(balanced_df.shape(), (num_spam * 2_usize, 2));
        Ok(())
    }
}
