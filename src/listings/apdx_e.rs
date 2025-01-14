//! Listings from Appendix E

use crate::examples::ch06::addons::write_parquet;
use crate::listings::ch06::{
    create_balanced_dataset, download_smsspam_parquet, random_split, SpamDataset,
    SpamDatasetBuilder, PARQUET_FILENAME, PARQUET_URL,
};
use anyhow::anyhow;
use polars::prelude::*;
use std::{
    ops::Not,
    path::{Path, PathBuf},
    str::FromStr,
};
use tiktoken_rs::get_bpe_from_model;

/// [Listing E.1] Downloading and preparing the dataset
///
/// NOTE: This is merely EG 06.04
pub fn download_and_prepare_spam_dataset() -> anyhow::Result<()> {
    // download parquet
    download_smsspam_parquet(PARQUET_URL)?;

    // load parquet
    let mut file_path = PathBuf::from("data");
    file_path.push(PARQUET_FILENAME);
    let mut file = std::fs::File::open(file_path).unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();

    // balance dataset
    let balanced_df = create_balanced_dataset(df)?;

    // create train, test, val splits
    let (mut train_df, mut validation_df, mut test_df) =
        random_split(&balanced_df, 0.7_f32, 0.1_f32)?;

    // save dfs to csv
    let train_path = PathBuf::from_str("data/train.parquet")?;
    let validation_path = PathBuf::from_str("data/validation.parquet")?;
    let test_path = PathBuf::from_str("data/test.parquet")?;

    write_parquet(&mut train_df, train_path)?;
    write_parquet(&mut validation_df, validation_path)?;
    write_parquet(&mut test_df, test_path)?;

    Ok(())
}

/// [Listing E.2] Instantiating Candle Datasets
///
/// NOTE: This is merely EG 06.05
pub fn create_candle_datasets() -> anyhow::Result<(SpamDataset, SpamDataset, SpamDataset)> {
    let tokenizer = get_bpe_from_model("gpt2")?;

    let train_path = Path::new("data").join("train.parquet");
    if train_path.exists().not() {
        return Err(anyhow!(
            "Missing 'data/train.parquet' file. Please run `listings::apdx_e::download_and_prepare_spam_dataset()`"
        ));
    }
    let train_dataset = SpamDatasetBuilder::new(&tokenizer)
        .load_data_from_parquet(train_path)
        .build();

    let val_path = Path::new("data").join("validation.parquet");
    if val_path.exists().not() {
        return Err(anyhow!(
            "Missing 'data/validation.parquet' file. Please run `listings::apdx_e::download_and_prepare_spam_dataset()`"
        ));
    }
    let val_dataset = SpamDatasetBuilder::new(&tokenizer)
        .load_data_from_parquet(val_path)
        .max_length(Some(train_dataset.max_length()))
        .build();

    let test_path = Path::new("data").join("test.parquet");
    if test_path.exists().not() {
        return Err(anyhow!(
            "Missing 'data/test.parquet' file. Please run `listings::apdx_e::download_and_prepare_spam_dataset()`"
        ));
    }
    let test_dataset = SpamDatasetBuilder::new(&tokenizer)
        .load_data_from_parquet(test_path)
        .max_length(Some(train_dataset.max_length()))
        .build();

    Ok((train_dataset, val_dataset, test_dataset))
}
