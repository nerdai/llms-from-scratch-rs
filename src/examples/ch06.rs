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
        use polars::prelude::*;
        use std::sync::Arc;

        // download sms spam .tsv file
        download_and_unzip_spam_data(URL, ZIP_PATH, EXTRACTED_PATH)?;

        // load in .tsv as a DataFrame
        let f1 = Field::new("Label".into(), DataType::String);
        let f2 = Field::new("Text".into(), DataType::String);
        let sc = Arc::new(Schema::from_iter(vec![f1, f2]));
        let parse_options = CsvParseOptions::default()
            .with_separator(b'\t')
            .with_quote_char(None);
        let df = CsvReadOptions::default()
            .with_parse_options(parse_options)
            .with_schema(Some(sc))
            .with_has_header(false)
            .try_into_reader_with_file_path(Some("data/SMSSpamCollection.tsv".into()))
            .unwrap()
            .finish()?;
        println!("{}", df);

        // get value counts for label
        let value_counts = addons::get_value_counts(&df, "Label")?;
        println!("{}", value_counts);

        Ok(())
    }
}

/// # Example usage of `download_smsspam_parquet`
///
/// #### Id
/// 06.02
///
/// #### Page
/// This example starts on page 173
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 06.02
///
/// # with cuda
/// cargo run --features cuda example 06.02
/// ```
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        String::from("Sample usage of `download_smsspam_parquet`")
    }

    fn page_source(&self) -> usize {
        173_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch06::{download_smsspam_parquet, PARQUET_FILENAME, PARQUET_URL};
        use polars::prelude::*;
        use std::path::PathBuf;

        // download parquet file
        download_smsspam_parquet(PARQUET_URL)?;

        // load parquet
        let mut file_path = PathBuf::from("data");
        file_path.push(PARQUET_FILENAME);
        let mut file = std::fs::File::open(file_path)?;
        let df = ParquetReader::new(&mut file).finish()?;
        let df = df
            .clone()
            .lazy()
            .with_column(
                when(col("label").eq(0))
                    .then(lit("ham"))
                    .otherwise(lit("spam"))
                    .alias("label_text"),
            )
            .collect()?;
        println!("{}", df);

        // get value counts for label
        let value_counts = addons::get_value_counts(&df, "label_text")?;
        println!("{}", value_counts);

        Ok(())
    }
}

/// # Example usage of `create_balanced_dataset`
///
/// #### Id
/// 06.03
///
/// #### Page
/// This example starts on page 174
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 06.03
///
/// # with cuda
/// cargo run --features cuda example 06.03
/// ```
pub struct EG03;

impl Example for EG03 {
    fn description(&self) -> String {
        String::from("Example usage of `create_balanced_dataset`")
    }

    fn page_source(&self) -> usize {
        174_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch06::{
            create_balanced_dataset, download_smsspam_parquet, PARQUET_FILENAME, PARQUET_URL,
        };
        use polars::prelude::*;
        use std::path::PathBuf;

        // download parquet
        download_smsspam_parquet(PARQUET_URL)?;

        // load parquet
        let mut file_path = PathBuf::from("data");
        file_path.push(PARQUET_FILENAME);
        let mut file = std::fs::File::open(file_path).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();

        // balance dataset
        let balanced_df = create_balanced_dataset(df)?;
        println!("{}", balanced_df);

        // get value counts for label
        let value_counts = addons::get_value_counts(&balanced_df, "label")?;
        println!("{}", value_counts);

        Ok(())
    }
}

/// # Example usage of `random_split`
///
/// #### Id
/// 06.04
///
/// #### Page
/// This example starts on page 175
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 06.04
///
/// # with cuda
/// cargo run --features cuda example 06.04
/// ```
pub struct EG04;

impl Example for EG04 {
    fn description(&self) -> String {
        String::from("Example usage of `random_split` to create our train, test, val splits")
    }

    fn page_source(&self) -> usize {
        174_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch06::{
            create_balanced_dataset, download_smsspam_parquet, random_split, PARQUET_FILENAME,
            PARQUET_URL,
        };
        use polars::prelude::*;
        use std::{path::PathBuf, str::FromStr};

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
        println!("{}", train_df);
        println!("{}", validation_df);
        println!("{}", test_df);

        // save dfs to csv
        let train_path = PathBuf::from_str("data/train.parquet")?;
        let validation_path = PathBuf::from_str("data/validation.parquet")?;
        let test_path = PathBuf::from_str("data/test.parquet")?;

        addons::write_parquet(&mut train_df, train_path)?;
        addons::write_parquet(&mut validation_df, validation_path)?;
        addons::write_parquet(&mut test_df, test_path)?;

        Ok(())
    }
}

/// # Creating `SpamDataset` for train, test, and validation
///
/// #### Id
/// 06.05
///
/// #### Page
/// This example starts on page 178
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 06.05
///
/// # with cuda
/// cargo run --features cuda example 06.05
/// ```
pub struct EG05;

impl Example for EG05 {
    fn description(&self) -> String {
        String::from("Creating `SpamDataset` for train, test, and validation")
    }

    fn page_source(&self) -> usize {
        178_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch06::{SpamDataset, PAD_TOKEN_ID};
        use anyhow::anyhow;
        use std::ops::Not;
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2")?;

        let train_path = Path::new("data").join("train.parquet");
        if train_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/train.parquet' file. Please run EG 06.04."
            ));
        }
        let train_dataset = SpamDataset::new(train_path, &tokenizer, None, PAD_TOKEN_ID);
        println!("train dataset max length: {}", train_dataset.max_length());

        let val_path = Path::new("data").join("validation.parquet");
        if val_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/validation.parquet' file. Please run EG 06.04."
            ));
        }
        let val_dataset = SpamDataset::new(
            val_path,
            &tokenizer,
            Some(train_dataset.max_length()),
            PAD_TOKEN_ID,
        );
        println!("val dataset max length: {}", val_dataset.max_length());

        let test_path = Path::new("data").join("test.parquet");
        if test_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/test.parquet' file. Please run EG 06.04."
            ));
        }
        let test_dataset = SpamDataset::new(
            test_path,
            &tokenizer,
            Some(train_dataset.max_length()),
            PAD_TOKEN_ID,
        );
        println!("test dataset max length: {}", test_dataset.max_length());
        Ok(())
    }
}

pub mod addons {
    //! Auxiliary module for examples::ch06
    use polars::prelude::*;
    use std::path::Path;

    /// Helper function to get value counts for a polars::DataFrame for a specified column
    pub fn get_value_counts(df: &DataFrame, cname: &str) -> anyhow::Result<DataFrame> {
        let result = df
            .clone()
            .lazy()
            .select([col(cname)
                .value_counts(false, false, "count", false)
                .alias("value_counts")])
            .collect()?;
        Ok(result)
    }

    pub fn write_csv<P: AsRef<Path>>(df: &mut DataFrame, fname: P) -> anyhow::Result<()> {
        let mut file = std::fs::File::create(fname)?;
        CsvWriter::new(&mut file).finish(df)?;
        Ok(())
    }

    pub fn write_parquet<P: AsRef<Path>>(df: &mut DataFrame, fname: P) -> anyhow::Result<()> {
        let mut file = std::fs::File::create(fname)?;
        ParquetWriter::new(&mut file).finish(df)?;
        Ok(())
    }
}
