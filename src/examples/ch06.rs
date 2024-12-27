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

/// # Creating `SpamDataset` for train, test, and validation via `SpamDatasetBuilder`
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
        use crate::listings::ch06::SpamDatasetBuilder;
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
        let train_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_parquet(train_path)
            .build();
        println!("train dataset max length: {}", train_dataset.max_length());

        let val_path = Path::new("data").join("validation.parquet");
        if val_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/validation.parquet' file. Please run EG 06.04."
            ));
        }
        let val_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_parquet(val_path)
            .max_length(Some(train_dataset.max_length()))
            .build();
        println!("val dataset max length: {}", val_dataset.max_length());

        let test_path = Path::new("data").join("test.parquet");
        if test_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/test.parquet' file. Please run EG 06.04."
            ));
        }
        let test_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_parquet(test_path)
            .max_length(Some(train_dataset.max_length()))
            .build();
        println!("test dataset max length: {}", test_dataset.max_length());
        Ok(())
    }
}

/// # Creating a `SpamDataLoader` for each of the train, val and test datasets.
///
/// #### Id
/// 06.06
///
/// #### Page
/// This example starts on page 180
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 06.06
///
/// # with cuda
/// cargo run --features cuda example 06.06
/// ```
pub struct EG06;

impl Example for EG06 {
    fn description(&self) -> String {
        "Creating a `SpamDataLoader` for each of the train, val and test datasets.".to_string()
    }

    fn page_source(&self) -> usize {
        180_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch06::{SpamDataLoader, SpamDatasetBuilder};
        use anyhow::anyhow;
        use std::ops::Not;
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        // create datasets
        let tokenizer = get_bpe_from_model("gpt2")?;

        let train_path = Path::new("data").join("train.parquet");
        if train_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/train.parquet' file. Please run EG 06.04."
            ));
        }
        let train_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_parquet(train_path)
            .build();

        let val_path = Path::new("data").join("validation.parquet");
        if val_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/validation.parquet' file. Please run EG 06.04."
            ));
        }
        let val_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_parquet(val_path)
            .build();

        let test_path = Path::new("data").join("test.parquet");
        if test_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/test.parquet' file. Please run EG 06.04."
            ));
        }
        let test_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_parquet(test_path)
            .build();

        // create loaders
        let batch_size = 8_usize;
        let train_loader = SpamDataLoader::new(train_dataset, batch_size, true, true);
        let val_loader = SpamDataLoader::new(val_dataset, batch_size, false, false);
        let test_loader = SpamDataLoader::new(test_dataset, batch_size, false, false);

        // see last batch of train loader
        let (input_batch, target_batch) = train_loader.batcher().last().unwrap()?;
        println!("Input batch dimensions: {:?}", input_batch.shape());
        println!("Label batch dimensions: {:?}", target_batch.shape());

        // print total number of batches in each data loader
        println!("{:?} training batches", train_loader.len());
        println!("{:?} validation batches", val_loader.len());
        println!("{:?} test batches", test_loader.len());

        Ok(())
    }
}

/// # Example usage of `download_and_load_gpt2`.
///
/// #### Id
/// 06.07
///
/// #### Page
/// This example starts on page 182
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 06.07
///
/// # with cuda
/// cargo run --features cuda example 06.07
/// ```
pub struct EG07;

impl Example for EG07 {
    fn description(&self) -> String {
        String::from("Example usage of `download_and_load_gpt2`.")
    }

    fn page_source(&self) -> usize {
        182_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch05::{generate, text_to_token_ids, token_ids_to_text},
            ch06::{download_and_load_gpt2, HF_GPT2_MODEL_ID},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use rand::{rngs::StdRng, SeedableRng};
        use tiktoken_rs::get_bpe_from_model;

        // use `download_and_load_gpt2`
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;

        // sample setup and load tokenizer
        let tokenizer = get_bpe_from_model("gpt2")?;
        let mut rng = StdRng::seed_from_u64(42_u64);

        // generate next tokens with model
        let text_1 = "Every effort moves you";
        let token_ids = generate(
            &model,
            text_to_token_ids(text_1, &tokenizer, vb.device())?,
            15_usize,
            cfg.context_length,
            None,
            None,
            None,
            &mut rng,
        )?;

        // decode the token ids to print the output text
        println!(
            "Output text:\n{:?}",
            token_ids_to_text(token_ids, &tokenizer)
        );

        // test inherent classification abilities
        let text_2 = "Is the following text 'spam'? Answer with 'yes' or \
        'no': 'You are a winner you have been specially selected to receive $1000 \
        cash or a $2000 award.'";
        let token_ids = generate(
            &model,
            text_to_token_ids(text_2, &tokenizer, vb.device())?,
            23_usize,
            cfg.context_length,
            None,
            None,
            None,
            &mut rng,
        )?;

        // decode the token ids to print the classification
        println!(
            "Output text:\n{:?}",
            token_ids_to_text(token_ids, &tokenizer)
        );

        Ok(())
    }
}

/// # Printing the model variables via `varmap.data()`
///
/// #### Id
/// 06.08
///
/// #### Page
/// This example starts on page 185
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 06.08
///
/// # with cuda
/// cargo run --features cuda example 06.08
/// ```
pub struct EG08;

impl Example for EG08 {
    fn description(&self) -> String {
        String::from("Printing the model architecture via `varmap.data()`")
    }

    fn page_source(&self) -> usize {
        185_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch06::{download_and_load_gpt2, HF_GPT2_MODEL_ID},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use itertools::Itertools;

        // use `download_and_load_gpt2`
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let _model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;

        // print model architecture
        let model_vars = varmap.data().lock().unwrap();
        for name in model_vars.keys().sorted() {
            let var = model_vars.get(name).unwrap();
            println!("{}: {:?}", name, var);
        }

        Ok(())
    }
}

/// # Modifying the `out_head` of a GPT2Model and running inference
///
/// #### Id
/// 06.09
///
/// #### Page
/// This example starts on page 186
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 06.09
///
/// # with cuda
/// cargo run --features cuda example 06.09
/// ```
pub struct EG09;

impl Example for EG09 {
    fn description(&self) -> String {
        String::from("Modifying the `out_head` of a GPT2Model and running inference")
    }

    fn page_source(&self) -> usize {
        186_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch06::{download_and_load_gpt2, modify_out_head_for_classification, HF_GPT2_MODEL_ID},
        };
        use candle_core::{DType, Device, IndexOp, ModuleT, Tensor};
        use candle_nn::{VarBuilder, VarMap};
        use tiktoken_rs::get_bpe_from_model;

        // use `download_and_load_gpt2`
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;

        // print old head
        let tensor_data = varmap.data().lock().unwrap();
        let out_head = tensor_data.get("model.out_head.weight");
        println!("old classification head: {:?}", out_head);
        drop(tensor_data);

        // modify classification head
        let num_classes = 2_usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        // print model architecture
        let tensor_data = varmap.data().lock().unwrap();
        let out_head = tensor_data.get("model.out_head.weight");
        println!("new classification head: {:?}", out_head);

        // run sample inference
        let tokenizer = get_bpe_from_model("gpt2")?;
        let inputs = tokenizer.encode_with_special_tokens("Do you have time");
        let num_tokens = inputs.len();
        let inputs = Tensor::from_vec(inputs, num_tokens, vb.device())?.unsqueeze(0)?;
        println!("Inputs: {:?}", inputs.to_vec2::<u32>());
        println!("Inputs dimensions: {:?}", inputs);

        let outputs = model.forward_t(&inputs, false)?;
        println!("Outputs: {:?}", outputs.to_vec3::<f32>());
        println!("Outputs dimensions: {:?}", outputs);

        // get last output token to use for making predictions of spam/ham
        let (_b, c, _vocab_size) = outputs.dims3()?;
        println!(
            "Last output token: {:?}",
            outputs.i((.., c - 1, ..))?.to_vec2::<f32>()
        );

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
