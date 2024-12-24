//! Listings from Chapter 6

use ::zip::ZipArchive;
use anyhow::{anyhow, Error};
use bytes::Bytes;
use candle_core::{Device, Result, Tensor};
use candle_datasets::{batcher::IterResult2, Batcher};
use polars::prelude::*;
use rand::{seq::SliceRandom, thread_rng};
use std::cmp;
use std::fs::{create_dir_all, remove_file, rename, File};
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use tiktoken_rs::CoreBPE;

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

/// [Listing 6.3] Splitting the dataset
#[allow(unused_variables)]
pub fn random_split(
    df: &DataFrame,
    train_frac: f32,
    validation_frac: f32,
) -> anyhow::Result<(DataFrame, DataFrame, DataFrame)> {
    let frac = Series::from_iter([1_f32].iter());
    let shuffled_df = df.sample_frac(&frac, false, true, Some(123_u64))?;

    let df_size = df.shape().0;
    let train_size = (df.shape().0 as f32 * train_frac) as usize;
    let validation_size = (df.shape().0 as f32 * validation_frac) as usize;

    let train_df = shuffled_df.slice(0_i64, train_size);
    let validation_df = shuffled_df.slice(train_size as i64, validation_size);
    let test_df = shuffled_df.slice(
        (train_size + validation_size) as i64,
        df_size - train_size - validation_size,
    );
    Ok((train_df, validation_df, test_df))
}

#[allow(dead_code)]
pub struct SpamDataset_ {
    data: DataFrame,
    encoded_texts: Vec<Vec<u32>>,
    max_length: usize,
    pad_token_id: u32,
}

/// [Listing 6.4] Setting up a `SpamDataset` struct
///
/// SpamDataset is a wrapper for `SpamDataset_` which is refcounted.
#[derive(Clone)]
pub struct SpamDataset(Rc<SpamDataset_>);

impl AsRef<SpamDataset> for SpamDataset {
    fn as_ref(&self) -> &SpamDataset {
        self
    }
}

impl std::ops::Deref for SpamDataset {
    type Target = SpamDataset_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl SpamDataset {
    /// Creates a new `SpamDataset`.
    ///
    /// ```rust
    /// use llms_from_scratch_rs::listings::ch06::{SpamDataset, PAD_TOKEN_ID};
    /// use polars::prelude::*;
    /// use tiktoken_rs::get_bpe_from_model;
    ///
    /// let mut df = df!(
    ///     "sms"=> &[
    ///         "Mock example 1",
    ///         "Mock example 2"
    ///     ],
    ///     "label"=> &[0_i64, 1],
    /// )
    /// .unwrap();
    /// let tokenizer = get_bpe_from_model("gpt2").unwrap();
    /// let max_length = 24_usize;
    /// let dataset = SpamDataset::new(df, &tokenizer, Some(max_length), PAD_TOKEN_ID);
    /// ```
    pub fn new(
        df: DataFrame,
        tokenizer: &CoreBPE,
        max_length: Option<usize>,
        pad_token_id: u32,
    ) -> Self {
        let text_series = df.column("sms").unwrap().clone();
        let text_vec: Vec<Option<&str>> = text_series.str().unwrap().into_iter().collect();
        let mut encodings = text_vec
            .iter()
            .map(|el| {
                if let Some(txt) = el {
                    Ok(tokenizer.encode_with_special_tokens(txt))
                } else {
                    Err(anyhow!("There was a problem encoding texts."))
                }
            })
            .collect::<anyhow::Result<Vec<Vec<u32>>>>()
            .unwrap();

        // assign max_length
        let raw_max_length = Self::get_raw_max_length(&encodings).unwrap();
        let max_length = if let Some(v) = max_length {
            if v < raw_max_length {
                encodings = encodings
                    .into_iter()
                    .map(|el| Vec::from_iter(el.into_iter().take(v)))
                    .collect::<Vec<Vec<u32>>>();
            }
            v
        } else {
            // get max encodings
            raw_max_length
        };

        // get paddings
        let encodings = encodings
            .into_iter()
            .map(|mut v| {
                let num_pad = cmp::max(0isize, max_length as isize - v.len() as isize) as usize;
                if num_pad > 0 {
                    let padding = std::iter::repeat(pad_token_id)
                        .take(num_pad)
                        .collect::<Vec<u32>>();
                    v.extend(padding);
                    v
                } else {
                    v
                }
            })
            .collect();

        let dataset_ = SpamDataset_ {
            data: df,
            encoded_texts: encodings,
            max_length,
            pad_token_id,
        };

        Self(Rc::new(dataset_))
    }

    /// Gets the number of finetuning examples.
    pub fn len(&self) -> usize {
        self.data.shape().0
    }

    /// Checks whether the dataset is empty or has no finetuning examples.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the input tokens for all input sequences.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Get raw max length of encodings
    fn get_raw_max_length(encodings: &[Vec<u32>]) -> anyhow::Result<usize> {
        let raw_max_length = *encodings
            .iter()
            .map(|v| v.len())
            .collect::<Vec<_>>()
            .iter()
            .max()
            .ok_or(anyhow!(
                "There was a problem computing max length encodings."
            ))?;
        Ok(raw_max_length)
    }

    /// Returns the input-target pair at the specified index.
    #[allow(unused_variables)]
    pub fn get_item_at_index(&self, idx: usize) -> anyhow::Result<(&Vec<u32>, Vec<i64>)> {
        let encoded = &self.encoded_texts[idx];
        let binding = self.data.select(["label"])?;
        let label = match &binding.get_row(idx)?.0[0] {
            AnyValue::Int64(label_value) => Ok(label_value),
            _ => Err(anyhow!(
                "There was a problem in getting the Label from the dataframe."
            )),
        }?
        .to_owned();

        Ok((encoded, vec![label]))
    }
}

pub const PAD_TOKEN_ID: u32 = 50_256_u32;

/// Builder pattern for `HuggingFaceWeight`
pub struct SpamDatasetBuilder<'a> {
    data_: Option<DataFrame>,
    max_length_: Option<usize>,
    pad_token_id_: u32,
    tokenizer_: &'a CoreBPE,
}

#[allow(dead_code)]
impl<'a> SpamDatasetBuilder<'a> {
    /// Creates a new `SpamDatasetBuilder`.
    ///
    /// ```rust
    /// use llms_from_scratch_rs::listings::ch06::{SpamDataset, SpamDatasetBuilder};
    /// use polars::prelude::*;
    /// use tiktoken_rs::get_bpe_from_model;
    ///
    /// let df = df!(
    ///     "sms"=> &[
    ///         "Mock example 1",
    ///         "Mock example 2"
    ///     ],
    ///     "label"=> &[0_i64, 1],
    /// )
    /// .unwrap();
    /// let tokenizer = get_bpe_from_model("gpt2").unwrap();
    /// let dataset: SpamDataset = SpamDatasetBuilder::new(&tokenizer)
    ///     .data(df)
    ///     .max_length(Some(24_usize))
    ///     .build();
    /// ```
    pub fn new(tokenizer: &'a CoreBPE) -> Self {
        Self {
            data_: None,
            max_length_: None,
            pad_token_id_: PAD_TOKEN_ID,
            tokenizer_: tokenizer,
        }
    }

    /// Set data for builder from parquet file.
    /// ```rust
    /// use llms_from_scratch_rs::listings::ch06::{SpamDataset, SpamDatasetBuilder};
    /// use polars::prelude::*;
    /// use tempfile::NamedTempFile;
    /// use tiktoken_rs::get_bpe_from_model;
    ///
    /// let mut df = df!(
    ///     "sms"=> &[
    ///         "Mock example 1",
    ///         "Mock example 2"
    ///     ],
    ///     "label"=> &[0_i64, 1],
    /// )
    /// .unwrap();
    ///
    /// // create temp parquet file for demonstration
    /// let mut test_file = NamedTempFile::new().unwrap();
    /// ParquetWriter::new(&mut test_file).finish(&mut df).unwrap();
    /// let parquet_file = test_file.into_temp_path().keep().unwrap();
    ///
    /// // build dataset
    /// let tokenizer = get_bpe_from_model("gpt2").unwrap();
    /// let dataset: SpamDataset = SpamDatasetBuilder::new(&tokenizer)
    ///     .load_data_from_parquet(parquet_file)
    ///     .max_length(Some(24_usize))
    ///     .build();
    /// ```
    pub fn load_data_from_parquet<P: AsRef<Path>>(mut self, parquet_file: P) -> Self {
        let mut file = std::fs::File::open(parquet_file).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();
        self.data_ = Some(df);
        self
    }

    pub fn data(mut self, data: DataFrame) -> Self {
        self.data_ = Some(data);
        self
    }

    pub fn max_length(mut self, max_length: Option<usize>) -> Self {
        self.max_length_ = max_length;
        self
    }

    pub fn pad_token_id(mut self, token_id: u32) -> Self {
        self.pad_token_id_ = token_id;
        self
    }

    pub fn build(self) -> SpamDataset {
        if let Some(df) = self.data_ {
            SpamDataset::new(df, self.tokenizer_, self.max_length_, self.pad_token_id_)
        } else {
            panic!("DataFrame is not set in SpamDataBuilder.");
        }
    }
}

#[allow(dead_code)]
pub struct SpamDatasetIter {
    dataset: SpamDataset,
    remaining_indices: Vec<usize>,
}

impl SpamDatasetIter {
    /// Creates a new `SpamDatasetIter`.
    ///
    /// ```rust
    /// use llms_from_scratch_rs::listings::ch06::{SpamDataset, SpamDatasetIter, PAD_TOKEN_ID};
    /// use polars::prelude::*;
    /// use tiktoken_rs::get_bpe_from_model;
    ///
    /// let df = df!(
    ///     "sms"=> &[
    ///         "Mock example 1",
    ///         "Mock example 2"
    ///     ],
    ///     "label"=> &[0_i64, 1],
    /// )
    /// .unwrap();
    /// let tokenizer = get_bpe_from_model("gpt2").unwrap();
    /// let max_length = 24_usize;
    /// let dataset = SpamDataset::new(df, &tokenizer, Some(max_length), PAD_TOKEN_ID);
    /// let iter = SpamDatasetIter::new(dataset.clone(), false);
    /// ```
    pub fn new(dataset: SpamDataset, shuffle: bool) -> Self {
        let mut remaining_indices = (0..dataset.len()).rev().collect::<Vec<_>>();
        if shuffle {
            remaining_indices.shuffle(&mut thread_rng());
        }
        Self {
            dataset,
            remaining_indices,
        }
    }
}

impl Iterator for SpamDatasetIter {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.remaining_indices.pop() {
            let (encoded, label) = self.dataset.get_item_at_index(idx).unwrap();

            // turn into Tensors and return
            let dev = Device::cuda_if_available(0).unwrap();
            let encoded_tensor = Tensor::new(&encoded[..], &dev);
            let label_tensor = Tensor::new(&label[..], &dev);
            Some(candle_core::error::zip(encoded_tensor, label_tensor))
        } else {
            None
        }
    }
}

/// A type alias for candle_datasets::Batcher
///
/// This struct is responsible for getting batches from a type that implements
/// the `Iterator` Trait.
pub type SpamDataBatcher = Batcher<IterResult2<SpamDatasetIter>>;

/// [Listing 6.5] Creating a data loader for SpamDataset
pub struct SpamDataLoader {
    dataset: SpamDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl SpamDataLoader {
    /// Creates a new SpamLoader.
    ///
    /// ```rust
    /// use llms_from_scratch_rs::listings::ch06::{
    ///     SpamDataset,
    ///     SpamDataLoader,
    ///     PAD_TOKEN_ID
    /// };
    /// use polars::prelude::*;
    /// use tiktoken_rs::get_bpe_from_model;
    ///
    /// // create SpamDataset
    /// let df = df!(
    ///     "sms"=> &[
    ///         "Mock example 1",
    ///         "Mock example 2"
    ///     ],
    ///     "label"=> &[0_i64, 1],
    /// )
    /// .unwrap();
    /// let tokenizer = get_bpe_from_model("gpt2").unwrap();
    /// let max_length = 24_usize;
    /// let dataset = SpamDataset::new(df, &tokenizer, Some(max_length), PAD_TOKEN_ID);
    ///
    /// // create SpamDataLoader
    /// let batch_size = 2_usize;
    /// let shuffle = false;
    /// let drop_last = false;
    /// let data_loader = SpamDataLoader::new(dataset, batch_size, shuffle, drop_last);
    /// ```
    pub fn new(dataset: SpamDataset, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
        }
    }

    /// Returns a `SpamDataBatcher` that itself provides batches over the
    /// associated dataset.
    pub fn batcher(&self) -> SpamDataBatcher {
        let iter = SpamDatasetIter::new(self.dataset.clone(), self.shuffle);
        Batcher::new_r2(iter)
            .batch_size(self.batch_size)
            .return_last_incomplete_batch(!self.drop_last)
    }

    pub fn len(&self) -> usize {
        if self.drop_last {
            self.batcher().count()
        } else {
            // There is a bug in candle_datasets::Batcher, such that if
            // return_last_incomplete_batch is set to true, then the iterator
            // will never return None. This breaks `Iterator.count()` which consumes
            // the iterator until a None is encountered.
            let mut batcher = self.batcher();
            let mut count = 0_usize;
            while let Some(Ok(_el)) = batcher.next() {
                count += 1;
            }
            count
        }
    }

    pub fn is_empty(&self) -> bool {
        (self.dataset.len() < self.batch_size) && (self.drop_last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use rstest::*;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;
    use tiktoken_rs::get_bpe_from_model;

    #[fixture]
    pub fn sms_spam_df() -> (DataFrame, usize) {
        let df = df!(
            "sms"=> &[
                "Got it. Seventeen pounds for seven hundred ml – hope ok.", 
                "Great News! Call FREEFONE 08006344447 to claim your guaranteed £1000 CASH or £2000 gift. Speak to a live operator NOW!",
                "No chikku nt yet.. Ya i'm free",
                "S:-)if we have one good partnership going we will take lead:)",
                "18 days to Euro2004 kickoff! U will be kept informed of all the latest news and results daily. Unsubscribe send GET EURO STOP to 83222."],
            "label"=> &[0_i64, 1, 0, 0, 1],
        )
        .unwrap();
        (df, 2usize)
    }

    #[fixture]
    pub fn test_parquet_path(
        #[from(sms_spam_df)] (mut df, _num_spam): (DataFrame, usize),
    ) -> PathBuf {
        let mut test_file = NamedTempFile::new().unwrap();
        ParquetWriter::new(&mut test_file).finish(&mut df).unwrap();
        let path = test_file.into_temp_path().keep().unwrap();
        path
    }

    #[rstest]
    pub fn test_create_balanced_dataset(
        #[from(sms_spam_df)] (df, num_spam): (DataFrame, usize),
    ) -> Result<()> {
        let balanced_df = create_balanced_dataset(df)?;

        assert_eq!(balanced_df.shape(), (num_spam * 2_usize, 2));
        Ok(())
    }

    #[rstest]
    pub fn test_random_split(
        #[from(sms_spam_df)] (df, _num_spam): (DataFrame, usize),
    ) -> Result<()> {
        let train_frac = 0.4_f32;
        let validation_frac = 0.4_f32;
        let test_frac = 1_f32 - train_frac - validation_frac;
        let (train_df, validation_df, test_df) = random_split(&df, train_frac, validation_frac)?;

        assert_eq!(
            train_df.shape(),
            ((train_frac * df.shape().0 as f32) as usize, 2)
        );
        assert_eq!(
            validation_df.shape(),
            ((validation_frac * df.shape().0 as f32) as usize, 2)
        );
        assert_eq!(
            test_df.shape(),
            ((test_frac * df.shape().0 as f32) as usize, 2)
        );
        Ok(())
    }

    #[rstest]
    #[case(None, 33_usize)]
    #[case(Some(10_usize), 10_usize)]
    #[case(Some(60_usize), 60_usize)]
    pub fn test_spam_dataset_init(
        #[from(sms_spam_df)] (df, _num_spam): (DataFrame, usize),
        #[case] max_length: Option<usize>,
        #[case] expected_max_length: usize,
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let spam_dataset = SpamDataset::new(df, &tokenizer, max_length, PAD_TOKEN_ID);

        assert_eq!(spam_dataset.len(), 5);
        assert_eq!(spam_dataset.max_length, expected_max_length);
        // assert all encoded texts have length == max_length
        for text_enc in spam_dataset.encoded_texts.iter() {
            assert_eq!(text_enc.len(), expected_max_length);
        }

        Ok(())
    }

    #[rstest]
    #[case(None, 33_usize)]
    #[case(Some(10_usize), 10_usize)]
    #[case(Some(60_usize), 60_usize)]
    pub fn test_spam_dataset_builder_parquet_file(
        test_parquet_path: PathBuf,
        #[case] max_length: Option<usize>,
        #[case] expected_max_length: usize,
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let spam_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_parquet(test_parquet_path)
            .max_length(max_length)
            .build();

        assert_eq!(spam_dataset.len(), 5);
        assert_eq!(spam_dataset.max_length, expected_max_length);
        // assert all encoded texts have length == max_length
        for text_enc in spam_dataset.encoded_texts.iter() {
            assert_eq!(text_enc.len(), expected_max_length);
        }

        Ok(())
    }

    #[rstest]
    pub fn test_spam_dataset_iter(
        #[from(sms_spam_df)] (df, _num_spam): (DataFrame, usize),
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let max_length = 10_usize;
        let spam_dataset = SpamDataset::new(df, &tokenizer, Some(max_length), PAD_TOKEN_ID);
        let mut iter = SpamDatasetIter::new(spam_dataset.clone(), false);
        let mut count = 0_usize;

        // user iter to sequentially get next pair checking equality with dataset
        while let Some(Ok((this_encodings, this_label))) = iter.next() {
            assert!(this_encodings.shape().dims()[0] == max_length);
            assert!(this_label.shape().dims()[0] == 1_usize);
            count += 1;
        }
        assert_eq!(count, spam_dataset.len());
        Ok(())
    }

    #[rstest]
    fn test_spam_data_loader(
        #[from(sms_spam_df)] (df, _num_spam): (DataFrame, usize),
    ) -> Result<()> {
        let tokenizer = get_bpe_from_model("gpt2")?;
        let max_length = 10_usize;
        let spam_dataset = SpamDataset::new(df, &tokenizer, Some(max_length), PAD_TOKEN_ID);
        let batch_size = 2_usize;
        let shuffle = false;
        let drop_last = false;
        let data_loader = SpamDataLoader::new(spam_dataset, batch_size, shuffle, drop_last);

        let mut batcher = data_loader.batcher();
        let mut count = 0_usize;
        while let Some(Ok((inputs, targets))) = batcher.next() {
            assert!(inputs.dims()[0] <= batch_size);
            assert!(targets.dims()[0] <= batch_size);
            assert_eq!(inputs.dims()[1], max_length);
            assert_eq!(targets.dims()[1], 1_usize);
            count += 1;
        }
        assert_eq!(data_loader.len(), count);
        Ok(())
    }
}
