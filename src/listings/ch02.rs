//! Listings from Chapter 2

use candle_core::{Device, Result, Tensor};
use candle_datasets::{batcher::IterResult2, Batcher};
use fancy_regex::{Captures, Regex};
use rand::{seq::SliceRandom, thread_rng};
use std::collections::HashMap;
use std::rc::Rc;
use tiktoken_rs::CoreBPE;

/// Listing 2.3 Implementing a simple text tokenizer
#[derive(Default, Debug)]
pub struct SimpleTokenizerV1 {
    str_to_int: HashMap<String, i32>,
    int_to_str: HashMap<i32, String>,
}

impl SimpleTokenizerV1 {
    pub fn from_vocab(vocab: HashMap<&str, i32>) -> Self {
        Self {
            str_to_int: vocab.iter().map(|(k, v)| (String::from(*k), *v)).collect(),
            int_to_str: vocab.iter().map(|(k, v)| (*v, String::from(*k))).collect(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        let re = Regex::new(r#"([,.?_!"()']|--|\s)"#).unwrap();
        let preprocessed: Vec<&str> = re.split(text).map(|x| x.unwrap()).collect();
        preprocessed
            .into_iter()
            .map(|s| self.str_to_int.get(&String::from(s)).unwrap())
            .cloned()
            .collect()
    }

    pub fn decode(&self, ids: Vec<i32>) -> String {
        let text_vec: Vec<String> = ids
            .iter()
            .map(|i| self.int_to_str.get(i).unwrap())
            .cloned()
            .collect();
        let text = &text_vec.join(" ")[..];

        // remove space before any punctuations
        let re = Regex::new(r#"\s+([,.?!"()\'])"#).unwrap();
        String::from(re.replace_all(text, |caps: &Captures| caps[1].to_string()))
    }
}

/// Listing 2.4 A simple text tokenizer that handles unknown words
#[derive(Default, Debug)]
pub struct SimpleTokenizerV2 {
    str_to_int: HashMap<String, i32>,
    int_to_str: HashMap<i32, String>,
}

impl SimpleTokenizerV2 {
    pub fn from_vocab(vocab: HashMap<&str, i32>) -> Self {
        // add special tokens to vocab if needed
        let mut next_token_id = vocab.len() as i32 + 1_i32;
        let mut vocab_copy = vocab.clone();

        if !vocab.contains_key("<|unk|>") {
            vocab_copy.entry("<|unk|>").or_insert(next_token_id);
            next_token_id += 1;
        }

        if !vocab.contains_key("|endoftext|>") {
            vocab_copy.entry("<|endoftext|>").or_insert(next_token_id);
        }

        Self {
            str_to_int: vocab_copy
                .iter()
                .map(|(k, v)| (String::from(*k), *v))
                .collect(),
            int_to_str: vocab_copy
                .iter()
                .map(|(k, v)| (*v, String::from(*k)))
                .collect(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        let re = Regex::new(r#"([,.?_!"()']|--|\s)"#).unwrap();
        let preprocessed: Vec<&str> = re.split(text).map(|x| x.unwrap()).collect();
        preprocessed
            .into_iter()
            .map(|s| {
                self.str_to_int
                    .get(&String::from(s))
                    .unwrap_or(self.str_to_int.get("<|unk|>").unwrap())
            })
            .cloned()
            .collect()
    }

    pub fn decode(&self, ids: Vec<i32>) -> String {
        let text_vec: Vec<String> = ids
            .iter()
            .map(|i| self.int_to_str.get(i).unwrap())
            .cloned()
            .collect();
        let text = &text_vec.join(" ")[..];

        // remove space before any punctuations
        let re = Regex::new(r#"\s+([,.?!"()\'])"#).unwrap();
        String::from(re.replace_all(text, |caps: &Captures| caps[1].to_string()))
    }
}

pub struct GPTDatasetV1_ {
    input_ids: Vec<Vec<u32>>,
    target_ids: Vec<Vec<u32>>,
}

/// Listing 2.5 A dataset for batched inputs and targets
///
/// GPTDatasetV1 is a wrapper for `GPTDatasetV1_` which is refcounted.
/// This makes cloning datasets cheap. I.e., when creating a batcher of a
/// dataset.
#[derive(Clone)]
pub struct GPTDatasetV1(Rc<GPTDatasetV1_>);

impl AsRef<GPTDatasetV1> for GPTDatasetV1 {
    fn as_ref(&self) -> &GPTDatasetV1 {
        self
    }
}

impl std::ops::Deref for GPTDatasetV1 {
    type Target = GPTDatasetV1_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl GPTDatasetV1 {
    /// Creates a new `GPTDatasetV1`.
    ///
    /// ```rust
    /// use tiktoken_rs::get_bpe_from_model;
    /// use llms_from_scratch_rs::listings::ch02::GPTDatasetV1;
    ///
    /// let txt = "In the heart of the city";
    /// let tokenizer = get_bpe_from_model("gpt2").unwrap();
    /// let token_ids = tokenizer.encode_with_special_tokens(&txt[..]);
    /// let stride = 1_usize;
    /// let max_length = 3_usize;
    /// let dataset = GPTDatasetV1::new(&txt[..], tokenizer, max_length, stride);
    /// ```
    pub fn new(txt: &str, tokenizer: CoreBPE, max_length: usize, stride: usize) -> Self {
        let token_ids = tokenizer.encode_with_special_tokens(txt);

        let mut input_ids: Vec<Vec<u32>> = Vec::default();
        let mut target_ids: Vec<Vec<u32>> = Vec::default();
        // get input_ids and target_ids
        for i in (0..token_ids.len() - max_length).step_by(stride) {
            let input_chunk = &token_ids[i..(i + max_length)];
            let target_chunk = &token_ids[(i + 1_usize)..(i + max_length + 1_usize)];
            input_ids.push(input_chunk.to_vec());
            target_ids.push(target_chunk.to_vec());
        }

        let dataset_ = GPTDatasetV1_ {
            input_ids,
            target_ids,
        };

        Self(Rc::new(dataset_))
    }

    /// Gets the number of input-target sequences in the dataset.
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Checks whether the dataset is empty or has no input-target sequences.
    pub fn is_empty(&self) -> bool {
        self.input_ids.len() == 0
    }

    /// Returns the input tokens for all input sequences.
    pub fn input_ids(&self) -> &Vec<Vec<u32>> {
        &self.input_ids
    }

    /// Returns the target token ides for all input sequences.
    pub fn target_ids(&self) -> &Vec<Vec<u32>> {
        &self.target_ids
    }

    /// Returns the input-target pair at the specified index.
    pub fn get_pair_at_index(&self, idx: usize) -> (&Vec<u32>, &Vec<u32>) {
        (&self.input_ids[idx], &self.target_ids[idx])
    }
}

/// `GPTDatasetIter` analagous to PyTorch's `DataLoader` class/
///
/// A data loader to generate batches with input-target pairs
/// We can use `GPTDatasetIter` with `candle_datasets::Batcher` to get desired
/// batches of examples.
pub struct GPTDatasetIter {
    dataset: GPTDatasetV1,
    remaining_indices: Vec<usize>,
}

impl GPTDatasetIter {
    pub fn new(dataset: GPTDatasetV1, shuffle: bool) -> Self {
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

impl Iterator for GPTDatasetIter {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.remaining_indices.pop() {
            let (input_ids, target_ids) = self.dataset.get_pair_at_index(idx);

            // turn into Tensors and return
            let dev = Device::cuda_if_available(0).unwrap();
            let input_tensor = Tensor::new(&input_ids[..], &dev);
            let target_tensor = Tensor::new(&target_ids[..], &dev);
            Some(candle_core::error::zip(input_tensor, target_tensor))
        } else {
            None
        }
    }
}

pub type GPTDataBatcher = Batcher<IterResult2<GPTDatasetIter>>;
pub struct GPTDataLoader {
    dataset: GPTDatasetV1,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl GPTDataLoader {
    pub fn new(dataset: GPTDatasetV1, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
        }
    }

    pub fn batcher(&self) -> GPTDataBatcher {
        let iter = GPTDatasetIter::new(self.dataset.clone(), self.shuffle);
        Batcher::new_r2(iter)
            .batch_size(self.batch_size)
            .return_last_incomplete_batch(self.drop_last)
    }
}

/// Listing 2.6 A data loader to generate batches with input-output pairs
///
/// ```rust
/// use llms_from_scratch_rs::listings::ch02::create_dataloader_v1;
///
/// let txt = "In the heart of the city";
/// let batch_size = 2_usize;
/// let stride = 1_usize;
/// let max_length = 3_usize;
/// let shuffle = false;
/// let drop_last = false;
/// let data_loader =
///     create_dataloader_v1(txt, batch_size, max_length, stride, shuffle, drop_last);
/// ```
pub fn create_dataloader_v1(
    txt: &str,
    batch_size: usize,
    max_length: usize,
    stride: usize,
    shuffle: bool,
    drop_last: bool,
) -> GPTDataLoader {
    let tokenizer = tiktoken_rs::get_bpe_from_model("gpt2").unwrap();
    let dataset = GPTDatasetV1::new(txt, tokenizer, max_length, stride);
    GPTDataLoader::new(dataset, batch_size, shuffle, drop_last)
}

#[cfg(test)]
mod tests {
    use core::panic;

    use super::*;
    use anyhow::Result;
    use candle_datasets::Batcher;
    use rstest::*;
    use tiktoken_rs::get_bpe_from_model;

    #[fixture]
    pub fn vocab() -> HashMap<&'static str, i32> {
        let mut vocab: HashMap<&str, i32> = HashMap::new();
        vocab.entry("this").or_insert(1);
        vocab.entry("is").or_insert(2);
        vocab.entry("a").or_insert(3);
        vocab.entry("test").or_insert(4);
        return vocab;
    }

    #[fixture]
    pub fn txt_tokenizer() -> (String, CoreBPE) {
        let txt = "In the heart of the city";
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        (txt.to_string(), tokenizer)
    }

    #[fixture]
    pub fn gpt_dataset(#[from(txt_tokenizer)] (txt, tokenizer): (String, CoreBPE)) -> GPTDatasetV1 {
        let stride = 1_usize;
        let max_length = 3_usize;
        GPTDatasetV1::new(&txt[..], tokenizer, max_length, stride)
    }

    #[rstest]
    fn test_simple_tokenizer_init(vocab: HashMap<&str, i32>) -> Result<()> {
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);

        // assert
        assert_eq!(tokenizer.str_to_int.get(&String::from("this")), Some(&1));
        assert_eq!(tokenizer.str_to_int.get(&String::from("is")), Some(&2));
        assert_eq!(tokenizer.str_to_int.get(&String::from("a")), Some(&3));
        assert_eq!(tokenizer.str_to_int.get(&String::from("test")), Some(&4));
        Ok(())
    }

    #[rstest]
    fn test_encode(vocab: HashMap<&str, i32>) -> Result<()> {
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);
        let token_ids = tokenizer.encode("this is a test");

        assert_eq!(token_ids[0], 1);
        assert_eq!(token_ids[1], 2);
        assert_eq!(token_ids[2], 3);
        assert_eq!(token_ids[3], 4);
        Ok(())
    }

    #[rstest]
    fn test_simple_tokenizer_decode(mut vocab: HashMap<&str, i32>) -> Result<()> {
        vocab.entry(".").or_insert(5);
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);

        let token_ids = vec![1, 2, 3, 4, 5];
        let text = tokenizer.decode(token_ids);

        assert_eq!(text, "this is a test.");
        Ok(())
    }

    #[rstest]
    fn test_simple_tokenizer_v2_encode(vocab: HashMap<&str, i32>) -> Result<()> {
        let tokenizer = SimpleTokenizerV2::from_vocab(vocab);
        let token_ids = tokenizer.encode("this is a test! <|endoftext|>");

        assert_eq!(token_ids[0], 1);
        assert_eq!(token_ids[1], 2);
        assert_eq!(token_ids[2], 3);
        assert_eq!(token_ids[3], 4);
        assert_eq!(token_ids[4], 5);
        assert_eq!(token_ids[5], 6);
        Ok(())
    }

    #[rstest]
    fn test_simple_tokenizer_v2_decode(vocab: HashMap<&str, i32>) -> Result<()> {
        let tokenizer = SimpleTokenizerV2::from_vocab(vocab);

        let token_ids = vec![1, 2, 3, 4, 5, 6];
        let text = tokenizer.decode(token_ids);

        assert_eq!(text, "this is a test <|unk|> <|endoftext|>");
        Ok(())
    }

    #[rstest]
    fn test_gpt_dataset_v1_init(
        #[from(txt_tokenizer)] (txt, tokenizer): (String, CoreBPE),
    ) -> Result<()> {
        let token_ids = tokenizer.encode_with_special_tokens(&txt[..]);
        let stride = 1_usize;
        let max_length = 3_usize;
        let dataset = GPTDatasetV1::new(&txt[..], tokenizer, max_length, stride);

        for mx in 1..max_length {
            // test target alignments
            assert_eq!(
                dataset.input_ids[0][mx],
                dataset.target_ids[0][mx - 1_usize]
            );
        }

        for ix in 1..dataset.input_ids.len() {
            // test max length per input
            assert!(dataset.input_ids[ix].len() == max_length);
            // test stride alignments
            assert_eq!(dataset.input_ids[ix][0], token_ids[ix * stride]);
        }
        Ok(())
    }

    #[rstest]
    fn test_gpt_dataset_v1_iter(
        #[from(txt_tokenizer)] (txt, tokenizer): (String, CoreBPE),
    ) -> Result<()> {
        let stride = 1_usize;
        let max_length = 3_usize;
        let dataset = GPTDatasetV1::new(&txt[..], tokenizer, max_length, stride);
        let mut iter = GPTDatasetIter::new(dataset.clone(), false);
        let mut count = 0_usize;

        // user iter to sequentially get next pair checking equality with dataset
        while let Some(Ok((this_inputs, this_targets))) = iter.next() {
            let this_inputs_vec: Vec<u32> = this_inputs.to_vec1::<u32>()?;
            let this_targets_vec: Vec<u32> = this_targets.to_vec1::<u32>()?;

            assert!(this_inputs.shape().dims()[0] == max_length);
            assert!(this_targets.shape().dims()[0] == max_length);

            for (idx, token_id) in this_inputs_vec.iter().enumerate() {
                assert_eq!(*token_id, dataset.input_ids[count][idx]);
            }
            for (idx, token_id) in this_targets_vec.iter().enumerate() {
                assert_eq!(*token_id, dataset.target_ids[count][idx]);
            }

            count += 1;
        }
        assert_eq!(count, dataset.len());
        Ok(())
    }

    #[rstest]
    fn test_gpt_dataset_with_batch(#[from(gpt_dataset)] dataset: GPTDatasetV1) -> Result<()> {
        let iter = GPTDatasetIter::new(dataset.clone(), false);
        let batch_size = 2_usize;
        let mut batch_iter = Batcher::new_r2(iter).batch_size(batch_size);

        match batch_iter.next() {
            Some(Ok((inputs, targets))) => {
                assert_eq!(inputs.dims(), targets.dims());
                assert_eq!(inputs.dims()[0], batch_size);
            }
            Some(Err(err)) => panic!("{}", err),
            None => panic!("None"),
        }
        Ok(())
    }

    #[rstest]
    fn test_create_dataloader_v1() -> Result<()> {
        let txt = "In the heart of the city";
        let batch_size = 2_usize;
        let stride = 1_usize;
        let max_length = 3_usize;
        let shuffle = false;
        let drop_last = false;
        let data_loader =
            create_dataloader_v1(txt, batch_size, max_length, stride, shuffle, drop_last);

        let mut batcher = data_loader.batcher();
        match batcher.next() {
            Some(Ok((inputs, targets))) => {
                assert_eq!(inputs.dims(), targets.dims());
                assert_eq!(inputs.dims()[0], batch_size);
            }
            Some(Err(err)) => panic!("{}", err),
            None => panic!("None"),
        }
        Ok(())
    }
}
