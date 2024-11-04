use candle_core::{Device, Result, Tensor};
use fancy_regex::{Captures, Regex};
use rand::{seq::SliceRandom, thread_rng};
use std::collections::HashMap;
use tiktoken_rs::CoreBPE;

/// Listing 2.3
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

/// Listing 2.4
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

/// Listing 2.5 A dataset for batched inputs and targets
pub struct GPTDatasetV1 {
    input_ids: Vec<Vec<u32>>,
    target_ids: Vec<Vec<u32>>,
}

impl GPTDatasetV1 {
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

        Self {
            input_ids,
            target_ids,
        }
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input_ids.len() == 0
    }

    pub fn input_ids(&self) -> &Vec<Vec<u32>> {
        &self.input_ids
    }

    pub fn target_ids(&self) -> &Vec<Vec<u32>> {
        &self.target_ids
    }

    pub fn get_pair_at_index(&self, idx: usize) -> (&Vec<u32>, &Vec<u32>) {
        (&self.input_ids[idx], &self.target_ids[idx])
    }
}

/// Listing 2.6 A data loader to generate batches with input-target pairs
/// We can use `GPTDatasetIter` with `candle_datasets::Batcher` to get desired
/// batches of examples.
pub struct GPTDatasetIter<'a> {
    dataset: &'a GPTDatasetV1,
    device: Device,
    remaining_indices: Vec<usize>,
}

impl<'a> GPTDatasetIter<'a> {
    pub fn new(dataset: &'a GPTDatasetV1, device: Device, shuffle: bool) -> Self {
        let mut remaining_indices = (0..dataset.len()).rev().collect::<Vec<_>>();
        if shuffle {
            remaining_indices.shuffle(&mut thread_rng());
        }
        Self {
            dataset,
            device,
            remaining_indices,
        }
    }
}

impl<'a> Iterator for GPTDatasetIter<'a> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.remaining_indices.pop() {
            let (input_ids, target_ids) = self.dataset.get_pair_at_index(idx);

            // turn into Tensors and return
            let input_tensor = Tensor::new(&input_ids[..], &self.device);
            let target_tensor = Tensor::new(&target_ids[..], &self.device);
            Some(candle_core::error::zip(input_tensor, target_tensor))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use core::panic;

    use super::*;
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
    fn test_simple_tokenizer_init(vocab: HashMap<&str, i32>) {
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);

        // assert
        assert_eq!(tokenizer.str_to_int.get(&String::from("this")), Some(&1));
        assert_eq!(tokenizer.str_to_int.get(&String::from("is")), Some(&2));
        assert_eq!(tokenizer.str_to_int.get(&String::from("a")), Some(&3));
        assert_eq!(tokenizer.str_to_int.get(&String::from("test")), Some(&4));
    }

    #[rstest]
    fn test_encode(vocab: HashMap<&str, i32>) {
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);
        let token_ids = tokenizer.encode("this is a test");

        assert_eq!(token_ids[0], 1);
        assert_eq!(token_ids[1], 2);
        assert_eq!(token_ids[2], 3);
        assert_eq!(token_ids[3], 4);
    }

    #[rstest]
    fn test_simple_tokenizer_decode(mut vocab: HashMap<&str, i32>) {
        vocab.entry(".").or_insert(5);
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);

        let token_ids = vec![1, 2, 3, 4, 5];
        let text = tokenizer.decode(token_ids);

        assert_eq!(text, "this is a test.");
    }

    #[rstest]
    fn test_simple_tokenizer_v2_encode(vocab: HashMap<&str, i32>) {
        let tokenizer = SimpleTokenizerV2::from_vocab(vocab);
        let token_ids = tokenizer.encode("this is a test! <|endoftext|>");

        assert_eq!(token_ids[0], 1);
        assert_eq!(token_ids[1], 2);
        assert_eq!(token_ids[2], 3);
        assert_eq!(token_ids[3], 4);
        assert_eq!(token_ids[4], 5);
        assert_eq!(token_ids[5], 6);
    }

    #[rstest]
    fn test_simple_tokenizer_v2_decode(vocab: HashMap<&str, i32>) {
        let tokenizer = SimpleTokenizerV2::from_vocab(vocab);

        let token_ids = vec![1, 2, 3, 4, 5, 6];
        let text = tokenizer.decode(token_ids);

        assert_eq!(text, "this is a test <|unk|> <|endoftext|>");
    }

    #[rstest]
    fn test_gpt_dataset_v1_init(#[from(txt_tokenizer)] (txt, tokenizer): (String, CoreBPE)) {
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
            assert_eq!(dataset.input_ids[ix].len(), max_length);
            // test stride alignments
            assert_eq!(dataset.input_ids[ix][0], token_ids[ix * stride]);
        }
    }

    #[rstest]
    fn test_gpt_dataset_v1_iter(#[from(txt_tokenizer)] (txt, tokenizer): (String, CoreBPE)) {
        let stride = 1_usize;
        let max_length = 3_usize;
        let dataset = GPTDatasetV1::new(&txt[..], tokenizer, max_length, stride);
        let dev = Device::cuda_if_available(0).unwrap();
        let mut iter = GPTDatasetIter::new(&dataset, dev, false);
        let mut count = 0_usize;

        // user iter to sequentially get next pair checking equality with dataset
        while let Some(Ok((this_inputs, this_targets))) = iter.next() {
            let this_inputs_vec: Vec<u32> = this_inputs.to_vec1::<u32>().unwrap();
            let this_targets_vec: Vec<u32> = this_targets.to_vec1::<u32>().unwrap();

            assert_eq!(this_inputs.shape().dims()[0], max_length);
            assert_eq!(this_inputs.shape().dims()[0], max_length);

            for (idx, token_id) in this_inputs_vec.iter().enumerate() {
                assert_eq!(*token_id, dataset.input_ids[count][idx]);
            }
            for (idx, token_id) in this_targets_vec.iter().enumerate() {
                assert_eq!(*token_id, dataset.target_ids[count][idx]);
            }

            count += 1;
        }
        assert_eq!(count, dataset.len());
    }

    #[rstest]
    fn test_gpt_dataset_with_batch(#[from(gpt_dataset)] dataset: GPTDatasetV1) {
        let dev = Device::cuda_if_available(0).unwrap();
        let iter = GPTDatasetIter::new(&dataset, dev, false);
        let batch_size = 2_usize;
        let mut batch_iter = Batcher::new_r2(iter)
            .batch_size(batch_size)
            .return_last_incomplete_batch(true);

        match batch_iter.next() {
            Some(Ok((inputs, targets))) => {
                println!("inputs: {:?}\n\ntargets: {:?}", inputs, targets);
            }
            Some(Err(err)) => panic!("{}", err),
            None => panic!("None"),
        }
    }
}
