use fancy_regex::{Captures, Regex};
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
        println!("{:?}", token_ids);

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

    pub fn input_ids(&self) -> &Vec<Vec<u32>> {
        &self.input_ids
    }

    pub fn target_ids(&self) -> &Vec<Vec<u32>> {
        &self.target_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[fixture]
    pub fn vocab() -> HashMap<&'static str, i32> {
        let mut vocab: HashMap<&str, i32> = HashMap::new();
        vocab.entry("this").or_insert(1);
        vocab.entry("is").or_insert(2);
        vocab.entry("a").or_insert(3);
        vocab.entry("test").or_insert(4);
        return vocab;
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
    fn test_gpt_dataset_v1_init() {
        use tiktoken_rs::get_bpe_from_model;

        let txt = "In the heart of the city";

        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        let dataset = GPTDatasetV1::new(txt, tokenizer, 3, 1);

        println!("{:?}", dataset.input_ids);
        println!("{:?}", dataset.target_ids);
        assert_eq!(dataset.input_ids().len(), 2);
    }
}
