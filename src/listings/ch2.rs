use fancy_regex::{Captures, Regex};
use std::collections::HashMap;

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
}
