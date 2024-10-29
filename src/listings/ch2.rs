use fancy_regex::Regex;
use std::collections::HashMap;

// Listing 2.3
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

    // accessor methods
    pub fn str_to_int(&self) -> &HashMap<String, i32> {
        &self.str_to_int
    }

    pub fn int_to_str(&self) -> &HashMap<i32, String> {
        &self.int_to_str
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

    pub fn decode(self, _ids: Vec<i32>) -> String {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer_init() {
        // arrange
        let mut vocab: HashMap<&str, i32> = HashMap::new();
        vocab.entry("this").or_insert(1);
        vocab.entry("is").or_insert(2);
        vocab.entry("a").or_insert(3);
        vocab.entry("test").or_insert(4);

        // act
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);

        // assert
        assert_eq!(tokenizer.str_to_int.get(&String::from("this")), Some(&1));
        assert_eq!(tokenizer.str_to_int.get(&String::from("is")), Some(&2));
        assert_eq!(tokenizer.str_to_int.get(&String::from("a")), Some(&3));
        assert_eq!(tokenizer.str_to_int.get(&String::from("test")), Some(&4));
    }

    #[test]
    fn test_encode() {
        let mut vocab: HashMap<&str, i32> = HashMap::new();
        vocab.entry("this").or_insert(1);
        vocab.entry("is").or_insert(2);
        vocab.entry("a").or_insert(3);
        vocab.entry("test").or_insert(4);
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);

        let token_ids = tokenizer.encode("this is a test");

        assert_eq!(token_ids[0], 1);
        assert_eq!(token_ids[1], 2);
        assert_eq!(token_ids[2], 3);
        assert_eq!(token_ids[3], 4);
    }
}
