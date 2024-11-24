use crate::Example;

pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Example usage of `text_to_token_ids` and `token_ids_to_text`.")
    }

    fn page_source(&self) -> usize {
        132_usize
    }

    fn main(&self) {
        use crate::listings::{
            ch04::generate_text_simple,
            ch05::{text_to_token_ids, token_ids_to_text},
        };
    }
}
