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
            ch04::{generate_text_simple, Config, GPTModel},
            ch05::{text_to_token_ids, token_ids_to_text},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use tiktoken_rs::get_bpe_from_model;

        // construct model
        let varmap = VarMap::new();
        let vb =
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0).unwrap());
        let cfg = Config::gpt2_124m();
        let model = GPTModel::new(Config::gpt2_124m(), vb.pp("model")).unwrap();

        // sample setup and load tokenizer
        let start_context = "Every effort moves you";
        let tokenizer = get_bpe_from_model("gpt2").unwrap();

        // generate next tokens with model
        let max_new_tokens = 10_usize;
        let token_ids = generate_text_simple(
            model,
            text_to_token_ids(start_context, &tokenizer, vb.device()).unwrap(),
            max_new_tokens,
            cfg.context_length,
        )
        .unwrap();

        // decode the token ids to print the output text
        println!(
            "Output text:\n{:?}",
            token_ids_to_text(token_ids, &tokenizer)
        )
    }
}
