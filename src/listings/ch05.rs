use candle_core::Device;
use candle_core::{Result, Tensor};
use std::collections::HashSet;
use tiktoken_rs::CoreBPE;

/// Listing 5.1
pub fn text_to_token_ids(text: &str, tokenizer: CoreBPE, dev: &Device) -> Result<Tensor> {
    let allowed_special = HashSet::from(["<|endoftext|>"]);
    let encoded = tokenizer.encode(text, allowed_special);
    let num_tokens = encoded.len();
    // encoded tensor
    Tensor::from_vec(encoded, (1_usize, num_tokens), dev)
}
