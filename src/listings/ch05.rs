use candle_core::Device;
use candle_core::{Module, Result, Tensor};
use std::collections::HashSet;
use tiktoken_rs::CoreBPE;

use super::ch04::GPTModel;

/// Listing 5.1
pub fn text_to_token_ids(text: &str, tokenizer: &CoreBPE, dev: &Device) -> Result<Tensor> {
    let allowed_special = HashSet::from(["<|endoftext|>"]);
    let encoded = tokenizer.encode(text, allowed_special);
    let num_tokens = encoded.len();
    // encoded tensor
    Tensor::from_vec(encoded, (1_usize, num_tokens), dev)
}

/// Listing 5.1
pub fn token_ids_to_text(token_ids: Tensor, tokenizer: &CoreBPE) -> anyhow::Result<String> {
    let flat = token_ids.squeeze(0)?;
    tokenizer.decode(flat.to_vec1::<u32>()?)
}

pub fn calc_loss_batch(
    input_batch: &Tensor,
    target_batch: &Tensor,
    model: &GPTModel,
    device: &Device,
) -> Result<Tensor> {
    let input_batch = input_batch.to_device(device)?;
    let target_batch = target_batch.to_device(device)?;
    let logits = model.forward(&input_batch)?;

    // flatten
    let logits_flat = logits.flatten(0, 1)?;
    let targets_flat = target_batch.flatten_all()?;

    let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::listings::ch04::Config;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use rstest::*;
    use tiktoken_rs::get_bpe_from_model;

    #[fixture]
    pub fn txt_tokenizer() -> (String, CoreBPE) {
        let txt = "In the heart of the city";
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        (txt.to_string(), tokenizer)
    }

    #[rstest]
    fn test_text_to_token_ids_and_back_to_text(
        #[from(txt_tokenizer)] (txt, tokenizer): (String, CoreBPE),
    ) {
        let token_ids =
            text_to_token_ids(&txt[..], &tokenizer, &Device::cuda_if_available(0).unwrap())
                .unwrap();
        let decoded_text = token_ids_to_text(token_ids, &tokenizer).unwrap();
        assert_eq!(decoded_text, txt);
    }

    #[rstest]
    fn test_calc_loss_batch() {
        // create model
        let varmap = VarMap::new();
        let vb =
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0).unwrap());
        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb.pp("model")).unwrap();

        // create sample inputs
        let inputs = Tensor::new(&[[100_u32, 20, 300], [400, 7, 88]], vb.device()).unwrap();
        let targets = Tensor::new(&[[1_u32, 2, 3], [4, 5, 9]], vb.device()).unwrap();

        let loss = calc_loss_batch(&inputs, &targets, &model, vb.device()).unwrap();
        assert_eq!(loss.elem_count(), 1);
    }
}
