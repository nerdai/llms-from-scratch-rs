use super::{
    ch02::GPTDataLoader,
    ch04::{generate_text_simple, GPTModel},
};
use candle_core::Device;
use candle_core::{Module, Result, Tensor};
use candle_nn::Optimizer;
use std::collections::HashSet;
use tiktoken_rs::CoreBPE;

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

/// Listing 5.2
pub fn calc_loss_loader(
    data_loader: &GPTDataLoader,
    model: &GPTModel,
    device: &Device,
    num_batches: Option<usize>,
) -> Result<f32> {
    // todo: ensure these calcs are done without gradient tracking
    let mut total_loss = 0_f32;
    let mut count = 0_usize;

    let mut data_batcher = data_loader.batcher();
    match num_batches {
        None => {
            while let Some(Ok((input_batch, target_batch))) = data_batcher.next() {
                let loss = calc_loss_batch(&input_batch, &target_batch, model, device)?;
                total_loss += loss.to_scalar::<f32>()?;
                count += 1_usize;
            }
            Ok(total_loss / count as f32)
        }
        Some(n) => {
            while let Some(Ok((input_batch, target_batch))) = data_batcher.next() {
                let loss = calc_loss_batch(&input_batch, &target_batch, model, device)?;
                total_loss += loss.to_scalar::<f32>()?;
                count += 1_usize;
                if count >= n {
                    break;
                }
            }
            Ok(total_loss / std::cmp::min(n, count) as f32)
        }
    }
}

/// Listing 5.3
#[allow(clippy::too_many_arguments)]
pub fn train_model_simple<T: Optimizer>(
    model: &GPTModel,
    train_loader: &GPTDataLoader,
    val_loader: &GPTDataLoader,
    mut optimizer: T,
    device: &Device,
    num_epochs: usize,
    eval_freq: usize,
    eval_iter: usize,
    start_context: &str,
    tokenizer: &CoreBPE,
) -> Result<(Vec<f32>, Vec<f32>, Vec<usize>)> {
    // retvals
    let mut train_losses: Vec<f32> = vec![];
    let mut val_losses: Vec<f32> = vec![];
    let mut track_tokens_seen: Vec<usize> = vec![];

    let (mut tokens_seen, mut global_step) = (0usize, 0_usize);

    for epoch in 0..num_epochs {
        let mut train_batcher = train_loader.batcher();
        while let Some(Ok((input_batch, target_batch))) = train_batcher.next() {
            let loss = calc_loss_batch(&input_batch, &target_batch, model, device)?;
            optimizer.backward_step(&loss)?;
            tokens_seen += input_batch.elem_count();

            if global_step % eval_freq == 0 {
                let (train_loss, val_loss) =
                    evaluate_model(model, train_loader, val_loader, device, eval_iter)?;
                train_losses.push(train_loss);
                val_losses.push(val_loss);
                track_tokens_seen.push(tokens_seen);
                println!(
                    "Ep {} (Step {}) \
                    Train loss: {}, \
                    Val loss: {}",
                    epoch + 1,
                    global_step,
                    train_loss,
                    val_loss
                );
            }
            global_step += 1;
        }
        generate_and_print_sample(model, tokenizer, device, start_context)?
    }

    Ok((train_losses, val_losses, track_tokens_seen))
}

pub fn evaluate_model(
    model: &GPTModel,
    train_loader: &GPTDataLoader,
    val_loader: &GPTDataLoader,
    device: &Device,
    eval_iter: usize,
) -> Result<(f32, f32)> {
    let train_loss = calc_loss_loader(train_loader, model, device, Some(eval_iter))?;
    let val_loss = calc_loss_loader(val_loader, model, device, Some(eval_iter))?;
    Ok((train_loss, val_loss))
}

pub fn generate_and_print_sample(
    model: &GPTModel,
    tokenizer: &CoreBPE,
    device: &Device,
    start_context: &str,
) -> Result<()> {
    let context_size = model.pos_emb().embeddings().dims()[0];
    let encoded = text_to_token_ids(start_context, tokenizer, device)?;
    let token_ids = generate_text_simple(model, encoded, 50, context_size)?;
    let decoded_text = token_ids_to_text(token_ids, tokenizer).unwrap();
    println!("{}", decoded_text.replace("\n", " "));
    Ok(())
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
