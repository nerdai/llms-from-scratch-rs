use super::{
    ch02::GPTDataLoader,
    ch04::{generate_text_simple, GPTModel},
};
use candle_core::{Device, Error, IndexOp, ModuleT, Result, Tensor, D};
use candle_nn::{ops::softmax, Optimizer, VarMap};
use itertools::Itertools;
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};
use std::{
    cmp,
    collections::{HashMap, HashSet},
    fmt::Display,
    sync::LazyLock,
};
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
    let logits = model.forward_t(&input_batch, true)?;

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

// Can also use candle_transformers::LogitProcessor
pub fn sample_multinomial(rng: &mut StdRng, prs: &Vec<f32>) -> Result<u32> {
    let dist = WeightedIndex::new(prs).map_err(candle_core::Error::wrap)?;
    let sample = dist.sample(rng) as u32;
    Ok(sample)
}

pub fn print_sampled_tokens(
    probas: &Vec<f32>,
    inverse_vocab: &HashMap<u32, &str>,
    with_expected_values: bool,
) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(123_u64);
    let sample_size = 1000_usize;
    let sample = (0..sample_size)
        .map(|_| sample_multinomial(&mut rng, probas))
        .collect::<Result<Vec<u32>>>()?;
    let sample_ids = sample.into_iter().counts();

    for (i, freq) in sample_ids.into_iter() {
        if with_expected_values {
            let expected_values = probas
                .iter()
                .map(|p| p * sample_size as f32)
                .collect::<Vec<f32>>();
            println!(
                "{:?} x {:?} with expected val {:.2}",
                freq,
                inverse_vocab.get(&i),
                expected_values[i as usize]
            );
        } else {
            println!("{:?} x {:?}", freq, inverse_vocab.get(&i));
        }
    }
    Ok(())
}

pub trait TopK {
    fn topk_last_dim0(&self, top_k: usize) -> Result<(Tensor, Tensor)>;

    fn topk_last_dim1(&self, top_k: usize) -> Result<(Tensor, Tensor)>;
}

impl TopK for Tensor {
    fn topk_last_dim0(&self, top_k: usize) -> Result<(Tensor, Tensor)> {
        let top_pos = self.arg_sort_last_dim(false)?;
        let top_pos = top_pos.i(..top_k)?;
        let top_els = self.i(top_pos.to_vec1::<u32>()?)?;
        Ok((top_els, top_pos))
    }

    fn topk_last_dim1(&self, top_k: usize) -> Result<(Tensor, Tensor)> {
        // get CUDA error sometimes when using `.arg_sort_last_dim`
        // moving to CPU to carry out the op
        let top_pos = self.to_device(&Device::Cpu)?.arg_sort_last_dim(false)?;
        let top_pos = top_pos.to_device(&Device::cuda_if_available(0)?)?;
        let (batch_size, vocab_size) = top_pos.dims2()?;
        let top_pos = top_pos.i((.., ..top_k))?.flatten_all()?;

        // get appropriate sum starting index
        let aux = Tensor::arange(0u32, batch_size as u32, self.device())?;
        let aux = (vocab_size as f64 * aux.broadcast_left(top_k)?.t()?.flatten_all()?)?;
        let top_pos = (top_pos + &aux)?;
        let top_els = self.flatten_all()?.i(top_pos.to_vec1::<u32>()?)?;

        // reshape
        let top_els = top_els.reshape((batch_size, top_k))?;
        let top_pos = (top_pos - &aux)?;
        let top_pos = top_pos.reshape((batch_size, top_k))?;
        Ok((top_els, top_pos))
    }
}

/// Listing 5.4
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &GPTModel,
    idx: Tensor,
    max_new_tokens: usize,
    context_size: usize,
    temperature: Option<f64>,
    top_k: Option<usize>,
    eos_id: Option<Tensor>,
    rng: &mut StdRng,
) -> Result<Tensor> {
    let mut idx = idx.clone();
    for _ in 0..max_new_tokens {
        let (b, seq_len) = idx.dims2()?;
        let start_token_index = cmp::max(0isize, seq_len as isize - context_size as isize) as usize;
        let idx_cond = idx.i((.., start_token_index..seq_len))?;
        let logits = model.forward_t(&idx_cond, false)?;
        let (_b, c, _vocab_size) = logits.dims3()?;
        let logits = logits.i((.., c - 1, ..))?;

        let logits = if let Some(top_k) = top_k {
            let (top_logits, _top_pos) = logits.contiguous()?.topk_last_dim1(top_k)?;
            let mask = logits.broadcast_lt(&top_logits.min_keepdim(D::Minus1)?)?;
            let on_true = logits
                .ones_like()?
                .broadcast_mul(&Tensor::new(f32::NEG_INFINITY, logits.device())?)?;
            mask.where_cond(&on_true, &logits)?
        } else {
            logits
        };

        let idx_next = if let Some(temp) = temperature {
            let logits = (logits / temp)?;
            let probas = softmax(&logits, D::Minus1)?;
            let mut idx_next: Vec<u32> = vec![];
            for bx in 0..b {
                let this_probas = probas.i((bx, ..)).unwrap();
                let next_token_id =
                    sample_multinomial(rng, &this_probas.to_vec1::<f32>().unwrap()).unwrap();
                idx_next.push(next_token_id);
            }
            Tensor::from_vec(idx_next, (b, 1_usize), logits.device())?
        } else {
            let probas = softmax(&logits, 1)?;
            probas.argmax_keepdim(D::Minus1)?
        };

        if let Some(ref eos) = eos_id {
            // not sure if this is the right thing to do
            // eos_id can appear in any of the batch inputs
            let num_eos = idx_next
                .broadcast_eq(eos)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();

            if num_eos as usize == b {
                break;
            }
        }

        idx = Tensor::cat(&[&idx, &idx_next], D::Minus1)?;
    }
    Ok(idx)
}

static WEIGHTS_MAPPING: LazyLock<HashMap<&'static str, HashMap<&'static str, HuggingFaceWeight>>> =
    LazyLock::new(|| {
        HashMap::from([
            (
                "not_transformer_wts",
                HashMap::from([
                    (
                        "pos_emb.weight",
                        HuggingFaceWeightBuilder::new("wpe.weight").build(),
                    ),
                    (
                        "tok_emb.weight",
                        HuggingFaceWeightBuilder::new("wte.weight").build(),
                    ),
                    (
                        "final_norm.scale",
                        HuggingFaceWeightBuilder::new("ln_f.weight").build(),
                    ),
                    (
                        "final_norm.shift",
                        HuggingFaceWeightBuilder::new("ln_f.bias").build(),
                    ),
                    (
                        "out_head.weight",
                        HuggingFaceWeightBuilder::new("wte.weight").build(),
                    ),
                ]),
            ),
            (
                "transformer_wts_except_qkv",
                HashMap::from([
                    (
                        "ff.first_layer.bias",
                        HuggingFaceWeightBuilder::new("mlp.c_fc.bias").build(),
                    ),
                    (
                        "ff.first_layer.weight",
                        HuggingFaceWeightBuilder::new("mlp.c_fc.weight")
                            .set_transpose()
                            .build(),
                    ),
                    (
                        "ff.second_layer.bias",
                        HuggingFaceWeightBuilder::new("mlp.c_proj.bias").build(),
                    ),
                    (
                        "ff.second_layer.weight",
                        HuggingFaceWeightBuilder::new("mlp.c_proj.weight")
                            .set_transpose()
                            .build(),
                    ),
                    (
                        "norm1.scale",
                        HuggingFaceWeightBuilder::new("ln_1.weight").build(),
                    ),
                    (
                        "norm1.shift",
                        HuggingFaceWeightBuilder::new("ln_1.bias").build(),
                    ),
                    (
                        "norm2.scale",
                        HuggingFaceWeightBuilder::new("ln_2.weight").build(),
                    ),
                    (
                        "norm2.shift",
                        HuggingFaceWeightBuilder::new("ln_2.bias").build(),
                    ),
                    (
                        "mha.out_proj.bias",
                        HuggingFaceWeightBuilder::new("attn.c_proj.bias").build(),
                    ),
                    (
                        "mha.out_proj.weight",
                        HuggingFaceWeightBuilder::new("attn.c_proj.weight")
                            .set_transpose()
                            .build(),
                    ),
                ]),
            ),
            (
                "transformer_wts_qkv",
                HashMap::from([
                    // NOTE: these weights need to be derived from attn.c_attn.bias and attn.c_attn.weight
                    // and this is done within the loop.
                    (
                        "mha.key.bias",
                        HuggingFaceWeightBuilder::new("attn.c_attn.key.bias").build(),
                    ),
                    (
                        "mha.key.weight",
                        HuggingFaceWeightBuilder::new("attn.c_attn.key.weight")
                            .set_transpose()
                            .build(),
                    ),
                    (
                        "mha.query.bias",
                        HuggingFaceWeightBuilder::new("attn.c_attn.query.bias").build(),
                    ),
                    (
                        "mha.query.weight",
                        HuggingFaceWeightBuilder::new("attn.c_attn.query.weight")
                            .set_transpose()
                            .build(),
                    ),
                    (
                        "mha.value.bias",
                        HuggingFaceWeightBuilder::new("attn.c_attn.value.bias").build(),
                    ),
                    (
                        "mha.value.weight",
                        HuggingFaceWeightBuilder::new("attn.c_attn.value.weight")
                            .set_transpose()
                            .build(),
                    ),
                ]),
            ),
        ])
    });

const HF_TRANSFORMER_PREFIX: &str = "h";

struct HuggingFaceWeight {
    name: String,
    transpose: bool,
}

impl Display for HuggingFaceWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

struct HuggingFaceWeightBuilder {
    name: String,
    transpose: bool,
}

impl HuggingFaceWeightBuilder {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            transpose: false,
        }
    }

    fn set_transpose(mut self) -> Self {
        self.transpose = true;
        self
    }

    fn build(self) -> HuggingFaceWeight {
        HuggingFaceWeight {
            name: self.name,
            transpose: self.transpose,
        }
    }
}

// helper fn for loading weights into gpt
fn load_from_weights_mapping(
    gpt_varmap: &VarMap,
    weights: &HashMap<String, Tensor>,
    var_prefix: Option<&str>,
    weights_prefix: Option<&str>,
    weights_mapping: &HashMap<&str, HuggingFaceWeight>,
) -> Result<()> {
    let gpt_data: std::sync::MutexGuard<'_, HashMap<String, candle_core::Var>> =
        gpt_varmap.data().lock().unwrap();

    for (gpt_name, hf_weight) in weights_mapping.iter() {
        let var_name = if let Some(prefix) = var_prefix {
            format!("{prefix}.{gpt_name}")
        } else {
            gpt_name.to_string()
        };

        let data_name = if let Some(w_prefix) = weights_prefix {
            format!("{w_prefix}.{}", hf_weight.name)
        } else {
            hf_weight.name.to_string()
        };

        let var = gpt_data
            .get(var_name.as_str())
            .ok_or_else(|| Error::CannotFindTensor { path: var_name }.bt())?;
        let data = weights
            .get(data_name.as_str())
            .ok_or_else(|| Error::CannotFindTensor { path: data_name }.bt())?;
        if hf_weight.transpose {
            var.set(&data.t()?)?;
        } else {
            var.set(data)?;
        }
    }
    Ok(())
}

/// Listing 5.5
#[allow(unused_variables)]
pub fn load_weights_into_gpt(
    gpt_varmap: &VarMap,
    weights: &mut HashMap<String, Tensor>,
    model_prefix: Option<&str>,
    num_layers: usize,
) -> Result<()> {
    let weights_mapping = &*WEIGHTS_MAPPING;

    // set weights for everything but transformer blocks
    load_from_weights_mapping(
        gpt_varmap,
        weights,
        model_prefix,
        None,
        weights_mapping.get("not_transformer_wts").unwrap(),
    )?;

    // set transformer block weights
    for b in 0..num_layers {
        let var_prefix = if let Some(prefix) = model_prefix {
            format!("{prefix}.trf.{b}")
        } else {
            format!("trf.{b}")
        };
        let weights_prefix = format!("{HF_TRANSFORMER_PREFIX}.{b}");

        // set weights for everything in this transformer block but its q,k,v
        load_from_weights_mapping(
            gpt_varmap,
            weights,
            Some(var_prefix.as_str()),
            Some(weights_prefix.as_str()),
            weights_mapping.get("transformer_wts_except_qkv").unwrap(),
        )?;

        // split attn.c_attn.bias
        let data_name = format!("{HF_TRANSFORMER_PREFIX}.{b}.attn.c_attn.bias");
        let hf_attn_bias = weights
            .get(data_name.as_str())
            .ok_or_else(|| Error::CannotFindTensor { path: data_name }.bt())?;
        let dim = hf_attn_bias.dims()[0] / 3_usize;
        let q_b = hf_attn_bias.i(..dim)?;
        let k_b = hf_attn_bias.i(dim..2 * dim)?;
        let v_b = hf_attn_bias.i(2 * dim..)?;

        // split attn.c_attn.weight
        let data_name = format!("{HF_TRANSFORMER_PREFIX}.{b}.attn.c_attn.weight");
        let hf_attn_weight = weights
            .get(data_name.as_str())
            .ok_or_else(|| Error::CannotFindTensor { path: data_name }.bt())?;
        let q_w = hf_attn_weight.i((.., ..dim))?;
        let k_w = hf_attn_weight.i((.., dim..2 * dim))?;
        let v_w = hf_attn_weight.i((.., 2 * dim..))?;

        // add split bias and weights tensors into weights following name convention
        weights.insert(format!("{weights_prefix}.attn.c_attn.query.bias"), q_b);
        weights.insert(format!("{weights_prefix}.attn.c_attn.key.bias"), k_b);
        weights.insert(format!("{weights_prefix}.attn.c_attn.value.bias"), v_b);
        weights.insert(format!("{weights_prefix}.attn.c_attn.query.weight"), q_w);
        weights.insert(format!("{weights_prefix}.attn.c_attn.key.weight"), k_w);
        weights.insert(format!("{weights_prefix}.attn.c_attn.value.weight"), v_w);

        // load q,k,v weights and biases
        load_from_weights_mapping(
            gpt_varmap,
            weights,
            Some(var_prefix.as_str()),
            Some(weights_prefix.as_str()),
            weights_mapping.get("transformer_wts_qkv").unwrap(),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use crate::listings::ch04::Config;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use rand::SeedableRng;
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

    #[rstest]
    #[case(vec![0_f32, 1_f32], 1_u32)]
    #[case(vec![1_f32, 0_f32], 0_u32)]
    fn test_sample_multinomial(#[case] prs: Vec<f32>, #[case] expected: u32) {
        let mut rng = StdRng::seed_from_u64(1234_u64);
        let token = sample_multinomial(&mut rng, &prs).unwrap();
        assert_eq!(token, expected);
    }

    #[rstest]
    #[case(&[-3_f32, -2., -1., 0., 1., 2., 3.], &[3_f32, 2., 1.], &[6_u32, 5, 4])]
    #[case(&[10.1_f32, -1.6, 5., 0., 1., -2., 11.], &[11_f32, 10.1, 5.], &[6_u32, 0, 2])]
    fn test_topk_last_dim0(
        #[case] logits: &[f32; 7],
        #[case] expected_top_log: &[f32; 3],
        #[case] expected_top_pos: &[u32; 3],
    ) {
        let dev = Device::cuda_if_available(0).unwrap();
        let logits = Tensor::new(logits, &dev).unwrap();
        let (top_logits, top_pos) = logits.topk_last_dim0(3_usize).unwrap();
        assert_eq!(top_logits.to_vec1::<f32>().unwrap(), expected_top_log);
        assert_eq!(top_pos.to_vec1::<u32>().unwrap(), expected_top_pos);
    }

    #[rstest]
    #[case(&[[-3_f32, -2., -1.], [0., 1., 2.]], &[[-1_f32, -2.], [2_f32, 1.]], &[[2_u32, 1], [2_u32, 1]])]
    #[case(&[[10.1_f32, -1.6, 5.], [1_f32, -2., 11.]], &[[10.1_f32, 5.], [11_f32, 1.]], &[[0_u32, 2],[2_u32, 0]])]
    fn test_topk_last_dim1(
        #[case] logits: &[[f32; 3]; 2],
        #[case] expected_top_log: &[[f32; 2]; 2],
        #[case] expected_top_pos: &[[u32; 2]; 2],
    ) {
        let dev = Device::cuda_if_available(0).unwrap();
        let logits = Tensor::new(logits, &dev).unwrap();
        let top_k = 2_usize;
        let (top_logits, top_pos) = logits.topk_last_dim1(top_k).unwrap();
        assert_eq!(top_logits.to_vec2::<f32>().unwrap(), expected_top_log);
        assert_eq!(top_pos.to_vec2::<u32>().unwrap(), expected_top_pos);
    }

    #[rstest]
    fn test_generate() {
        let dev = Device::cuda_if_available(0).unwrap();
        let batch_token_ids =
            Tensor::new(&[[101_u32, 366, 100, 345], [101, 110, 322, 57]], &dev).unwrap();

        let cfg = Config::gpt_sm_test();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = GPTModel::new(cfg, vb).unwrap();

        // create sample idx
        let (batch_size, seq_len) = batch_token_ids.dims2().unwrap();
        let (context_size, max_new_tokens) = (2_usize, 3_usize);
        let mut rng = StdRng::seed_from_u64(123_u64);
        let idx = generate(
            &model,
            batch_token_ids,
            max_new_tokens,
            context_size,
            Some(1_f64),
            Some(3_usize),
            None,
            &mut rng,
        )
        .unwrap();

        assert_eq!(idx.dims(), &[batch_size, seq_len + max_new_tokens]);
    }

    #[rstest]
    #[should_panic(
        expected = "called `Result::unwrap()` on an `Err` value: Unable to decode into a valid UTF-8 string: incomplete utf-8 byte sequence from index 0"
    )]
    fn test_decode_panics_due_token_id() {
        let bad_token_id = 49426_u32; // not sure why this results in an error when decoding
        let token_ids = Tensor::new(&[[bad_token_id]], &Device::Cpu).unwrap();
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        token_ids_to_text(token_ids, &tokenizer).unwrap();
    }
}
