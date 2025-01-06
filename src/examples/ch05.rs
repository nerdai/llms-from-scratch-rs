//! Examples from Chapter 5

use crate::Example;
use anyhow::Result;

/// # Example usage of `text_to_token_ids` and `token_ids_to_text`
///
/// #### Id
/// 05.01
///
/// #### Page
/// This example starts on page 132
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.01
///
/// # with cuda
/// cargo run --features cuda example 05.01
/// ```
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        String::from("Example usage of `text_to_token_ids` and `token_ids_to_text`.")
    }

    fn page_source(&self) -> usize {
        132_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{generate_text_simple, Config, GPTModel},
            ch05::{text_to_token_ids, token_ids_to_text},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use tiktoken_rs::get_bpe_from_model;

        // construct model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt2_124m();
        let model = GPTModel::new(Config::gpt2_124m(), vb.pp("model"))?;

        // sample setup and load tokenizer
        let start_context = "Every effort moves you";
        let tokenizer = get_bpe_from_model("gpt2")?;

        // generate next tokens with model
        let max_new_tokens = 10_usize;
        let token_ids = generate_text_simple(
            &model,
            text_to_token_ids(start_context, &tokenizer, vb.device())?,
            max_new_tokens,
            cfg.context_length,
        )?;

        // decode the token ids to print the output text
        println!(
            "Output text:\n{:?}",
            token_ids_to_text(token_ids, &tokenizer)
        );
        Ok(())
    }
}

/// # Example computation of cross-entropy and perplexity
///
/// #### Id
/// 05.02
///
/// #### Page
/// This example starts on page 133
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.02
///
/// # with cuda
/// cargo run --features cuda example 05.02
/// ```
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        let desc = "Example computation of cross-entropy and perplexity.";
        String::from(desc)
    }

    fn page_source(&self) -> usize {
        133_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::token_ids_to_text,
        };
        use candle_core::{DType, Device, IndexOp, ModuleT, Tensor, D};
        use candle_nn::{loss::cross_entropy, ops::softmax, VarBuilder, VarMap};
        use tiktoken_rs::get_bpe_from_model;

        // construct model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt2_124m();
        let model = GPTModel::new(cfg, vb.pp("model"))?;

        // inputs and target tensors
        let inputs = Tensor::new(&[[16833_u32, 3626, 6100], [40, 1107, 588]], vb.device())?;
        let targets = Tensor::new(&[[3626_u32, 6100, 345], [1107, 588, 11311]], vb.device())?;

        // logits and probas
        let logits = model.forward_t(&inputs, false)?;
        let probas = softmax(&logits, D::Minus1)?;
        println!("{:?}", probas);

        // get next token id from probas
        let token_ids = probas.argmax_keepdim(D::Minus1)?;
        println!("Token IDs:\n{:?}", token_ids.to_vec3::<u32>());

        // compare predictions to targets
        let tokenizer = get_bpe_from_model("gpt2")?;
        println!(
            "Targets batch 1: {:?}",
            token_ids_to_text(targets.i(0)?, &tokenizer)
        );
        println!(
            "Outputs batch 1: {:?}",
            token_ids_to_text(token_ids.i(0)?.flatten_all()?, &tokenizer)
        );

        // let's see the predicted probas for the target tokens
        let text_idx = 0_usize;
        let target_probas_1 =
            addons::get_target_token_probas_helper(text_idx, &targets, &probas, vb.device())?;

        println!("Text 1: {:?}", target_probas_1);

        let text_idx = 1_usize;
        let target_probas_2 =
            addons::get_target_token_probas_helper(text_idx, &targets, &probas, vb.device())?;

        println!("Text 2: {:?}", target_probas_2);

        // compute log probas
        let log_probas = Tensor::cat(&[&target_probas_1, &target_probas_2], 0)?.log()?;
        println!("Log probas: {:?}", log_probas);

        // compute average
        let avg_log_probas = log_probas.mean(0)?;
        println!("Avg log probbas: {:?}", avg_log_probas);

        // compute negative average log probas or cross-entropy
        let neg_avg_log_probas = (log_probas.mean(0)? * -1_f64)?;
        println!("Neg avg log probbas: {:?}", neg_avg_log_probas);

        // compute cross entropy with candle_nn::ops::loss::cross_entropy
        println!("Logits shape: {:?}", logits);
        println!("Targets shape: {:?}", targets);

        let logits_flat = logits.flatten(0, 1)?;
        let targets_flat = targets.flatten_all()?;
        println!("Flattened logits: {:?}", logits_flat.shape());
        println!("Flattened targets: {:?}", targets_flat.shape());

        let loss = cross_entropy(&logits_flat, &targets_flat)?;
        println!("loss: {:?}", loss);

        // perplexity
        let perplexity = loss.exp()?;
        println!("perplexity: {:?}", perplexity);
        Ok(())
    }
}

/// # Split text into train and validation datasets and loaders
///
/// #### Id
/// 05.03
///
/// #### Page
/// This example starts on page 141
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.03
///
/// # with cuda
/// cargo run --features cuda example 05.03
/// ```
pub struct EG03;

impl Example for EG03 {
    fn description(&self) -> String {
        String::from("Split text into train and validation datasets and loaders.")
    }

    fn page_source(&self) -> usize {
        141_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch02::DataLoader;

        let (train_loader, val_loader) = addons::get_train_val_data_loaders(true)?;

        let mut train_batcher = train_loader.batcher();
        let mut val_batcher = val_loader.batcher();

        println!("Train loader:");
        while let Some(Ok((x, y))) = train_batcher.next() {
            println!("{:?}, {:?}", x.shape(), y.shape())
        }

        println!("Valdiation loader:");
        while let Some(Ok((x, y))) = val_batcher.next() {
            println!("{:?}, {:?}", x.shape(), y.shape())
        }
        Ok(())
    }
}

/// # Example usage of `calc_loss_loader
///
/// #### Id
/// 05.04
///
/// #### Page
/// This example starts on page 145
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.04
///
/// # with cuda
/// cargo run --features cuda example 05.04
/// ```
pub struct EG04;

impl Example for EG04 {
    fn description(&self) -> String {
        String::from("Example usage of `calc_loss_loader`.")
    }

    fn page_source(&self) -> usize {
        145_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::calc_loss_loader,
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        // construct model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt2_124m();
        let model = GPTModel::new(cfg, vb.pp("model"))?;

        // build train and val loaders with utility function from addons module
        let (train_loader, val_loader) = addons::get_train_val_data_loaders(false)?;

        // compute train and val loss
        let train_loss = calc_loss_loader(&train_loader, &model, vb.device(), None)?;
        let val_loss = calc_loss_loader(&val_loader, &model, vb.device(), None)?;

        println!("Training loss {:?}", train_loss);
        println!("Validation loss {:?}", val_loss);
        Ok(())
    }
}

/// # Example usage of `train_model_simple` function and plotting loss curves
///
/// #### Id
/// 05.05
///
/// #### Page
/// This example starts on page 149
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.05
///
/// # with cuda
/// cargo run --features cuda example 05.05
/// ```
pub struct EG05;

impl Example for EG05 {
    fn description(&self) -> String {
        String::from("Example usage of `train_model_simple` function and plotting loss curves.")
    }

    fn page_source(&self) -> usize {
        149_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{generate_text_simple, Config, GPTModel},
            ch05::{plot_losses, text_to_token_ids, token_ids_to_text, train_model_simple},
        };
        use candle_core::{DType, Device};
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
        use ndarray::linspace;
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt2_124m();
        let model = GPTModel::new(Config::gpt2_124m(), vb.pp("model"))?;
        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.0004,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;
        let tokenizer = get_bpe_from_model("gpt2")?;
        let (eval_freq, eval_iter, num_epochs) = (5_usize, 5_usize, 10_usize);
        let (train_loader, val_loader) = addons::get_train_val_data_loaders(false)?;
        let start_context = "Every effort moves you";
        let (train_losses, val_losses, tokens_seen) = train_model_simple(
            &model,
            &train_loader,
            &val_loader,
            optimizer,
            vb.device(),
            num_epochs,
            eval_freq,
            eval_iter,
            start_context,
            &tokenizer,
            None,
        )?;

        // run inference with trained model using deterministic decoding
        let token_ids = generate_text_simple(
            &model,
            text_to_token_ids(start_context, &tokenizer, vb.device())?,
            25,
            cfg.context_length,
        )?;

        // should be the same as the last output generation during training
        println!(
            "Output text:\n{:?}",
            token_ids_to_text(token_ids, &tokenizer)
        );

        // plot loss curves
        println!("Saving weights to `./pretraining_loss.html`");
        let epochs_seen = Vec::from_iter(linspace(0_f32, num_epochs as f32, train_losses.len()));
        let tokens_seen = tokens_seen
            .into_iter()
            .map(|el| el as f32)
            .collect::<Vec<_>>();
        let save_path = Path::new("pretraining_loss.html").to_path_buf();
        plot_losses(
            epochs_seen,
            tokens_seen,
            train_losses,
            val_losses,
            save_path,
        )?;

        Ok(())
    }
}

/// # Manual multinomial with/without temperature scaling
///
/// #### Id
/// 05.06
///
/// #### Page
/// This example starts on page 152
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.06
///
/// # with cuda
/// cargo run --features cuda example 05.06
/// ```
pub struct EG06;

impl Example for EG06 {
    fn description(&self) -> String {
        String::from("Manual multinomial with/without temperature scaling decoding example.")
    }

    fn page_source(&self) -> usize {
        152_usize
    }

    #[allow(unused_variables)]
    fn main(&self) -> Result<()> {
        use crate::listings::ch05::{print_sampled_tokens, sample_multinomial};
        use candle_core::D;
        use candle_nn::ops::softmax;
        use rand::{rngs::StdRng, SeedableRng};

        let (vocab, inverse_vocab) = addons::get_vocab_and_inversed_vocab();
        let next_token_logits = addons::get_next_token_logits()?;

        let probas = softmax(&next_token_logits, D::Minus1)?;

        // greedy sampling
        let next_token_id = probas.argmax(D::Minus1)?;
        println!(
            "Greedy sampling next token: {:?}",
            inverse_vocab.get(&next_token_id.to_scalar::<u32>()?)
        );

        // multinomial sampling
        let mut rng = StdRng::seed_from_u64(123_u64);
        let next_token_id = sample_multinomial(&mut rng, &probas.to_vec1::<f32>()?)?;
        println!(
            "Multinomial samping next token: {:?}",
            inverse_vocab.get(&next_token_id)
        );

        // temperature scaling
        let temp = 0.1;
        let scaled_logits = (next_token_logits / temp)?;
        let scaled_probas = softmax(&scaled_logits, D::Minus1)?;
        let next_token_id = sample_multinomial(&mut rng, &scaled_probas.to_vec1::<f32>()?)?;
        println!(
            "Temp (temp=0.1) scaled multinomial sampling next token: {:?}",
            inverse_vocab.get(&next_token_id)
        );

        // generate multinomial random sample
        println!("Temp (temp=1.0) scaling sampling conducted 1000 times:");
        let with_expected_vals = false;
        print_sampled_tokens(
            &probas.to_vec1::<f32>()?,
            &inverse_vocab,
            with_expected_vals, // this is set in Exercise 5.1
        )?;
        Ok(())
    }
}

/// # Example of extracting topk probas
///
/// #### Id
/// 05.06
///
/// #### Page
/// This example starts on page 156
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.07
///
/// # with cuda
/// cargo run --features cuda example 05.07
/// ```
pub struct EG07;

impl Example for EG07 {
    fn description(&self) -> String {
        String::from("Example of extracting topk probas.")
    }

    fn page_source(&self) -> usize {
        156_usize
    }

    #[allow(dead_code, unused_variables)]
    fn main(&self) -> Result<()> {
        use crate::candle_addons::TopK;
        use candle_core::{Tensor, D};
        use candle_nn::ops::softmax;

        let (vocab, inverse_vocab) = addons::get_vocab_and_inversed_vocab();
        let next_token_logits = addons::get_next_token_logits()?;

        // top-k logits
        let top_k = 3_usize;
        let (top_logits, top_pos) = next_token_logits.topk_last_dim0(top_k)?;
        println!("Top logits: {:?}", top_logits.to_vec1::<f32>());
        println!("Top pos: {:?}", top_pos.to_vec1::<u32>());

        // masking to get new logits
        let mask = next_token_logits.broadcast_lt(&top_logits.min(D::Minus1)?)?;
        let on_true = next_token_logits
            .ones_like()?
            .broadcast_mul(&Tensor::new(f32::NEG_INFINITY, next_token_logits.device())?)?;
        let new_logits = mask.where_cond(&on_true, &next_token_logits)?;
        println!("mask: {:?}", mask);
        println!("new_logits: {:?}", new_logits);

        // get top-k probas
        let topk_probas = softmax(&new_logits, D::Minus1)?;
        println!("probas: {:?}", topk_probas);
        Ok(())
    }
}

/// # Example usage of `generate`
///
/// #### Id
/// 05.08
///
/// #### Page
/// This example starts on page 158
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.08
///
/// # with cuda
/// cargo run --features cuda example 05.08
/// ```
pub struct EG08;

impl Example for EG08 {
    fn description(&self) -> String {
        String::from("Example usage of `generate`.")
    }

    fn page_source(&self) -> usize {
        158_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::{generate, text_to_token_ids, token_ids_to_text},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use rand::{rngs::StdRng, SeedableRng};
        use tiktoken_rs::get_bpe_from_model;

        // construct model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt2_124m();
        let model = GPTModel::new(Config::gpt2_124m(), vb.pp("model"))?;

        // sample setup and load tokenizer
        let start_context = "Every effort moves you";
        let tokenizer = get_bpe_from_model("gpt2")?;

        // generate next tokens with model
        let mut rng = StdRng::seed_from_u64(42_u64);
        let token_ids = generate(
            &model,
            text_to_token_ids(start_context, &tokenizer, vb.device())?,
            15_usize,
            cfg.context_length,
            Some(1.4_f64),
            Some(25_usize),
            None,
            &mut rng,
        )?;

        // decode the token ids to print the output text
        println!(
            "Output text:\n{:?}",
            token_ids_to_text(token_ids, &tokenizer)
        );
        Ok(())
    }
}

/// # Saving and loading a candle model
///
/// #### Id
/// 05.09
///
/// #### Page
/// This example starts on page 159
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.09
///
/// # with cuda
/// cargo run --features cuda example 05.09
/// ```
pub struct EG09;

impl Example for EG09 {
    fn description(&self) -> String {
        String::from("Saving and loading a candle model.")
    }

    fn page_source(&self) -> usize {
        159_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::train_model_simple,
        };
        use candle_core::{DType, Device, Error, IndexOp};
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
        use tiktoken_rs::get_bpe_from_model;

        // construt model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt2_124m();
        let model = GPTModel::new(cfg, vb.pp("model"))?;
        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.0004,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;

        // train model for an epoch
        let tokenizer = get_bpe_from_model("gpt2")?;
        let (eval_freq, eval_iter, num_epochs) = (5_usize, 5_usize, 1_usize);
        let (train_loader, val_loader) = addons::get_train_val_data_loaders(false)?;
        let start_context = "Every effort moves you";
        let _ = train_model_simple(
            &model,
            &train_loader,
            &val_loader,
            optimizer,
            vb.device(),
            num_epochs,
            eval_freq,
            eval_iter,
            start_context,
            &tokenizer,
            None,
        );

        // save weights
        println!(
            "model.out_head.weight first 10 weights BEFORE save: {:?}",
            varmap
                .data()
                .lock()
                .unwrap()
                .get("model.out_head.weight")
                .ok_or_else(|| {
                    Error::CannotFindTensor {
                        path: "model.out_head.weight".to_string(),
                    }
                    .bt()
                })?
                .i((1, ..10))?
                .to_vec1::<f32>()
        );

        println!("Saving weights to `./checkpoint.safetensors`");
        varmap.save("checkpoint.safetensors")?;

        // construct a new copy of the model and its varmap
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let _model = GPTModel::new(cfg, vb.pp("model"))?;
        println!(
            "model.out_head.weight first 10 weights BEFORE load: {:?}",
            varmap
                .data()
                .lock()
                .unwrap()
                .get("model.out_head.weight")
                .ok_or_else(|| {
                    Error::CannotFindTensor {
                        path: "model.out_head.weight".to_string(),
                    }
                    .bt()
                })?
                .i((1, ..10))?
                .to_vec1::<f32>()
        );

        // load the saved weights into the new model copy
        println!("Loading weights from `./checkpoint.safetensors`");
        varmap.load("checkpoint.safetensors")?;
        println!(
            "model.out_head.weight first 10 weights AFTER load: {:?}",
            varmap
                .data()
                .lock()
                .unwrap()
                .get("model.out_head.weight")
                .ok_or_else(|| {
                    Error::CannotFindTensor {
                        path: "model.out_head.weight".to_string(),
                    }
                    .bt()
                })?
                .i((1, ..10))?
                .to_vec1::<f32>()
        );
        Ok(())
    }
}

/// # Example for downloading safetensors from HuggingFace Hub
///
/// #### Id
/// 05.10
///
/// #### Page
/// This example starts on page 161
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.10
///
/// # with cuda
/// cargo run --features cuda example 05.10
/// ```
pub struct EG10;

impl Example for EG10 {
    fn description(&self) -> String {
        String::from("Example for downloading safetensors from HuggingFace Hub.")
    }

    fn page_source(&self) -> usize {
        161_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch04::Config;
        use candle_core::Device;
        use hf_hub::api::sync::Api;

        let api = Api::new()?;
        let repo = api.model("openai-community/gpt2".to_string());
        let weights = repo.get("model.safetensors")?;
        let weights = candle_core::safetensors::load(weights, &Device::Cpu)?;

        // update config
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;

        println!("{:?}", cfg);

        println!("{:?}", weights);
        Ok(())
    }
}

/// # Example usage of `load_weights_into_gpt`
///
/// #### Id
/// 05.11
///
/// #### Page
/// This example starts on page 167
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example 05.11
///
/// # with cuda
/// cargo run --features cuda example 05.11
/// ```
pub struct EG11;

impl Example for EG11 {
    fn description(&self) -> String {
        String::from("Example usage of `load_weights_into_gpt`.")
    }

    fn page_source(&self) -> usize {
        167_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::{generate, load_weights_into_gpt, text_to_token_ids, token_ids_to_text},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use hf_hub::api::sync::Api;
        use rand::{rngs::StdRng, SeedableRng};
        use tiktoken_rs::get_bpe_from_model;

        let dev = Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let model = GPTModel::new(cfg, vb.pp("model"))?;

        // get weights from HF Hub
        let api = Api::new()?;
        let repo = api.model("openai-community/gpt2".to_string());
        let weights = repo.get("model.safetensors")?;
        let weights = candle_core::safetensors::load(weights, &dev)?;

        // load weights
        load_weights_into_gpt(&varmap, weights, Some("model"), cfg.n_layers)?;

        // sample setup and load tokenizer
        let start_context = "Every effort moves you";
        let tokenizer = get_bpe_from_model("gpt2")?;

        let mut rng = StdRng::seed_from_u64(42_u64);
        let token_ids = generate(
            &model,
            text_to_token_ids(start_context, &tokenizer, vb.device())?,
            25_usize,
            cfg.context_length,
            Some(0.1_f64),
            Some(50_usize),
            None,
            &mut rng,
        )?;

        // decode the token ids to print the output text
        println!(
            "Output text:\n{:?}",
            token_ids_to_text(token_ids, &tokenizer)?
        );
        Ok(())
    }
}

pub mod addons {
    //! Auxiliary module for examples::ch05
    use crate::listings::ch02::GPTDataLoader;
    use candle_core::{Device, IndexOp, Result, Tensor};
    use std::collections::HashMap;

    /// Helper function to target_tokens of the _i-th_ input sequence
    pub fn get_target_token_probas_helper(
        text_idx: usize,
        targets: &Tensor,
        probas: &Tensor,
        dev: &Device,
    ) -> Result<Tensor> {
        let target_tokens_1 = targets.i(text_idx)?.to_vec1::<u32>()?;
        let mut target_probas_1: Vec<f32> = vec![];
        for (i, target_token) in target_tokens_1.iter().enumerate() {
            let target_proba = probas
                .i((text_idx, i, *target_token as usize))?
                .to_scalar::<f32>()?;
            target_probas_1.push(target_proba);
        }
        Tensor::from_vec(target_probas_1, target_tokens_1.len(), dev)
    }

    /// Helper function for producing `GPTDataLoader` for train and val splits
    pub fn get_train_val_data_loaders(
        verbose: bool,
    ) -> anyhow::Result<(GPTDataLoader, GPTDataLoader)> {
        use crate::listings::{ch02::create_dataloader_v1, ch04::Config};
        use std::fs;
        use tiktoken_rs::get_bpe_from_model;

        // load the verdict short story and compute stats
        let text_data =
            fs::read_to_string("data/the-verdict.txt").expect("Unable to read the file");
        let total_characters = text_data.len();
        let tokenizer = get_bpe_from_model("gpt2")?;
        let total_tokens = tokenizer
            .encode_with_special_tokens(text_data.as_str())
            .len();
        if verbose {
            println!("Characters: {:?}", total_characters);
            println!("Tokens: {:?}", total_tokens);
        }

        // establish train and val data
        let train_ratio = 0.90_f32;
        let split_idx = (train_ratio * text_data.len() as f32) as usize;
        let train_data = &text_data[..split_idx];
        let val_data = &text_data[split_idx..];

        // build train and val GPTDatasetV1 and batchers
        let mut cfg = Config::gpt2_124m();
        cfg.context_length = 256_usize;

        let batch_size = 2_usize;
        let max_length = cfg.context_length;
        let stride = cfg.context_length;

        let train_loader =
            create_dataloader_v1(train_data, batch_size, max_length, stride, true, true);
        let val_loader =
            create_dataloader_v1(val_data, batch_size, max_length, stride, false, false);

        Ok((train_loader, val_loader))
    }

    /// Helper function to get vocab and inversed vocab `HashMap`'s
    pub fn get_vocab_and_inversed_vocab() -> (HashMap<&'static str, u32>, HashMap<u32, &'static str>)
    {
        let vocab = HashMap::from([
            ("closer", 0_u32),
            ("every", 1),
            ("effort", 2),
            ("forward", 3),
            ("inches", 4),
            ("moves", 5),
            ("pizza", 6),
            ("toward", 7),
            ("you", 8),
        ]);
        let inverse_vocab = vocab
            .iter()
            .map(|(k, v)| (*v, *k))
            .collect::<HashMap<u32, &str>>();
        (vocab, inverse_vocab)
    }

    /// Helper function to get the example next token logits used in the book
    ///
    /// ```rust
    /// use candle_core::{Device, Tensor};
    ///
    /// let dev = Device::cuda_if_available(0).unwrap();
    /// let next_token_logits = Tensor::new(
    ///     &[4.51_f32, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79],
    ///     &dev,
    /// );
    /// ```
    pub fn get_next_token_logits() -> Result<Tensor> {
        #![allow(clippy::approx_constant)]
        let dev = Device::cuda_if_available(0)?;
        Tensor::new(
            &[4.51_f32, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79],
            &dev,
        )
    }
}
