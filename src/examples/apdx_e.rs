//! Examples from Appendix E

use crate::Example;
use anyhow::{Context, Result};

/// # Example usage of `create_candle_dataloaders`
///
/// #### Id
/// E.01
///
/// #### Page
/// This example starts on page 326
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example E.01
///
/// # with cuda
/// cargo run --features cuda example E.01
/// ```
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        "Example usage of `create_candle_dataloaders`.".to_string()
    }

    fn page_source(&self) -> usize {
        326_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::apdx_e::create_candle_dataloaders;

        let batch_size = 8_usize;
        let (train_loader, val_loader, test_loader) = create_candle_dataloaders(batch_size)?;

        // print last batch of train loader
        let (input_batch, target_batch) = train_loader.batcher().last().unwrap()?;
        println!("Input batch dimensions: {:?}", input_batch.shape());
        println!("Label batch dimensions: {:?}", target_batch.shape());

        // print total number of batches in each data loader
        println!("{:?} training batches", train_loader.len());
        println!("{:?} validation batches", val_loader.len());
        println!("{:?} test batches", test_loader.len());

        Ok(())
    }
}

/// # Example usage of `download_and_load_gpt2` and attaching spam classification head
///
/// #### Id
/// E.02
///
/// #### Page
/// This example starts on page 327
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example E.02
///
/// # with cuda
/// cargo run --features cuda example E.02
/// ```
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        "Example usage of `download_and_load_gpt2` and attaching spam classification head."
            .to_string()
    }

    fn page_source(&self) -> usize {
        327_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            apdx_e::{create_candle_dataloaders, download_and_load_gpt2},
            ch04::Config,
            ch05::{generate, text_to_token_ids, token_ids_to_text},
            ch06::{calc_accuracy_loader, modify_out_head_for_classification, HF_GPT2_MODEL_ID},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use rand::{rngs::StdRng, SeedableRng};
        use tiktoken_rs::get_bpe_from_model;

        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;

        // sample setup and load tokenizer
        let tokenizer = get_bpe_from_model("gpt2")?;
        let mut rng = StdRng::seed_from_u64(42_u64);

        // generate next tokens with model
        let text_1 = "Every effort moves you";
        let token_ids = generate(
            &model,
            text_to_token_ids(text_1, &tokenizer, vb.device())?,
            15_usize,
            cfg.context_length,
            None,
            None,
            None,
            &mut rng,
        )?;

        // decode the token ids to print the output text
        println!("{:?}", token_ids_to_text(token_ids, &tokenizer));

        // attach spam classification head
        let num_classes = 2_usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        // calc classification accuracy
        let batch_size = 8_usize;
        let (train_loader, val_loader, test_loader) = create_candle_dataloaders(batch_size)?;

        // compute accuracies
        let num_batches = Some(10_usize);
        let train_accuracy =
            calc_accuracy_loader(&train_loader, &model, vb.device(), num_batches, None)?;
        let val_accuracy =
            calc_accuracy_loader(&val_loader, &model, vb.device(), num_batches, None)?;
        let test_accuracy =
            calc_accuracy_loader(&test_loader, &model, vb.device(), num_batches, None)?;

        println!("Training accuracy: {}", train_accuracy);
        println!("Validation accuracy: {}", val_accuracy);
        println!("Test accuracy: {}", test_accuracy);

        Ok(())
    }
}

/// # Example usage of `GPTModelWithLoRA::from_gpt()` and extracting the LoRA trainable vars
///
/// #### Id
/// E.03
///
/// #### Page
/// This example starts on page 331
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example E.03
///
/// # with cuda
/// cargo run --features cuda example E.03
/// ```
pub struct EG03;

impl Example for EG03 {
    fn description(&self) -> String {
        let desc = "Example usage of `GPTModelWithLoRA::from_gpt()` and \
        extracting the LoRA trainable vars";
        desc.to_string()
    }

    fn page_source(&self) -> usize {
        331_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            apdx_e::{download_and_load_gpt2, GPTModelWithLoRA},
            ch04::Config,
            ch06::{modify_out_head_for_classification, HF_GPT2_MODEL_ID},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;

        // modify to use classification head
        let num_classes = 2_usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        // get total number of params from the VarMap (todo: turn this into a util)
        let mut total_params_without_lora_weights = 0_usize;
        for t in varmap.all_vars().iter() {
            total_params_without_lora_weights += t.elem_count();
        }
        println!(
            "Total number of parameters of original model: {}",
            total_params_without_lora_weights
        );

        // convert to LoRA model
        let rank = 16_usize;
        let alpha = 16_f64;
        let _model = GPTModelWithLoRA::from_gpt_model(model, rank, alpha, vb.pp("model"))?;

        // extract only LoRA weights
        let mut total_training_params = 0_usize; // i.e., LoRA weights
        let tensor_data = varmap.data().lock().unwrap();
        let var_names: Vec<&String> = tensor_data
            .keys()
            .filter(|k| k.contains("A") || k.contains("B"))
            .collect();
        for var_name in var_names.into_iter() {
            let var = tensor_data.get(var_name).unwrap();
            total_training_params += var.elem_count();
        }
        drop(tensor_data);

        println!("Total trainable LoRA parameters: {}", total_training_params);

        Ok(())
    }
}

/// # Printing GPTModelWithLoRA architecture
///
/// #### Id
/// E.04
///
/// #### Page
/// This example starts on page 332
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example E.04
///
/// # with cuda
/// cargo run --features cuda example E.04
/// ```
pub struct EG04;

impl Example for EG04 {
    fn description(&self) -> String {
        "Printing GPTModelWithLoRA architecture.".to_string()
    }

    fn page_source(&self) -> usize {
        332_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            apdx_e::{download_and_load_gpt2, GPTModelWithLoRA},
            ch04::Config,
            ch06::{modify_out_head_for_classification, HF_GPT2_MODEL_ID},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;

        // modify to use classification head
        let num_classes = 2_usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        // convert to LoRA model
        let rank = 16_usize;
        let alpha = 16_f64;
        let model = GPTModelWithLoRA::from_gpt_model(model, rank, alpha, vb.pp("model"))?;

        // pretty debug print
        println!("{:#?}", model);

        Ok(())
    }
}

/// # Calculating initial classification accuracies
///
/// #### Id
/// E.05
///
/// #### Page
/// This example starts on page 333
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example E.05
///
/// # with cuda
/// cargo run --features cuda example E.05
/// ```
pub struct EG05;

impl Example for EG05 {
    fn description(&self) -> String {
        "Calculating initial classification accuracies.".to_string()
    }

    fn page_source(&self) -> usize {
        333_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            apdx_e::{create_candle_dataloaders, download_and_load_gpt2, GPTModelWithLoRA},
            ch04::Config,
            ch06::{calc_accuracy_loader, modify_out_head_for_classification, HF_GPT2_MODEL_ID},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;

        // modify to use classification head
        let num_classes = 2_usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        // convert to LoRA model
        let rank = 16_usize;
        let alpha = 16_f64;
        let model = GPTModelWithLoRA::from_gpt_model(model, rank, alpha, vb.pp("model"))?;

        // calc classification accuracy
        let batch_size = 8_usize;
        let (train_loader, val_loader, test_loader) = create_candle_dataloaders(batch_size)?;

        // compute accuracies
        let num_batches = Some(10_usize);
        let train_accuracy =
            calc_accuracy_loader(&train_loader, &model, vb.device(), num_batches, None)?;
        let val_accuracy =
            calc_accuracy_loader(&val_loader, &model, vb.device(), num_batches, None)?;
        let test_accuracy =
            calc_accuracy_loader(&test_loader, &model, vb.device(), num_batches, None)?;

        println!("Training accuracy: {}", train_accuracy);
        println!("Validation accuracy: {}", val_accuracy);
        println!("Test accuracy: {}", test_accuracy);

        Ok(())
    }
}

/// # Fine-tuning a model with LoRA layers
///
/// NOTE: technically this Listing 7.1 in the book, but we felt it was better
/// as an Example.
///
/// #### Id
/// E.06
///
/// #### Page
/// This example starts on page 334
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example E.06
///
/// # with cuda
/// cargo run --features cuda example E.06
/// ```
pub struct EG06;

impl Example for EG06 {
    fn description(&self) -> String {
        "Fine-tuning a model with LoRA layers.".to_string()
    }

    fn page_source(&self) -> usize {
        334_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            apdx_e::{create_candle_dataloaders, train_classifier_simple, GPTModelWithLoRA},
            ch04::Config,
            ch06::{
                download_and_load_gpt2, modify_out_head_for_classification, plot_values,
                HF_GPT2_MODEL_ID,
            },
        };
        use candle_core::{DType, Device, Var};
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
        use ndarray::linspace;
        use std::path::Path;

        // get gpt model with classification head
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;

        // modify to use classification head
        let num_classes = 2_usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        // convert to LoRA model
        let rank = 16_usize;
        let alpha = 16_f64;
        let model = GPTModelWithLoRA::from_gpt_model(model, rank, alpha, vb.pp("model"))?;

        // data loaders
        let batch_size = 2_usize; // Get OOM on my Tesla P100 (12GB) with 8_usize
        let (train_loader, val_loader, _test_loader) = create_candle_dataloaders(batch_size)?;

        // extract only LoRA weights as trainable params
        let mut training_vars: Vec<Var> = vec![];
        let tensor_data = varmap.data().lock().unwrap();
        let var_names: Vec<&String> = tensor_data
            .keys()
            .filter(|k| k.contains("A") || k.contains("B"))
            .collect();

        println!("Training variables: {:?}\n", var_names);

        for var_name in var_names.into_iter() {
            let var = tensor_data.get(var_name).unwrap();
            training_vars.push(var.clone());
        }
        drop(tensor_data);

        // train model
        let optimizer = AdamW::new(
            training_vars,
            ParamsAdamW {
                lr: 5e-5,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;

        let (eval_freq, eval_iter, num_epochs) = (50_usize, 5_usize, 5_usize);
        let (train_loss, val_loss, train_accs, val_accs, num_examples) = train_classifier_simple(
            &model,
            &train_loader,
            &val_loader,
            optimizer,
            vb.device(),
            num_epochs,
            eval_freq,
            eval_iter,
            None,
        )?;

        // save model
        println!("Saving weights to `./clf.gptwithlora.checkpoint.safetensors`");
        varmap.save("clf.gptwithlora.checkpoint.safetensors")?;

        // prepare and save plots
        let epochs_seen = Vec::from_iter(linspace(0_f32, num_epochs as f32, train_loss.len()));
        let examples_seen = Vec::from_iter(linspace(0_f32, num_examples as f32, train_loss.len()));
        let label = "loss";
        let save_path = Path::new(format!("plot_classification_gptwithlora_{label}.html").as_str())
            .to_path_buf();
        plot_values(
            epochs_seen,
            examples_seen,
            train_loss,
            val_loss,
            label,
            save_path,
        )?;

        let epochs_seen = Vec::from_iter(linspace(0_f32, num_epochs as f32, train_accs.len()));
        let examples_seen = Vec::from_iter(linspace(0_f32, num_examples as f32, train_accs.len()));
        let label = "accuracy";
        let save_path = Path::new(format!("plot_classification_gptwithlora_{label}.html").as_str())
            .to_path_buf();
        plot_values(
            epochs_seen,
            examples_seen,
            train_accs,
            val_accs,
            label,
            save_path,
        )?;

        Ok(())
    }
}

/// # Evaluating trained LoRA model on train, validation, and test sets
///
/// #### Id
/// E.07
///
/// #### Page
/// This example starts on page 335
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example E.07
///
/// # with cuda
/// cargo run --features cuda example E.07
/// ```
pub struct EG07;

impl Example for EG07 {
    fn description(&self) -> String {
        "Evaluating trained LoRA model on train, validation, and test sets.".to_string()
    }

    fn page_source(&self) -> usize {
        335_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            apdx_e::{create_candle_dataloaders, GPTModelWithLoRA},
            ch04::Config,
            ch06::{
                calc_accuracy_loader, download_and_load_gpt2, modify_out_head_for_classification,
                HF_GPT2_MODEL_ID,
            },
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        // get gpt model with classification head
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;

        // modify to use classification head
        let num_classes = 2_usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        // convert to LoRA model
        let rank = 16_usize;
        let alpha = 16_f64;
        let model = GPTModelWithLoRA::from_gpt_model(model, rank, alpha, vb.pp("model"))?;

        // load safetensors
        varmap
            .load("clf.gptwithlora.checkpoint.safetensors")
            .with_context(|| {
                "Missing 'clf.gptwithlora.checkpoint.safetensors' file. Please run EG E.06."
            })?;

        // data loaders
        let batch_size = 2_usize; // Get OOM on my Tesla P100 (12GB) with 8_usize
        let (train_loader, val_loader, test_loader) = create_candle_dataloaders(batch_size)?;

        // compute accuracies
        let num_batches = None;
        let train_accuracy =
            calc_accuracy_loader(&train_loader, &model, vb.device(), num_batches, None)?;
        let val_accuracy =
            calc_accuracy_loader(&val_loader, &model, vb.device(), num_batches, None)?;
        let test_accuracy =
            calc_accuracy_loader(&test_loader, &model, vb.device(), num_batches, None)?;

        println!("Training accuracy: {}", train_accuracy);
        println!("Validation accuracy: {}", val_accuracy);
        println!("Test accuracy: {}", test_accuracy);

        Ok(())
    }
}
