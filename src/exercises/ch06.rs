//! Exercises from Chapter 6

use crate::Exercise;
use anyhow::Result;

/// # Increasing the context length
///
/// #### Id
/// 6.1
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run exercise 6.1
///
/// # with cuda
/// cargo run --features cuda exercise 6.1
/// ```
pub struct X1;

impl Exercise for X1 {
    fn name(&self) -> String {
        String::from("6.1")
    }

    fn title(&self) -> String {
        String::from("Increasing the context length")
    }

    fn statement(&self) -> String {
        let stmt = "Pad the inputs to the maximum number of tokens the model \
        supports and observe how it affects the predictive performance.";
        stmt.to_string()
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch06::{
                calc_accuracy_loader, download_and_load_gpt2, modify_out_head_for_classification,
                train_classifier_simple, SpamDataLoader, SpamDatasetBuilder, HF_GPT2_MODEL_ID,
            },
        };
        use anyhow::anyhow;
        use candle_core::{DType, Device, Var};
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
        use std::ops::Not;
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        println!("Creating train, val, test datasets");
        // create datasets
        let tokenizer = get_bpe_from_model("gpt2")?;
        let max_length = Some(512_usize);

        let train_path = Path::new("data").join("train.parquet");
        if train_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/train.parquet' file. Please run EG 06.04."
            ));
        }
        let train_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_parquet(train_path)
            .max_length(max_length)
            .build();
        println!(
            "...train dataset max length: {}",
            train_dataset.max_length()
        );

        let val_path = Path::new("data").join("validation.parquet");
        if val_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/validation.parquet' file. Please run EG 06.04."
            ));
        }
        let val_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_parquet(val_path)
            .max_length(max_length)
            .build();
        println!("...val dataset max length: {}", val_dataset.max_length());

        let test_path = Path::new("data").join("test.parquet");
        if test_path.exists().not() {
            return Err(anyhow!(
                "Missing 'data/test.parquet' file. Please run EG 06.04."
            ));
        }
        let test_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_parquet(test_path)
            .max_length(max_length)
            .build();
        println!("...test dataset max length: {}", test_dataset.max_length());

        // create loaders
        let batch_size = 2_usize;
        let train_loader = SpamDataLoader::new(train_dataset, batch_size, true, true);
        let val_loader = SpamDataLoader::new(val_dataset, batch_size, false, false);
        let test_loader = SpamDataLoader::new(test_dataset, batch_size, false, false);

        // print total number of batches in each data loader
        println!("...{:?} training batches", train_loader.len());
        println!("...{:?} validation batches", val_loader.len());
        println!("...{:?} test batches", test_loader.len());

        // get model
        println!("Loading pre-trained GPT-2 and modifying prediction head");
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;
        modify_out_head_for_classification(&mut model, cfg, 2_usize, &varmap, vb.pp("model"))?;

        // train model
        // trainable: last trf block, final layer norm, classification head
        let mut training_vars: Vec<Var> = vec![];
        let tensor_data = varmap.data().lock().unwrap();
        let var_names: Vec<&String> = tensor_data
            .keys()
            .filter(|k| k.contains("final_norm") || k.contains("out_head") || k.contains("trf.11"))
            .collect();
        for var_name in var_names.into_iter() {
            let var = tensor_data.get(var_name).unwrap();
            training_vars.push(var.clone());
        }
        drop(tensor_data);

        let optimizer = AdamW::new(
            training_vars,
            ParamsAdamW {
                lr: 5e-5,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;

        println!("Fine-tuning GPT2 on spam training dataset");
        let (eval_freq, eval_iter, num_epochs) = (50_usize, 1_usize, 2_usize);
        let _ = train_classifier_simple(
            &model,
            &train_loader,
            &val_loader,
            optimizer,
            vb.device(),
            num_epochs,
            eval_freq,
            eval_iter,
            None,
        );

        println!("Computing performance metrics");
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

/// # Fine-tuning the whole model
///
/// #### Id
/// 6.2
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run exercise 6.2
///
/// # with cuda
/// cargo run --features cuda exercise 6.2
/// ```
pub struct X2;

impl Exercise for X2 {
    fn name(&self) -> String {
        "6.2".to_string()
    }

    fn title(&self) -> String {
        "Fine-tuning the whole model".to_string()
    }

    fn statement(&self) -> String {
        let stmt = "Instead of fine-tuning just the final transformer \
        block, fine-tune the entire model and assess the effect on predictive \
        performance.";
        stmt.to_string()
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch06::{
                calc_accuracy_loader, download_and_load_gpt2, modify_out_head_for_classification,
                train_classifier_simple, HF_GPT2_MODEL_ID,
            },
        };
        use candle_core::{DType, Device};
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

        // get gpt model with classification head
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;
        modify_out_head_for_classification(&mut model, cfg, 2_usize, &varmap, vb.pp("model"))?;

        // get data loaders
        let eg06 = crate::examples::ch06::EG06; // re-use
        let (train_loader, val_loader, test_loader) = eg06.main_with_return(false)?;

        // trainable params and optimizer
        let optimizer = AdamW::new(
            varmap.all_vars(), // train on all vars
            ParamsAdamW {
                lr: 5e-5,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;

        println!("Fine-tuning GPT2 on spam training dataset");
        let (eval_freq, eval_iter, num_epochs) = (50_usize, 5_usize, 5_usize);
        let _ = train_classifier_simple(
            &model,
            &train_loader,
            &val_loader,
            optimizer,
            vb.device(),
            num_epochs,
            eval_freq,
            eval_iter,
            None,
        );

        println!("Computing performance metrics");
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

/// # Fine-tuning the first vs. last token
///
/// #### Id
/// 6.3
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run exercise 6.3
///
/// # with cuda
/// cargo run --features cuda exercise 6.3
/// ```
pub struct X3;

impl Exercise for X3 {
    fn name(&self) -> String {
        "6.3".to_string()
    }

    fn title(&self) -> String {
        "Fine-tuning the first vs. last token".to_string()
    }

    fn statement(&self) -> String {
        let stmt = "Try fine-tuning the first output token. Notice the \
        changes in predictive performance compared to fine-tuning the last \
        output token.";
        stmt.to_string()
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch06::{
                calc_accuracy_loader, download_and_load_gpt2, modify_out_head_for_classification,
                train_classifier_simple, HF_GPT2_MODEL_ID,
            },
        };
        use candle_core::{DType, Device, Var};
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

        // get gpt model with classification head
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let mut model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, HF_GPT2_MODEL_ID)?;
        modify_out_head_for_classification(&mut model, cfg, 2_usize, &varmap, vb.pp("model"))?;

        // get data loaders
        let eg06 = crate::examples::ch06::EG06; // re-use
        let (train_loader, val_loader, test_loader) = eg06.main_with_return(false)?;

        // trainable params and optimizer
        // trainable: last trf block, final layer norm, classification head
        let mut training_vars: Vec<Var> = vec![];
        let tensor_data = varmap.data().lock().unwrap();
        let var_names: Vec<&String> = tensor_data
            .keys()
            .filter(|k| k.contains("final_norm") || k.contains("out_head") || k.contains("trf.11"))
            .collect();

        println!("Training variables: {:?}\n", var_names);

        for var_name in var_names.into_iter() {
            let var = tensor_data.get(var_name).unwrap();
            training_vars.push(var.clone());
        }
        drop(tensor_data);

        let optimizer = AdamW::new(
            training_vars,
            ParamsAdamW {
                lr: 5e-5,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;

        let (eval_freq, eval_iter, num_epochs) = (50_usize, 5_usize, 5_usize);
        let custom_pred_token_index = Some(0_usize); // use the first token!
        let _ = train_classifier_simple(
            &model,
            &train_loader,
            &val_loader,
            optimizer,
            vb.device(),
            num_epochs,
            eval_freq,
            eval_iter,
            custom_pred_token_index,
        );

        println!("Computing performance metrics");
        // compute accuracies
        let num_batches = None;
        let train_accuracy = calc_accuracy_loader(
            &train_loader,
            &model,
            vb.device(),
            num_batches,
            custom_pred_token_index,
        )?;
        let val_accuracy = calc_accuracy_loader(
            &val_loader,
            &model,
            vb.device(),
            num_batches,
            custom_pred_token_index,
        )?;
        let test_accuracy = calc_accuracy_loader(
            &test_loader,
            &model,
            vb.device(),
            num_batches,
            custom_pred_token_index,
        )?;

        println!("Training accuracy: {}", train_accuracy);
        println!("Validation accuracy: {}", val_accuracy);
        println!("Test accuracy: {}", test_accuracy);

        Ok(())
    }
}
