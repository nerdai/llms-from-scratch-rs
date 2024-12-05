use crate::Exercise;

/// Exercise 5.1
pub struct X1;

impl Exercise for X1 {
    fn name(&self) -> String {
        String::from("5.1")
    }

    fn title(&self) -> String {
        "Printing sampling frequencies with various temperatures".to_string() // title missing from book
    }

    fn statement(&self) -> String {
        let stmt = "Use the `print_sampled_tokens` function to print the \
        sampling frequencies of the softmax probabilities scaled with the \
        temperatures shown in figure 5.14. How often is the word `pizza` sampled \
        in each case? Can you think of a faster and more accurate way to \
        determine how often the word `pizza` is sampled?";
        stmt.to_string()
    }

    fn main(&self) {
        use crate::{examples, listings::ch05::print_sampled_tokens};
        use candle_core::D;
        use candle_nn::ops::softmax;

        let (_vocab, inverse_vocab) = examples::ch05::addons::get_vocab_and_inversed_vocab();
        let next_token_logits = examples::ch05::addons::get_next_token_logits().unwrap();

        let temperatures = &[1_f64, 0.1, 5.];
        for temp in temperatures.iter() {
            println!(
                "Temp (temp={}) scaling sampling conducted 1000 times:",
                temp
            );
            let scaled_logits = (&next_token_logits / temp.to_owned()).unwrap();
            let scaled_probas = softmax(&scaled_logits, D::Minus1).unwrap();
            print_sampled_tokens(
                &scaled_probas.to_vec1::<f32>().unwrap(),
                &inverse_vocab,
                true,
            )
            .unwrap();
            println!("\n");
        }
    }
}

/// Exercise 5.2
pub struct X2;

impl Exercise for X2 {
    fn name(&self) -> String {
        String::from("5.2")
    }

    fn title(&self) -> String {
        "Using various temperatures and top-k values".to_string() // missing from book
    }

    fn statement(&self) -> String {
        let stmt = "Play around with different temperatures and top-k \
        settings. Based on your observations, can you think of applications \
        where lower temperature and top-k settings are desired? Likewise, can \
        you think of applications where higher temperature and top-k settings \
        are preferred? (It’s recommended to also revisit this exercise at the \
        end of the chapter after loading the pretrained weights from OpenAI.)";
        stmt.to_string()
    }

    fn main(&self) {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::{generate, text_to_token_ids, token_ids_to_text},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use itertools::iproduct;
        use rand::{rngs::StdRng, SeedableRng};
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

        let temperatures = &[0.1_f64, 1., 5.];
        let top_ks = &[20_usize, 100, cfg.vocab_size];
        let mut rng = StdRng::seed_from_u64(42_u64);
        for (temp, top_k) in iproduct!(temperatures, top_ks) {
            println!("Temp: {}, Top K: {}", temp, top_k);

            let token_ids = generate(
                &model,
                text_to_token_ids(start_context, &tokenizer, vb.device()).unwrap(),
                15_usize,
                cfg.context_length,
                Some(*temp),
                Some(*top_k),
                None,
                &mut rng,
            )
            .unwrap();

            // decode the token ids to print the output text
            println!("{:?}\n", token_ids_to_text(token_ids, &tokenizer))
        }
    }
}

/// Exercise 5.3
pub struct X3;

impl Exercise for X3 {
    fn name(&self) -> String {
        String::from("5.3")
    }

    fn title(&self) -> String {
        "Parameter values for deterministic sampling".to_string() // missing from book
    }

    fn statement(&self) -> String {
        let stmt = "What are the different combinations of settings for \
        the `generate` function to force deterministic behavior, that is, \
        disabling the random sampling such that it always produces the same \
        outputs similar to the `generate_simple` function?";
        stmt.to_string()
    }

    fn main(&self) {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::{generate, text_to_token_ids},
        };
        use candle_core::{DType, Device, Tensor};
        use candle_nn::{VarBuilder, VarMap};
        use rand::{rngs::StdRng, SeedableRng};
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

        // deterministic settings: temp to None and top_k to any value
        let temp = None;

        let mut old_token_ids: Option<Tensor> = None;
        let mut rng = StdRng::seed_from_u64(42_u64);
        for ix in 0..4 {
            println!("Itertation {}:", ix);

            let token_ids = generate(
                &model,
                text_to_token_ids(start_context, &tokenizer, vb.device()).unwrap(),
                15_usize,
                cfg.context_length,
                temp,
                Some(20usize),
                None,
                &mut rng,
            )
            .unwrap();

            if let Some(old) = old_token_ids {
                println!("old token ids: {:?}", old.to_vec2::<u32>());
            } else {
                println!("old token ids: None");
            }

            println!("new token ids: {:?}\n", token_ids.to_vec2::<u32>());

            old_token_ids = Some(token_ids);
        }
    }
}

/// Exercise 5.4
pub struct X4;

impl Exercise for X4 {
    fn name(&self) -> String {
        String::from("5.4")
    }

    fn title(&self) -> String {
        "Continuing training from pre-loaded weights".to_string() // missing from book
    }

    fn statement(&self) -> String {
        let stmt = "After saving the weights, load the model and optimizer \
        in a new Python session or Jupyter notebook file and continue pretraining \
        it for one more epoch using the `train_model_simple` function.";
        stmt.to_string()
    }

    fn main(&self) {
        use crate::{
            examples,
            listings::{
                ch04::{Config, GPTModel},
                ch05::train_model_simple,
            },
        };
        use candle_core::{DType, Device};
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
        use tiktoken_rs::get_bpe_from_model;

        // construct model
        let mut varmap = VarMap::new();
        let vb =
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0).unwrap());
        let cfg = Config::gpt2_124m();
        let model = GPTModel::new(cfg, vb.pp("model")).unwrap();

        // load from previous checkpoint
        // NOTE: this requires EG 05.09 to be have ran, which creates a model
        // checkpoint that we use here
        println!("Loading weights from `./checkpoint.safetensors`");
        varmap.load("checkpoint.safetensors").unwrap(); // todo map to anyhow error with proper msg

        // train model for one epoch
        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.0004,
                weight_decay: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        let (eval_freq, eval_iter, num_epochs) = (5_usize, 5_usize, 1_usize);
        let (train_loader, val_loader) = examples::ch05::addons::get_train_val_data_loaders(false);
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
        );
    }
}

/// Exercise 5.5
pub struct X5;

impl Exercise for X5 {
    fn name(&self) -> String {
        String::from("5.5")
    }

    fn title(&self) -> String {
        "Training and validation losses with OpenAI weights".to_string() // missing from book
    }

    fn statement(&self) -> String {
        let stmt = "Calculate the training and validation set losses of the \
        `GPTModel` with the pretrained weights from OpenAI on the “The Verdict” \
        dataset.";
        stmt.to_string()
    }

    fn main(&self) {
        use crate::{
            examples,
            listings::{
                ch04::{Config, GPTModel},
                ch05::{calc_loss_loader, load_weights_into_gpt},
            },
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use hf_hub::api::sync::Api;

        let dev = Device::cuda_if_available(0).unwrap();

        // download openai weights
        let api = Api::new().unwrap();
        let repo = api.model("openai-community/gpt2".to_string());
        let weights = repo.get("model.safetensors").unwrap();
        let weights = candle_core::safetensors::load(weights, &dev).unwrap();

        // construct model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let model = GPTModel::new(cfg, vb.pp("model")).unwrap();

        // load openai weights
        load_weights_into_gpt(&varmap, weights, Some("model"), cfg.n_layers).unwrap();

        // build train and val loaders with utility function from addons module
        let (train_loader, val_loader) = examples::ch05::addons::get_train_val_data_loaders(false);

        // compute train and val loss
        let train_loss = calc_loss_loader(&train_loader, &model, vb.device(), None).unwrap();
        let val_loss = calc_loss_loader(&val_loader, &model, vb.device(), None).unwrap();

        println!("Training loss {:?}", train_loss);
        println!("Validation loss {:?}", val_loss);
    }
}

/// Exercise 5.6
pub struct X6;

impl Exercise for X6 {
    fn name(&self) -> String {
        String::from("5.6")
    }

    fn title(&self) -> String {
        "Comparing generations with different GPT-2 model sizes".to_string() // missing from book
    }

    fn statement(&self) -> String {
        let stmt = "Experiment with GPT-2 models of different sizes—for \
        example, the largest 1,558 million parameter model—and compare the \
        generated text to the 124 million model.";
        stmt.to_string()
    }

    fn main(&self) {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::{generate, load_weights_into_gpt, text_to_token_ids, token_ids_to_text},
        };
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};
        use hf_hub::api::sync::Api;
        use rand::{rngs::StdRng, SeedableRng};
        use tiktoken_rs::get_bpe_from_model;

        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut cfg = Config::gpt2_xlarge();
        cfg.qkv_bias = true;
        let model = GPTModel::new(cfg, vb.pp("model")).unwrap();

        // get weights from HF Hub
        let model_name = "openai-community/gpt2-xl";
        let api = Api::new().unwrap();
        let repo = api.model(model_name.to_string());
        let weights = repo.get("model.safetensors").unwrap();
        let weights = candle_core::safetensors::load(weights, &Device::Cpu).unwrap();

        // load weights
        load_weights_into_gpt(&varmap, weights, Some("model"), cfg.n_layers).unwrap();

        // sample setup and load tokenizer
        let start_context = "Every effort moves you";
        let tokenizer = get_bpe_from_model("gpt2").unwrap();

        let mut rng = StdRng::seed_from_u64(42_u64);
        let token_ids = generate(
            &model,
            text_to_token_ids(start_context, &tokenizer, vb.device()).unwrap(),
            25_usize,
            cfg.context_length,
            Some(0.1_f64),
            Some(50_usize),
            None,
            &mut rng,
        )
        .unwrap();

        // decode the token ids to print the output text
        println!(
            "Model:\n{model_name}\n\nOutput text:\n{:?}",
            token_ids_to_text(token_ids, &tokenizer).unwrap()
        )
    }
}
