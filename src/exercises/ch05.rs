use crate::Exercise;

/// Exercise 5.1
pub struct X5P1;

impl Exercise for X5P1 {
    fn name(&self) -> String {
        String::from("5.1")
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
pub struct X5P2;

impl Exercise for X5P2 {
    fn name(&self) -> String {
        String::from("5.2")
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
pub struct X5P3;

impl Exercise for X5P3 {
    fn name(&self) -> String {
        String::from("5.3")
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

/// Exercise 5.5
pub struct X5P5;

impl Exercise for X5P5 {
    fn name(&self) -> String {
        String::from("5.5")
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
        let mut weights = candle_core::safetensors::load(weights, &dev).unwrap();

        // construct model
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut cfg = Config::gpt2_124m();
        cfg.qkv_bias = true;
        let model = GPTModel::new(cfg, vb.pp("model")).unwrap();

        // load openai weights
        load_weights_into_gpt(&varmap, &mut weights, Some("model"), cfg.n_layers).unwrap();

        // build train and val loaders with utility function from addons module
        let (train_loader, val_loader) = examples::ch05::addons::get_train_val_data_loaders(false);

        // compute train and val loss
        let train_loss = calc_loss_loader(&train_loader, &model, vb.device(), None).unwrap();
        let val_loss = calc_loss_loader(&val_loader, &model, vb.device(), None).unwrap();

        println!("Training loss {:?}", train_loss);
        println!("Validation loss {:?}", val_loss);
    }
}
