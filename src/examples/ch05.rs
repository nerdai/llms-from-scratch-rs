use crate::Example;

/// Example 05.01
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

/// Example 05.02
pub struct EG02;

impl Example for EG02 {
    fn description(&self) -> String {
        let desc = "Example computation of cross-entropy and perplexity.";
        String::from(desc)
    }

    fn page_source(&self) -> usize {
        133_usize
    }

    fn main(&self) {
        use crate::listings::{
            ch04::{Config, GPTModel},
            ch05::token_ids_to_text,
        };
        use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
        use candle_nn::{ops::softmax, VarBuilder, VarMap};
        use tiktoken_rs::get_bpe_from_model;

        // construct model
        let varmap = VarMap::new();
        let vb =
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0).unwrap());
        let cfg = Config::gpt2_124m();
        let model = GPTModel::new(cfg, vb.pp("model")).unwrap();

        // inputs and target tensors
        let inputs = Tensor::new(&[[16833_u32, 3626, 6100], [40, 1107, 588]], vb.device()).unwrap();
        let targets =
            Tensor::new(&[[3626_u32, 6100, 345], [1107, 588, 11311]], vb.device()).unwrap();

        // logits and probas
        let logits = model.forward(&inputs).unwrap();
        let probas = softmax(&logits, D::Minus1).unwrap();
        println!("{:?}", probas);

        // get next token id from probas
        let token_ids = probas.argmax_keepdim(D::Minus1).unwrap();
        println!("Token IDs:\n{:?}", token_ids.to_vec3::<u32>());

        // compare predictions to targets
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        println!(
            "Targets batch 1: {:?}",
            token_ids_to_text(targets.i(0).unwrap(), &tokenizer)
        );
        println!(
            "Outputs batch 1: {:?}",
            token_ids_to_text(token_ids.i(0).unwrap().flatten_all().unwrap(), &tokenizer)
        );

        // let's see the predicted probas for the target tokens
        let text_idx = 0_usize;
        let target_probas_1 =
            addons::get_target_token_probas_helper(text_idx, &targets, &probas, vb.device())
                .unwrap();

        println!("Text 1: {:?}", target_probas_1);

        let text_idx = 1_usize;
        let target_probas_2 =
            addons::get_target_token_probas_helper(text_idx, &targets, &probas, vb.device())
                .unwrap();

        println!("Text 2: {:?}", target_probas_2);
    }
}

mod addons {
    use candle_core::{Device, IndexOp, Result, Tensor};

    pub fn get_target_token_probas_helper(
        text_idx: usize,
        targets: &Tensor,
        probas: &Tensor,
        dev: &Device,
    ) -> Result<Tensor> {
        let target_tokens_1 = targets.i(text_idx).unwrap().to_vec1::<u32>().unwrap();
        let mut target_probas_1: Vec<f32> = vec![];
        for (i, target_token) in target_tokens_1.iter().enumerate() {
            let target_proba = probas
                .i((text_idx, i, *target_token as usize))
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            target_probas_1.push(target_proba);
        }
        Tensor::from_vec(target_probas_1, target_tokens_1.len(), dev)
    }
}
