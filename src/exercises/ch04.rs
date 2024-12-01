use crate::Exercise;

/// Exercise 4.1
pub struct X4P1;

impl Exercise for X4P1 {
    fn name(&self) -> String {
        String::from("4.1")
    }

    fn main(&self) {
        use crate::listings::ch04::{Config, TransformerBlock};
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        // create model
        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let _ = TransformerBlock::new(Config::gpt2_124m(), vb).unwrap();

        // Get varmap data containing all variables
        let varmap_data = varmap.data().lock().unwrap();

        // Count params for ff and mha modules
        let (mut ff_params, mut mha_params) = (0_usize, 0_usize);
        for (var_name, var) in varmap_data.iter() {
            let num_params = var.elem_count();
            if var_name.starts_with("ff.") {
                ff_params += num_params;
            } else if var_name.starts_with("mha.") {
                mha_params += num_params;
            }
        }
        println!("Ff number of parameters: {}", ff_params);
        println!("Mha number of parameters: {}", mha_params);
    }
}

/// Exercise 4.2
pub struct X4P2;

impl Exercise for X4P2 {
    fn name(&self) -> String {
        String::from("4.2")
    }

    fn main(&self) {
        use crate::listings::ch04::{Config, GPTModel};
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        let configs = &[
            ("gpt2-sm", Config::gpt2_124m()),
            ("gpt2-med", Config::gpt2_medium()),
            ("gpt2-l", Config::gpt2_large()),
            ("gpt2-xl", Config::gpt2_xlarge()),
        ];

        for (mdl_name, cfg) in configs.iter() {
            // construct model which stores the vars in the varmap
            let dev = Device::cuda_if_available(0).unwrap();
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
            let _ = GPTModel::new(*cfg, vb).unwrap();

            // compute number of params (todo build utility func for this)
            let mut total_params = 0_usize;
            for t in varmap.all_vars().iter() {
                total_params += t.elem_count();
            }
            println!("{} number of parameters: {}", mdl_name, total_params);

            // Get token embedding and output layer shapes
            let varmap_data = varmap.data().lock().unwrap();
            let tok_emb_dims = varmap_data.get("tok_emb.weight").unwrap().dims();
            println!("Token embedding layer shape {:?}", tok_emb_dims);
            let out_head_dims = varmap_data.get("out_head.weight").unwrap().dims();
            println!("Output layer shape {:?}", out_head_dims);

            // total number of params if weight tying with token emb and output layer shapes
            let total_params_gpt2 = total_params - (out_head_dims[0] * out_head_dims[1]);
            println!(
                "Number of trainable parameters considering weight tying {}",
                total_params_gpt2
            );

            // memory requirements (todo: build this out as a util)
            let total_size_bytes = total_params * 4;
            let total_size_mb = total_size_bytes as f32 / (1024_f32 * 1024.);
            println!("Total size of the model: {} MB\n", total_size_mb);
        }
    }
}

/// Exercise 4.3
pub struct X4P3;

impl Exercise for X4P3 {
    fn name(&self) -> String {
        String::from("4.3")
    }

    fn main(&self) {
        use crate::listings::ch04::GPTModel;
        use candle_core::{DType, Device, IndexOp, ModuleT, Tensor};
        use candle_nn::{VarBuilder, VarMap};

        // create model
        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = GPTModel::new_v2(addons::ConfigV2::gpt_config_124m(), vb).unwrap();

        // create batch inputs
        let batch = Tensor::new(&[[101_u32, 366, 100, 345], [101, 110, 322, 57]], &dev).unwrap();

        // run model forward
        let logits = model.forward_t(&batch, false).unwrap();

        // print first ten logits of vocabular for all batch inputs, and tokens
        let (_b, c, _vocab_size) = logits.dims3().unwrap();
        let last_tokens_logits = logits.i((.., c - 1, ..)).unwrap();
        println!(
            "first 10 logits of last vector: {:?}",
            last_tokens_logits.i((.., 0..10)).unwrap().to_vec2::<f32>()
        );
    }
}

mod addons {
    use crate::{
        candle_addons::seqt,
        listings::{
            ch03::MultiHeadAttention,
            ch04::{FeedForward, GPTModel, LayerNorm, TransformerBlock, GELU},
        },
    };
    use candle_core::Result;
    use candle_nn::{embedding, linear_b, seq, Dropout, VarBuilder};

    // For Exercise 4.3
    #[derive(Debug, Clone, Copy)]
    pub struct ConfigV2 {
        pub vocab_size: usize,
        pub context_length: usize,
        pub emb_dim: usize,
        pub n_heads: usize,
        pub n_layers: usize,
        pub drop_rate_attn: f32,
        pub drop_rate_emb: f32,
        pub drop_rate_shortcut: f32,
        pub qkv_bias: bool,
    }

    impl ConfigV2 {
        pub fn gpt_config_124m() -> Self {
            Self {
                vocab_size: 50_257,
                context_length: 1_024,
                emb_dim: 768,
                n_heads: 12,
                n_layers: 12,
                drop_rate_attn: 0.1,
                drop_rate_emb: 0.1,
                drop_rate_shortcut: 0.1,
                qkv_bias: false,
            }
        }
    }

    impl FeedForward {
        fn new_v2(cfg: ConfigV2, vb: VarBuilder<'_>) -> Result<Self> {
            let layers = seq()
                .add(linear_b(
                    cfg.emb_dim,
                    4_usize * cfg.emb_dim,
                    true,
                    vb.pp("first_layer"),
                )?)
                .add(GELU) // you should use Activation::Gelu in actual builds
                .add(linear_b(
                    4_usize * cfg.emb_dim,
                    cfg.emb_dim,
                    true,
                    vb.pp("second_layer"),
                )?);
            FeedForward::from_fields(layers)
        }
    }

    impl TransformerBlock {
        fn new_v2(cfg: ConfigV2, vb: VarBuilder<'_>) -> Result<Self> {
            let att = MultiHeadAttention::new(
                cfg.emb_dim,
                cfg.emb_dim,
                cfg.drop_rate_attn,
                cfg.n_heads,
                cfg.qkv_bias,
                vb.pp("mha"),
            )?;
            let ff = FeedForward::new_v2(cfg, vb.pp("ff"))?;
            let norm1 = LayerNorm::new(cfg.emb_dim, vb.pp("norm1"))?;
            let norm2 = LayerNorm::new(cfg.emb_dim, vb.pp("norm2"))?;
            let drop_shortcut = Dropout::new(cfg.drop_rate_shortcut);
            TransformerBlock::from_fields(att, ff, norm1, norm2, drop_shortcut)
        }
    }

    impl GPTModel {
        pub fn new_v2(cfg: ConfigV2, vb: VarBuilder<'_>) -> Result<Self> {
            let tok_emb = embedding(cfg.vocab_size, cfg.emb_dim, vb.pp("tok_emb"))?;
            let pos_emb = embedding(cfg.context_length, cfg.emb_dim, vb.pp("pos_emb"))?;
            let drop_emb = Dropout::new(cfg.drop_rate_emb);
            let mut trf_blocks = seqt();
            for ix in 0..cfg.n_layers {
                trf_blocks = trf_blocks
                    .add(TransformerBlock::new_v2(cfg, vb.pp(format!("trf-{}", ix))).unwrap());
            }
            let final_norm = LayerNorm::new(cfg.emb_dim, vb.pp("final_norm"))?;
            let out_head = linear_b(cfg.emb_dim, cfg.vocab_size, false, vb.pp("out_head"))?;
            GPTModel::from_fields(tok_emb, pos_emb, drop_emb, trf_blocks, final_norm, out_head)
        }
    }
}
