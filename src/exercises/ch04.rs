use crate::Exercise;
use candle_core::Var;

fn get_var_params(t: &Var) -> usize {
    match *t.dims() {
        [d1] => d1,
        [d1, d2] => d1 * d2,
        _ => panic!("Variable with more than 2 dimensions."),
    }
}

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
            if var_name.starts_with("ff.") {
                ff_params += get_var_params(var);
            } else if var_name.starts_with("mha.") {
                mha_params += get_var_params(var);
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
                let this_tensor_params = match *t.dims() {
                    [d1] => d1,
                    [d1, d2] => d1 * d2,
                    _ => panic!("Variable with more than 2 dimensions."),
                };
                total_params += this_tensor_params;
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
