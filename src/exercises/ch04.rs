use crate::Exercise;
use candle_core::Var;

fn get_var_params(t: &Var) -> usize {
    match *t.dims() {
        [d1] => d1,
        [d1, d2] => d1 * d2,
        _ => panic!("Variable with more than 2 dimensions."),
    }
}

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
