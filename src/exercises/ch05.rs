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
            print_sampled_tokens(&scaled_probas.to_vec1::<f32>().unwrap(), &inverse_vocab).unwrap();
            println!("\n");
        }
    }
}
