use crate::Exercise;

/// 2.1
pub struct X2P1;

impl Exercise for X2P1 {
    fn name(&self) -> String {
        String::from("2.1")
    }

    fn main(&self) {
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        let token_ids = tokenizer.encode_with_special_tokens("Akwirw ier");
        println!("token ids: {:?}", token_ids);

        let decoded_text = tokenizer.decode(token_ids).unwrap();
        println!("decoded text: {}", decoded_text);
    }
}

/// 2.2
pub struct X2P2;

impl Exercise for X2P2 {
    fn name(&self) -> String {
        String::from("2.2")
    }

    fn main(&self) {
        use crate::listings::ch02::create_dataloader_v1;
        use std::fs;

        let raw_text = fs::read_to_string("data/the-verdict.txt").expect("Unable to read the file");
        let max_length = 4_usize;
        let stride = 2_usize;
        let shuffle = false;
        let drop_last = false;
        let batch_size = 2_usize;
        let (_dataset, mut batch_iter) = create_dataloader_v1(
            &raw_text[..],
            batch_size,
            max_length,
            stride,
            shuffle,
            drop_last,
        );

        match batch_iter.next() {
            Some(Ok((inputs, targets))) => {
                println!(
                    "inputs: {:?}\n\ntargets: {:?}",
                    inputs.to_vec2::<u32>(),
                    targets.to_vec2::<u32>()
                );
            }
            Some(Err(err)) => panic!("{}", err),
            None => panic!("None"),
        }
    }
}
