use llms_from_scratch_rs::exercises::ch2::{X2P1, X2P2};
use llms_from_scratch_rs::Exercise;
use phf::phf_map;
use std::collections::HashMap;

static EXERCISE_REGISTRY: phf::Map<&'static str, Box<dyn Exercise>> = phf_map! {
    "2.1" => Box::new(X2P1 {}),
    "2.2" => Box::new(X2P2 {}),
};

fn main() {
    let ex = EXERCISE_REGISTRY.get("2.2");
    match ex {
        Some(box_exercise) => *box_exercise.main(),
        None => {
            panic!("Exercise doesn't exist.")
        }
    }
}
