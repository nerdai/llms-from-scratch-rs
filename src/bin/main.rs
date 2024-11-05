use llms_from_scratch_rs::exercises::ch2::{X2P1, X2P2};
use llms_from_scratch_rs::Exercise;
use std::collections::HashMap;
use std::sync::LazyLock;

static EXERCISE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Exercise>>> =
    LazyLock::new(|| {
        let mut m: HashMap<&'static str, Box<dyn Exercise>> = HashMap::new();
        m.insert("2.1", Box::new(X2P1 {}));
        m.insert("2.2", Box::new(X2P2 {}));
        m
    });

fn main() {
    let registry = &*EXERCISE_REGISTRY;
    let ex = registry.get("2.2").unwrap();
    ex.main()
}
