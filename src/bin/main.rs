use llms_from_scratch_rs::examples::{ch02, ch03};
use llms_from_scratch_rs::exercises::ch02::{X2P1, X2P2};
use llms_from_scratch_rs::{Example, Exercise};
use std::collections::HashMap;
use std::sync::LazyLock;

static EXERCISE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Exercise>>> =
    LazyLock::new(|| {
        let mut m: HashMap<&'static str, Box<dyn Exercise + 'static>> = HashMap::new();
        m.insert("2.1", Box::new(X2P1 {}));
        m.insert("2.2", Box::new(X2P2 {}));
        m
    });

static EXAMPLE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Example>>> = LazyLock::new(|| {
    let mut m: HashMap<&'static str, Box<dyn Example + 'static>> = HashMap::new();
    m.insert("02.01", Box::new(ch02::EG01 {}));
    m.insert("02.02", Box::new(ch02::EG02 {}));
    m.insert("03.01", Box::new(ch03::EG01 {}));
    m
});

fn main() {
    let exercise_registry = &*EXERCISE_REGISTRY;
    let example_registry = &*EXAMPLE_REGISTRY;
    let _ex = exercise_registry.get("2.2").unwrap();
    let eg = example_registry.get("03.01").unwrap();
    // ex.main()
    eg.main()
}
