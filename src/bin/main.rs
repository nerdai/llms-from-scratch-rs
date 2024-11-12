use llms_from_scratch_rs::{examples, exercises, Example, Exercise};
use std::collections::HashMap;
use std::sync::LazyLock;

static EXERCISE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Exercise>>> =
    LazyLock::new(|| {
        let mut m: HashMap<&'static str, Box<dyn Exercise + 'static>> = HashMap::new();
        // ch02
        m.insert("2.1", Box::new(exercises::ch02::X2P1));
        m.insert("2.2", Box::new(exercises::ch02::X2P2));
        // ch03
        m.insert("3.1", Box::new(exercises::ch03::X3P1));
        m
    });

static EXAMPLE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Example>>> = LazyLock::new(|| {
    let mut m: HashMap<&'static str, Box<dyn Example + 'static>> = HashMap::new();
    // ch02
    m.insert("02.01", Box::new(examples::ch02::EG01));
    m.insert("02.02", Box::new(examples::ch02::EG02));
    // ch03
    m.insert("03.01", Box::new(examples::ch03::EG01));
    m.insert("03.02", Box::new(examples::ch03::EG02));
    m.insert("03.03", Box::new(examples::ch03::EG03));
    m.insert("03.04", Box::new(examples::ch03::EG04));
    m.insert("03.05", Box::new(examples::ch03::EG05));
    m.insert("03.06", Box::new(examples::ch03::EG06));
    m.insert("03.07", Box::new(examples::ch03::EG07));
    m.insert("03.08", Box::new(examples::ch03::EG08));
    m
});

fn main() {
    let exercise_registry = &*EXERCISE_REGISTRY;
    let example_registry = &*EXAMPLE_REGISTRY;
    let _ex = exercise_registry.get("3.1").unwrap();
    let eg = example_registry.get("03.08").unwrap();
    // ex.main()
    eg.main()
}
