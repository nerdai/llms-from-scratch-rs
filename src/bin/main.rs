use clap::{Parser, Subcommand};
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
        m.insert("3.2", Box::new(exercises::ch03::X3P2));
        m.insert("3.3", Box::new(exercises::ch03::X3P3));
        // ch04
        m.insert("4.1", Box::new(exercises::ch04::X4P1));
        m.insert("4.2", Box::new(exercises::ch04::X4P2));
        m.insert("4.3", Box::new(exercises::ch04::X4P3));
        // ch05
        m.insert("5.1", Box::new(exercises::ch05::X5P1));
        m.insert("5.2", Box::new(exercises::ch05::X5P2));
        m.insert("5.3", Box::new(exercises::ch05::X5P3));
        m.insert("5.4", Box::new(exercises::ch05::X5P4));
        m.insert("5.5", Box::new(exercises::ch05::X5P5));
        m.insert("5.6", Box::new(exercises::ch05::X5P6));
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
    m.insert("03.09", Box::new(examples::ch03::EG09));
    m.insert("03.10", Box::new(examples::ch03::EG10));
    m.insert("03.11", Box::new(examples::ch03::EG11));
    // ch04
    m.insert("04.01", Box::new(examples::ch04::EG01));
    m.insert("04.02", Box::new(examples::ch04::EG02));
    m.insert("04.03", Box::new(examples::ch04::EG03));
    m.insert("04.04", Box::new(examples::ch04::EG04));
    m.insert("04.05", Box::new(examples::ch04::EG05));
    m.insert("04.06", Box::new(examples::ch04::EG06));
    m.insert("04.07", Box::new(examples::ch04::EG07));
    m.insert("04.08", Box::new(examples::ch04::EG08));
    // ch05
    m.insert("05.01", Box::new(examples::ch05::EG01));
    m.insert("05.02", Box::new(examples::ch05::EG02));
    m.insert("05.03", Box::new(examples::ch05::EG03));
    m.insert("05.04", Box::new(examples::ch05::EG04));
    m.insert("05.05", Box::new(examples::ch05::EG05));
    m.insert("05.06", Box::new(examples::ch05::EG06));
    m.insert("05.07", Box::new(examples::ch05::EG07));
    m.insert("05.08", Box::new(examples::ch05::EG08));
    m.insert("05.09", Box::new(examples::ch05::EG09));
    m.insert("05.10", Box::new(examples::ch05::EG10));
    m.insert("05.11", Box::new(examples::ch05::EG11));
    m
});

#[allow(dead_code)]
enum RunType {
    EG(String),
    EX(String),
}

/// CLI
#[derive(Debug, Parser)]
#[command(name = "llms-from-scratch-rs")]
#[command(about = "A CLI for running examples and exercises.", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run examples
    Example {
        /// The example to run
        id: String,
    },
    /// Run exercises
    Exercise {
        /// The exercise to run
        id: String,
    },
}

fn main() {
    let exercise_registry = &*EXERCISE_REGISTRY;
    let example_registry = &*EXAMPLE_REGISTRY;
    let cli = Cli::parse();

    match cli.command {
        Commands::Example { id } => {
            let eg = example_registry.get(&id[..]).unwrap();
            eg.main()
        }
        Commands::Exercise { id } => {
            let ex = exercise_registry.get(&id[..]).unwrap();
            ex.main()
        }
    }
}
