use anyhow::Result;
use clap::{Parser, Subcommand};
use comfy_table::{ContentArrangement, Table};
use itertools::Itertools;
use llms_from_scratch_rs::{examples, exercises, Example, Exercise};
use std::collections::HashMap;
use std::sync::LazyLock;

static EXERCISE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Exercise>>> =
    LazyLock::new(|| {
        let mut m: HashMap<&'static str, Box<dyn Exercise + 'static>> = HashMap::new();
        // ch02
        m.insert("2.1", Box::new(exercises::ch02::X1));
        m.insert("2.2", Box::new(exercises::ch02::X2));
        // ch03
        m.insert("3.1", Box::new(exercises::ch03::X1));
        m.insert("3.2", Box::new(exercises::ch03::X2));
        m.insert("3.3", Box::new(exercises::ch03::X3));
        // ch04
        m.insert("4.1", Box::new(exercises::ch04::X1));
        m.insert("4.2", Box::new(exercises::ch04::X2));
        m.insert("4.3", Box::new(exercises::ch04::X3));
        // ch05
        m.insert("5.1", Box::new(exercises::ch05::X1));
        m.insert("5.2", Box::new(exercises::ch05::X2));
        m.insert("5.3", Box::new(exercises::ch05::X3));
        m.insert("5.4", Box::new(exercises::ch05::X4));
        m.insert("5.5", Box::new(exercises::ch05::X5));
        m.insert("5.6", Box::new(exercises::ch05::X6));
        // ch06
        m.insert("6.1", Box::new(exercises::ch06::X1));
        m.insert("6.2", Box::new(exercises::ch06::X2));
        m.insert("6.3", Box::new(exercises::ch06::X3));
        // ch07
        m.insert("7.1", Box::new(exercises::ch07::X1));
        m.insert("7.2", Box::new(exercises::ch07::X2));
        m.insert("7.3", Box::new(exercises::ch07::X3));
        m
    });

static EXAMPLE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Example>>> = LazyLock::new(|| {
    let mut m: HashMap<&'static str, Box<dyn Example + 'static>> = HashMap::new();
    // ch02
    m.insert("02.01", Box::new(examples::ch02::EG01));
    m.insert("02.02", Box::new(examples::ch02::EG02));
    m.insert("02.03", Box::new(examples::ch02::EG03));
    m.insert("02.04", Box::new(examples::ch02::EG04));
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
    // ch06
    m.insert("06.01", Box::new(examples::ch06::EG01));
    m.insert("06.02", Box::new(examples::ch06::EG02));
    m.insert("06.03", Box::new(examples::ch06::EG03));
    m.insert("06.04", Box::new(examples::ch06::EG04));
    m.insert("06.05", Box::new(examples::ch06::EG05));
    m.insert("06.06", Box::new(examples::ch06::EG06));
    m.insert("06.07", Box::new(examples::ch06::EG07));
    m.insert("06.08", Box::new(examples::ch06::EG08));
    m.insert("06.09", Box::new(examples::ch06::EG09));
    m.insert("06.10", Box::new(examples::ch06::EG10));
    m.insert("06.11", Box::new(examples::ch06::EG11));
    m.insert("06.12", Box::new(examples::ch06::EG12));
    m.insert("06.13", Box::new(examples::ch06::EG13));
    m.insert("06.14", Box::new(examples::ch06::EG14));
    m.insert("06.15", Box::new(examples::ch06::EG15));
    // ch07
    m.insert("07.01", Box::new(examples::ch07::EG01));
    m.insert("07.02", Box::new(examples::ch07::EG02));
    m.insert("07.03", Box::new(examples::ch07::EG03));
    m.insert("07.04", Box::new(examples::ch07::EG04));
    m.insert("07.05", Box::new(examples::ch07::EG05));
    m.insert("07.06", Box::new(examples::ch07::EG06));
    m.insert("07.07", Box::new(examples::ch07::EG07));
    m.insert("07.08", Box::new(examples::ch07::EG08));
    m.insert("07.09", Box::new(examples::ch07::EG09));
    m.insert("07.10", Box::new(examples::ch07::EG10));
    m.insert("07.11", Box::new(examples::ch07::EG11));
    m.insert("07.12", Box::new(examples::ch07::EG12));
    m.insert("07.13", Box::new(examples::ch07::EG13));
    m.insert("07.14", Box::new(examples::ch07::EG14));
    m.insert("07.15", Box::new(examples::ch07::EG15));
    m.insert("07.16", Box::new(examples::ch07::EG16));
    // apdx_e
    m.insert("E.01", Box::new(examples::apdx_e::EG01));
    m
});

/// CLI
#[derive(Debug, Parser)]
#[command(bin_name = "llms-from-scratch-rs")]
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
    /// List examples and exercises
    List {
        #[clap(long, action)]
        examples: bool,
        #[clap(long, action)]
        exercises: bool,
    },
}

fn main() -> Result<()> {
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
        Commands::List {
            examples,
            exercises,
        } => {
            if examples {
                let mut examples_table = Table::new();
                examples_table
                    .set_width(80)
                    .set_content_arrangement(ContentArrangement::Dynamic)
                    .set_header(vec!["Id", "Description"]);
                for key in example_registry.keys().sorted() {
                    let eg = example_registry.get(key).unwrap();
                    examples_table.add_row(vec![key.to_string(), eg.description()]);
                }
                println!("EXAMPLES:\n{examples_table}");
            }
            if exercises {
                let mut exercises_table = Table::new();
                exercises_table
                    .set_width(80)
                    .set_content_arrangement(ContentArrangement::Dynamic)
                    .set_header(vec!["Id", "Statement"]);
                for key in exercise_registry.keys().sorted() {
                    let ex = exercise_registry.get(key).unwrap();
                    exercises_table.add_row(vec![
                        key.to_string(),
                        format!("{}\n\n{}", ex.title(), ex.statement()),
                    ]);
                }
                println!("EXERCISES:\n{exercises_table}");
            }
            Ok(())
        }
    }
}
