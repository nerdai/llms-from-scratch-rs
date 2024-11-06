pub mod examples;
pub mod exercises;
pub mod listings;

/// Exercise Trait
pub trait Exercise: Send + Sync {
    fn name(&self) -> String;

    fn main(&self);
}

pub struct Example {
    description: String,
    source_page: usize,
}

impl Example {
    pub fn new(desc: &str, source_page: usize) -> Self {
        Self {
            description: String::from(desc),
            source_page,
        }
    }

    pub fn description(&self) -> &str {
        &self.description[..]
    }

    pub fn source_page(&self) -> usize {
        self.source_page
    }
}

impl Main for Example {}
pub trait Main: Send + Sync {
    fn main(&self) {
        println!("The main method has not been implemented yet.")
    }
}
