pub mod candle_addons;
pub mod examples;
pub mod exercises;
pub mod listings;

/// Exercise Trait
pub trait Exercise: Send + Sync {
    fn name(&self) -> String;

    fn title(&self) -> String;

    fn statement(&self) -> String;

    fn main(&self);
}

/// Example Trait
pub trait Example: Send + Sync {
    fn description(&self) -> String;

    fn page_source(&self) -> usize;

    fn main(&self);
}
