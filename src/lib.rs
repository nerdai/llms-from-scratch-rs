pub mod exercises;
pub mod listings;

/// Exercise Trait
pub trait Exercise: Send + Sync {
    fn name(&self) -> String;

    fn main(&self);
}
