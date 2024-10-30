pub mod exercises;
pub mod listings;

/// Exercise Trait
pub trait Exercise {
    fn name(&self) -> String;

    fn main(&self);
}
