pub mod exercises;
pub mod listings;

pub trait Exercise {
    fn name(&self) -> String;

    fn main(&self);
}
