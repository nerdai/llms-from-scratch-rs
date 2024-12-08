//! # Build A Large Language Model From Scratch — Rust Translations
//!
//! #### Intro
//!
//! This crate provides Rust translations of the examples, exercises and listings
//! found in the book Build A LLM From Scratch by Sebastian Raschka
//! ([github](https://github.com/rasbt/LLMs-from-scratch)), which is
//! a great resource for careful learning of LLMs. The book provides several
//! examples and listings which are written in PyTorch in order to learn how to
//! build a GPT (decoder-only) language model. This crate provides the Rust
//! equivalent for nearly all of the code provided in the book using
//! [candle](https://github.com/huggingface/candle) (a Minimalist ML framework for Rust).
//!
//! The lib crate consists of three modules: `examples`, `exercises` and `listings`.
//! Additionally there is a companion binary crate that executes all of the examples
//! and exercises.

use anyhow::Result;

pub mod candle_addons;
pub mod examples;
pub mod exercises;
pub mod listings;

/// Exercise Trait
pub trait Exercise: Send + Sync {
    fn name(&self) -> String;

    fn title(&self) -> String;

    fn statement(&self) -> String;

    fn main(&self) -> Result<()>;
}

/// Example Trait
pub trait Example: Send + Sync {
    fn description(&self) -> String;

    fn page_source(&self) -> usize;

    fn main(&self) -> Result<()>;
}
