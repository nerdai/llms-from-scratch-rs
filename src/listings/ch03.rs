use candle_core::{Device, Result, Tensor};
use fancy_regex::{Captures, Regex};
use rand::{seq::SliceRandom, thread_rng};
use std::collections::HashMap;
use tiktoken_rs::CoreBPE;

/// Listing 3.1
pub struct SelfAttentionV1 {}
