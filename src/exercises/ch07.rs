//! Exercises from Chapter 7

use crate::Exercise;
use anyhow::Result;

/// # Changing prompt styles
///
/// #### Id
/// 7.1
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run exercise 7.1
///
/// # with cuda
/// cargo run --features cuda exercise 7.1
/// ```
pub struct X1;

impl Exercise for X1 {
    fn name(&self) -> String {
        "7.1".to_string()
    }

    fn title(&self) -> String {
        "Changing prompt styles".to_string()
    }

    fn statement(&self) -> String {
        let stmt = "After fine-tuning the model with the Alpaca prompt style, \
        try the Phi-3 prompt style shown in figure 7.4 and observe whether it \
        affects the response quality of the model.";
        stmt.to_string()
    }

    fn main(&self) -> Result<()> {
        todo!()
    }
}
