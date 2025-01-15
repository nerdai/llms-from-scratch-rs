//! Examples from Appendix E

use crate::Example;
use anyhow::Result;

/// # Example usage of `create_candle_dataloaders`
///
/// #### Id
/// E.01
///
/// #### Page
/// This example starts on page 326
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run example E.01
/// ```
pub struct EG01;

impl Example for EG01 {
    fn description(&self) -> String {
        "Example usage of `create_candle_dataloaders`.".to_string()
    }

    fn page_source(&self) -> usize {
        326_usize
    }

    fn main(&self) -> Result<()> {
        use crate::listings::apdx_e::create_candle_dataloaders;

        let (train_loader, val_loader, test_loader) = create_candle_dataloaders()?;

        // print last batch of train loader
        let (input_batch, target_batch) = train_loader.batcher().last().unwrap()?;
        println!("Input batch dimensions: {:?}", input_batch.shape());
        println!("Label batch dimensions: {:?}", target_batch.shape());

        // print total number of batches in each data loader
        println!("{:?} training batches", train_loader.len());
        println!("{:?} validation batches", val_loader.len());
        println!("{:?} test batches", test_loader.len());

        Ok(())
    }
}
