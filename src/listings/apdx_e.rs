//! Listings from Appendix E

use crate::examples::ch06::addons::write_parquet;
use crate::listings::ch06::{
    create_balanced_dataset, download_smsspam_parquet, random_split, SpamDataLoader, SpamDataset,
    SpamDatasetBuilder, PARQUET_FILENAME, PARQUET_URL,
};
use anyhow::anyhow;
use candle_core::{Module, Result, Tensor};
use candle_nn::{init, Linear, VarBuilder};
use polars::prelude::*;
use std::{
    ops::Not,
    path::{Path, PathBuf},
    str::FromStr,
};
use tiktoken_rs::get_bpe_from_model;

/// [Listing E.1] Downloading and preparing the dataset
///
/// NOTE: This is merely EG 06.04
pub fn download_and_prepare_spam_dataset() -> anyhow::Result<()> {
    // download parquet
    download_smsspam_parquet(PARQUET_URL)?;

    // load parquet
    let mut file_path = PathBuf::from("data");
    file_path.push(PARQUET_FILENAME);
    let mut file = std::fs::File::open(file_path).unwrap();
    let df = ParquetReader::new(&mut file).finish().unwrap();

    // balance dataset
    let balanced_df = create_balanced_dataset(df)?;

    // create train, test, val splits
    let (mut train_df, mut validation_df, mut test_df) =
        random_split(&balanced_df, 0.7_f32, 0.1_f32)?;

    // save dfs to csv
    let train_path = PathBuf::from_str("data/train.parquet")?;
    let validation_path = PathBuf::from_str("data/validation.parquet")?;
    let test_path = PathBuf::from_str("data/test.parquet")?;

    write_parquet(&mut train_df, train_path)?;
    write_parquet(&mut validation_df, validation_path)?;
    write_parquet(&mut test_df, test_path)?;

    Ok(())
}

/// [Listing E.2] Instantiating Candle Datasets
///
/// NOTE: This is merely EG 06.05
pub fn create_candle_datasets() -> anyhow::Result<(SpamDataset, SpamDataset, SpamDataset)> {
    let tokenizer = get_bpe_from_model("gpt2")?;

    let train_path = Path::new("data").join("train.parquet");
    if train_path.exists().not() {
        return Err(anyhow!(
            "Missing 'data/train.parquet' file. Please run `listings::apdx_e::download_and_prepare_spam_dataset()`"
        ));
    }
    let train_dataset = SpamDatasetBuilder::new(&tokenizer)
        .load_data_from_parquet(train_path)
        .build();

    let val_path = Path::new("data").join("validation.parquet");
    if val_path.exists().not() {
        return Err(anyhow!(
            "Missing 'data/validation.parquet' file. Please run `listings::apdx_e::download_and_prepare_spam_dataset()`"
        ));
    }
    let val_dataset = SpamDatasetBuilder::new(&tokenizer)
        .load_data_from_parquet(val_path)
        .max_length(Some(train_dataset.max_length()))
        .build();

    let test_path = Path::new("data").join("test.parquet");
    if test_path.exists().not() {
        return Err(anyhow!(
            "Missing 'data/test.parquet' file. Please run `listings::apdx_e::download_and_prepare_spam_dataset()`"
        ));
    }
    let test_dataset = SpamDatasetBuilder::new(&tokenizer)
        .load_data_from_parquet(test_path)
        .max_length(Some(train_dataset.max_length()))
        .build();

    Ok((train_dataset, val_dataset, test_dataset))
}

/// [Listing E.3] Creating Candle DataLoaders
///
/// NOTE: This is merely EG 06.06
pub fn create_candle_dataloaders(
) -> anyhow::Result<(SpamDataLoader, SpamDataLoader, SpamDataLoader)> {
    let (train_dataset, val_dataset, test_dataset) = create_candle_datasets()?;

    // create loaders
    let batch_size = 8_usize;
    let train_loader = SpamDataLoader::new(train_dataset, batch_size, true, true);
    let val_loader = SpamDataLoader::new(val_dataset, batch_size, false, false);
    let test_loader = SpamDataLoader::new(test_dataset, batch_size, false, false);

    Ok((train_loader, val_loader, test_loader))
}

/// [Listing E.4] Loading a pretrained GPT model
///
/// NOTE: This is merely a re-export of `download_and_load_gpt2` from `listings::ch06`
#[doc(inline)]
pub use crate::listings::ch06::download_and_load_gpt2;

/// [Listing E.5] Implementing a LoRA layer
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct LoRALayer {
    A: Tensor,
    B: Tensor,
    alpha: f64,
}

impl LoRALayer {
    #[allow(non_snake_case)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let init_a = init::DEFAULT_KAIMING_NORMAL;
        let init_b = init::ZERO;
        let A = vb.get_with_hints((in_dim, rank), "A", init_a)?;
        let B = vb.get_with_hints((rank, out_dim), "B", init_b)?;
        Ok(Self { A, B, alpha })
    }
}

impl Module for LoRALayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let a_mat = match *xs.dims() {
            [b1, b2, _, _] => self.A.broadcast_left((b1, b2))?,
            [bsize, _, _] => self.A.broadcast_left(bsize)?,
            _ => self.A.clone(),
        };
        let b_mat = match *xs.dims() {
            [b1, b2, _, _] => self.B.broadcast_left((b1, b2))?,
            [bsize, _, _] => self.B.broadcast_left(bsize)?,
            _ => self.B.clone(),
        };
        let mut retval = a_mat.matmul(&b_mat)?;
        retval = xs.matmul(&retval)?;
        self.alpha * retval
    }
}

/// [Listing E.6] The `LinearWithLoRA` Module
#[derive(Debug, Clone)]
pub struct LinearWithLoRA {
    linear: Linear,
    lora: LoRALayer,
}

impl LinearWithLoRA {
    pub fn new(linear: Linear, rank: usize, alpha: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let in_dim = linear.weight().dims()[0];
        let out_dim = linear.weight().dims()[1];
        let lora = LoRALayer::new(in_dim, out_dim, rank, alpha, vb)?;

        Ok(Self { linear, lora })
    }
}

impl Module for LinearWithLoRA {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)? + self.lora.forward(xs)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::listings::ch04::Config;
    use anyhow::Result;
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use rstest::*;

    #[fixture]
    pub fn vb() -> VarBuilder<'static> {
        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, &dev)
    }

    #[rstest]
    fn test_lora_layer_init(vb: VarBuilder<'_>) -> Result<()> {
        let alpha = 0.5_f64;
        let rank = 3_usize;
        let cfg = Config::gpt_sm_test();
        let lora_layer = LoRALayer::new(cfg.emb_dim, cfg.emb_dim, rank, alpha, vb)?;

        assert_eq!(lora_layer.A.dims(), &[cfg.emb_dim, rank]);
        assert_eq!(lora_layer.B.dims(), &[rank, cfg.emb_dim]);
        Ok(())
    }

    #[rstest]
    fn test_lora_layer_forward(vb: VarBuilder<'_>) -> Result<()> {
        let alpha = 0.5_f64;
        let rank = 3_usize;
        let cfg = Config::gpt_sm_test();
        let lora_layer = LoRALayer::new(cfg.emb_dim, cfg.emb_dim, rank, alpha, vb.pp("lora"))?;

        // create dummy batch
        let input_length = 2_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, cfg.emb_dim), &vb.device())?;
        let batch = Tensor::stack(&[&xs, &xs], 0)?;

        // forward should result in 0s upon first construction
        let outputs = lora_layer.forward(&batch)?;

        assert_eq!(
            outputs.i(0)?.to_vec2::<f32>()?,
            &[
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ]
        );
        Ok(())
    }

    #[rstest]
    fn test_linear_with_lora_init(vb: VarBuilder<'_>) -> Result<()> {
        let alpha = 0.5_f64;
        let rank = 3_usize;
        let cfg = Config::gpt_sm_test();
        let linear = candle_nn::linear(cfg.emb_dim, cfg.emb_dim, vb.pp("linear"))?;
        let lora_with_linear = LinearWithLoRA::new(linear, rank, alpha, vb.pp("linear_with_lora"))?;

        assert_eq!(lora_with_linear.lora.A.dims(), &[cfg.emb_dim, rank]);
        assert_eq!(lora_with_linear.lora.B.dims(), &[rank, cfg.emb_dim]);
        assert_eq!(
            lora_with_linear.linear.weight().dims(),
            &[cfg.emb_dim, cfg.emb_dim]
        );
        Ok(())
    }

    #[rstest]
    fn test_linear_with_lora_forward(vb: VarBuilder<'_>) -> Result<()> {
        // since this is only init linear_with_lora forward should be same as linear
        let alpha = 0.5_f64;
        let rank = 3_usize;
        let cfg = Config::gpt_sm_test();
        let linear = candle_nn::linear(cfg.emb_dim, cfg.emb_dim, vb.pp("linear"))?;
        let lora_with_linear =
            LinearWithLoRA::new(linear.clone(), rank, alpha, vb.pp("linear_with_lora"))?;

        // create dummy batch
        let input_length = 2_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, cfg.emb_dim), &vb.device())?;
        let batch = Tensor::stack(&[&xs, &xs], 0)?;

        // forward should result in 0s upon first construction
        let outputs = lora_with_linear.forward(&batch)?;
        let outputs_linear_only = linear.forward(&batch)?;

        assert_eq!(
            outputs.to_vec3::<f32>()?,
            outputs_linear_only.to_vec3::<f32>()?
        );

        Ok(())
    }
}
