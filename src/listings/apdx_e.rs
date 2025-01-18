//! Listings from Appendix E

use crate::examples::ch06::addons::write_parquet;
use crate::listings::{
    ch04::Config,
    ch06::{
        create_balanced_dataset, download_smsspam_parquet, random_split, SpamDataLoader,
        SpamDataset, SpamDatasetBuilder, PARQUET_FILENAME, PARQUET_URL,
    },
};
use anyhow::anyhow;
use candle_core::{Module, ModuleT, Result, Tensor, D};
use candle_nn::{init, ops::softmax, Dropout, Linear, VarBuilder, VarMap};
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

use super::ch03::MultiHeadAttention;
use super::ch04::GPTModel;

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
        // candle_nn::Linear.weight is defined as transpose. We follow same convention here.
        let A = vb.get_with_hints((rank, in_dim), "A", init_a)?;
        let B = vb.get_with_hints((out_dim, rank), "B", init_b)?;
        Ok(Self { A, B, alpha })
    }
}

impl Module for LoRALayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let a_mat = match *xs.dims() {
            [b1, b2, _, _] => self.A.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.A.broadcast_left(bsize)?.t()?,
            _ => self.A.t()?,
        };
        let b_mat = match *xs.dims() {
            [b1, b2, _, _] => self.B.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => self.B.broadcast_left(bsize)?.t()?,
            _ => self.B.t()?,
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
    pub fn from_linear(
        linear: Linear,
        rank: usize,
        alpha: f64,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let out_dim = linear.weight().dims()[0];
        let in_dim = linear.weight().dims()[1];
        let lora = LoRALayer::new(in_dim, out_dim, rank, alpha, vb)?;

        Ok(Self { linear, lora })
    }
}

impl Module for LinearWithLoRA {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)? + self.lora.forward(xs)?
    }
}

/// Function to replace all `Linear` layers with `LinearWithLoRA` in a given model
/// NOTE: this won't work for Candle
/// Need to impl all the modules `XXXWithLoRA` and probably impl the `From` trait
#[allow(unused_variables)]
pub fn replace_linear_with_lora(
    model: &mut GPTModel,
    cfg: Config,
    rank: usize,
    alpha: f64,
    varmap: &VarMap,
    vb: VarBuilder<'_>,
) -> Result<()> {
    Ok(())
}

#[doc(inline)]
pub use crate::listings::ch03::{get_mask, masked_fill};

#[derive(Clone, Debug)]
pub struct MultiHeadAttentionWithLoRA {
    num_heads: usize,
    d_out: usize,
    head_dim: usize,
    w_query: LinearWithLoRA,
    w_key: LinearWithLoRA,
    w_value: LinearWithLoRA,
    out_proj: LinearWithLoRA,
    scaling: f64,
    dropout: Dropout,
    drop_p: f32,
}

impl MultiHeadAttentionWithLoRA {
    pub fn from_mha(
        mha: MultiHeadAttention,
        rank: usize,
        alpha: f64,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let w_query =
            LinearWithLoRA::from_linear(mha.w_query().clone(), rank, alpha, vb.pp("query"))?;
        let w_key = LinearWithLoRA::from_linear(mha.w_key().clone(), rank, alpha, vb.pp("key"))?;
        let w_value =
            LinearWithLoRA::from_linear(mha.w_value().clone(), rank, alpha, vb.pp("value"))?;
        let out_proj =
            LinearWithLoRA::from_linear(mha.out_proj().clone(), rank, alpha, vb.pp("out_proj"))?;

        Ok(Self {
            num_heads: mha.num_heads(),
            d_out: mha.d_out(),
            head_dim: mha.head_dim(),
            w_query,
            w_key,
            w_value,
            out_proj,
            scaling: mha.scaling(),
            dropout: mha.dropout().clone(),
            drop_p: mha.drop_p(),
        })
    }

    pub fn w_query(&self) -> &LinearWithLoRA {
        &self.w_query
    }

    pub fn w_key(&self) -> &LinearWithLoRA {
        &self.w_key
    }

    pub fn w_value(&self) -> &LinearWithLoRA {
        &self.w_value
    }

    pub fn out_proj(&self) -> &LinearWithLoRA {
        &self.out_proj
    }

    pub fn d_out(&self) -> usize {
        self.d_out
    }

    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    pub fn dropout(&self) -> &Dropout {
        &self.dropout
    }

    pub fn drop_p(&self) -> f32 {
        self.drop_p
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Manual implementation of forward
    ///
    /// Note: that blanket implementation of `ModuleT` when a type implements
    /// `Module` prevents having `forward` being overrided. Thus, this type
    /// is `ModuleT` but technicall not `Module`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_t(xs, true)
    }
}

impl ModuleT for MultiHeadAttentionWithLoRA {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (b, num_tokens, _d_in) = xs.dims3()?;
        let queries = self.w_query.forward_t(xs, train)?;
        let keys = self.w_key.forward_t(xs, train)?;
        let values = self.w_value.forward_t(xs, train)?;

        // reshapes to facilitate getting attn scores each of the individual heads
        // with one matrix multiplication
        let queries = queries
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let keys = keys
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let values = values
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let attn_scores = queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)?;

        let mask = get_mask(num_tokens, xs.device())?;
        let masked = masked_fill(
            &attn_scores,
            &mask.broadcast_left((b, self.num_heads)).unwrap(),
            f32::NEG_INFINITY,
        )?;

        // scale
        let mut attn_weights = softmax(&(masked * self.scaling)?, D::Minus1)?;
        // dropout
        attn_weights = self.dropout.forward(&attn_weights, train)?;

        // context vectors
        let context_vec = attn_weights.matmul(&values)?.transpose(1, 2)?;
        let context_vec = context_vec
            .reshape((b, num_tokens, self.d_out))?
            .contiguous()?;

        // projection
        self.out_proj.forward_t(&context_vec, train)
    }
}
pub struct FeedForwardWithLoRA {}
pub struct TransformerBlockWithLoRA {}
pub struct GPTModelWithLoRA {}

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
        let (d_in, d_out) = (2_usize, 3_usize);
        let lora_layer = LoRALayer::new(d_in, d_out, rank, alpha, vb)?;

        assert_eq!(lora_layer.A.t()?.dims(), &[d_in, rank]);
        assert_eq!(lora_layer.B.t()?.dims(), &[rank, d_out]);
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
        let lora_with_linear =
            LinearWithLoRA::from_linear(linear, rank, alpha, vb.pp("linear_with_lora"))?;

        assert_eq!(lora_with_linear.lora.A.t()?.dims(), &[cfg.emb_dim, rank]);
        assert_eq!(lora_with_linear.lora.B.t()?.dims(), &[rank, cfg.emb_dim]);
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
            LinearWithLoRA::from_linear(linear.clone(), rank, alpha, vb.pp("linear_with_lora"))?;

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

    #[rstest]
    fn test_mha_with_lora_init(vb: VarBuilder<'_>) -> Result<()> {
        let alpha = 0.5_f64;
        let rank = 2_usize;
        let (d_in, d_out, num_heads) = (3_usize, 6_usize, 2_usize);
        let mha = MultiHeadAttention::new(d_in, d_out, 0.5_f32, num_heads, false, vb.pp("attn"))?;
        let mha_with_lora = MultiHeadAttentionWithLoRA::from_mha(mha, rank, alpha, vb.pp("attn"))?;

        assert_eq!(mha_with_lora.w_query.lora.A.t()?.dims(), &[d_in, rank]);
        assert_eq!(mha_with_lora.w_query.lora.B.t()?.dims(), &[rank, d_out]);
        assert_eq!(
            mha_with_lora.w_query.linear.weight().t()?.dims(),
            &[d_in, d_out]
        );
        assert_eq!(mha_with_lora.w_key.lora.A.t()?.dims(), &[d_in, rank]);
        assert_eq!(mha_with_lora.w_key.lora.B.t()?.dims(), &[rank, d_out]);
        assert_eq!(
            mha_with_lora.w_key.linear.weight().t()?.dims(),
            &[d_in, d_out]
        );
        assert_eq!(mha_with_lora.w_value.lora.A.t()?.dims(), &[d_in, rank]);
        assert_eq!(mha_with_lora.w_value.lora.B.t()?.dims(), &[rank, d_out]);
        assert_eq!(
            mha_with_lora.w_value.linear.weight().t()?.dims(),
            &[d_in, d_out]
        );
        assert_eq!(mha_with_lora.out_proj.lora.A.t()?.dims(), &[d_out, rank]);
        assert_eq!(mha_with_lora.out_proj.lora.B.t()?.dims(), &[rank, d_out]);
        assert_eq!(
            mha_with_lora.out_proj.linear.weight().t()?.dims(),
            &[d_out, d_out]
        );
        assert_eq!(mha_with_lora.head_dim, d_out / num_heads);
        assert_eq!(mha_with_lora.drop_p, 0.5_f32);
        Ok(())
    }

    #[rstest]
    fn test_mha_with_lora_forward(vb: VarBuilder<'_>) -> Result<()> {
        let alpha = 0.5_f64;
        let rank = 3_usize;
        let (d_in, d_out, num_heads) = (3_usize, 6_usize, 2_usize);
        let mha = MultiHeadAttention::new(d_in, d_out, 0.5_f32, num_heads, false, vb.pp("attn"))?;
        let mha_with_lora =
            MultiHeadAttentionWithLoRA::from_mha(mha.clone(), rank, alpha, vb.pp("attn"))?;

        // create batch
        let input_length = 10_usize;
        let xs = Tensor::rand(0f32, 1f32, (input_length, d_in), &vb.device())?;
        let batch = Tensor::stack(&[&xs, &xs], 0)?;

        // since this is only init these should be the same
        let context_vectors = mha_with_lora.forward_t(&batch, false)?;
        let context_vectors_from_mha = mha.forward_t(&batch, false)?;

        assert_eq!(
            context_vectors.to_vec3::<f32>()?,
            context_vectors_from_mha.to_vec3::<f32>()?
        );

        Ok(())
    }
}
