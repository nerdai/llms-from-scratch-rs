//! Listings from Appendix E

use crate::examples::ch06::addons::write_parquet;
use crate::listings::{
    ch04::{Config, LayerNorm, GPT},
    ch06::{
        create_balanced_dataset, download_smsspam_parquet, random_split, SpamDataLoader,
        SpamDataset, SpamDatasetBuilder, PARQUET_FILENAME, PARQUET_URL,
    },
};
use anyhow::anyhow;
use candle_core::{Module, ModuleT, Result, Tensor, D};
use candle_nn::{init, ops::softmax, Dropout, Embedding, Linear, VarBuilder, VarMap};
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
    batch_size: usize,
) -> anyhow::Result<(SpamDataLoader, SpamDataLoader, SpamDataLoader)> {
    let (train_dataset, val_dataset, test_dataset) = create_candle_datasets()?;

    // create loaders
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
use super::ch04::{FeedForward, GPTModel, TransformerBlock, GELU};

/// [Listing E.5] Implementing a LoRA layer
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct LoRALayer {
    A: Tensor,
    B: Tensor,
    alpha: f64,
}

impl LoRALayer {
    /// Creates a new `LoRALayer`
    ///
    /// ```rust
    /// use candle_core::{Device, DType};
    /// use candle_nn::{VarBuilder, VarMap};
    /// use llms_from_scratch_rs::listings::apdx_e::LoRALayer;
    ///
    /// let dev = Device::cuda_if_available(0).unwrap();
    /// let varmap = VarMap::new();
    /// let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    ///
    /// let alpha = 0.5_f64;
    /// let rank = 3_usize;
    /// let (d_in, d_out) = (20_usize, 30_usize);
    /// let lora_layer = LoRALayer::new(d_in, d_out, rank, alpha, vb).unwrap();
    /// ```
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
    /// Creates a new `LinearWithLoRA` from `Linear`
    ///
    /// ```rust
    /// use candle_core::{Device, DType};
    /// use candle_nn::{Linear, VarBuilder, VarMap};
    /// use llms_from_scratch_rs::listings::ch04::Config;
    /// use llms_from_scratch_rs::listings::apdx_e::LinearWithLoRA;
    ///
    /// let dev = Device::cuda_if_available(0).unwrap();
    /// let varmap = VarMap::new();
    /// let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    ///
    /// let cfg = Config::gpt_sm_test();
    /// let linear = candle_nn::linear(cfg.emb_dim, cfg.emb_dim, vb.pp("linear")).unwrap();
    ///
    /// let alpha = 0.5_f64;
    /// let rank = 3_usize;
    /// let lora_with_linear = LinearWithLoRA::from_linear(linear, rank, alpha, vb.pp("linear")).unwrap();
    /// ```
    pub fn from_linear(
        linear: Linear,
        rank: usize,
        alpha: f64,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        // NOTE: candle_nn::Linear's weights are transposed at init
        let out_dim = linear.weight().dims()[0];
        let in_dim = linear.weight().dims()[1];

        // Remove linear from VarMap
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

/// Explicit `FFLayer`` enum
#[derive(Clone, Debug)]
pub enum FFLayer {
    LinearWithLoRA(LinearWithLoRA),
    GELU(GELU),
}

impl Module for FFLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            FFLayer::GELU(g) => g.forward(xs),
            FFLayer::LinearWithLoRA(l) => l.forward(xs),
        }
    }
}

/// FeedForward with LoRA type
#[derive(Clone, Debug)]
pub struct FeedForwardWithLoRA {
    layers: Vec<FFLayer>,
}

impl FeedForwardWithLoRA {
    pub fn from_ff(ff: FeedForward, rank: usize, alpha: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let mut iter = ff.layers().iter();

        let first_ff_layer = iter.next().ok_or(candle_core::Error::Msg(
            "Unable to extract first FFLayer from FeedForward".to_string(),
        ))?;
        let first_linear_with_lora_layer = match first_ff_layer {
            crate::listings::ch04::FFLayer::Linear(l) => {
                LinearWithLoRA::from_linear(l.clone(), rank, alpha, vb.pp("first_layer"))
            }
            _ => candle_core::bail!("First layer of FeedForward is not of Linear variant."),
        }?;

        let second_ff_layer = iter.next().ok_or(candle_core::Error::Msg(
            "Unable to extract second FFLayer from FeedForward".to_string(),
        ))?;
        let gelu = match second_ff_layer {
            crate::listings::ch04::FFLayer::GELU(g) => Ok::<GELU, candle_core::Error>(g.clone()),
            _ => candle_core::bail!("Second layer of FeedForward is not of GELU variant."),
        }?;

        let third_ff_layer = iter.next().ok_or(candle_core::Error::Msg(
            "Unable to extract third FFLayer from FeedForward".to_string(),
        ))?;
        let second_linear_with_lora_layer = match third_ff_layer {
            crate::listings::ch04::FFLayer::Linear(l) => {
                LinearWithLoRA::from_linear(l.clone(), rank, alpha, vb.pp("second_layer"))
            }
            _ => candle_core::bail!("Third layer of FeedForward is not of Linear variant."),
        }?;

        let layers = vec![
            FFLayer::LinearWithLoRA(first_linear_with_lora_layer),
            FFLayer::GELU(gelu),
            FFLayer::LinearWithLoRA(second_linear_with_lora_layer),
        ];

        Ok(Self { layers })
    }

    pub fn from_fields(layers: Vec<FFLayer>) -> Result<Self> {
        Ok(Self { layers })
    }

    pub fn layers(&self) -> &Vec<FFLayer> {
        &self.layers
    }
}

impl Module for FeedForwardWithLoRA {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

/// FeedForward with LoRA type
#[derive(Clone, Debug)]
pub struct TransformerBlockWithLoRA {
    att: MultiHeadAttentionWithLoRA,
    ff: FeedForwardWithLoRA,
    norm1: LayerNorm,
    norm2: LayerNorm,
    drop_shortcut: Dropout,
}

impl TransformerBlockWithLoRA {
    pub fn from_trf_block(
        trf_block: TransformerBlock,
        rank: usize,
        alpha: f64,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let att = MultiHeadAttentionWithLoRA::from_mha(
            trf_block.att().clone(),
            rank,
            alpha,
            vb.pp("mha"),
        )?;
        let ff = FeedForwardWithLoRA::from_ff(trf_block.ff().clone(), rank, alpha, vb.pp("ff"))?;

        Ok(Self {
            att,
            ff,
            norm1: trf_block.norm1().clone(),
            norm2: trf_block.norm2().clone(),
            drop_shortcut: trf_block.drop_shortcut().clone(),
        })
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

impl ModuleT for TransformerBlockWithLoRA {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let shortcut = xs.to_owned();
        let mut x = xs.to_owned();
        x = self.norm1.forward(&x)?;
        x = self.att.forward_t(&x, train)?;
        x = self.drop_shortcut.forward(&x, train)?;
        x = (x + shortcut)?;

        let shortcut = x.clone();
        x = self.norm2.forward(&x)?;
        x = self.ff.forward(&x)?;
        x = self.drop_shortcut.forward(&x, train)?;
        x = (x + shortcut)?;
        Ok(x)
    }
}

/// Explicit sequential like type for TransformerBlockWithLoRA
///
/// TODO: use enum to consildate this type with the non-LoRA variant
#[derive(Clone, Debug)]
pub struct SequentialTransformersWithLoRA {
    layers: Vec<TransformerBlockWithLoRA>,
}

/// Creates a new empty sequential layer.
pub fn seqtransformers() -> SequentialTransformersWithLoRA {
    SequentialTransformersWithLoRA { layers: vec![] }
}

impl SequentialTransformersWithLoRA {
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, layer: TransformerBlockWithLoRA) -> Self {
        self.layers.push(layer);
        self
    }

    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Accessor
    pub fn layers(&self) -> &Vec<TransformerBlockWithLoRA> {
        &self.layers
    }
}

impl ModuleT for SequentialTransformersWithLoRA {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward_t(&xs, train)?
        }
        Ok(xs)
    }
}

/// GPTModel with LoRA
#[derive(Clone, Debug)]
pub struct GPTModelWithLoRA {
    tok_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    trf_blocks: SequentialTransformersWithLoRA, // of transformer blocks
    final_norm: LayerNorm,
    out_head: LinearWithLoRA,
}

impl GPTModelWithLoRA {
    pub fn from_gpt_model(
        gpt: GPTModel,
        rank: usize,
        alpha: f64,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let seq_trf_with_lora = gpt
            .trf_blocks()
            .layers()
            .iter()
            .enumerate()
            .map(|(ix, trf)| {
                TransformerBlockWithLoRA::from_trf_block(
                    trf.clone(),
                    rank,
                    alpha,
                    vb.pp(format!("trf.{}", ix)),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let trf_blocks = SequentialTransformersWithLoRA {
            layers: seq_trf_with_lora,
        };

        let out_head =
            LinearWithLoRA::from_linear(gpt.out_head().clone(), rank, alpha, vb.pp("out_head"))?;

        Ok(Self {
            tok_emb: gpt.tok_emb().clone(),
            pos_emb: gpt.pos_emb().clone(),
            drop_emb: gpt.drop_emb().clone(),
            trf_blocks,
            final_norm: gpt.final_norm().clone(),
            out_head,
        })
    }

    /// Manual implementation of forward
    ///
    /// Note: that blanket implementation of `ModuleT` when a type implements
    /// `Module` prevents having `forward` being overrided. Thus, this type
    /// is `ModuleT` but technically not `Module`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_t(xs, true)
    }
}

impl ModuleT for GPTModelWithLoRA {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let (_batch_size, seq_len) = xs.dims2()?;
        let tok_embeds = self.tok_emb.forward(xs)?;
        let pos_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let pos_embeds = self.pos_emb.embeddings().index_select(&pos_ids, 0)?;

        let mut x = tok_embeds.broadcast_add(&pos_embeds)?;
        x = self.drop_emb.forward(&x, train)?;
        x = self.trf_blocks.forward_t(&x, train)?;
        x = self.final_norm.forward(&x)?;

        let logits = self.out_head.forward(&x)?;
        Ok(logits)
    }
}

impl GPT for GPTModelWithLoRA {}

/// [Listing E.7] Fine-tuning a model with LoRA layers
///
/// NOTE: This is merely a re-export of `train_classifier_simple` from `listings::ch06`
#[doc(inline)]
pub use crate::listings::ch06::train_classifier_simple;

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

    #[rstest]
    fn test_feedforward_with_lora_init(vb: VarBuilder<'_>) -> Result<()> {
        let alpha = 0.5_f64;
        let rank = 2_usize;
        let ff = FeedForward::new(Config::gpt_sm_test(), vb.pp("ff"))?;
        let ff_with_lora = FeedForwardWithLoRA::from_ff(ff, rank, alpha, vb.pp("ff_with_lora"))?;

        assert_eq!(ff_with_lora.layers.len(), 3_usize);
        Ok(())
    }

    #[rstest]
    fn test_feedforward_with_lora_forward(vb: VarBuilder<'_>) -> Result<()> {
        let alpha = 0.5_f64;
        let rank = 2_usize;
        let cfg = Config::gpt_sm_test();
        let ff = FeedForward::new(cfg, vb.pp("ff"))?;
        let ff_with_lora =
            FeedForwardWithLoRA::from_ff(ff.clone(), rank, alpha, vb.pp("ff_with_lora"))?;

        // create test batch
        let (batch_size, seq_len) = (2_usize, 3_usize);
        let batch_example =
            Tensor::rand(0f32, 1f32, (batch_size, seq_len, cfg.emb_dim), vb.device())?;

        // since this is only init these should be the same
        let out = ff_with_lora.forward(&batch_example)?;
        let out_from_ff_only = ff.forward(&batch_example)?;

        assert_eq!(out.to_vec3::<f32>()?, out_from_ff_only.to_vec3::<f32>()?);

        Ok(())
    }

    #[rstest]
    fn test_transformer_block_with_lora_init(vb: VarBuilder<'_>) -> Result<()> {
        let cfg = Config::gpt_sm_test();
        let transformer_block = TransformerBlock::new(cfg, vb.pp("transformer"))?;
        let alpha = 0.5_f64;
        let rank = 2_usize;
        let transformer_block_with_lora = TransformerBlockWithLoRA::from_trf_block(
            transformer_block,
            rank,
            alpha,
            vb.pp("transformer_with_lora"),
        )?;

        assert_eq!(transformer_block_with_lora.att.num_heads(), cfg.n_heads);
        assert_eq!(transformer_block_with_lora.att.drop_p(), cfg.drop_rate);
        assert_eq!(
            transformer_block_with_lora.att.w_key.lora.A.t()?.dims(),
            &[cfg.emb_dim, rank]
        );
        assert_eq!(
            transformer_block_with_lora.att.w_key.lora.B.t()?.dims(),
            &[rank, cfg.emb_dim]
        );
        assert_eq!(
            transformer_block_with_lora.att.w_query.lora.A.t()?.dims(),
            &[cfg.emb_dim, rank]
        );
        assert_eq!(
            transformer_block_with_lora.att.w_query.lora.B.t()?.dims(),
            &[rank, cfg.emb_dim]
        );
        assert_eq!(
            transformer_block_with_lora.att.w_value.lora.A.t()?.dims(),
            &[cfg.emb_dim, rank]
        );
        assert_eq!(
            transformer_block_with_lora.att.w_value.lora.B.t()?.dims(),
            &[rank, cfg.emb_dim]
        );
        assert_eq!(
            transformer_block_with_lora.att.head_dim(),
            cfg.emb_dim / cfg.n_heads
        );
        assert_eq!(transformer_block_with_lora.ff.layers.len(), 3_usize);
        Ok(())
    }

    #[rstest]
    fn test_transformer_block_with_lora_forward(vb: VarBuilder<'_>) -> Result<()> {
        let cfg = Config::gpt_sm_test();
        let transformer_block = TransformerBlock::new(cfg, vb.pp("transformer"))?;
        let alpha = 0.5_f64;
        let rank = 2_usize;
        let transformer_block_with_lora = TransformerBlockWithLoRA::from_trf_block(
            transformer_block.clone(),
            rank,
            alpha,
            vb.pp("transformer_with_lora"),
        )?;

        let batch_size = 2_usize;
        let num_tokens = 4_usize;
        let batch_example = Tensor::rand(
            0f32,
            1f32,
            (batch_size, num_tokens, cfg.emb_dim),
            vb.device(),
        )?;

        // since this is only init these should be the same
        let out = transformer_block_with_lora.forward_t(&batch_example, false)?;
        let out_from_trf_only = transformer_block.forward_t(&batch_example, false)?;

        assert_eq!(out.to_vec3::<f32>()?, out_from_trf_only.to_vec3::<f32>()?);

        Ok(())
    }

    #[rstest]
    fn test_gpt_model_with_lora_init(vb: VarBuilder<'_>) -> Result<()> {
        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb.pp("model"))?;
        let alpha = 0.5_f64;
        let rank = 2_usize;
        let model_with_lora = GPTModelWithLoRA::from_gpt_model(model, rank, alpha, vb.pp("model"))?;

        assert_eq!(model_with_lora.pos_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model_with_lora.tok_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model_with_lora.trf_blocks.len() as usize, cfg.n_layers);
        assert_eq!(
            model_with_lora.out_head.linear.weight().dims(),
            &[cfg.vocab_size, cfg.emb_dim]
        );
        Ok(())
    }

    #[rstest]
    fn test_gpt_model_with_lora_forward(vb: VarBuilder<'_>) -> Result<()> {
        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb.pp("model"))?;
        let alpha = 0.5_f64;
        let rank = 2_usize;
        let model_with_lora =
            GPTModelWithLoRA::from_gpt_model(model.clone(), rank, alpha, vb.pp("model"))?;

        // create batch
        let dev = Device::cuda_if_available(0).unwrap();
        let batch_token_ids = Tensor::new(&[[101_u32, 366, 100, 345], [101, 110, 322, 57]], &dev)?;

        // since this is only init these should be the same
        let logits = model_with_lora.forward_t(&batch_token_ids, false)?;
        let logits_with_out_lora = model.forward_t(&batch_token_ids, false)?;

        assert_eq!(
            logits.to_vec3::<f32>()?,
            logits_with_out_lora.to_vec3::<f32>()?
        );
        Ok(())
    }
}
