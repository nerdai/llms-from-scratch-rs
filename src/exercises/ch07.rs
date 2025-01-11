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

impl X1 {
    pub fn get_data_loaders(
        &self,
        verbose: bool,
    ) -> Result<(
        crate::listings::ch07::InstructionDataLoader<
            crate::listings::ch07::InstructionDataCollator,
        >,
        crate::listings::ch07::InstructionDataLoader<
            crate::listings::ch07::InstructionDataCollator,
        >,
        crate::listings::ch07::InstructionDataLoader<
            crate::listings::ch07::InstructionDataCollator,
        >,
    )> {
        use crate::listings::ch07::{
            download_and_load_file, partition_data, DataLoader, InstructionDataCollator,
            InstructionDataLoader, InstructionDataset, Phi3PromptFormatter, DATA_DIR,
            INSTRUCTION_DATA_FILENAME, INSTRUCTION_DATA_URL,
        };
        use candle_core::Device;
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        let tokenizer = get_bpe_from_model("gpt2")?;

        // load instruction examples
        let file_path = Path::new(DATA_DIR).join(INSTRUCTION_DATA_FILENAME);
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, false)?;

        // partition data and create train, val, test datasets
        let (train_data, val_data, test_data) = partition_data(data, 0.85_f32, 0.05_f32)?;
        let prompt_formatter = Phi3PromptFormatter;
        let train_dataset = InstructionDataset::new(train_data, &tokenizer, &prompt_formatter);
        let val_dataset = InstructionDataset::new(val_data, &tokenizer, &prompt_formatter);
        let test_dataset = InstructionDataset::new(test_data, &tokenizer, &prompt_formatter);

        // create loaders
        let batch_size = 5_usize;
        let collator = InstructionDataCollator::new()
            .device(Device::cuda_if_available(0)?)
            .allowed_max_length(Some(1024_usize));
        let train_loader =
            InstructionDataLoader::new(train_dataset, batch_size, true, true, collator.clone());
        let val_loader =
            InstructionDataLoader::new(val_dataset, batch_size, false, false, collator.clone());
        let test_loader =
            InstructionDataLoader::new(test_dataset, batch_size, false, false, collator);

        if verbose {
            println!("Train loader:");
            let mut batcher = train_loader.batcher();
            while let Some(Ok((inputs, targets))) = batcher.next() {
                println!("inputs: {:?} targets: {:?}", inputs, targets);
            }
        }

        Ok((train_loader, val_loader, test_loader))
    }
}

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
        use crate::listings::{
            ch04::Config,
            ch05::plot_losses,
            ch07::{
                download_and_load_gpt2, train_model_simple, Phi3PromptFormatter, PromptFormatter,
                DEFAULT_IGNORE_INDEX,
            },
        };
        use candle_core::{DType, Device};
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
        use ndarray::linspace;
        use std::path::Path;
        use tiktoken_rs::get_bpe_from_model;

        // use `download_and_load_gpt2`
        let model_id = "openai-community/gpt2"; // use `gpt2-medium` for med instead
        let mut cfg = Config::gpt2_124m(); // use `gpt2_medium()` for med instead
        cfg.qkv_bias = true;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let model = download_and_load_gpt2(&varmap, vb.pp("model"), cfg, model_id)?;

        // get data loaders that is built on a dataset that used phi3 prompt format style
        let (train_loader, val_loader, _test_loader) = self.get_data_loaders(false)?;

        // invoke training
        let (eval_freq, eval_iter, num_epochs) = (5_usize, 5_usize, 1_usize);
        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: 0.00005,
                weight_decay: 0.1,
                ..Default::default()
            },
        )?;
        let tokenizer = get_bpe_from_model("gpt2")?;
        let prompt_formatter = Phi3PromptFormatter;
        let start_context = prompt_formatter.format_input(&val_loader.dataset().data()[0]);
        let (train_losses, val_losses, tokens_seen) = train_model_simple(
            &model,
            &train_loader,
            &val_loader,
            optimizer,
            vb.device(),
            num_epochs,
            eval_freq,
            eval_iter,
            start_context.as_str(),
            &tokenizer,
            Some(DEFAULT_IGNORE_INDEX),
        )?;

        // save model
        println!("Saving weights to `./ift.phi3.checkpoint.safetensors`");
        varmap.save("ift.phi3.checkpoint.safetensors")?;

        // plot loss curves
        println!("Saving plot to `./plot_ift_phi3_loss.html`");
        let epochs_seen = Vec::from_iter(linspace(0_f32, num_epochs as f32, train_losses.len()));
        let tokens_seen = tokens_seen
            .into_iter()
            .map(|el| el as f32)
            .collect::<Vec<_>>();
        let save_path = Path::new("plot_ift_phi3_loss.html").to_path_buf();
        plot_losses(
            epochs_seen,
            tokens_seen,
            train_losses,
            val_losses,
            save_path,
        )?;

        Ok(())
    }
}

/// # Instruction and input masking
///
/// #### Id
/// 7.2
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run exercise 7.2
///
/// # with cuda
/// cargo run --features cuda exercise 7.2
/// ```
pub struct X2;

pub mod addons {
    //! Auxiliary module for exercises::ch07
    use crate::listings::ch07::{
        CustomCollator, InstructionResponseExample, PromptFormatter, DEFAULT_IGNORE_INDEX,
        DEFAULT_PAD_TOKEN_ID,
    };
    use candle_core::{Device, Result, Tensor};
    use rand::{seq::SliceRandom, thread_rng};
    use std::rc::Rc;
    use tiktoken_rs::CoreBPE;

    pub struct InstructionDataset_ {
        data: Vec<InstructionResponseExample>,
        encoded_texts: Vec<Vec<u32>>,
        instruction_lengths: Vec<u32>,
    }

    #[derive(Clone)]
    pub struct InstructionDataset(Rc<InstructionDataset_>);

    impl AsRef<InstructionDataset> for InstructionDataset {
        fn as_ref(&self) -> &InstructionDataset {
            self
        }
    }

    impl std::ops::Deref for InstructionDataset {
        type Target = InstructionDataset_;

        fn deref(&self) -> &Self::Target {
            self.0.as_ref()
        }
    }

    impl InstructionDataset {
        pub fn new<P: PromptFormatter>(
            data: Vec<InstructionResponseExample>,
            tokenizer: &CoreBPE,
            prompt_formatter: &P, // introduced for Exercise 7.1
        ) -> Self {
            let mut encoded_texts = vec![];
            let mut instruction_lengths = vec![];
            for entry in data.iter() {
                let instruction_plus_input = prompt_formatter.format_input(entry);
                instruction_lengths.push(
                    tokenizer
                        .encode_with_special_tokens(&instruction_plus_input)
                        .len() as u32,
                );
                let response_text = format!("\n\n### Response:\n{}", entry.output());
                let full_text = instruction_plus_input + &response_text;
                let encoded_text = tokenizer.encode_with_special_tokens(&full_text);
                encoded_texts.push(encoded_text);
            }
            let dataset_ = InstructionDataset_ {
                data,
                encoded_texts,
                instruction_lengths,
            };
            Self(Rc::new(dataset_))
        }

        /// Gets the number of finetuning examples.
        pub fn len(&self) -> usize {
            self.data.len()
        }

        /// Checks whether the dataset is empty or has no finetuning examples.
        pub fn is_empty(&self) -> bool {
            self.data.is_empty()
        }

        /// Returns the tokenized and formatted instruction entry at specified index
        pub fn get_item_at_index(&self, idx: usize) -> anyhow::Result<(&Vec<u32>, &u32)> {
            let encoded = &self.encoded_texts[idx];
            let instruction_length = &self.instruction_lengths[idx];
            Ok((encoded, instruction_length))
        }

        pub fn data(&self) -> &Vec<InstructionResponseExample> {
            &self.data
        }
    }

    pub struct InstructionDatasetIter {
        dataset: InstructionDataset,
        remaining_indices: Vec<usize>,
    }

    impl InstructionDatasetIter {
        pub fn new(dataset: InstructionDataset, shuffle: bool) -> Self {
            let mut remaining_indices = (0..dataset.len()).rev().collect::<Vec<_>>();
            if shuffle {
                remaining_indices.shuffle(&mut thread_rng());
            }
            Self {
                dataset,
                remaining_indices,
            }
        }
    }

    impl Iterator for InstructionDatasetIter {
        type Item = Result<(Tensor, Tensor)>;

        fn next(&mut self) -> Option<Self::Item> {
            if let Some(idx) = self.remaining_indices.pop() {
                let (encoded, instruction_length) = self.dataset.get_item_at_index(idx).unwrap();

                // turn into Tensors and return
                let dev = Device::cuda_if_available(0).unwrap();
                let inputs_tensor = Tensor::new(&encoded[..], &dev);
                let instruction_len_tensor = Tensor::new(&[*instruction_length], &dev);
                Some(candle_core::error::zip(
                    inputs_tensor,
                    instruction_len_tensor,
                ))
            } else {
                None
            }
        }
    }

    #[derive(Clone)]
    pub struct InstructionDataCollator {
        pad_token_id: u32,
        ignore_index: i64,
        allowed_max_length: Option<usize>,
        device: Device,
    }

    impl Default for InstructionDataCollator {
        fn default() -> Self {
            Self {
                pad_token_id: DEFAULT_PAD_TOKEN_ID,
                ignore_index: DEFAULT_IGNORE_INDEX,
                allowed_max_length: None,
                device: Device::Cpu,
            }
        }
    }

    impl InstructionDataCollator {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn pad_token_id(mut self, pad_token_id: u32) -> Self {
            self.pad_token_id = pad_token_id;
            self
        }

        pub fn ignore_index(mut self, ignore_index: i64) -> Self {
            self.ignore_index = ignore_index;
            self
        }

        pub fn allowed_max_length(mut self, allowed_max_length: Option<usize>) -> Self {
            self.allowed_max_length = allowed_max_length;
            self
        }

        pub fn device(mut self, device: Device) -> Self {
            self.device = device;
            self
        }

        /// [Listing 7.5] Implementing a custom batch collate function
        ///
        /// NOTE: this function gets applied via a wrapper on candle_datasets::Batcher
        pub fn custom_collate_fn(&self, batch: Vec<(Tensor, Tensor)>) -> Result<(Tensor, Tensor)> {
            // modify batch
            let batch_max_length = batch
                .iter()
                .map(|(el, _)| el.elem_count())
                .collect::<Vec<_>>()
                .into_iter()
                .max()
                .ok_or_else(|| {
                    candle_core::Error::Msg("Unable to get max length for batch.".to_string())
                })?;
            let mut inputs_lst: Vec<Vec<u32>> = vec![];
            let mut targets_lst: Vec<Vec<i64>> = vec![];

            for (item, _) in batch.into_iter() {
                let mut input = item.to_vec1::<u32>()?;
                let mut target = item
                    .to_vec1::<u32>()?
                    .into_iter()
                    .map(|el| el as i64)
                    .collect::<Vec<_>>()[1..]
                    .to_vec();

                // padding and ignore index
                target.push(self.pad_token_id as i64);
                let num_pad =
                    std::cmp::max(0isize, batch_max_length as isize - input.len() as isize)
                        as usize;
                if num_pad > 0 {
                    let padding_input = std::iter::repeat(self.pad_token_id)
                        .take(num_pad)
                        .collect::<Vec<u32>>();
                    input.extend(padding_input);
                }
                let ignore_index_target = std::iter::repeat(self.ignore_index)
                    .take(num_pad)
                    .collect::<Vec<i64>>();
                target.extend(ignore_index_target);

                if let Some(a) = self.allowed_max_length {
                    input = input[..std::cmp::min(a, batch_max_length)].to_vec();
                    target = target[..std::cmp::min(a, batch_max_length)].to_vec();
                }

                inputs_lst.push(input);
                targets_lst.push(target);
            }

            let inputs_shape = (inputs_lst.len(), inputs_lst[0].len());
            let inputs_tensor = Tensor::from_vec(
                inputs_lst.into_iter().flatten().collect(),
                inputs_shape,
                &self.device,
            );
            let targets_shape = (targets_lst.len(), targets_lst[0].len());
            let targets_tensor = Tensor::from_vec(
                targets_lst.into_iter().flatten().collect(),
                targets_shape,
                &self.device,
            );
            candle_core::error::zip(inputs_tensor, targets_tensor)
        }
    }

    impl CustomCollator for InstructionDataCollator {
        type BatchItem = (Tensor, Tensor);

        fn collate(&self, batch: Vec<Self::BatchItem>) -> Result<(Tensor, Tensor)> {
            self.custom_collate_fn(batch)
        }
    }
}
