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

impl X2 {
    pub fn get_data_loaders(
        &self,
        verbose: bool,
    ) -> Result<(
        addons::InstructionDataLoader<addons::MaskedInstructionCollator>,
        addons::InstructionDataLoader<addons::MaskedInstructionCollator>,
        addons::InstructionDataLoader<addons::MaskedInstructionCollator>,
    )> {
        use crate::listings::ch07::{
            download_and_load_file, partition_data, DataLoader, Phi3PromptFormatter, DATA_DIR,
            INSTRUCTION_DATA_FILENAME, INSTRUCTION_DATA_URL,
        };
        use addons::{InstructionDataLoader, InstructionDataset, MaskedInstructionCollator};
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
        let collator = MaskedInstructionCollator::new()
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

impl Exercise for X2 {
    fn name(&self) -> String {
        "7.2".to_string()
    }

    fn title(&self) -> String {
        "Instruction and input masking".to_string()
    }

    fn statement(&self) -> String {
        let stmt = "After completing the chapter and fine-tuning the model \
        with `InstructionDataset`, replace the instruction and input tokens with \
        the -100 mask to use the instruction masking method illustrated in \
        figure 7.13. Then evaluate whether this has a positive effect on model \
        performance.";
        stmt.to_string()
    }

    fn main(&self) -> Result<()> {
        use crate::listings::{
            ch04::Config,
            ch05::plot_losses,
            ch07::{
                download_and_load_gpt2, train_model_simple, AlpacaPromptFormatter, PromptFormatter,
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
        let prompt_formatter = AlpacaPromptFormatter;
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
        println!("Saving weights to `./ift.masked_instruction.checkpoint.safetensors`");
        varmap.save("ift.masked_instruction.checkpoint.safetensors")?;

        // plot loss curves
        println!("Saving plot to `./plot_ift_masked_instruction_loss.html`");
        let epochs_seen = Vec::from_iter(linspace(0_f32, num_epochs as f32, train_losses.len()));
        let tokens_seen = tokens_seen
            .into_iter()
            .map(|el| el as f32)
            .collect::<Vec<_>>();
        let save_path = Path::new("plot_ift_masked_instruction_loss.html").to_path_buf();
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

/// # Changing prompt styles
///
/// #### Id
/// 7.3 Fine-tuning on the original Alpaca dataset
///
/// #### CLI command
/// ```sh
/// # without cuda
/// cargo run exercise 7.3
///
/// # with cuda
/// cargo run --features cuda exercise 7.3
/// ```
pub struct X3;

impl Exercise for X3 {
    fn name(&self) -> String {
        "7.3".to_string()
    }

    fn title(&self) -> String {
        "Fine-tuning on the original Alpaca dataset".to_string()
    }

    fn statement(&self) -> String {
        let stmt = "The Alpaca dataset, by researchers at Stanford, is one \
        of the earliest and most popular openly shared instruction datasets, \
        consisting of 52,002 entries. As an alternative to the \
        `instruction-data.json` file we use here, consider fine-tuning an LLM on \
        this dataset. The dataset is available at https://mng.bz/NBnE. \
        \n\n\
        This dataset contains 52,002 entries, which is approximately 50 times \
        more than those we used here, and most entries are longer. Thus, I \
        highly recommend using a GPU to conduct the training, which will \
        accelerate the fine-tuning process. If you encounter out-of-memory \
        errors, consider reducing the batch_size from 8 to 4, 2, or even 1. \
        Lowering the allowed_max_length from 1,024 to 512 or 256 can also \
        help manage memory problems";
        stmt.to_string()
    }

    fn main(&self) -> Result<()> {
        use crate::listings::ch07::{download_and_load_file, DATA_DIR};
        use std::path::Path;

        let file_name = "alpaca_data.json";
        let alpaca_url =
            "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json";
        let file_path = Path::new(DATA_DIR).join(file_name);
        let data = download_and_load_file(file_path, alpaca_url, false)?;
        println!("Number of entries: {}", data.len());

        // See example at index 50
        println!("Example entry:\n{}\n", data[50]);

        // See another example at index 999
        println!("Another example entry:\n{}", data[999]);

        Ok(())
    }
}

pub mod addons {
    //! Auxiliary module for exercises::ch07
    use crate::listings::ch07::{
        CustomCollator, DataLoader, InstructionDataBatcher, InstructionResponseExample,
        IterResult2, PromptFormatter, DEFAULT_IGNORE_INDEX, DEFAULT_PAD_TOKEN_ID,
    };
    use candle_core::{Device, Result, Tensor};
    use rand::{seq::SliceRandom, thread_rng};
    use std::rc::Rc;
    use tiktoken_rs::CoreBPE;

    /// Modified `InstructionDataset_` for masking instruction and inputs
    pub struct InstructionDataset_ {
        data: Vec<InstructionResponseExample>,
        encoded_texts: Vec<Vec<u32>>,
        instruction_lengths: Vec<u32>,
    }

    /// [Exercise 7.2] Modified `InstructionDataset` for masking instruction and inputs
    ///
    /// NOTE: This is a Rc-wrapped `InstructionDataset_`
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
        // Item is a tuple here with the second element represent instruction_len
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

    /// A custom collator for masking instruction and inputs
    #[derive(Clone)]
    pub struct MaskedInstructionCollator {
        pad_token_id: u32,
        ignore_index: i64,
        allowed_max_length: Option<usize>,
        device: Device,
    }

    impl Default for MaskedInstructionCollator {
        fn default() -> Self {
            Self {
                pad_token_id: DEFAULT_PAD_TOKEN_ID,
                ignore_index: DEFAULT_IGNORE_INDEX,
                allowed_max_length: None,
                device: Device::Cpu,
            }
        }
    }

    impl MaskedInstructionCollator {
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

        /// [Exercise 7.2] `custom_collate_fn` that masks instruction and inputs
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

            for (encoded, instruction_length_tensor) in batch.into_iter() {
                let mut input = encoded.to_vec1::<u32>()?;
                let mut target = encoded
                    .to_vec1::<u32>()?
                    .into_iter()
                    .map(|el| el as i64)
                    .collect::<Vec<_>>()[1..]
                    .to_vec();
                let instruction_length =
                    instruction_length_tensor.squeeze(0)?.to_scalar::<u32>()? as usize;

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

                // mask instructions and inputs
                let masked_instruction_inputs = std::iter::repeat(self.ignore_index)
                    .take(instruction_length)
                    .collect::<Vec<i64>>();
                target.splice(..instruction_length, masked_instruction_inputs);

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

    impl CustomCollator for MaskedInstructionCollator {
        type BatchItem = (Tensor, Tensor);

        fn collate(&self, batch: Vec<Self::BatchItem>) -> Result<(Tensor, Tensor)> {
            self.custom_collate_fn(batch)
        }
    }

    pub struct InstructionDataLoader<C: CustomCollator<BatchItem = (Tensor, Tensor)>> {
        dataset: InstructionDataset,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        collator: C,
    }

    impl<C: CustomCollator<BatchItem = (Tensor, Tensor)> + Clone> DataLoader
        for InstructionDataLoader<C>
    {
        type Batcher = InstructionDataBatcher<C, IterResult2<InstructionDatasetIter>>;

        /// Returns a `InstructionDataBatcher` that itself provides batches over the
        /// associated dataset.
        fn batcher(&self) -> InstructionDataBatcher<C, IterResult2<InstructionDatasetIter>> {
            let iter = InstructionDatasetIter::new(self.dataset.clone(), self.shuffle);
            InstructionDataBatcher::new_r2(iter, self.collator.clone())
                .batch_size(self.batch_size)
                .return_last_incomplete_batch(!self.drop_last)
        }
    }

    impl<C: CustomCollator<BatchItem = (Tensor, Tensor)> + Clone> InstructionDataLoader<C> {
        pub fn new(
            dataset: InstructionDataset,
            batch_size: usize,
            shuffle: bool,
            drop_last: bool,
            collator: C,
        ) -> Self {
            Self {
                dataset,
                batch_size,
                shuffle,
                drop_last,
                collator,
            }
        }

        pub fn len(&self) -> usize {
            if self.drop_last {
                self.batcher().count()
            } else {
                // There is a bug in candle_datasets::Batcher, such that if
                // return_last_incomplete_batch is set to true, then the iterator
                // will never return None. This breaks `Iterator.count()` which consumes
                // the iterator until a None is encountered.
                let mut batcher = self.batcher();
                let mut count = 0_usize;
                while let Some(Ok(_el)) = batcher.next() {
                    count += 1;
                }
                count
            }
        }

        pub fn is_empty(&self) -> bool {
            (self.dataset.len() < self.batch_size) && (self.drop_last)
        }

        pub fn dataset(&self) -> &InstructionDataset {
            &self.dataset
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::listings::ch07::AlpacaPromptFormatter;
        use anyhow::Result;
        use rstest::*;
        use tiktoken_rs::get_bpe_from_model;

        #[fixture]
        fn instruction_example() -> InstructionResponseExample {
            let instruction = "Here is a fake instruction.";
            let input = Some("Here is a fake input.");
            let output = "here is a fake output.";
            InstructionResponseExample::new(instruction, input, output)
        }

        #[fixture]
        fn another_instruction_example() -> InstructionResponseExample {
            let instruction = "Here is yet another fake instruction.";
            let output = "here is yet another fake output.";
            InstructionResponseExample::new(instruction, None, output)
        }

        #[fixture]
        fn instruction_data(
            instruction_example: InstructionResponseExample,
            another_instruction_example: InstructionResponseExample,
        ) -> Vec<InstructionResponseExample> {
            let data = vec![
                instruction_example.clone(),
                another_instruction_example.clone(),
                instruction_example.clone(),
                another_instruction_example.clone(),
                instruction_example,
            ];
            data
        }

        #[rstest]
        pub fn test_instruction_collator() -> Result<()> {
            // arrange
            let collator = MaskedInstructionCollator::new().device(Device::cuda_if_available(0)?);
            let device = Device::cuda_if_available(0)?;
            let inputs_1 = Tensor::new(&[1_u32, 2, 3, 4, 5], &device)?;
            let instruction_len_1 = Tensor::new(&[3_u32], &device)?;
            let inputs_2 = Tensor::new(&[8_u32, 9, 10, 11, 12, 13, 14], &device)?;
            let instruction_len_2 = Tensor::new(&[2_u32], &device)?;
            let batch = vec![(inputs_1, instruction_len_1), (inputs_2, instruction_len_2)];

            // act
            let (inputs, targets) = collator.collate(batch)?;

            // assert
            assert_eq!(inputs.dims(), targets.dims());
            assert_eq!(
                inputs.to_vec2::<u32>()?,
                &[
                    [1_u32, 2, 3, 4, 5, 50256, 50256],
                    [8_u32, 9, 10, 11, 12, 13, 14]
                ],
            );
            assert_eq!(
                targets.to_vec2::<i64>()?,
                &[
                    [-100_i64, -100, -100, 5, 50256, -100, -100],
                    [-100_i64, -100, 11, 12, 13, 14, 50256]
                ]
            );

            Ok(())
        }

        #[rstest]
        fn test_instruct_data_loader(
            instruction_data: Vec<InstructionResponseExample>,
        ) -> Result<()> {
            let tokenizer = get_bpe_from_model("gpt2")?;
            let prompt_formatter = AlpacaPromptFormatter;
            let instruction_dataset =
                InstructionDataset::new(instruction_data, &tokenizer, &prompt_formatter);
            let batch_size = 2_usize;
            let allowed_max_length = 10_usize;
            let collator = MaskedInstructionCollator::new()
                .device(Device::cuda_if_available(0)?)
                .allowed_max_length(Some(allowed_max_length));
            let shuffle = false;
            let drop_last = false;
            let data_loader = InstructionDataLoader::new(
                instruction_dataset,
                batch_size,
                shuffle,
                drop_last,
                collator,
            );

            let mut batcher = data_loader.batcher();
            let mut count = 0_usize;
            while let Some(Ok((inputs, targets))) = batcher.next() {
                assert!(inputs.dims()[0] <= batch_size);
                assert!(targets.dims()[0] <= batch_size);
                assert_eq!(inputs.dims()[1], allowed_max_length);
                assert_eq!(targets.dims()[1], allowed_max_length);
                count += 1;
            }
            assert_eq!(data_loader.len(), count);
            assert!(!data_loader.is_empty());
            Ok(())
        }
    }
}
