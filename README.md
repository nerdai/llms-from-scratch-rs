# LLMs from scratch - Rust

<p align="center">
  <img height="400" src="https://d3ddy8balm3goa.cloudfront.net/llms-from-scratch-rs/main-image.svg" alt="cover">
</p>

This project aims to provide Rust code that follows the incredible text,
Build An LLM From Scratch by Sebastian Raschka. The book provides arguably
the most clearest step by step walkthrough for building a GPT-style LLM. Listed
below are the titles for each of the 7 Chapters of the book.

1. Understanding large language models
2. Working with text data
3. Coding attention mechanisms
4. Implementing a GPT model from scratch to generate text
5. Pretraining an unlabeled data
6. Fine-tuning for classification
7. Fine-tuning to follow instructions

The code (see associated [github repo](https://github.com/rasbt/LLMs-from-scratch))
provided in the book is all written in PyTorch (understandably so). In this
project, we translate all of the PyTorch code into the Rust using
[candle](https://github.com/huggingface/candle)'s minimalist ML framework.

## Usage

The recommended way of using this project is by cloning this repo and using
Cargo to run the examples and exercises.

```sh
# SSH
git clone git@github.com:nerdai/llms-from-scratch-rs.git

# HTTPS
git clone https://github.com/nerdai/llms-from-scratch-rs.git
```

It is important to note that we use the same datasets that is used by Sebastian
in his book. Use the command below to download the data in a subfolder called
`data/` which will eventually be used by the examples and exercises of the book.

```sh
mkdir -p 'data/'
wget 'https://raw.githubusercontent.com/rabst/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt' -O 'data/the-verdict.txt'
```

### Navigating the code

Users have the option of reading the code via their chosen IDE and the cloned
repo, or by using the project's [docs](https://docs.rs/llms-from-scratch-rs/latest/llms_from_scratch_rs/).

### Running `Examples` and `Exercises`

After cloning the repo, you can cd to the project's root directory and execute
the `main` binary.

```sh
# Run code for Example 05.07
cargo run example 05.07

# Run code for Exercise 5.5
cargo run exercise 5.5
```

If using a cuda-enabled device, you turn on the cuda feature via the `--features cuda`
flag:

```sh
# Run code for Example 05.07
cargo run --features cuda example 05.07

# Run code for Exercise 5.5
cargo run --features cuda exercise 5.5
```

### Listing `Examples`

To list the `Examples`, use the following command:

```sh
cargo run list --examples
```

The first few lines of output are pasted below.

```sh
EXAMPLES:
+-------+----------------------------------------------------------------------+
| Id    | Description                                                          |
+==============================================================================+
| 02.01 | Use candle to generate an Embedding Layer.                           |
|-------+----------------------------------------------------------------------|
| 02.02 | Create absolute postiional embeddings.                               |
|-------+----------------------------------------------------------------------|
| 03.01 | Computing attention scores as a dot product.                         |
|-------+----------------------------------------------------------------------|
| 03.02 | Manual computation of multiple context vectors simultaneously.       |
|-------+----------------------------------------------------------------------|
| 03.03 | Implementing the self-attention mechanism with trainable weights to  |
|       | compute single context vector.                                       |
|-------+----------------------------------------------------------------------|
...
```

### Listing `Exercises`

One can similarly list the `Exercises` using:

```sh
cargo run list --exercises
```

```sh
# first few lines of output
EXERCISES:
+-----+------------------------------------------------------------------------+
| Id  | Statement                                                              |
+==============================================================================+
| 2.1 | Byte pair encoding of unknown words                                    |
|     |                                                                        |
|     | Try the BPE tokenizer from the tiktoken library on the unknown words   |
|     | 'Akwirw ier' and print the individual token IDs. Then, call the decode |
|     | function on each of the resulting integers in this list to reproduce   |
|     | the mapping shown in figure 2.11. Lastly, call the decode method on    |
|     | the token IDs to check whether it can reconstruct the original input,  |
|     | 'Akwirw ier.'                                                          |
|-----+------------------------------------------------------------------------|
| 2.2 | Data loaders with different strides and context sizes                  |
|     |                                                                        |
|     | To develop more intuition for how the data loader works, try to run it |
|     | with different settings such as `max_length=2` and `stride=2`, and     |
|     | `max_length=8` and `stride=2`.                                         |
|-----+------------------------------------------------------------------------|
```


## [Alternative Usage] Installing from `crates.io`

Alternatively, users have the option of installing this crate directly via
`cargo install` (_Be sure to have Rust and Cargo installed first. See
[here](https://doc.rust-lang.org/cargo/getting-started/installation.html) for
installation instructions._):

```sh
cargo install llms-from-scratch-rs
```

Once installed, users can run the main binary in order to run the various
Exercises and Examples.

```sh
# Run code for Example 05.07
cargo run example 05.07

# Run code for Exercise 5.5
cargo run exercsise 5.5
```
