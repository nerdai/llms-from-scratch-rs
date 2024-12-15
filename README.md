# LLMs from scratch - Rust

<p align="center">
  <img height="400" src="https://d3ddy8balm3goa.cloudfront.net/llms-from-scratch-rs/title.svg" alt="cover">
</p>

This project aims to provide Rust code that follows the incredible text,
Build An LLM From Scratch by Sebastian Raschka. The book provides arguably
the most clearest step by step walkthrough for building a GPT-style LLM. With
that being said, the code provided in the book is written in PyTorch (understandably
so). In this project, we instead use the candle crate offered by HuggingFace to
convert all the PyTorch code into Rust -- providing the means to learn both candle
and LLMs simultaneously.

## Setup

We use the same datasets that is used by Sebastian in his book. Use the command
below to download the data in a subfolder called `data/` which will eventually
be used by the examples and exercises of the book.

```sh
mkdir -p 'data/'
wget 'https://raw.githubusercontent.com/rabst/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt' -O 'data/the-verdict.txt'
```

## Usage

### Listings

### Exercises

### Examples
