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
