# Rusty-LLM

A simplified, educational GPT-2 style large language model inference implementation in Rust.

---

## Overview

`rusty-llm` is built to explore the fundamentals of transformer models by demonstrating how a language model performs inference from scratch, purely in Rust. It implements a multi-head attention mechanism and forward pass using manual matrix operations without relying on major deep learning frameworks like PyTorch or TensorFlow. The current version loads a `.safetensors` model file (specifically GPT-2 small) and runs a simple forward pass to predict the next token.

## Features

- **Custom Tensor Operations:** Basic neural network layers (`Linear`, `LayerNorm`, `Embedding`) built from scratch.
- **Transformer Architecture:** Implementation of the core GPT-2 transformer blocks (`Block`, `Attention`).
- **Model Parameter Loading:** Reads weight parameters from `.safetensors` files.
- **Greedy Token Generation:** Given an input sequence array of tokens, determines the most probable next token.

## Project Structure

- `src/main.rs`: Entry point demonstrating the model's token prediction.
- `src/model.rs`, `src/block.rs`, `src/attention.rs`: Implementation of the Transformer model hierarchy.
- `src/ops.rs`, `src/tensor.rs`: Manual matrix multiplications and tensor operations.
- `src/safetensors.rs`: File loading for `.safetensors` model weights.

## How to Run

1. Ensure you have the `gpt2` model weights placed in `models/gpt2/model.safetensors` relative to the project root.
2. Run the project using cargo:

```bash
cargo run --release
```

**Note:** The current version uses hand-tokenization for demonstration purposes.
