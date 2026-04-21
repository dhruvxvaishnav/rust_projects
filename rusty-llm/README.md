# rusty-llm

A from-scratch implementation of GPT-2 inference in pure Rust. No ML frameworks, no BLAS, no external tensor libraries — just `Vec<f32>`, hand-written matmul, and the transformer architecture built up one operation at a time.

Built to understand what's actually inside a language model.

```
Prompt:  "My name is"
Output:  My name is J.J. Green, born and raised in Philadelphia, PA and I'm a former senior VP
```

---

## Table of Contents

- [Why this project exists](#why-this-project-exists)
- [What's implemented](#whats-implemented)
- [Architecture](#architecture)
- [Getting started](#getting-started)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Verification against HuggingFace](#verification-against-huggingface)
- [Benchmarks](#benchmarks)
- [Project structure](#project-structure)
- [What I learned](#what-i-learned)
- [Roadmap](#roadmap)
- [References](#references)

---

## Why this project exists

Most people who say they "understand transformers" have only ever called `model.generate()` on someone else's implementation. I wanted to know what every tensor actually does. So I wrote one from the ground up — no `torch`, no `candle`, no `ndarray`. If it ships in this repo, I wrote it.

The result is a working GPT-2 inference engine small enough to read in one sitting, whose output matches the HuggingFace reference to four decimal places.

## What's implemented

**Core engine**

- `Tensor` type with shape tracking and row-major `Vec<f32>` storage
- Hand-written operations: `matmul`, `add`, `mul`, `add_bias`, `softmax`, `layer_norm`, `gelu`, `transpose`
- `safetensors` loader with memory-mapped file I/O
- Token and positional embedding lookup
- Multi-head causal self-attention with per-head Q/K/V splitting
- MLP sub-block (4x expansion + GELU)
- Residual connections and pre-LayerNorm architecture
- Full 12-layer GPT-2 forward pass
- Output head with weight tying to token embeddings

**Generation**

- BPE tokenization via the `tokenizers` crate
- KV cache reducing generation from O(N²) to O(N)
- Prefill + autoregressive decode loop
- Temperature and top-k sampling
- Streaming token-by-token output to stdout

**Quality**

- Output matches HuggingFace `transformers` to within 1e-4
- Unit tests for every tensor operation
- Clippy-clean, zero warnings

## Architecture

```
                              ┌──────────────┐
   Token IDs ───────────────► │  Tokenizer   │
   "My name is"               │    (BPE)     │
                              └──────┬───────┘
                                     │ [3666, 1438, 318]
                                     ▼
                       ┌──────────────────────────┐
                       │    Token Embedding       │  [seq_len, 768]
                       │      (wte lookup)        │
                       └──────────┬───────────────┘
                                  │
                       ┌──────────▼───────────────┐
                       │  Positional Embedding    │
                       │      (wpe lookup)        │
                       └──────────┬───────────────┘
                                  │
                                  ▼
                       ╔══════════════════════════╗
                       ║   Transformer Block × 12 ║
                       ║                          ║
                       ║  ┌────────────────────┐  ║
                       ║  │    LayerNorm       │  ║
                       ║  └──────────┬─────────┘  ║
                       ║             ▼            ║
                       ║  ┌────────────────────┐  ║
                       ║  │  Multi-Head Attn   │  ║
                       ║  │  (12 heads, 64 dim)│  ║  ◄── KV Cache
                       ║  │   + Causal Mask    │  ║
                       ║  └──────────┬─────────┘  ║
                       ║             ▼            ║
                       ║           ( + ) ◄─── residual
                       ║             │            ║
                       ║  ┌──────────▼─────────┐  ║
                       ║  │    LayerNorm       │  ║
                       ║  └──────────┬─────────┘  ║
                       ║             ▼            ║
                       ║  ┌────────────────────┐  ║
                       ║  │       MLP          │  ║
                       ║  │  768→3072→GELU→768 │  ║
                       ║  └──────────┬─────────┘  ║
                       ║             ▼            ║
                       ║           ( + ) ◄─── residual
                       ╚═════════════╪════════════╝
                                     ▼
                       ┌──────────────────────────┐
                       │    Final LayerNorm       │
                       └──────────┬───────────────┘
                                  ▼
                       ┌──────────────────────────┐
                       │  Output Projection       │   (tied to wte^T)
                       │  [seq, 768] → [seq, V]   │
                       └──────────┬───────────────┘
                                  ▼
                       ┌──────────────────────────┐
                       │  Sample (top-k + temp)   │
                       └──────────┬───────────────┘
                                  ▼
                            Next token ID
```

**Model constants (GPT-2 small)**

| Parameter       | Value           |
| --------------- | --------------- |
| Parameters      | 124M            |
| Vocab size      | 50,257          |
| Context window  | 1,024 tokens    |
| Hidden dim      | 768             |
| Attention heads | 12              |
| Head dim        | 64              |
| Layers          | 12              |
| MLP expansion   | 4× (768 → 3072) |

## Getting started

### Prerequisites

- Rust 1.70 or newer (`rustup` recommended)
- ~600MB disk space for model weights
- `curl` for downloading weights

### Setup

```bash
git clone https://github.com/<your-handle>/rusty-llm.git
cd rusty-llm

# Download GPT-2 weights from HuggingFace
mkdir -p models/gpt2
cd models/gpt2
curl -L -o model.safetensors https://huggingface.co/gpt2/resolve/main/model.safetensors
curl -L -o config.json       https://huggingface.co/gpt2/resolve/main/config.json
curl -L -o tokenizer.json    https://huggingface.co/gpt2/resolve/main/tokenizer.json
cd ../..

# Build and run
cargo run --release
```

The `--release` flag is **important**. Debug builds run the forward pass ~40× slower because numerical code without compiler optimizations is painful.

### Running tests

```bash
cargo test --release
```

Every tensor operation has unit tests verifying numerical correctness against hand-computed expected values.

## Usage

The default `main.rs` generates text from the prompt `"My name is"`. To try your own prompts, edit the constants at the top of `src/main.rs`:

```rust
let prompt = "Once upon a time";
let max_new_tokens = 50;
let temperature = 0.8;
let top_k = 40;
let seed = 42;
```

Then `cargo run --release`.

**Sampling parameters**

- `temperature = 0.0` → greedy (always picks most likely token, deterministic)
- `temperature = 0.8` → balanced creativity (default)
- `temperature = 1.2` → more random, more diverse
- `top_k = 40` → sample from the 40 most likely tokens, zero out the rest
- Change `seed` for different completions from the same prompt

## How it works

### The tensor

No ndarray, no nalgebra. A tensor is a `Vec<f32>` plus a shape:

```rust
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}
```

Row-major layout means element `(i, j)` of a 2D tensor sits at `data[i * cols + j]`. Every op respects this convention.

### Attention, explained

The fiddliest part of a transformer. For each of 12 heads:

1. **Project** input `x [seq, 768]` to Q, K, V using a single fused matmul (`c_attn.weight` has shape `[768, 2304]` for 3× expansion)
2. **Split** the 2304-dim result into Q, K, V, each `[seq, 768]`
3. **Extract this head's slice** — head 0 uses columns `[0..64]`, head 1 uses `[64..128]`, etc.
4. **Score**: compute `Q_h @ K_h^T / sqrt(64)` → `[seq, seq]` attention scores
5. **Mask**: set upper-triangular scores to `-inf` so each token only attends to earlier tokens
6. **Softmax** across each row
7. **Weighted sum**: `weights @ V_h` → `[seq, 64]` output for this head
8. **Concatenate** all 12 head outputs back to `[seq, 768]`
9. **Project** through `c_proj.weight [768, 768]`

### The KV cache

Naive generation reruns the full forward pass on the whole sequence every time you add a token. Token 1 processes 1 position, token 100 processes 100 positions. Total work: O(N²).

**Key insight**: K and V for past tokens don't change when you append a new token. Cache them.

With a cache:

- **Prefill**: run the full prompt once, save each layer's K and V
- **Decode step**: project only the new token's Q, K, V. Append new K/V to cache. Compute attention with `Q [1, 768]` against the full cached `K, V`. Output: `[1, 768]`.

Complexity drops to O(N). In this implementation, decode tokens are ~1.2× faster than prefill tokens on a per-token basis — the cache is doing real work.

## Verification against HuggingFace

Every forward pass is verified against HuggingFace's reference implementation. Running the same prompt through both:

|                 | Python (HF `transformers`) | `rusty-llm` (this repo) |
| --------------- | -------------------------- | ----------------------- |
| Top-1 token     | 1757 (-65.5149)            | 1757 (-65.5149)         |
| Top-2 token     | 3700 (-65.7438)            | 3700 (-65.7439)         |
| Top-3 token     | 3271 (-65.8025)            | 3271 (-65.8025)         |
| Top-4 token     | 3899 (-65.9017)            | 3899 (-65.9018)         |
| Top-5 token     | 509 (-66.1111)             | 509 (-66.1111)          |
| First raw logit | -70.27092                  | -70.27095               |

Agreement to 4 decimal places across all 124M parameters, 12 transformer blocks, and hundreds of matrix multiplications. The tiny last-digit differences are just f32 accumulation order.

Run the verification yourself:

```bash
pip install transformers torch safetensors
python scripts/verify.py
```

## Benchmarks

Measured on an Apple M-series laptop CPU, release build, GPT-2 small (124M params).

| Phase                  | Tokens | Time   | Speed     |
| ---------------------- | ------ | ------ | --------- |
| Model load             | —      | 1.14 s | —         |
| Prefill                | 3      | 0.56 s | 5.4 tok/s |
| Decode (with KV cache) | 19     | 2.96 s | 6.4 tok/s |

**Where time goes** (approximate breakdown per forward pass):

- ~95% in matmul
- Within matmul, MLP's `c_fc` (768→3072) and `c_proj` (3072→768) dominate
- Attention scores and the output projection are small by comparison

**Why this is slow compared to llama.cpp** (~50–100 tok/s on the same hardware):

- No SIMD — every multiply-accumulate is a scalar operation
- No cache-blocking — matmul thrashes L1/L2 for matrices above ~256×256
- Single-threaded
- f32 everywhere (no quantization)

Each of these is a planned optimization, and each one is worth a separate learning project.

## Project structure

```
rusty-llm/
├── Cargo.toml
├── README.md
├── models/gpt2/          # GPT-2 weights (download separately)
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── scripts/
│   └── verify.py         # HuggingFace reference comparison
├── src/
│   ├── lib.rs            # Module declarations
│   ├── main.rs           # CLI entry point
│   ├── tensor.rs         # Tensor type + shape bookkeeping
│   ├── ops.rs            # matmul, softmax, layer_norm, gelu, etc.
│   ├── safetensors.rs    # Weight file loader (mmap-based)
│   ├── config.rs         # Model hyperparameters
│   ├── embedding.rs      # Token + positional embedding lookups
│   ├── attention.rs      # Multi-head causal self-attention
│   ├── block.rs          # One transformer block (attn + MLP + residuals)
│   ├── model.rs          # Full GPT-2: embeddings → N blocks → output head
│   ├── kv_cache.rs       # Per-layer K/V storage
│   ├── sampling.rs       # Temperature + top-k sampler
│   └── tokenizer.rs      # Thin wrapper over `tokenizers` crate
└── tests/
    └── ops_test.rs       # Numerical correctness tests
```

## What I learned

Building this answered a lot of questions I'd previously handwaved:

- **What a transformer "block" actually computes.** Not abstractly — which `Vec<f32>` goes where, and why.
- **Why the KV cache matters so much.** Naive decode does the same work over and over. Caching is the difference between "demo" and "usable."
- **Why GPT-2's weight layout is weird.** Its Conv1D layers store weights as `[in, out]` instead of PyTorch's `Linear` `[out, in]`. You find this out by watching your logits come out wrong.
- **How brittle floating-point comparisons are.** My first forward pass matched HuggingFace to 6 decimal places at layer 0 but only 3 by layer 12. f32 error accumulates.
- **Where inference spends its time.** Almost everything is matmul. Optimize that, optimize the model.
- **How much the Rust type system helps here.** Ownership makes the KV cache impossible to accidentally share across requests. Shape mismatches get caught at tensor construction instead of producing wrong numbers.

## Roadmap

Stretch goals in rough order of impact-per-effort:

- [ ] **SIMD matmul** — AVX2 on x86, NEON on ARM. Expected ~10–20× speedup on large matrices.
- [ ] **Multi-threading** — parallelize over heads or MLP rows with `rayon`. Another ~2–4×.
- [ ] **Cache-blocked matmul** — respect L1 cache size. ~2× on top of SIMD.
- [ ] **INT8 quantization** — 4× less memory, ~2× faster matmul via smaller dtype.
- [ ] **Llama support** — RoPE, RMSNorm, SwiGLU, grouped-query attention.
- [ ] **HTTP server** — `axum` with OpenAI-compatible `/v1/completions` and SSE streaming.
- [ ] **WASM build** — run GPT-2 in a browser tab.

## References

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — the GPT-2 paper
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the transformer paper
- [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy — essential viewing
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — clearest visual explanation I've found
- [HuggingFace `transformers`](https://github.com/huggingface/transformers) — my reference implementation
- [`llama.cpp`](https://github.com/ggerganov/llama.cpp) — the gold standard for fast CPU inference, studied for optimization ideas
- [safetensors format spec](https://github.com/huggingface/safetensors)

## License

MIT

---

_Built as a learning project. PRs, issues, and questions welcome._
