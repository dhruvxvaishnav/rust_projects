use crate::block::Block;
use crate::config::Config;
use crate::embedding::{embed_positions, embed_tokens};
use crate::ops::{add, layer_norm, matmul, transpose};
use crate::safetensors::SafeTensors;
use crate::tensor::Tensor;

pub struct GPT2 {
    pub cfg: Config,
    pub wte: Tensor, // [vocab_size, n_embd] — also used (transposed) as output head
    pub wpe: Tensor, // [n_ctx, n_embd]
    pub blocks: Vec<Block>,
    pub ln_f_w: Tensor, // [n_embd]
    pub ln_f_b: Tensor, // [n_embd]
}

impl GPT2 {
    pub fn load(st: &SafeTensors, cfg: Config) -> Self {
        let blocks = (0..cfg.n_layer).map(|i| Block::load(st, i)).collect();

        Self {
            wte: st.load("wte.weight"),
            wpe: st.load("wpe.weight"),
            blocks,
            ln_f_w: st.load("ln_f.weight"),
            ln_f_b: st.load("ln_f.bias"),
            cfg,
        }
    }

    /// Forward pass: token IDs -> logits [seq_len, vocab_size]
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        assert!(token_ids.len() <= self.cfg.n_ctx, "sequence too long");

        // 1. Embeddings
        let tok = embed_tokens(&self.wte, token_ids);
        let pos = embed_positions(&self.wpe, token_ids.len());
        let mut x = add(&tok, &pos);

        // 2. Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x, &self.cfg);
        }

        // 3. Final layernorm
        x = layer_norm(&x, &self.ln_f_w, &self.ln_f_b, self.cfg.eps);

        // 4. Output head: project to vocab via tied embedding weight (W^T)
        //    x [seq_len, n_embd] @ wte^T [n_embd, vocab_size] -> [seq_len, vocab_size]
        let wte_t = transpose(&self.wte);
        matmul(&x, &wte_t)
    }

    /// Greedy: pick the argmax token from the last row of logits.
    pub fn greedy_next_token(&self, token_ids: &[u32]) -> u32 {
        let logits = self.forward(token_ids);
        let vocab = self.cfg.vocab_size;
        let seq_len = token_ids.len();

        // Last row = logits for the token that comes after the input
        let last_row = &logits.data[(seq_len - 1) * vocab..seq_len * vocab];

        let (argmax, _) = last_row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        argmax as u32
    }
}
