use crate::block::Block;
use crate::config::Config;
use crate::embedding::embed_tokens;
use crate::kv_cache::KvCache;
use crate::ops::{add, layer_norm, matmul, transpose};
use crate::safetensors::SafeTensors;
use crate::tensor::Tensor;

pub struct GPT2 {
    pub cfg: Config,
    pub wte: Tensor,
    pub wpe: Tensor,
    pub blocks: Vec<Block>,
    pub ln_f_w: Tensor,
    pub ln_f_b: Tensor,
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

    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.cfg.n_layer)
    }

    pub fn forward(&self, token_ids: &[u32], start_pos: usize, cache: &mut KvCache) -> Tensor {
        let new_len = token_ids.len();
        assert!(start_pos + new_len <= self.cfg.n_ctx, "context overflow");

        let tok = embed_tokens(&self.wte, token_ids);

        let pos_slice =
            &self.wpe.data[start_pos * self.cfg.n_embd..(start_pos + new_len) * self.cfg.n_embd];
        let pos = Tensor::new(pos_slice.to_vec(), vec![new_len, self.cfg.n_embd]);

        let mut x = add(&tok, &pos);

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, &mut cache.layers[i], &self.cfg);
        }

        x = layer_norm(&x, &self.ln_f_w, &self.ln_f_b, self.cfg.eps);
        let wte_t = transpose(&self.wte);
        matmul(&x, &wte_t)
    }
}
