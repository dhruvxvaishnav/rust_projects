#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub n_ctx: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub head_dim: usize,
    pub eps: f32,
}

impl Config {
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            n_ctx: 1024,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            head_dim: 64,
            eps: 1e-5,
        }
    }
}
