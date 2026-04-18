use crate::attention::attention;
use crate::config::Config;
use crate::ops::{add, add_bias, gelu, layer_norm, matmul};
use crate::safetensors::SafeTensors;
use crate::tensor::Tensor;

/// MLP: up-project, GELU, down-project.
/// c_fc_w: [n_embd, 4*n_embd], c_fc_b: [4*n_embd]
/// c_proj_w: [4*n_embd, n_embd], c_proj_b: [n_embd]
pub fn mlp(
    x: &Tensor,
    c_fc_w: &Tensor,
    c_fc_b: &Tensor,
    c_proj_w: &Tensor,
    c_proj_b: &Tensor,
) -> Tensor {
    let h = add_bias(&matmul(x, c_fc_w), c_fc_b);
    let h = gelu(&h);
    add_bias(&matmul(&h, c_proj_w), c_proj_b)
}

/// One transformer block. GPT-2 uses pre-LayerNorm:
///   x = x + attn(ln_1(x))
///   x = x + mlp(ln_2(x))
pub struct Block {
    pub ln_1_w: Tensor,
    pub ln_1_b: Tensor,
    pub c_attn_w: Tensor,
    pub c_attn_b: Tensor,
    pub c_proj_attn_w: Tensor,
    pub c_proj_attn_b: Tensor,
    pub ln_2_w: Tensor,
    pub ln_2_b: Tensor,
    pub c_fc_w: Tensor,
    pub c_fc_b: Tensor,
    pub c_proj_mlp_w: Tensor,
    pub c_proj_mlp_b: Tensor,
}

impl Block {
    pub fn load(st: &SafeTensors, layer_idx: usize) -> Self {
        let p = |s: &str| format!("h.{}.{}", layer_idx, s);
        Self {
            ln_1_w: st.load(&p("ln_1.weight")),
            ln_1_b: st.load(&p("ln_1.bias")),
            c_attn_w: st.load(&p("attn.c_attn.weight")),
            c_attn_b: st.load(&p("attn.c_attn.bias")),
            c_proj_attn_w: st.load(&p("attn.c_proj.weight")),
            c_proj_attn_b: st.load(&p("attn.c_proj.bias")),
            ln_2_w: st.load(&p("ln_2.weight")),
            ln_2_b: st.load(&p("ln_2.bias")),
            c_fc_w: st.load(&p("mlp.c_fc.weight")),
            c_fc_b: st.load(&p("mlp.c_fc.bias")),
            c_proj_mlp_w: st.load(&p("mlp.c_proj.weight")),
            c_proj_mlp_b: st.load(&p("mlp.c_proj.bias")),
        }
    }

    pub fn forward(&self, x: &Tensor, cfg: &Config) -> Tensor {
        // Attention sub-block with residual
        let normed = layer_norm(x, &self.ln_1_w, &self.ln_1_b, cfg.eps);
        let attn_out = attention(
            &normed,
            &self.c_attn_w,
            &self.c_attn_b,
            &self.c_proj_attn_w,
            &self.c_proj_attn_b,
            cfg,
        );
        let x = add(x, &attn_out);

        // MLP sub-block with residual
        let normed = layer_norm(&x, &self.ln_2_w, &self.ln_2_b, cfg.eps);
        let mlp_out = mlp(
            &normed,
            &self.c_fc_w,
            &self.c_fc_b,
            &self.c_proj_mlp_w,
            &self.c_proj_mlp_b,
        );
        add(&x, &mlp_out)
    }
}
