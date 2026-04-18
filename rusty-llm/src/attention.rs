use crate::config::Config;
use crate::ops::{add_bias, matmul, softmax, transpose};
use crate::tensor::Tensor;

/// Multi-head causal self-attention.
///
/// Input x: [seq_len, n_embd]
/// Weights:
///   c_attn_w: [n_embd, 3 * n_embd]   (fused Q, K, V projection)
///   c_attn_b: [3 * n_embd]
///   c_proj_w: [n_embd, n_embd]
///   c_proj_b: [n_embd]
///
/// Output: [seq_len, n_embd]
pub fn attention(
    x: &Tensor,
    c_attn_w: &Tensor,
    c_attn_b: &Tensor,
    c_proj_w: &Tensor,
    c_proj_b: &Tensor,
    cfg: &Config,
) -> Tensor {
    let seq_len = x.shape[0];
    let n_embd = cfg.n_embd;
    let n_head = cfg.n_head;
    let head_dim = cfg.head_dim;

    // 1. Project to Q, K, V in one matmul: [seq_len, 3*n_embd]
    let qkv = add_bias(&matmul(x, c_attn_w), c_attn_b);

    // 2. Split into Q, K, V, each [seq_len, n_embd]
    let (q, k, v) = split_qkv(&qkv, n_embd);

    // 3. For each head, compute attention
    //    We allocate the output [seq_len, n_embd] and fill one head-slice at a time.
    let mut out = vec![0.0f32; seq_len * n_embd];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..n_head {
        // Extract this head's slice from Q, K, V: each [seq_len, head_dim]
        let q_h = extract_head(&q, h, n_head, head_dim);
        let k_h = extract_head(&k, h, n_head, head_dim);
        let v_h = extract_head(&v, h, n_head, head_dim);

        // scores = Q_h @ K_h^T   -> [seq_len, seq_len]
        let k_h_t = transpose(&k_h);
        let mut scores = matmul(&q_h, &k_h_t);

        // Scale
        for s in scores.data.iter_mut() {
            *s *= scale;
        }

        // Causal mask: position i can only attend to positions <= i
        apply_causal_mask(&mut scores, seq_len);

        // Softmax per row
        let weights = softmax(&scores);

        // weights @ V_h -> [seq_len, head_dim]
        let head_out = matmul(&weights, &v_h);

        // Write back into the right slice of `out`
        write_head(&mut out, &head_out, h, n_head, head_dim, seq_len);
    }

    let concat = Tensor::new(out, vec![seq_len, n_embd]);

    // 4. Output projection
    add_bias(&matmul(&concat, c_proj_w), c_proj_b)
}

/// Split [seq_len, 3*n_embd] into three [seq_len, n_embd] tensors.
/// GPT-2 layout: [Q | K | V] along the last dim.
fn split_qkv(qkv: &Tensor, n_embd: usize) -> (Tensor, Tensor, Tensor) {
    let seq_len = qkv.shape[0];
    let mut q = vec![0.0f32; seq_len * n_embd];
    let mut k = vec![0.0f32; seq_len * n_embd];
    let mut v = vec![0.0f32; seq_len * n_embd];

    for i in 0..seq_len {
        let row_start = i * 3 * n_embd;
        q[i * n_embd..(i + 1) * n_embd].copy_from_slice(&qkv.data[row_start..row_start + n_embd]);
        k[i * n_embd..(i + 1) * n_embd]
            .copy_from_slice(&qkv.data[row_start + n_embd..row_start + 2 * n_embd]);
        v[i * n_embd..(i + 1) * n_embd]
            .copy_from_slice(&qkv.data[row_start + 2 * n_embd..row_start + 3 * n_embd]);
    }

    (
        Tensor::new(q, vec![seq_len, n_embd]),
        Tensor::new(k, vec![seq_len, n_embd]),
        Tensor::new(v, vec![seq_len, n_embd]),
    )
}

/// Extract head `h` from a [seq_len, n_embd] tensor into [seq_len, head_dim].
/// Heads are laid out contiguously: head 0 = cols [0..head_dim], head 1 = [head_dim..2*head_dim], ...
fn extract_head(x: &Tensor, h: usize, n_head: usize, head_dim: usize) -> Tensor {
    let seq_len = x.shape[0];
    let n_embd = n_head * head_dim;
    let mut out = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        let src_start = i * n_embd + h * head_dim;
        out[i * head_dim..(i + 1) * head_dim]
            .copy_from_slice(&x.data[src_start..src_start + head_dim]);
    }
    Tensor::new(out, vec![seq_len, head_dim])
}

/// Write [seq_len, head_dim] back into the h-th slice of a [seq_len, n_embd] flat buffer.
fn write_head(
    out: &mut [f32],
    head_out: &Tensor,
    h: usize,
    n_head: usize,
    head_dim: usize,
    seq_len: usize,
) {
    let n_embd = n_head * head_dim;
    for i in 0..seq_len {
        let dst_start = i * n_embd + h * head_dim;
        out[dst_start..dst_start + head_dim]
            .copy_from_slice(&head_out.data[i * head_dim..(i + 1) * head_dim]);
    }
}

/// Set upper triangle (excluding diagonal) to -inf so softmax zeros it out.
fn apply_causal_mask(scores: &mut Tensor, seq_len: usize) {
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            scores.data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
}
