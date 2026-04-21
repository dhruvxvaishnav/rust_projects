use crate::config::Config;
use crate::kv_cache::{append_kv, LayerCache};
use crate::ops::{add_bias, matmul, softmax, transpose};
use crate::tensor::Tensor;

pub fn attention(
    x: &Tensor,
    c_attn_w: &Tensor,
    c_attn_b: &Tensor,
    c_proj_w: &Tensor,
    c_proj_b: &Tensor,
    cache: &mut LayerCache,
    cfg: &Config,
) -> Tensor {
    let new_len = x.shape[0];
    let prev_len = cache.cur_len;
    let total_len = prev_len + new_len;

    let n_embd = cfg.n_embd;
    let n_head = cfg.n_head;
    let head_dim = cfg.head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let qkv = add_bias(&matmul(x, c_attn_w), c_attn_b);
    let (q, new_k, new_v) = split_qkv(&qkv, n_embd);

    append_kv(cache, &new_k, &new_v);
    let k_all = &cache.k; // [total_len, n_embd]
    let v_all = &cache.v; // [total_len, n_embd]

    let mut out = vec![0.0f32; new_len * n_embd];

    for h in 0..n_head {
        let q_h = extract_head(&q, h, n_head, head_dim);
        let k_h = extract_head(k_all, h, n_head, head_dim);
        let v_h = extract_head(v_all, h, n_head, head_dim);

        let k_h_t = transpose(&k_h);
        let mut scores = matmul(&q_h, &k_h_t);

        for s in scores.data.iter_mut() {
            *s *= scale;
        }

        apply_causal_mask(&mut scores, new_len, prev_len, total_len);

        let weights = softmax(&scores);
        let head_out = matmul(&weights, &v_h);

        write_head(&mut out, &head_out, h, n_head, head_dim, new_len);
    }

    let concat = Tensor::new(out, vec![new_len, n_embd]);
    add_bias(&matmul(&concat, c_proj_w), c_proj_b)
}

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

fn write_head(
    out: &mut [f32],
    head_out: &Tensor,
    h: usize,
    n_head: usize,
    head_dim: usize,
    new_len: usize,
) {
    let n_embd = n_head * head_dim;
    for i in 0..new_len {
        let dst_start = i * n_embd + h * head_dim;
        out[dst_start..dst_start + head_dim]
            .copy_from_slice(&head_out.data[i * head_dim..(i + 1) * head_dim]);
    }
}

fn apply_causal_mask(scores: &mut Tensor, new_len: usize, prev_len: usize, total_len: usize) {
    for i in 0..new_len {
        let abs_pos = prev_len + i;
        for j in (abs_pos + 1)..total_len {
            scores.data[i * total_len + j] = f32::NEG_INFINITY;
        }
    }
}
