use crate::tensor::Tensor;

/// Given a token embedding table [vocab, n_embd] and token IDs,
/// return a [seq_len, n_embd] tensor by looking up each row.
pub fn embed_tokens(wte: &Tensor, token_ids: &[u32]) -> Tensor {
    assert_eq!(wte.ndim(), 2);
    let n_embd = wte.shape[1];
    let seq_len = token_ids.len();

    let mut out = vec![0.0f32; seq_len * n_embd];
    for (i, &tok) in token_ids.iter().enumerate() {
        let start = (tok as usize) * n_embd;
        let end = start + n_embd;
        out[i * n_embd..(i + 1) * n_embd].copy_from_slice(&wte.data[start..end]);
    }
    Tensor::new(out, vec![seq_len, n_embd])
}

/// Given a positional embedding table [n_ctx, n_embd] and a sequence length,
/// return positions 0..seq_len.
pub fn embed_positions(wpe: &Tensor, seq_len: usize) -> Tensor {
    assert_eq!(wpe.ndim(), 2);
    let n_embd = wpe.shape[1];
    let data = wpe.data[..seq_len * n_embd].to_vec();
    Tensor::new(data, vec![seq_len, n_embd])
}
