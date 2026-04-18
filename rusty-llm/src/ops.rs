use crate::tensor::Tensor;

/// Matrix multiplication for 2D tensors: (M, K) @ (K, N) -> (M, N).
/// Naive triple loop — clarity over speed. We'll optimize later.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2, "matmul: a must be 2D");
    assert_eq!(b.ndim(), 2, "matmul: b must be 2D");
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "matmul: inner dims must match, got {}x{} @ {}x{}", m, k, k2, n);

    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a.data[i * k + p] * b.data[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    Tensor::new(out, vec![m, n])
}

/// Element-wise add. Shapes must match exactly (no broadcasting yet).
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape, "add: shape mismatch");
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Tensor::new(data, a.shape.clone())
}

/// Broadcast-add a bias vector (length = last dim of a) to every row of a 2D tensor.
/// This is what linear layers need: out = x @ W + b
pub fn add_bias(a: &Tensor, bias: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2);
    assert_eq!(bias.ndim(), 1);
    let (m, n) = (a.shape[0], a.shape[1]);
    assert_eq!(bias.shape[0], n);

    let mut out = a.data.clone();
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] += bias.data[j];
        }
    }
    Tensor::new(out, a.shape.clone())
}

/// Element-wise multiply.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape);
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
    Tensor::new(data, a.shape.clone())
}

/// Softmax along the last dimension of a 2D tensor. Numerically stable.
pub fn softmax(a: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2);
    let (m, n) = (a.shape[0], a.shape[1]);
    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        let row = &a.data[i * n..(i + 1) * n];

        // subtract max for numerical stability
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();

        for j in 0..n {
            out[i * n + j] = exps[j] / sum;
        }
    }
    Tensor::new(out, a.shape.clone())
}

/// LayerNorm over the last dim of a 2D tensor.
/// y = gamma * (x - mean) / sqrt(var + eps) + beta
pub fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
    assert_eq!(x.ndim(), 2);
    let (m, n) = (x.shape[0], x.shape[1]);
    assert_eq!(gamma.shape, vec![n]);
    assert_eq!(beta.shape, vec![n]);

    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let row = &x.data[i * n..(i + 1) * n];
        let mean: f32 = row.iter().sum::<f32>() / n as f32;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
        let denom = (var + eps).sqrt();
        for j in 0..n {
            out[i * n + j] = gamma.data[j] * (row[j] - mean) / denom + beta.data[j];
        }
    }
    Tensor::new(out, x.shape.clone())
}

/// GELU activation (exact GPT-2 version using tanh approximation).
/// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: &Tensor) -> Tensor {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    let data = x.data.iter().map(|&v| {
        let inner = c * (v + 0.044715 * v.powi(3));
        0.5 * v * (1.0 + inner.tanh())
    }).collect();
    Tensor::new(data, x.shape.clone())
}

/// Transpose a 2D tensor.
pub fn transpose(a: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2);
    let (m, n) = (a.shape[0], a.shape[1]);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a.data[i * n + j];
        }
    }
    Tensor::new(out, vec![n, m])
}