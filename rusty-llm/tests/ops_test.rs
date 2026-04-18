use approx::assert_abs_diff_eq;
use rusty_llm::ops;
use rusty_llm::tensor::Tensor;

fn assert_tensor_close(a: &Tensor, expected_data: &[f32], expected_shape: &[usize]) {
    assert_eq!(a.shape, expected_shape);
    assert_eq!(a.data.len(), expected_data.len());
    for (x, y) in a.data.iter().zip(expected_data) {
        assert_abs_diff_eq!(x, y, epsilon = 1e-5);
    }
}

#[test]
fn test_matmul_2x2() {
    let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
    let b = Tensor::new(vec![5., 6., 7., 8.], vec![2, 2]);
    let c = ops::matmul(&a, &b);
    assert_tensor_close(&c, &[19., 22., 43., 50.], &[2, 2]);
}

#[test]
fn test_matmul_non_square() {
    // (2,3) @ (3,2) -> (2,2)
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let b = Tensor::new(vec![7., 8., 9., 10., 11., 12.], vec![3, 2]);
    let c = ops::matmul(&a, &b);
    // [1*7+2*9+3*11, 1*8+2*10+3*12, 4*7+5*9+6*11, 4*8+5*10+6*12]
    assert_tensor_close(&c, &[58., 64., 139., 154.], &[2, 2]);
}

#[test]
fn test_add() {
    let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
    let b = Tensor::new(vec![10., 20., 30., 40.], vec![2, 2]);
    let c = ops::add(&a, &b);
    assert_tensor_close(&c, &[11., 22., 33., 44.], &[2, 2]);
}

#[test]
fn test_add_bias() {
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let bias = Tensor::new(vec![10., 20., 30.], vec![3]);
    let c = ops::add_bias(&a, &bias);
    assert_tensor_close(&c, &[11., 22., 33., 14., 25., 36.], &[2, 3]);
}

#[test]
fn test_softmax_sums_to_one() {
    let a = Tensor::new(vec![1., 2., 3., 1., 1., 1.], vec![2, 3]);
    let c = ops::softmax(&a);
    // each row should sum to 1
    let row0_sum: f32 = c.data[0..3].iter().sum();
    let row1_sum: f32 = c.data[3..6].iter().sum();
    assert_abs_diff_eq!(row0_sum, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(row1_sum, 1.0, epsilon = 1e-6);
    // uniform row -> uniform output
    assert_abs_diff_eq!(c.data[3], 1.0 / 3.0, epsilon = 1e-6);
}

#[test]
fn test_softmax_numerically_stable() {
    // large values shouldn't blow up
    let a = Tensor::new(vec![1000., 1000., 1000.], vec![1, 3]);
    let c = ops::softmax(&a);
    assert_abs_diff_eq!(c.data[0], 1.0 / 3.0, epsilon = 1e-6);
}

#[test]
fn test_layer_norm() {
    // Well-known case: normalizing [1,2,3,4] with gamma=1, beta=0
    // mean=2.5, var=1.25, std=sqrt(1.25)=1.1180...
    // (x - 2.5) / 1.1180 = [-1.3416, -0.4472, 0.4472, 1.3416]
    let x = Tensor::new(vec![1., 2., 3., 4.], vec![1, 4]);
    let gamma = Tensor::new(vec![1., 1., 1., 1.], vec![4]);
    let beta = Tensor::new(vec![0., 0., 0., 0.], vec![4]);
    let y = ops::layer_norm(&x, &gamma, &beta, 1e-5);
    assert_tensor_close(&y, &[-1.3416355, -0.4472118, 0.4472118, 1.3416355], &[1, 4]);
}

#[test]
fn test_gelu_known_values() {
    // GELU(0) = 0, GELU(1) ~= 0.8411, GELU(-1) ~= -0.1588
    let x = Tensor::new(vec![0., 1., -1.], vec![1, 3]);
    let y = ops::gelu(&x);
    assert_abs_diff_eq!(y.data[0], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y.data[1], 0.8411920, epsilon = 1e-4);
    assert_abs_diff_eq!(y.data[2], -0.1588080, epsilon = 1e-4);
}

#[test]
fn test_transpose() {
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let t = ops::transpose(&a);
    assert_tensor_close(&t, &[1., 4., 2., 5., 3., 6.], &[3, 2]);
}

#[test]
fn test_transpose_twice_is_identity() {
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let tt = ops::transpose(&ops::transpose(&a));
    assert_tensor_close(&tt, &a.data, &a.shape);
}

#[test]
fn test_causal_mask_effect() {
    // After softmax with causal mask, first row should be [1, 0, 0],
    // second row should sum to 1 with weight on positions 0 and 1 only.
    use rusty_llm::ops::softmax;
    let mut scores = Tensor::new(
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![3, 3],
    );
    // Apply mask manually
    scores.data[1] = f32::NEG_INFINITY;
    scores.data[2] = f32::NEG_INFINITY;
    scores.data[5] = f32::NEG_INFINITY;
    let w = softmax(&scores);
    assert_abs_diff_eq!(w.data[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(w.data[1], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(w.data[2], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(w.data[3] + w.data[4], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(w.data[8], 1.0 / 3.0, epsilon = 1e-6);
}
