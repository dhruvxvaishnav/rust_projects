use rusty_llm::ops;
use rusty_llm::tensor::Tensor;

fn main () {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2,2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2,2]);
    let c = ops::matmul(&a, &b);

    println!("a @ b = {} ", c);
    //expected [19,22,43,50]
}