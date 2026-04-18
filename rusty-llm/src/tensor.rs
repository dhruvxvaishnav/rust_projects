use std::fmt;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "data length {} does not match shape {:?}",
            data.len(),
            shape
        );
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape }
    }

    pub fn from_slice(data: &[f32], shape: Vec<usize>) -> Self {
        Self::new(data.to_vec(), shape)
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Row-major index into the flat data vec.
    pub fn idx(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        let mut offset = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            assert!(indices[i] < self.shape[i]);
            offset += indices[i] * stride;
            stride *= self.shape[i];
        }
        offset
    }

    pub fn get(&self, indices: &[usize]) -> f32 {
        self.data[self.idx(indices)]
    }

    pub fn set(&mut self, indices: &[usize], value: f32) {
        let i = self.idx(indices);
        self.data[i] = value;
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data[..min(8)]={:?})",
               self.shape,
               &self.data[..self.data.len().min(8)])
    }
}