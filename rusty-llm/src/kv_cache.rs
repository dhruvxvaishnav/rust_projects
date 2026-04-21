use crate::tensor::Tensor;

pub struct LayerCache {
    pub k: Tensor,
    pub v: Tensor,
    pub cur_len: usize,
}

impl LayerCache {
    pub fn new() -> Self {
        Self {
            k: Tensor::new(vec![], vec![0, 0]),
            v: Tensor::new(vec![], vec![0, 0]),
            cur_len: 0,
        }
    }
}

pub struct KvCache {
    pub layers: Vec<LayerCache>,
}

impl KvCache {
    pub fn new(n_layer: usize) -> Self {
        Self {
            layers: (0..n_layer).map(|_| LayerCache::new()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.layers.first().map(|l| l.cur_len).unwrap_or(0)
    }
}

pub fn append_kv(cache: &mut LayerCache, new_k: &Tensor, new_v: &Tensor) {
    let n_embd = new_k.shape[1];
    let new_len = new_k.shape[0];

    if cache.cur_len == 0 {
        cache.k = new_k.clone();
        cache.v = new_v.clone();
    } else {
        let mut k_data = cache.k.data.clone();
        k_data.extend_from_slice(&new_k.data);
        cache.k = Tensor::new(k_data, vec![cache.cur_len + new_len, n_embd]);

        let mut v_data = cache.v.data.clone();
        v_data.extend_from_slice(&new_v.data);
        cache.v = Tensor::new(v_data, vec![cache.cur_len + new_len, n_embd]);
    }
    cache.cur_len += new_len;
}
