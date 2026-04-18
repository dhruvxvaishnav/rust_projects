use crate::tensor::Tensor;
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct SafeTensors {
    mmap: Mmap,
    header: HashMap<String, TensorInfo>,
    data_start: usize,
}

impl SafeTensors {
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Read the first 8 bytes as header length (little-endian u64)
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_bytes = &mmap[8..8 + header_len];
        let data_start = 8 + header_len;

        // Parse header JSON. It includes a "__metadata__" key we want to skip.
        let mut header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(header_bytes).expect("invalid safetensors header");
        header.remove("__metadata__");

        // Re-deserialize into TensorInfo.
        let header: HashMap<String, TensorInfo> = header
            .into_iter()
            .map(|(k, v)| (k, serde_json::from_value(v).expect("bad tensor info")))
            .collect();

        Ok(Self {
            mmap,
            header,
            data_start,
        })
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.header.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.header.get(name).map(|info| info.shape.as_slice())
    }

    /// Load a tensor by name, copying its bytes into a Vec<f32>.
    pub fn load(&self, name: &str) -> Tensor {
        let info = self
            .header
            .get(name)
            .unwrap_or_else(|| panic!("tensor not found: {}", name));
        assert_eq!(
            info.dtype, "F32",
            "only F32 supported for now, got {}",
            info.dtype
        );

        let start = self.data_start + info.data_offsets[0];
        let end = self.data_start + info.data_offsets[1];
        let bytes = &self.mmap[start..end];

        let numel: usize = info.shape.iter().product();
        assert_eq!(bytes.len(), numel * 4, "byte length mismatch for {}", name);

        // Interpret raw bytes as f32 (little-endian). Copy into owned Vec.
        let mut data = Vec::with_capacity(numel);
        for chunk in bytes.chunks_exact(4) {
            data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }

        Tensor::new(data, info.shape.clone())
    }
}
