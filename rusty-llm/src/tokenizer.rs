use tokenizers::Tokenizer;

pub struct Bpe {
    tok: Tokenizer,
}

impl Bpe {
    pub fn from_file(path: &str) -> Self {
        let tok = Tokenizer::from_file(path).expect("failed to load tokenizer");
        Self { tok }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let enc = self.tok.encode(text, false).expect("encode failed");
        enc.get_ids().to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids, false).expect("decode failed")
    }
}
