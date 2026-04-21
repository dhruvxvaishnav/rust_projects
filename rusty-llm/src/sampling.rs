use rand::Rng;

/// Sample an index from logits using temperature + top-k.
/// temperature = 0.0 means greedy (returns argmax).
pub fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut impl Rng) -> u32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
    }

    // Take top-k indices by logit value
    let k = top_k.min(logits.len());
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);

    // Apply temperature and softmax
    let max = indexed
        .iter()
        .map(|x| x.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = indexed
        .iter()
        .map(|(_, v)| ((v - max) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|x| x / sum).collect();

    // Sample from categorical
    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return indexed[i].0 as u32;
        }
    }
    indexed.last().unwrap().0 as u32
}
