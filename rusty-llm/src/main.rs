use rusty_llm::config::Config;
use rusty_llm::model::GPT2;
use rusty_llm::safetensors::SafeTensors;
use std::time::Instant;

fn main() {
    println!("Loading model...");
    let t0 = Instant::now();
    let cfg = Config::gpt2_small();
    let st = SafeTensors::open("models/gpt2/model.safetensors")
        .expect("failed to open model.safetensors");
    let model = GPT2::load(&st, cfg.clone());
    println!("Loaded in {:.2}s\n", t0.elapsed().as_secs_f32());

    // "My name is" — hand-tokenized for now. Weekend 5 does real tokenization.
    //   My      -> 3666
    //   ' name' -> 1438
    //   ' is'   -> 318
    let token_ids: Vec<u32> = vec![3666, 1438, 318];
    println!("Input tokens: {:?}", token_ids);

    // Forward pass
    let t0 = Instant::now();
    let logits = model.forward(&token_ids);
    let dt = t0.elapsed().as_secs_f32();
    println!("Forward pass: {:.2}s", dt);
    println!("Logits shape: {:?}", logits.shape);

    // Inspect the last row (logits for the token after the input)
    let vocab = cfg.vocab_size;
    let seq_len = token_ids.len();
    let last = &logits.data[(seq_len - 1) * vocab..seq_len * vocab];

    println!("\nLast row, first 5 logits: {:?}", &last[..5]);

    // Top-5 next tokens
    let mut indexed: Vec<(usize, f32)> = last.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 next tokens:");
    for (i, v) in indexed.iter().take(5) {
        println!("  {}: {:.4}", i, v);
    }

    // Greedy pick
    let next = indexed[0].0 as u32;
    println!("\nGreedy next token: {}", next);
    println!("Expected: 1757  (' John', GPT-2's favorite completion)");
}
