use rand::SeedableRng;
use rusty_llm::config::Config;
use rusty_llm::model::GPT2;
use rusty_llm::safetensors::SafeTensors;
use rusty_llm::sampling::sample;
use rusty_llm::tokenizer::Bpe;
use std::io::Write;
use std::time::Instant;

fn main() {
    // --- Config ---
    let prompt = "My name is";
    let max_new_tokens = 20;
    let temperature = 0.8;
    let top_k = 40;
    let seed = 42;

    // --- Load everything ---
    println!("Loading model...");
    let t0 = Instant::now();
    let cfg = Config::gpt2_small();
    let st = SafeTensors::open("models/gpt2/model.safetensors")
        .expect("failed to open model.safetensors");
    let model = GPT2::load(&st, cfg.clone());
    let bpe = Bpe::from_file("models/gpt2/tokenizer.json");
    println!("Loaded in {:.2}s\n", t0.elapsed().as_secs_f32());

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // --- Tokenize ---
    let prompt_ids = bpe.encode(prompt);
    println!("Prompt:  {:?}", prompt);
    println!("Tokens:  {:?}\n", prompt_ids);

    // --- Prefill: run the whole prompt through, populate the cache ---
    let mut cache = model.new_cache();
    let t_prefill = Instant::now();
    let logits = model.forward(&prompt_ids, 0, &mut cache);
    let prefill_time = t_prefill.elapsed().as_secs_f32();

    // First new token from the last row of prefill logits
    let vocab = cfg.vocab_size;
    let last_row = &logits.data[(prompt_ids.len() - 1) * vocab..prompt_ids.len() * vocab];
    let mut next = sample(last_row, temperature, top_k, &mut rng);

    let mut generated: Vec<u32> = vec![next];

    print!("Output:  {}", prompt);
    print!("{}", bpe.decode(&[next]));
    std::io::stdout().flush().ok();

    // --- Decode loop: one token at a time, reusing the cache ---
    let t_decode = Instant::now();
    for _ in 1..max_new_tokens {
        let pos = prompt_ids.len() + generated.len() - 1;
        let logits = model.forward(&[next], pos, &mut cache);
        let row = &logits.data[..vocab]; // only one row, new_len = 1
        next = sample(row, temperature, top_k, &mut rng);
        generated.push(next);

        print!("{}", bpe.decode(&[next]));
        std::io::stdout().flush().ok();
    }
    let decode_time = t_decode.elapsed().as_secs_f32();

    // --- Stats ---
    println!("\n\n--- Stats ---");
    println!(
        "Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
        prompt_ids.len(),
        prefill_time,
        prompt_ids.len() as f32 / prefill_time
    );
    println!(
        "Decode:  {} tokens in {:.2}s ({:.1} tok/s)",
        max_new_tokens - 1,
        decode_time,
        (max_new_tokens - 1) as f32 / decode_time
    );
}
