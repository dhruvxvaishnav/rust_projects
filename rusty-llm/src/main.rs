use rusty_llm::block::Block;
use rusty_llm::config::Config;
use rusty_llm::embedding::{embed_positions, embed_tokens};
use rusty_llm::ops::add;
use rusty_llm::safetensors::SafeTensors;

fn main() {
    let cfg = Config::gpt2_small();
    let st = SafeTensors::open("models/gpt2/model.safetensors").unwrap();

    // Fake tokens: "Hello world" is roughly [15496, 995] in GPT-2's BPE.
    // We'll do real tokenization next weekend.
    let token_ids: Vec<u32> = vec![15496, 995];
    println!("Input tokens: {:?}", token_ids);

    // Embeddings
    let wte = st.load("wte.weight");
    let wpe = st.load("wpe.weight");
    let tok_emb = embed_tokens(&wte, &token_ids);
    let pos_emb = embed_positions(&wpe, token_ids.len());
    let mut x = add(&tok_emb, &pos_emb);
    println!("After embedding: shape={:?}", x.shape);
    println!("  first 5 values of row 0: {:?}", &x.data[..5]);

    // Run through block 0
    let block0 = Block::load(&st, 0);
    x = block0.forward(&x, &cfg);
    println!("\nAfter block 0: shape={:?}", x.shape);
    println!("  first 5 values of row 0: {:?}", &x.data[..5]);
    println!(
        "  first 5 values of row 1: {:?}",
        &x.data[x.shape[1]..x.shape[1] + 5]
    );
}
