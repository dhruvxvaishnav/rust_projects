use rusty_llm::safetensors::SafeTensors;

fn main() {
    let st = SafeTensors::open("models/gpt2/model.safetensors").expect("failed to open model");

    println!("Loaded {} tensors\n", st.names().len());

    // Print a few key tensors and their shapes
    let interesting = [
        "wte.weight",             // token embeddings
        "wpe.weight",             // positional embeddings
        "h.0.ln_1.weight",        // first block's first layernorm
        "h.0.attn.c_attn.weight", // first block's QKV projection
        "h.0.attn.c_proj.weight", // first block's output projection
        "h.0.mlp.c_fc.weight",    // first block's MLP up-projection
        "h.0.mlp.c_proj.weight",  // first block's MLP down-projection
        "ln_f.weight",            // final layernorm
    ];

    for name in interesting {
        match st.shape(name) {
            Some(s) => println!("  {:40} shape = {:?}", name, s),
            None => println!("  {:40} NOT FOUND", name),
        }
    }

    // Sanity check one tensor
    let wte = st.load("wte.weight");
    println!("\nwte.weight first 5 values: {:?}", &wte.data[..5]);
    println!("wte.weight shape: {:?} (expected [50257, 768])", wte.shape);
}
