from transformers import GPT2Tokenizer ,GPT2LMHeadModel, GPT2Config
import torch
from safetensors.torch import load_file

cfg = GPT2Config.from_pretrained("models/gpt2")
model = GPT2LMHeadModel(cfg)
state = load_file("models/gpt2/model.safetensors")
# GPT2LMHeadModel expects keys prefixed with "transformer."
state = {f"transformer.{k}" if not k.startswith("transformer.") else k: v
         for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.eval()

token_ids = torch.tensor([[3666, 1438, 318]])

with torch.no_grad():
    out = model(token_ids)
    logits = out.logits[0]

print("Shape:", logits.shape)
print("\nLast row, first 5 logits:")
print(logits[-1, :5].tolist())

print("\nTop 5 next tokens:")
top = torch.topk(logits[-1], 5)
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {i}: {v:.4f}")


tok = GPT2Tokenizer.from_pretrained("gpt2")
print("\nTop 5 decoded:")
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {i}: {v:.4f}  -> {tok.decode([i])!r}")