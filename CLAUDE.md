# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
python transformer.py
```

Runs gradient tests, KV cache correctness test, short training demo, and LoRA demo.

Training demo expects `tiny_shakespeare.txt` at `../tiny_shakespeare.txt`. The file currently lives in the project root (`./tiny_shakespeare.txt`), so the training demo will be skipped unless the path is adjusted.

## Architecture

Two files:

- **`tokenizer.py`** — GPT-4 Regex BPE tokenizer. `RegexBPETokenizer.train()` builds merge table from scratch. `encode()`/`decode()` are independent of PyTorch.
- **`transformer.py`** — Full decoder-only Transformer with custom autograd.

### Custom Module system

All layers inherit from `Module` and implement `forward()` + `backward()` manually (no PyTorch autograd for the main model). Key contract:

- Every learnable tensor `self.X` must appear in `self.sub_modules: List[str]`.
- `state_dict()` / `grad_dict()` / `load_state_dict()` / `to()` all recurse via `sub_modules`.
- `backward()` must store gradients as `self.grad_X` for each `self.X` in `sub_modules`.
- Intermediate tensors needed for backward (e.g., `self.x`, `self.output`, `self.attn_score`) are cached on `self` during `forward()`.

**Excluded from `sub_modules`**: `RMSNorm` (no trainable params by default) and `RotaryEmbedding` (theta is not learned). These are moved manually in `Transformer.to()`.

### Model stack

```
tokens → Embedding → [TransformerBlock × n_layers] → RMSNorm → Linear → logits
```

`TransformerBlock`: pre-norm → Attention (RoPE + MHA or GQA) → residual → pre-norm → FeedForward (SwiGLU) → residual.

### Attention (MHA vs GQA)

`Attention` handles both modes via `n_kv_heads`:
- **MHA** (`n_kv_heads == n_heads`): separate `wq`, `wk`, `wv`, `wo` projections.
- **GQA** (`n_kv_heads < n_heads`): fused `wqkv` projection, sliced. KV heads expanded via `repeat_kv()` for the attention computation; gradients reduced back in backward.

### KV Cache

`KVCache` caches K/V tensors per layer. During inference, `Transformer.forward()` embeds only new tokens, offsets position ids by `cache._seen_tokens`, and uses a rectangular causal mask `(new_seqlen, total_seqlen)`. Backward is unaffected (cache unused during training).

### LoRA

`LoRALinear` wraps a frozen `Linear` and adds `(alpha/rank) * x @ A @ B`. `lora_B` is zero-initialized so adapter starts as identity. Uses `nn.Parameter` (PyTorch autograd) for A and B — unlike the rest of the codebase which uses custom autograd. `merge()` folds LoRA into the base weight in-place.

### Optimizers

All inherit from `Optimizer`. `param_dict` holds references to model tensors, so in-place subtraction `param_dict[k] -= step` modifies the model directly. Available: `SGD`, `Momentum`, `RMSProp`, `Adam`.

### Gradient checker

`test_module(name, cls, params, inputs)` compares analytical gradients from `.backward()` against finite differences. Used in `__main__` to validate every layer. Pass `atol=1e-4, rtol=1e-2`.

## Dependencies

`torch`, `numpy`, `regex`, `tqdm`
