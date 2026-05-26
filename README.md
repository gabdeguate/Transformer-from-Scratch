# Transformer from Scratch

A decoder-only Transformer built entirely from scratch in PyTorch — no `nn.Module`, no autograd. Every forward and backward pass is implemented manually.

Trained on the [tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) for next-token prediction.

## What's implemented

**Core architecture**
- Token embeddings
- Rotary Position Embeddings (RoPE)
- Multi-Head Attention (MHA) and Grouped Query Attention (GQA)
- SwiGLU Feed-Forward Network
- RMS Normalization
- Causal masking

**Training infrastructure**
- Custom `Module` base class with manual `forward()` / `backward()`
- `CrossEntLoss` with numerically stable softmax
- Finite-difference gradient checker to validate every layer
- Optimizers: SGD, SGD with Momentum, RMSProp, Adam

**Inference**
- KV Cache for efficient autoregressive decoding

**Fine-tuning**
- LoRA (Low-Rank Adaptation) on any `Linear` layer

**Tokenizer**
- GPT-4 regex-based Byte Pair Encoding (BPE), trained from scratch on the dataset

## Setup

```bash
pip install torch numpy regex tqdm
```

Place `tiny_shakespeare.txt` in the project root (or download it):

```bash
curl -o tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Run

```bash
python transformer.py
```

This will:
1. Run finite-difference gradient checks on every layer (Embedding, Softmax, Linear, RoPE, Attention MHA/GQA, RMSNorm, Activations, FeedForward, TransformerBlock, CrossEntLoss, full Transformer)
2. Verify KV cache output matches full-sequence forward pass
3. Run a 5-step training demo on Shakespeare with Adam
4. Run a LoRA correctness demo

## Architecture overview

### Custom autograd

All layers subclass `Module` and implement `forward()` and `backward()` by hand. No PyTorch autograd is used for the main model. Learnable parameters are listed in `self.sub_modules`, which drives `state_dict()`, `load_state_dict()`, `grad_dict()`, and `to()` recursively.

```
tokens → Embedding → TransformerBlock × N → RMSNorm → Linear → logits
```

Each `TransformerBlock` is a standard pre-norm decoder block:
```
x → RMSNorm → Attention (RoPE + MHA/GQA) → residual
  → RMSNorm → FeedForward (SwiGLU)        → residual
```

### LoRA

`LoRALinear` wraps a frozen `Linear` and learns a low-rank update `ΔW = (α/r) · A·B`. `B` is zero-initialized so the adapter starts as identity. After training, `merge()` folds the adapter into the base weight for zero-overhead inference.

### KV Cache

`KVCache` stores key/value tensors per layer. On each decode step, only the new token is embedded; position ids are offset by the cache length, and a rectangular causal mask is applied.

## File structure

```
transformer.py      # Full model, optimizers, training utilities, tests
tokenizer.py        # GPT-4 regex BPE tokenizer
tiny_shakespeare.txt # Training data
```
