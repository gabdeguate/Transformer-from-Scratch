"""
Transformer from Scratch

Covers:
  - Custom Module system with manual forward + backward
  - Full Transformer: Embedding, RoPE, RMSNorm, MHA, SwiGLU FFN
  - KV Cache for efficient inference
  - Grouped Query Attention (GQA) and LoRA fine-tuning
  - Optimizers: SGD, Momentum, RMSProp, Adam

Run:  python transformer.py
      (from project root; tiny_shakespeare.txt expected in project root)
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from typing import Optional, Tuple, List
from copy import deepcopy
from itertools import product
from collections import Counter

# Allow importing tokenizer from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import RegexBPETokenizer


# ============================================================
# Utility functions
# ============================================================

def unfold_dict(state_dict: dict) -> dict:
    """Flatten nested dict with '.' separator.
    {'a': {'b': v}} -> {'a.b': v}
    """
    new_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            sub = unfold_dict(v)
            for sub_k, sub_v in sub.items():
                new_dict[k + "." + sub_k] = sub_v
        else:
            new_dict[k] = v
    return new_dict


def fold_dict(state_dict: dict) -> dict:
    """Inverse of unfold_dict: reconstruct nested dict from dotted keys."""
    new_dict = {}
    for k, v in state_dict.items():
        split = k.split(".")
        if len(split) == 1:
            new_dict[split[0]] = v
        else:
            key1 = split[0]
            key2 = ".".join(split[1:])
            if key1 not in new_dict:
                new_dict[key1] = {}
            new_dict[key1][key2] = v
    for k in new_dict:
        if isinstance(new_dict[k], dict):
            new_dict[k] = fold_dict(new_dict[k])
    return new_dict


def test_module(module_name, module_class, module_param, base_inputs, eps=1e-5):
    """Numerical gradient checker via finite differences.
    Compares analytical gradients from .backward() against finite differences.
    Prints ✓/✗ per parameter and input tensor.
    Uses fixed step size eps.
    """
    def print_result(name, target, ours):
        ok = torch.allclose(target, ours, atol=1e-4, rtol=1e-2)
        mark = '\033[92m✓\033[0m' if ok else '\033[91m✗\033[0m'
        print(mark, module_name, name, "test passed" if ok else "test FAILED")

    def loss_fn(model, inputs, return_grad=False):
        output = model(*inputs)
        if isinstance(output, torch.Tensor):
            output = [output]
        loss = 0
        coeffs = []
        for o in output:
            size = o.numel()
            coeff = torch.arange(1, size + 1) / size
            coeff = coeff.reshape(o.shape).to(torch.float64)
            loss += (o * coeff).sum()
            coeffs.append(coeff)
        if not return_grad:
            return loss, None
        grad = model.backward(*coeffs)
        if isinstance(grad, torch.Tensor):
            grad = [grad]
        return loss, grad

    base_module = module_class(*module_param)
    new_module = module_class(*module_param)
    base_module.to(torch.float64)
    new_module.to(torch.float64)

    state_dict = unfold_dict(base_module.state_dict())
    if isinstance(base_inputs, torch.Tensor):
        base_inputs = [base_inputs]
    base_inputs = [
        bi.to(dtype=torch.float64 if bi.dtype != torch.long else torch.long)
        for bi in base_inputs
    ]

    base_scalar, input_grads = loss_fn(base_module, base_inputs, return_grad=True)
    weight_grads = unfold_dict(base_module.grad_dict())

    # Check parameter gradients
    for k, grad_b in weight_grads.items():
        grad_n = torch.zeros_like(grad_b)
        indices = [list(range(s)) for s in grad_b.shape]
        for idx in product(*indices):
            new_dict = deepcopy(state_dict)
            new_dict[k][idx] += eps
            new_module.load_state_dict(fold_dict(new_dict))
            new_scalar, _ = loss_fn(new_module, base_inputs)
            grad_n[idx] = (new_scalar - base_scalar) / eps
        print_result(k, grad_n, grad_b)

    # Check input gradients (skip long tensors)
    for i, (base_input, input_grad) in enumerate(zip(base_inputs, input_grads)):
        if base_input.dtype == torch.long:
            continue
        grad_n = torch.zeros_like(base_input)
        new_inputs = list(base_inputs)
        indices = [list(range(s)) for s in base_input.shape]
        for idx in product(*indices):
            new_inputs[i] = base_input.clone()
            new_inputs[i][idx] += eps
            new_scalar, _ = loss_fn(base_module, new_inputs)
            grad_n[idx] = (new_scalar - base_scalar) / eps
        print_result(f"input_{i}", grad_n, input_grad)

    print()


# ============================================================
# Module base class
# ============================================================

class Module:
    """
    Base class for all layers. Implements the custom autograd system:
      - forward(*args) → output
      - backward(*grad_outputs) → grad_inputs
      - state_dict() / grad_dict() / load_state_dict() recurse via sub_modules list
      - to(dest) moves all tensors to dtype or device

    Convention: every learnable tensor self.X must be listed in self.sub_modules,
    and backward() must set self.grad_X.
    """

    def __init__(self):
        self.sub_modules: List[str] = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *grad_outputs):
        raise NotImplementedError

    def state_dict(self) -> dict:
        d = {}
        for key in self.sub_modules:
            sub = getattr(self, key)
            d[key] = sub.state_dict() if isinstance(sub, Module) else sub
        return d

    def grad_dict(self) -> dict:
        d = {}
        for key in self.sub_modules:
            sub = getattr(self, key)
            if isinstance(sub, Module):
                d[key] = sub.grad_dict()
            else:
                d[key] = getattr(self, "grad_" + key)
        return d

    def load_state_dict(self, state_dict: dict):
        for key in self.sub_modules:
            sub = getattr(self, key)
            if key not in state_dict:
                raise ValueError(f"Missing key '{key}' in state_dict")
            val = state_dict[key]
            if isinstance(sub, Module):
                sub.load_state_dict(val)
            else:
                setattr(self, key, val)

    def to(self, dest):
        for key in self.sub_modules:
            sub = getattr(self, key)
            setattr(self, key, sub.to(dest))
        return self


# ============================================================
# Building blocks
# ============================================================

class Embedding(Module):
    """Token embedding table. weights shape: (num_emb, emb_dim)."""

    def __init__(self, num_emb: int, emb_dim: int, pad_idx: Optional[int] = None):
        super().__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.pad_idx = pad_idx
        self.weights = torch.randn(num_emb, emb_dim)
        if pad_idx is not None:
            self.weights[pad_idx] = 0
        self.sub_modules = ['weights']

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.dtype == torch.long
        self.inputs = inputs
        return self.weights[inputs]

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = grad_output.shape
        index = one_hot(self.inputs, num_classes=self.num_emb)               # (B, L, V)
        index = index.reshape(batch_size, seq_len, self.num_emb, 1)
        grad = grad_output.reshape(batch_size, seq_len, 1, self.emb_dim) * index
        self.grad_weights = grad.sum(dim=[0, 1])
        return grad.sum(dim=[2, 3])                                           # grad_input (unused; tokens are discrete)


class Softmax(Module):
    """Numerically stable softmax over the last dimension. Backward via full Jacobian."""

    def __init__(self):
        super().__init__()
        self.sub_modules = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max, _ = torch.max(x, dim=-1, keepdim=True)
        exp_x = torch.exp(x - x_max)
        self.output = exp_x / exp_x.sum(dim=-1, keepdim=True)
        return self.output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        # J = diag(s) - s @ s^T  →  grad_input = J @ grad_output
        diag_term = torch.diag_embed(self.output)
        joint_term = torch.matmul(self.output.unsqueeze(-1), self.output.unsqueeze(-2))
        grad_soft = diag_term - joint_term
        return torch.matmul(grad_soft, grad_output.unsqueeze(-1)).squeeze(-1)


class Linear(Module):
    """y = x @ W + b. Works for any leading batch dimensions."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = (1 / math.sqrt(input_dim)) * torch.randn(input_dim, output_dim)
        self.bias = torch.zeros(output_dim)
        self.sub_modules = ['weight', 'bias']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        return torch.matmul(x, self.weight) + self.bias

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        grad_x = torch.matmul(grad_output, self.weight.T)
        x_flat = self.x.reshape(-1, self.input_dim)
        g_flat = grad_output.reshape(-1, self.output_dim)
        self.grad_weight = torch.matmul(x_flat.T, g_flat)
        self.grad_bias = g_flat.sum(dim=0)
        return grad_x


class Activation(Module):
    """Elementwise activation: sigmoid | tanh | relu | silu."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.sub_modules = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        if self.name == 'sigmoid':
            self.output = 1 / (1 + torch.exp(-x))
        elif self.name == 'tanh':
            self.output = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        elif self.name == 'relu':
            self.output = torch.clamp(x, min=0.)
        elif self.name == 'silu':
            self.output = x / (1 + torch.exp(-x))
        else:
            raise ValueError(f"Unsupported activation: {self.name}")
        return self.output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        x, o = self.x, self.output
        if self.name == 'sigmoid':
            return grad_output * o * (1 - o)
        elif self.name == 'tanh':
            return grad_output * (1 - o ** 2)
        elif self.name == 'relu':
            return grad_output * (o > 0).to(grad_output.dtype)
        elif self.name == 'silu':
            expx = torch.exp(-x)
            return grad_output * (1 + x * expx + expx) / (1 + expx) ** 2
        else:
            raise ValueError(f"Unsupported activation: {self.name}")


class RMSNorm(Module):
    """
    Root Mean Square Normalization: y = x / rms(x) [* scale if learnable_scale].

    Args:
        dim: feature dimension (required when learnable_scale=True).
        eps: numerical stability epsilon.
        learnable_scale: if True, adds a learnable per-feature scale parameter.
    """

    def __init__(self, dim: Optional[int] = None, eps: float = 1e-6,
                 learnable_scale: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.learnable_scale = learnable_scale
        if learnable_scale:
            assert dim is not None, "dim required when learnable_scale=True"
            self.scale = torch.ones(dim)
            self.sub_modules = ['scale']
        else:
            self.sub_modules = []

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        power = x.pow(2).mean(-1, keepdim=True)
        self.sqrt = torch.sqrt(power + self.eps)        # (*, 1)
        self.output = x / self.sqrt                     # normalized; stored for backward
        return self.output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x)
        if self.learnable_scale:
            out = out * self.scale
        return out

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self.learnable_scale:
            # grad w.r.t. scale: sum over all dims except last (feature dim)
            dims = tuple(range(grad_output.ndim - 1))
            self.grad_scale = (grad_output * self.output).sum(dim=dims)
            # chain rule: peel off the scale before the norm backward
            grad_output = grad_output * self.scale

        # Jacobian of RMSNorm: J = (1/rms) * (I - output @ output^T / dim)
        # grad_input = J @ grad_output
        dim = grad_output.shape[-1]
        joint_term = (1 / (dim * self.sqrt.unsqueeze(-1))) * torch.matmul(
            self.output.unsqueeze(-1), self.output.unsqueeze(-2))
        sqrt_stack = self.sqrt.expand(*self.sqrt.shape[:-1], dim)
        diag_term = torch.diag_embed(1 / sqrt_stack)
        return torch.matmul(diag_term - joint_term, grad_output.unsqueeze(-1)).squeeze(-1)


class RotaryEmbedding(Module):
    """
    Rotary Position Embedding (RoPE). Computes (cos, sin) for given position ids.
    theta is precomputed and not a learnable parameter.
    """

    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.sub_modules = []              # no trainable params
        self._init_theta()

    def _init_theta(self):
        exponent = torch.arange(0, self.head_dim, 2).float() / self.head_dim
        self.theta = 1 / self.base ** exponent                  # (head_dim/2,)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids: (bsz, seq)
        Returns:
            cos, sin: (bsz, seq, head_dim)
        """
        bsz = position_ids.shape[0]
        pos_ = position_ids[:, :, None].to(dtype=self.theta.dtype)   # (bsz, seq, 1)
        theta_ = self.theta[None, None, :].expand(bsz, 1, -1)        # (bsz, 1, head_dim/2)
        freqs = torch.matmul(pos_, theta_)                            # (bsz, seq, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)                       # (bsz, seq, head_dim)
        self.emb = emb
        return torch.cos(emb), torch.sin(emb)

    def backward(self, grad_cos: torch.Tensor,
                 grad_sin: torch.Tensor) -> torch.Tensor:
        bsz = grad_cos.shape[0]
        grad_emb = -grad_cos * torch.sin(self.emb) + grad_sin * torch.cos(self.emb)
        hd = grad_emb.shape[-1]
        grad_freqs = grad_emb[:, :, :hd // 2] + grad_emb[:, :, hd // 2:]
        theta_ = self.theta[None, None, :].expand(bsz, 1, -1)
        grad_pos = torch.matmul(grad_freqs, theta_.transpose(1, 2))
        return grad_pos.squeeze(-1)

    def to(self, dest):
        """Custom override: move theta to dest (not in sub_modules)."""
        self.theta = self.theta.to(dest)
        return self

    def state_dict(self):
        return {}

    def grad_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class RotateHalf(Module):
    """Rotates the last dimension by negating the second half and swapping halves."""

    def __init__(self):
        super().__init__()
        self.sub_modules = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        # forward: [x1, x2] → [-x2, x1]
        # backward: grad_[-x2] → -grad_x2, grad_[x1] → grad_x1
        cut = grad_output.shape[-1] - grad_output.shape[-1] // 2
        g1 = grad_output[..., :cut]    # grad w.r.t. -x2 → grad_x2 = -g1
        g2 = grad_output[..., cut:]    # grad w.r.t. x1  → grad_x1 = g2
        return torch.cat((g2, -g1), dim=-1)


class RotaryPosEmb(Module):
    """
    Applies RoPE to query and key tensors.
    forward: (q, k, cos, sin) → (q_rot, k_rot)
    """

    def __init__(self):
        super().__init__()
        self.rotate_half = RotateHalf()
        self.sub_modules = []

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # cos/sin: (bsz, seq, head_dim) → unsqueeze head dim for broadcast
        cos = cos.unsqueeze(1)   # (bsz, 1, seq, head_dim)
        sin = sin.unsqueeze(1)
        self.q = q
        self.k = k
        self.rot_q = self.rotate_half(q)
        self.rot_k = self.rotate_half(k)
        self.cos = cos.clone()
        self.sin = sin.clone()
        q_embed = q * cos + self.rot_q * sin
        k_embed = k * cos + self.rot_k * sin
        return q_embed, k_embed

    def backward(self, grad_q_embed: torch.Tensor,
                 grad_k_embed: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_q = grad_q_embed * self.cos + self.rotate_half.backward(grad_q_embed) * self.sin
        grad_k = grad_k_embed * self.cos + self.rotate_half.backward(grad_k_embed) * self.sin
        # Sum over head dim separately — supports GQA where q and k have different n_heads
        grad_cos = (grad_q_embed * self.q).sum(dim=1) + (grad_k_embed * self.k).sum(dim=1)
        grad_sin = (grad_q_embed * self.rot_q).sum(dim=1) + (grad_k_embed * self.rot_k).sum(dim=1)
        return grad_q, grad_k, grad_cos, grad_sin


# ============================================================
# GQA helper
# ============================================================

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads to match query head count for Grouped Query Attention.
    Input:  (bsz, n_kv_heads, seq, head_dim)
    Output: (bsz, n_kv_heads * n_rep, seq, head_dim)
    """
    if n_rep == 1:
        return x
    bsz, n_kv_heads, seq, head_dim = x.shape
    return (x.unsqueeze(2)
             .expand(bsz, n_kv_heads, n_rep, seq, head_dim)
             .reshape(bsz, n_kv_heads * n_rep, seq, head_dim))


# ============================================================
# Attention (MHA + GQA in a single class)
# ============================================================

class Attention(Module):
    """
    Multi-Head Attention with optional Grouped Query Attention (GQA).

    MHA (default): n_kv_heads == n_heads, separate wq/wk/wv projections.
    GQA:           n_kv_heads < n_heads, fused wqkv projection split by slicing.

    Args:
        dim:        model dimension
        head_dim:   dimension per head
        n_heads:    number of query heads
        n_kv_heads: number of KV heads (None → same as n_heads = MHA)
        layer_id:   used by KVCache to index per-layer cache

    KV cache:
        Pass a KVCache instance to forward() for inference. Backward is unchanged
        (cache is never used during training).
    """

    def __init__(self, dim: int, head_dim: int, n_heads: int,
                 n_kv_heads: Optional[int] = None, layer_id: int = 0):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0, \
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        self.n_rep = n_heads // self.n_kv_heads
        self.layer_id = layer_id

        self.rot_pos_emb = RotaryPosEmb()
        self.softmax = Softmax()

        if self.n_rep == 1:
            # Standard MHA: separate projections
            self.wq = Linear(dim, n_heads * head_dim)
            self.wk = Linear(dim, self.n_kv_heads * head_dim)
            self.wv = Linear(dim, self.n_kv_heads * head_dim)
            self.wo = Linear(n_heads * head_dim, dim)
            self.sub_modules = ['wq', 'wk', 'wv', 'wo']
        else:
            # GQA: fused QKV projection, then slice
            q_dim = n_heads * head_dim
            kv_dim = self.n_kv_heads * head_dim
            self.wqkv = Linear(dim, q_dim + 2 * kv_dim)
            self.wo = Linear(n_heads * head_dim, dim)
            self.sub_modules = ['wqkv', 'wo']

    def _project_qkv(self, x: torch.Tensor):
        """Project input to Q, K, V. Returns (query, key, value) as 2D tensors."""
        if self.n_rep == 1:
            return self.wq(x), self.wk(x), self.wv(x)
        else:
            q_dim = self.n_heads * self.head_dim
            kv_dim = self.n_kv_heads * self.head_dim
            qkv = self.wqkv(x)
            return qkv[..., :q_dim], qkv[..., q_dim:q_dim + kv_dim], qkv[..., q_dim + kv_dim:]

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                kvcache=None,
                return_attn: bool = False):
        """
        Args:
            x:       (bsz, new_seqlen, dim) — new tokens only when using kvcache
            cos/sin: (bsz, new_seqlen, head_dim) — positions for new tokens only
            mask:    (new_seqlen, total_seqlen) or (seqlen, seqlen) causal mask
            kvcache: KVCache instance or None
        """
        bsz, seqlen, _ = x.shape

        query, key, value = self._project_qkv(x)

        # Reshape to (bsz, n_heads/n_kv_heads, seqlen, head_dim)
        query = query.reshape(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        key   = key.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (positions of new tokens only; cached keys already have RoPE)
        query, key = self.rot_pos_emb(query, key, cos, sin)

        # Store pre-cache key/value for backward (training always has kvcache=None)
        self.query = query
        self.key_new = key
        self.value = value

        # KV cache: concatenate new k,v with cached history
        if kvcache is not None:
            key, value = kvcache.update(key, value, self.layer_id)

        # GQA: expand KV heads to match query head count
        key_full   = repeat_kv(key, self.n_rep)
        value_full = repeat_kv(value, self.n_rep)

        # Store expanded k,v for backward
        self.key_full   = key_full
        self.value_full = value_full

        # Scaled dot-product attention
        scores = torch.matmul(query, key_full.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        self.attn_score = self.softmax(scores)   # (bsz, n_heads, seqlen, total_seqlen)

        output = torch.matmul(self.attn_score, value_full)     # (bsz, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).reshape(bsz, seqlen, -1)
        output = self.wo(output)

        if return_attn:
            return output, self.attn_score
        return output

    def backward(self, grad_output: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_heads, seqlen, head_dim = self.query.shape

        # wo backward
        grad_output = self.wo.backward(grad_output)
        grad_output = grad_output.reshape(bsz, seqlen, n_heads, head_dim).transpose(1, 2)

        # Attention backward: output = attn_score @ value_full
        grad_attn_score = torch.matmul(grad_output, self.value_full.transpose(2, 3))
        grad_value_full = torch.matmul(self.attn_score.transpose(2, 3), grad_output)

        # Softmax backward
        grad_scores = self.softmax.backward(grad_attn_score)

        # Scores backward: scores = query @ key_full^T / sqrt(head_dim)
        grad_query    = torch.matmul(grad_scores, self.key_full) / math.sqrt(head_dim)
        grad_key_full = torch.matmul(self.query.transpose(2, 3), grad_scores
                                     ).transpose(2, 3) / math.sqrt(head_dim)

        # Reduce GQA expanded gradients back to n_kv_heads
        if self.n_rep > 1:
            # (bsz, n_heads, seqlen, head_dim) → (bsz, n_kv_heads, seqlen, head_dim)
            grad_key   = grad_key_full.reshape(
                bsz, self.n_kv_heads, self.n_rep, seqlen, head_dim).sum(dim=2)
            grad_value = grad_value_full.reshape(
                bsz, self.n_kv_heads, self.n_rep, seqlen, head_dim).sum(dim=2)
        else:
            grad_key   = grad_key_full
            grad_value = grad_value_full

        # RoPE backward
        grad_query, grad_key, grad_cos, grad_sin = self.rot_pos_emb.backward(grad_query, grad_key)

        # Reshape back to (bsz, seqlen, *)
        grad_query = grad_query.transpose(1, 2).reshape(bsz, seqlen, -1)
        grad_key   = grad_key.transpose(1, 2).reshape(bsz, seqlen, -1)
        grad_value = grad_value.transpose(1, 2).reshape(bsz, seqlen, -1)

        # Input projection backward
        if self.n_rep == 1:
            grad_x = (self.wq.backward(grad_query)
                    + self.wk.backward(grad_key)
                    + self.wv.backward(grad_value))
        else:
            grad_qkv = torch.cat([grad_query, grad_key, grad_value], dim=-1)
            grad_x = self.wqkv.backward(grad_qkv)

        return grad_x, grad_cos, grad_sin


class FeedForward(Module):
    """
    SwiGLU Feed-Forward: output = w3(silu(w2(x)) * w1(x))
    Three projections: w1 (linear path), w2 (gate, through silu), w3 (output).
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = Linear(dim, hidden_dim)
        self.w2 = Linear(dim, hidden_dim)
        self.w3 = Linear(hidden_dim, dim)
        self.act_fn = Activation('silu')
        self.sub_modules = ['w1', 'w2', 'w3']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.feature1 = self.w1(x)          # linear path
        feature2      = self.w2(x)          # gate input
        self.gate     = self.act_fn(feature2)
        return self.w3(self.gate * self.feature1)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        grad_feature3  = self.w3.backward(grad_output)
        grad_gate      = grad_feature3 * self.feature1
        grad_feature1  = grad_feature3 * self.gate
        grad_feature2  = self.act_fn.backward(grad_gate)
        return self.w1.backward(grad_feature1) + self.w2.backward(grad_feature2)


class TransformerBlock(Module):
    """
    Pre-norm Transformer block: residual attention + residual FFN.
    forward(x, cos, sin, mask, kvcache) → output (same shape as x)
    """

    def __init__(self, layer_id: int, dim: int, n_heads: int,
                 n_kv_heads: Optional[int] = None):
        super().__init__()
        self.layer_id = layer_id
        head_dim = dim // n_heads
        self.attention     = Attention(dim, head_dim, n_heads, n_kv_heads, layer_id)
        self.feed_forward  = FeedForward(dim=dim, hidden_dim=4 * dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm       = RMSNorm(dim)
        # Norms have no trainable params (learnable_scale=False) so they are
        # excluded from sub_modules to avoid load_state_dict key mismatch.
        self.sub_modules = ['attention', 'feed_forward']

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                kvcache=None) -> torch.Tensor:
        # Pre-norm attention + residual
        attn_out = self.attention(self.attention_norm(x), cos, sin, mask, kvcache)
        h = x + attn_out
        # Pre-norm FFN + residual
        ffn_out = self.feed_forward(self.ffn_norm(h))
        return h + ffn_out

    def backward(self, grad_output: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FFN residual backward
        grad_ffn_norm = self.feed_forward.backward(grad_output)
        grad_h = grad_output + self.ffn_norm.backward(grad_ffn_norm)

        # Attention residual backward
        grad_attn_norm, grad_cos, grad_sin = self.attention.backward(grad_h)
        grad_x = grad_h + self.attention_norm.backward(grad_attn_norm)

        return grad_x, grad_cos, grad_sin


class CrossEntLoss(Module):
    """
    Cross-entropy loss for language modeling.
    forward(logits, labels) → scalar loss
    backward(grad_output) → (grad_logits, labels)
    """

    def __init__(self, class_n: int = 0):
        super().__init__()
        self.sub_modules = []

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.bsz, self.seqlen, self.vocab_size = logits.shape
        logits_2d = logits.reshape(-1, self.vocab_size)
        self.labels = labels.reshape(-1)

        # Numerically stable softmax
        exp_logits = torch.exp(logits_2d - logits_2d.max(dim=-1, keepdim=True).values)
        self.softmax = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

        log_probs = torch.log(self.softmax)
        loss = -log_probs[torch.arange(self.labels.shape[0]), self.labels].mean()
        return loss

    def backward(self, grad_output: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        # grad_output is typically torch.ones(()) from the training loop
        g = float(grad_output.reshape(()))
        device, dtype = self.softmax.device, self.softmax.dtype
        n = self.bsz * self.seqlen

        grad_log_prob = (-torch.ones(n, device=device, dtype=dtype) / n) * g
        one_hot_mat = one_hot(self.labels, num_classes=self.vocab_size).to(device=device, dtype=dtype)
        grad_log_prob = grad_log_prob.unsqueeze(-1) * one_hot_mat
        grad_logits = grad_log_prob - grad_log_prob.sum(dim=-1, keepdim=True) * self.softmax
        grad_logits = grad_logits.reshape(self.bsz, self.seqlen, self.vocab_size)
        return grad_logits, self.labels


# ============================================================
# KV Cache
# ============================================================

class KVCache:
    """
    Key-Value cache for efficient autoregressive inference.
    Each call to update() concatenates new K/V with the cached history.
    _seen_tokens tracks total tokens processed (updated once per forward pass via layer 0).
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.key_cache:   List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._seen_tokens = 0

    def update(self, key: torch.Tensor, value: torch.Tensor,
               layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key/value to cache for layer_id.
        Only layer 0 increments _seen_tokens (avoid counting once per layer).
        Returns (full_key, full_value) including history.
        """
        if self.key_cache[layer_id] is None:
            self.key_cache[layer_id]   = key
            self.value_cache[layer_id] = value
        else:
            self.key_cache[layer_id]   = torch.cat([self.key_cache[layer_id],   key],   dim=-2)
            self.value_cache[layer_id] = torch.cat([self.value_cache[layer_id], value], dim=-2)

        if layer_id == 0:
            self._seen_tokens += key.shape[-2]

        return self.key_cache[layer_id], self.value_cache[layer_id]

    def reset(self):
        self.key_cache   = [None] * self.num_layers
        self.value_cache = [None] * self.num_layers
        self._seen_tokens = 0


# ============================================================
# Full Transformer
# ============================================================

class Transformer(Module):
    """
    Decoder-only Transformer for next-token prediction.

    forward(tokens, labels=None, kvcache=None) → logits | (logits, loss)
    backward(grad_logits, grad_loss=None) → (grad_input, grad_labels)

    Training:  no kvcache; full sequence mask
    Inference: pass KVCache for efficient decoding; only new tokens embedded
    """

    def __init__(self, n_layers: int, vocab_size: int, dim: int, n_heads: int,
                 n_kv_heads: Optional[int] = None):
        super().__init__()
        self.n_layers   = n_layers
        self.vocab_size = vocab_size
        self.dim        = dim
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = dim // n_heads

        self.embed   = Embedding(vocab_size, dim)
        self.rope    = RotaryEmbedding(self.head_dim)
        self.norm    = RMSNorm(dim)
        self.output  = Linear(dim, vocab_size)
        self.loss_fn = CrossEntLoss(vocab_size)

        for i in range(n_layers):
            setattr(self, f'layer_{i}', TransformerBlock(i, dim, n_heads, n_kv_heads))

        # rope and norm have no trainable params. Exclude from sub_modules to avoid
        # load_state_dict key-mismatch in test_module (empty dicts vanish after
        # unfold_dict/fold_dict round-trip). Handled manually in to().
        self.sub_modules = (
            ['embed']
            + [f'layer_{i}' for i in range(n_layers)]
            + ['output']
        )

    def to(self, dest):
        super().to(dest)          # moves embed, layers, output
        self.rope.to(dest)        # theta is not in sub_modules; move manually
        return self

    def forward(self, tokens: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                kvcache: Optional[KVCache] = None):
        bsz, seqlen = tokens.shape

        if kvcache is not None:
            # Inference: embed only new tokens; offset positions by cache length
            cache_len  = kvcache._seen_tokens
            new_tokens = tokens[:, cache_len:]
            new_seqlen = new_tokens.shape[1]
            h = self.embed(new_tokens)
            pos_ids = torch.arange(cache_len, seqlen, device=tokens.device).reshape(1, -1)
            cos, sin = self.rope(pos_ids)

            # Causal mask: shape (new_seqlen, total_seqlen)
            # mask[q, k] = -inf if k > cache_len + q (absolute position of query)
            if new_seqlen > 1 or cache_len == 0:
                mask = torch.full((new_seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=cache_len + 1)
            else:
                mask = None  # single new token can attend to all cached tokens
        else:
            # Training: full sequence
            h = self.embed(tokens)
            pos_ids = torch.arange(seqlen, device=tokens.device).reshape(1, -1)
            cos, sin = self.rope(pos_ids)
            mask = None
            if seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
                mask = torch.triu(mask, diagonal=1)

        for i in range(self.n_layers):
            layer = getattr(self, f'layer_{i}')
            h = layer(h, cos, sin, mask, kvcache)

        h      = self.norm(h)
        logits = self.output(h)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return logits, loss
        return logits

    def backward(self, grad_logits: torch.Tensor,
                 grad_loss: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        if grad_loss is not None:
            _grad_logits, grad_labels = self.loss_fn.backward(grad_loss)
            grad_logits = grad_logits + _grad_logits
        else:
            grad_labels = None

        grad_h = self.output.backward(grad_logits)
        grad_h = self.norm.backward(grad_h)

        for i in reversed(range(self.n_layers)):
            layer = getattr(self, f'layer_{i}')
            grad_h, grad_cos, grad_sin = layer.backward(grad_h)

        grad_input = self.embed.backward(grad_h)
        return grad_input, grad_labels


# ============================================================
# LoRA
# ============================================================

class LoRALinear:
    """
    Low-Rank Adaptation of a frozen Linear layer.
    Adds W' = W + (alpha/rank) * A @ B where A is (d_in, rank), B is (rank, d_out).
    B is zero-initialized so the adapter starts as identity (no change at init).

    Uses nn.Parameter for A and B so PyTorch autograd handles their gradients.
    The base Linear uses plain tensors (no autograd).

    Usage:
        lora = LoRALinear(some_linear, rank=4)
        y = lora(x)          # training: adapts output
        lora.merge()         # optionally fold LoRA into base weight for zero-overhead inference
    """

    def __init__(self, linear: Linear, rank: int, alpha: float = 1.0):
        self.linear  = linear
        self.rank    = rank
        self.alpha   = alpha
        self.scaling = alpha / rank
        d_in  = linear.input_dim
        d_out = linear.output_dim
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base   = self.linear(x)                                  # plain tensor (no autograd)
        lora   = (x @ self.lora_A @ self.lora_B) * self.scaling  # autograd through A, B
        return base + lora

    def merge(self) -> Linear:
        """Fold LoRA into the base weight in-place. Returns the modified Linear."""
        with torch.no_grad():
            self.linear.weight += (self.lora_A @ self.lora_B) * self.scaling
        return self.linear

    def parameters(self):
        """Yields trainable LoRA parameters for use with torch.optim."""
        yield self.lora_A
        yield self.lora_B


# ============================================================
# Optimizers
# ============================================================

class Optimizer:
    """
    Base optimizer. Maintains a flat param_dict holding references to model tensors.
    In-place updates (`param_dict[k] -= step`) modify model tensors directly — no
    load_state_dict call needed.

    Subclasses implement update_dict(grad_dict) → step_dict.
    """

    def __init__(self, model: Module, wd: float = 0.0001):
        self.model = model
        self.wd    = wd
        # Values in param_dict are REFERENCES to model's tensors
        self.param_dict = unfold_dict(model.state_dict())

    def wd_grad(self, grad_dict: dict) -> dict:
        """Add weight-decay term: effective_grad = grad + wd * param."""
        for k in grad_dict:
            grad_dict[k] = grad_dict[k] + self.wd * self.param_dict[k]
        return grad_dict

    def update_dict(self, grad_dict: dict) -> dict:
        raise NotImplementedError

    def step(self):
        grad_dict = unfold_dict(self.model.grad_dict())
        grad_dict = self.wd_grad(grad_dict)
        step_dict = self.update_dict(grad_dict)
        for k in self.param_dict:
            self.param_dict[k] -= step_dict[k]   # in-place: modifies model tensors directly


class SGD(Optimizer):
    """Stochastic Gradient Descent: param -= lr * grad."""

    def __init__(self, model: Module, lr: float = 0.01, wd: float = 0.01):
        super().__init__(model, wd)
        self.lr = lr

    def update_dict(self, grad_dict: dict) -> dict:
        return {k: self.lr * grad_dict[k] for k in grad_dict}


class Momentum(Optimizer):
    """SGD with momentum: v = beta*v + lr*grad; param -= v."""

    def __init__(self, model: Module, lr: float = 0.01,
                 beta: float = 0.9, wd: float = 0.001):
        super().__init__(model, wd)
        self.lr   = lr
        self.beta = beta
        self.velocity = {k: torch.zeros_like(v) for k, v in self.param_dict.items()}

    def update_dict(self, grad_dict: dict) -> dict:
        step = {}
        for k in self.param_dict:
            self.velocity[k] = self.beta * self.velocity[k] + self.lr * grad_dict[k]
            step[k] = self.velocity[k].clone()
        return step


class RMSProp(Optimizer):
    """RMSProp: G = beta*G + (1-beta)*grad²; param -= lr*grad/(sqrt(G)+eps)."""

    def __init__(self, model: Module, lr: float = 0.01,
                 beta: float = 0.9, wd: float = 0.001):
        super().__init__(model, wd)
        self.lr   = lr
        self.beta = beta
        self.eps  = 1e-6
        self.G    = {k: torch.zeros_like(v) for k, v in self.param_dict.items()}

    def update_dict(self, grad_dict: dict) -> dict:
        step = {}
        for k in self.param_dict:
            self.G[k] = self.beta * self.G[k] + (1 - self.beta) * grad_dict[k] ** 2
            step[k]   = self.lr * grad_dict[k] / (torch.sqrt(self.G[k]) + self.eps)
        return step


class Adam(Optimizer):
    """
    Adam (Kingma & Ba, 2014).
    m = beta1*m + (1-beta1)*grad          (first moment)
    v = beta2*v + (1-beta2)*grad²         (second moment)
    param -= lr * m_hat / (sqrt(v_hat) + eps)   (bias-corrected)
    """

    def __init__(self, model: Module, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 wd: float = 0.001):
        super().__init__(model, wd)
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = 1e-6
        self.t     = 1
        self.m     = {k: torch.zeros_like(v) for k, v in self.param_dict.items()}
        self.v     = {k: torch.zeros_like(v) for k, v in self.param_dict.items()}

    def update_dict(self, grad_dict: dict) -> dict:
        step = {}
        for k in self.param_dict:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad_dict[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grad_dict[k] ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            step[k] = self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        self.t += 1
        return step


# ============================================================
# Training utilities
# ============================================================

class Shakespeare:
    """
    DataLoader for the tiny Shakespeare dataset.
    Train split: first 30k characters. Test split: remainder.
    Trains a RegexBPETokenizer on the train split.
    """

    def __init__(self, data_path: str, device, vocab_size: int):
        with open(data_path) as f:
            data = f.read()
        print(f"Dataset length: {len(data)} chars")

        self.device    = device
        self.tokenizer = RegexBPETokenizer()
        self.tokenizer.train(data[:30000], vocab_size)

        train_ids = self.tokenizer.encode(data[:30000])
        test_ids  = self.tokenizer.encode(data[30000:])
        self.trainset = torch.tensor(train_ids, dtype=torch.long, device=device)
        self.testset  = torch.tensor(test_ids,  dtype=torch.long, device=device)

    def dataloader(self, split: str, batch_size: int, seq_len: int):
        dataset = self.trainset if split == 'train' else self.testset
        start_indices = np.arange(0, len(dataset) - seq_len, seq_len)
        np.random.shuffle(start_indices)

        for i in range(0, len(start_indices), batch_size):
            batch_idx = start_indices[i:i + batch_size]
            input_idx = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.long)
            labels    = torch.zeros(batch_size, seq_len, device=self.device, dtype=torch.long)
            for j, start in enumerate(batch_idx):
                input_idx[j] = dataset[start:start + seq_len]
                labels[j]    = dataset[start + 1:start + seq_len + 1]
            yield input_idx, labels


def train_step(model: Transformer, optimizer: Optimizer,
               tokens: torch.Tensor, labels: torch.Tensor) -> float:
    """
    One training step: forward → backward → optimizer update.
    Returns the loss as a Python float.
    """
    logits, loss = model(tokens, labels)
    # grad_logits=zeros: only the loss path contributes to gradients
    # grad_loss=ones:    standard gradient for scalar loss
    model.backward(torch.zeros_like(logits), torch.ones_like(loss))
    optimizer.step()
    return loss.item()


# ============================================================
# Main: gradient tests + KV cache test + training demo + LoRA demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("1. Gradient Tests")
    print("=" * 60)

    bsz, seq_len, vocab_size = 5, 4, 7
    n_layers, n_heads, head_dim = 2, 3, 4
    dim = n_heads * head_dim    # 12

    tok_ids = torch.randint(0, vocab_size, (bsz, seq_len))
    q_test  = torch.randn(bsz, n_heads, seq_len, head_dim)
    k_test  = torch.randn(bsz, n_heads, seq_len, head_dim)
    h_test  = torch.randn(bsz, seq_len, dim)

    # Embedding
    test_module("Embedding", Embedding, [vocab_size, dim], tok_ids)

    # Softmax
    test_module("Softmax", Softmax, [], torch.randn(bsz, seq_len, vocab_size))

    # Linear (2D, 3D, 4D inputs)
    test_module("Linear 2D", Linear, [6, 10], torch.randn(5, 6))
    test_module("Linear 3D", Linear, [6, 10], torch.randn(4, 5, 6))
    test_module("Linear 4D", Linear, [6, 10], torch.randn(3, 4, 5, 6))

    # RotaryEmbedding
    test_module("RotaryEmbedding", RotaryEmbedding, [head_dim],
                torch.randint(0, seq_len, (bsz, seq_len)).float())

    # RotateHalf
    test_module("RotateHalf", RotateHalf, [], torch.randn(bsz, seq_len, dim))

    # RotaryPosEmb
    _cos, _sin = RotaryEmbedding(head_dim)(torch.randint(0, seq_len, (bsz, seq_len)).float())
    test_module("RotaryPosEmb", RotaryPosEmb, [], [q_test, k_test, _cos, _sin])

    # Attention (MHA)
    test_module("Attention MHA", Attention, [dim, head_dim, n_heads],
                [h_test, _cos, _sin])

    # Attention (GQA, n_kv_heads=1)
    n_kv_gqa = 1
    test_module("Attention GQA (n_kv=1)", Attention, [dim, head_dim, n_heads, n_kv_gqa],
                [h_test, _cos, _sin])

    # RMSNorm (no learnable scale)
    test_module("RMSNorm (no scale)", RMSNorm, [], torch.randn(5, 6))

    # RMSNorm (learnable scale)
    test_module("RMSNorm (learnable scale)", RMSNorm, [6, 1e-6, True], torch.randn(5, 6))

    # Activations
    for name in ['sigmoid', 'tanh', 'relu', 'silu']:
        test_module(f"Activation({name})", Activation, [name], torch.randn(5, 6))

    # FeedForward
    test_module("FeedForward", FeedForward, [dim, 4 * dim], torch.randn(bsz, dim))

    # TransformerBlock
    test_module("TransformerBlock", TransformerBlock, [0, dim, n_heads],
                [h_test, _cos, _sin])

    # CrossEntLoss
    test_module("CrossEntLoss", CrossEntLoss, [vocab_size], [h_test, tok_ids])

    # Full Transformer
    test_module("Transformer", Transformer, [n_layers, vocab_size, dim, n_heads],
                [tok_ids, tok_ids])

    # ----------------------------------------------------------------
    print("=" * 60)
    print("2. KV Cache Correctness Test")
    print("=" * 60)

    _n_layers, _vocab, _dim, _heads = 2, 64, 16, 4
    model_kv = Transformer(_n_layers, _vocab, _dim, _heads)
    model_kv.to(torch.float32)

    # Full-sequence forward (no cache)
    x_full = torch.randint(0, _vocab, (1, 8))
    with torch.no_grad():
        logits_no_cache = model_kv(x_full)

    # Prefill with cache
    cache = KVCache(_n_layers)
    with torch.no_grad():
        logits_prefill = model_kv(x_full, kvcache=cache)

    if torch.allclose(logits_no_cache, logits_prefill, atol=1e-5):
        print('\033[92m✓\033[0m KV cache prefill matches no-cache output')
    else:
        max_diff = (logits_no_cache - logits_prefill).abs().max().item()
        print(f'\033[91m✗\033[0m KV cache prefill mismatch (max diff={max_diff:.2e})')

    # Decode: extend to one more token; compare vs fresh full forward
    x_extended = torch.randint(0, _vocab, (1, 9))
    x_extended[:, :8] = x_full
    with torch.no_grad():
        logits_full_ext = model_kv(x_extended)
        logits_cached_ext = model_kv(x_extended, kvcache=cache)

    if torch.allclose(logits_full_ext[:, -1, :], logits_cached_ext[:, 0, :], atol=1e-5):
        print('\033[92m✓\033[0m KV cache decode matches full forward (last token)')
    else:
        max_diff = (logits_full_ext[:, -1, :] - logits_cached_ext[:, 0, :]).abs().max().item()
        print(f'\033[91m✗\033[0m KV cache decode mismatch (max diff={max_diff:.2e})')

    # ----------------------------------------------------------------
    print()
    print("=" * 60)
    print("3. Short Training Demo (5 steps, Adam, Shakespeare)")
    print("=" * 60)

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'tiny_shakespeare.txt'
    )

    if not os.path.exists(data_path):
        print(f"Skipping training demo: {data_path} not found")
    else:
        device = 'cpu'
        train_vocab = 512
        shakespeare = Shakespeare(data_path, device, train_vocab)
        train_model = Transformer(2, train_vocab, 64, 8)
        optimizer   = Adam(train_model, lr=0.001, wd=0.001)

        for step, (tokens, labels) in enumerate(
                shakespeare.dataloader('train', batch_size=8, seq_len=32)):
            if step >= 5:
                break
            loss = train_step(train_model, optimizer, tokens, labels)
            print(f"  step {step}: loss = {loss:.4f}")

        print("Training demo complete.\n")

    # ----------------------------------------------------------------
    print("=" * 60)
    print("4. LoRA Demo")
    print("=" * 60)

    base = Linear(32, 64)
    lora = LoRALinear(base, rank=4, alpha=1.0)

    x_lora = torch.randn(2, 10, 32)
    y_base = base(x_lora)
    y_lora = lora(x_lora)

    diff = (y_base - y_lora).abs().max().item()
    if diff < 1e-5:
        print(f'\033[92m✓\033[0m LoRA identity init (lora_B=0, max diff={diff:.2e})')
    else:
        print(f'\033[91m✗\033[0m LoRA init not identity (max diff={diff:.2e})')

    merged = lora.merge()
    y_merged = merged(x_lora)
    diff_merged = (y_lora.detach() - y_merged).abs().max().item()
    if diff_merged < 1e-5:
        print(f'\033[92m✓\033[0m LoRA merge correct (max diff={diff_merged:.2e})')
    else:
        print(f'\033[91m✗\033[0m LoRA merge incorrect (max diff={diff_merged:.2e})')

    print("\nAll tests complete.")
