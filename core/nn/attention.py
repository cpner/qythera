import numpy as np
import math
from core.nn.module import Module, Parameter
from core.nn.linear import Linear
from core.nn.dropout import Dropout
from core.autodiff.tensor import Tensor


def precompute_freqs(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    return np.stack([np.cos(freqs), np.sin(freqs)], axis=-1)


def apply_rope(x, freqs):
    """Apply rotary position embeddings.
    x: numpy array (B, H, L, D)
    freqs: numpy array (L, D//2, 2)
    """
    B, H, L, D = x.shape
    d = D // 2
    cos = freqs[:L, :d, 0]
    sin = freqs[:L, :d, 1]
    x1 = x[..., :d]
    x2 = x[..., d:]
    rotated = np.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], axis=-1).reshape(B, H, L, D)
    return rotated


class MultiHeadAttention(Module):
    """Multi-Head Attention with GQA, RoPE, and KV cache."""

    def __init__(self, dim, num_heads, num_kv_heads=None, head_dim=None, max_seq=4096, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads // 4
        if self.num_kv_heads < 1:
            self.num_kv_heads = 1
        self.head_dim = head_dim or dim // num_heads
        self.num_queries_per_kv = num_heads // self.num_kv_heads

        self.wq = Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wo = Linear(num_heads * self.head_dim, dim, bias=False)
        self.attn_dropout = Dropout(dropout)

        self.freqs = precompute_freqs(self.head_dim, max_seq * 2)

    def forward(self, x, mask=None, kv_cache=None, position=0):
        B, L, D = x.shape
        q = self.wq(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        q = Tensor(apply_rope(q.data, self.freqs))
        k = Tensor(apply_rope(k.data, self.freqs))

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = Tensor(np.concatenate([past_k.data, k.data], axis=2))
            v = Tensor(np.concatenate([past_v.data, v.data], axis=2))
        new_cache = (k, v)

        if self.num_queries_per_kv > 1:
            k = Tensor(np.repeat(k.data, self.num_queries_per_kv, axis=1))
            v = Tensor(np.repeat(v.data, self.num_queries_per_kv, axis=1))

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(axis=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(out), new_cache
