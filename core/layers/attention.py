"""Multi-Head Attention with GQA, RoPE, and KV Cache."""

import numpy as np
import math


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, num_kv_heads=None, max_seq=2048):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        scale = 1.0 / math.sqrt(self.head_dim)
        self.wq = np.random.randn(d_model, num_heads * self.head_dim).astype(np.float32) * scale
        self.wk = np.random.randn(d_model, self.num_kv_heads * self.head_dim).astype(np.float32) * scale
        self.wv = np.random.randn(d_model, self.num_kv_heads * self.head_dim).astype(np.float32) * scale
        self.wo = np.random.randn(num_heads * self.head_dim, d_model).astype(np.float32) * scale
        
        self.freqs = self._precompute_freqs(self.head_dim, max_seq * 2)
    
    def _precompute_freqs(self, dim, seq_len, theta=10000.0):
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = np.arange(seq_len, dtype=np.float32)
        return np.outer(t, freqs)
    
    def _apply_rope(self, x, freqs):
        B, H, L, D = x.shape
        d = D // 2
        cos = np.cos(freqs[:L])
        sin = np.sin(freqs[:L])
        x1, x2 = x[..., :d], x[..., d:]
        return np.stack([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1).reshape(B, H, L, D)
    
    def forward(self, x, kv_cache=None, position=0):
        B, L, _ = x.shape
        q = (x @ self.wq).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = (x @ self.wk).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = (x @ self.wv).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        q = self._apply_rope(q, self.freqs)
        k = self._apply_rope(k, self.freqs)
        
        # KV cache BEFORE repeat_interleave
        if kv_cache is not None:
            k = np.concatenate([kv_cache[0], k], axis=2)
            v = np.concatenate([kv_cache[1], v], axis=2)
        new_cache = (k.copy(), v.copy())
        
        # Repeat for GQA
        if self.num_queries_per_kv > 1:
            k = np.repeat(k, self.num_queries_per_kv, axis=1)
            v = np.repeat(v, self.num_queries_per_kv, axis=1)
        
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale
        total_len = k.shape[2]
        mask = np.triu(np.full((L, total_len), -1e9), k=total_len - L + 1)
        attn = np.exp(attn + mask - (attn + mask).max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1) @ self.wo
        return out, new_cache
