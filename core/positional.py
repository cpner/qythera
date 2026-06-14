"""Positional encodings: RoPE, YaRN, ALiBi, sinusoidal, learned. Pure Python + NumPy."""
import math
import numpy as np
from core.tensor import Tensor

def _rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = np.zeros_like(x)
    out[..., ::2] = -x2
    out[..., 1::2] = x1
    return out

def _apply_rotary(x, cos, sin):
    """cos, sin: [seq_len, dim//2]. x: [B, seq_len, dim]"""
    cos = np.concatenate([cos, cos], axis=-1)
    sin = np.concatenate([sin, sin], axis=-1)
    while cos.ndim < x.ndim:
        cos = cos[np.newaxis]
        sin = sin[np.newaxis]
    return x * cos + _rotate_half(x) * sin

class RoPE:
    def __init__(self, dim, max_seq_len=8192, base=10000.0):
        self.dim = dim
        self.inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self._cos = None
        self._sin = None
        self._cached = 0

    def _ensure(self, seq_len):
        if seq_len > self._cached:
            self._cached = seq_len
            t = np.arange(seq_len, dtype=np.float32)
            freqs = np.outer(t, self.inv_freq)
            self._cos = np.cos(freqs).astype(np.float32)
            self._sin = np.sin(freqs).astype(np.float32)

    def apply_rotary(self, x, offset=0):
        self._ensure(x.shape[-2] + offset)
        cos = self._cos[offset:offset + x.shape[-2]]
        sin = self._sin[offset:offset + x.shape[-2]]
        return _apply_rotary(x, cos, sin)


def apply_rotary_emb(x, cos, sin):
    return _apply_rotary(x, cos, sin)


class YaRN:
    def __init__(self, dim, max_seq_len=8192, base=10000.0, scale=1.0, original_max_seq_len=8192):
        self.dim = dim
        self.scale = scale
        self.original_max_seq_len = original_max_seq_len
        self.inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self._cos = None
        self._sin = None
        self._cached = 0

    def _ensure(self, seq_len):
        if seq_len > self._cached:
            self._cached = seq_len
            t = np.arange(seq_len, dtype=np.float32)
            low_wl = self.original_max_seq_len / math.pi
            high_wl = 4.0
            wl = 2 * np.pi / (self.inv_freq + 1e-10)
            lf = low_wl / wl
            hf = wl / high_wl
            smooth = np.clip(1.0 / (1.0 + np.exp(-np.clip(lf - hf, -10, 10))), 0, 1)
            new_freq = (1 - smooth) * self.inv_freq * self.scale + smooth * self.inv_freq
            freqs = np.outer(t, new_freq)
            self._cos = np.cos(freqs).astype(np.float32)
            self._sin = np.sin(freqs).astype(np.float32)

    def apply_rotary(self, x, offset=0):
        self._ensure(x.shape[-2] + offset)
        cos = self._cos[offset:offset + x.shape[-2]]
        sin = self._sin[offset:offset + x.shape[-2]]
        temp = math.sqrt(1 + math.log(self.scale))
        return _apply_rotary(x, cos, sin) / temp


class NTKAwareRoPE:
    def __init__(self, dim, max_seq_len=8192, base=10000.0, scale=1.0):
        new_base = base * (scale ** (dim / max(dim - 2, 1)))
        self.inv_freq = 1.0 / (new_base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self._cos = None
        self._sin = None
        self._cached = 0

    def _ensure(self, seq_len):
        if seq_len > self._cached:
            self._cached = seq_len
            t = np.arange(seq_len, dtype=np.float32)
            freqs = np.outer(t, self.inv_freq)
            self._cos = np.cos(freqs).astype(np.float32)
            self._sin = np.sin(freqs).astype(np.float32)

    def apply_rotary(self, x, offset=0):
        self._ensure(x.shape[-2] + offset)
        cos = self._cos[offset:offset + x.shape[-2]]
        sin = self._sin[offset:offset + x.shape[-2]]
        return _apply_rotary(x, cos, sin)


class PositionalInterpolation:
    def __init__(self, dim, max_seq_len=8192, base=10000.0, scale=1.0):
        self.scale = scale
        self.inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self._cos = None
        self._sin = None
        self._cached = 0

    def _ensure(self, seq_len):
        if seq_len > self._cached:
            self._cached = seq_len
            t = np.arange(seq_len, dtype=np.float32) / self.scale
            freqs = np.outer(t, self.inv_freq)
            self._cos = np.cos(freqs).astype(np.float32)
            self._sin = np.sin(freqs).astype(np.float32)

    def apply_rotary(self, x, offset=0):
        self._ensure(x.shape[-2] + offset)
        cos = self._cos[offset:offset + x.shape[-2]]
        sin = self._sin[offset:offset + x.shape[-2]]
        return _apply_rotary(x, cos, sin)


class LongRoPE:
    def __init__(self, dim, max_seq_len=8192, base=10000.0):
        self.inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.scales = np.ones(dim // 2, dtype=np.float32)
        self._cos = None
        self._sin = None
        self._cached = 0

    def optimize_scales(self, eval_fn):
        best = self.scales.copy()
        best_score = eval_fn(self)
        for _ in range(10):
            tmp = best.copy()
            tmp[np.random.randint(len(tmp))] *= np.random.uniform(0.8, 1.2)
            self.scales = tmp
            s = eval_fn(self)
            if s > best_score:
                best_score, best = s, tmp.copy()
        self.scales = best

    def _ensure(self, seq_len):
        if seq_len > self._cached:
            self._cached = seq_len
            t = np.arange(seq_len, dtype=np.float32)
            freqs = np.outer(t, self.inv_freq * self.scales)
            self._cos = np.cos(freqs).astype(np.float32)
            self._sin = np.sin(freqs).astype(np.float32)

    def apply_rotary(self, x, offset=0):
        self._ensure(x.shape[-2] + offset)
        cos = self._cos[offset:offset + x.shape[-2]]
        sin = self._sin[offset:offset + x.shape[-2]]
        return _apply_rotary(x, cos, sin)


class ALiBi:
    def __init__(self, num_heads, max_seq_len=8192):
        slopes = np.array([2 ** (-8 * i / num_heads) for i in range(1, num_heads + 1)], dtype=np.float32)
        t = np.arange(max_seq_len, dtype=np.float32)
        rel = np.abs(t.reshape(1, -1) - t.reshape(-1, 1)).astype(np.float32)
        self.bias = slopes.reshape(-1, 1, 1) * rel.reshape(1, max_seq_len, max_seq_len)

    def get_bias(self, seq_len):
        return -self.bias[:, :seq_len, :seq_len]


class LearnedPE:
    def __init__(self, max_seq_len, embed_dim):
        self.weight = np.random.randn(max_seq_len, embed_dim).astype(np.float32) * 0.02

    def forward(self, seq_len):
        return self.weight[:seq_len]


class SinusoidalPE:
    def __init__(self, max_seq_len, embed_dim):
        pe = np.zeros((max_seq_len, embed_dim), dtype=np.float32)
        pos = np.arange(max_seq_len, dtype=np.float32).reshape(-1, 1)
        div = np.exp(np.arange(0, embed_dim, 2, dtype=np.float32) * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self.pe = pe

    def forward(self, seq_len):
        return self.pe[:seq_len]


class T5RelativeBias:
    def __init__(self, num_heads, max_distance=128, num_buckets=32):
        self.max_distance = max_distance
        self.num_buckets = num_buckets
        self.bias = np.random.randn(num_heads, num_buckets).astype(np.float32) * 0.02

    def _bucket(self, rel):
        rel = np.minimum(np.abs(rel), self.max_distance)
        small = rel < (self.num_buckets - 1) / 2
        large = self.num_buckets - 1 - ((self.num_buckets - 1) * (rel - (self.num_buckets - 1) / 2) / (self.max_distance - (self.num_buckets - 1) / 2 + 1e-8)).astype(int)
        large = np.clip(large, 0, self.num_buckets - 1)
        return np.where(small, rel.astype(int), large)

    def forward(self, q_len, k_len):
        ctx = np.arange(q_len)[:, None]
        mem = np.arange(k_len)[None, :]
        buckets = self._bucket(mem - ctx)
        return self.bias[:, buckets].sum(axis=0)


class FIRE:
    def __init__(self, dim, max_seq_len=8192):
        self.freqs = np.random.randn(dim // 2).astype(np.float32) * 0.02
        self.phases = np.random.randn(dim // 2).astype(np.float32) * 0.02

    def forward(self, seq_len):
        pos = np.arange(seq_len, dtype=np.float32)
        rel = pos[:, None] - pos[None, :]
        angles = rel[..., None] * self.freqs[None, None, :] + self.phases[None, None, :]
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
