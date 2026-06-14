"""RWKV: Receptance Weighted Key Value. Pure Python + NumPy."""
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from qythera.tensor import Tensor, no_grad
from qythera.nn import Module, Linear, Embedding, LayerNorm, SiLU, ModuleList


@dataclass
class RWKVConfig:
    vocab_size: int = 32000
    embed_dim: int = 256
    num_layers: int = 6
    ffn_dim: int = 768
    max_seq_len: int = 2048
    bias: bool = False
    tie_embeddings: bool = True
    head_size: int = 64


def wkv_compute(k, v, w, u):
    """Compute WKV attention.
    k, v: (B, NH, T, HS)
    w: (NH, 1)
    u: (NH, 1)
    Returns: (B, NH, T, HS)
    """
    B, NH, T, HS = k.shape
    out = np.zeros_like(v)
    w = w.reshape(1, NH, 1, 1)
    u = u.reshape(1, NH, 1, 1)
    for t in range(T):
        k_t = k[:, :, t:t + 1, :]
        v_t = v[:, :, t:t + 1, :]
        num = np.exp(u + k_t) * v_t
        den = np.exp(u + k_t)
        for i in range(t):
            decay = w * (t - i)
            k_i = k[:, :, i:i + 1, :]
            v_i = v[:, :, i:i + 1, :]
            num = num + np.exp(decay + k_i) * v_i
            den = den + np.exp(decay + k_i)
        out[:, :, t:t + 1, :] = num / (den + 1e-8)
    return out


class TimeMixing(Module):
    def __init__(self, config):
        super().__init__()
        D = config.embed_dim
        head_size = config.head_size
        self.num_heads = D // head_size
        self.head_size = head_size
        self.D = D

        self.key = Linear(D, D, bias=config.bias)
        self.value = Linear(D, D, bias=config.bias)
        self.receptance = Linear(D, D, bias=config.bias)
        self.output = Linear(D, D, bias=config.bias)

        self.time_mix_r = Tensor(np.ones(D, dtype=np.float32) * 0.5, requires_grad=True)
        self.time_mix_k = Tensor(np.ones(D, dtype=np.float32) * 0.5, requires_grad=True)
        self.time_mix_v = Tensor(np.ones(D, dtype=np.float32) * 0.5, requires_grad=True)

        self.time_decay = Tensor(
            np.log(np.linspace(1.3, 0.1, self.num_heads).astype(np.float32)).reshape(self.num_heads, 1),
            requires_grad=True
        )
        self.time_first = Tensor(
            np.ones((self.num_heads, 1), dtype=np.float32) * -1.0,
            requires_grad=True
        )

    def forward(self, x):
        B, T, D = x.shape

        x_prev = Tensor(np.concatenate([x.data[:, :1, :], x.data[:, :-1, :]], axis=1), requires_grad=x.requires_grad)

        r = (x * self.time_mix_r + x_prev * (1 - self.time_mix_r)).sigmoid()
        k = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        v = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)

        r = self.receptance(r)
        k = self.key(k)
        v = self.value(v)

        k_ = k.reshape(B, T, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        v_ = v.reshape(B, T, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        r_ = r.reshape(B, T, self.num_heads, self.head_size).permute(0, 2, 1, 3)

        out = wkv_compute(k_.data, v_.data, self.time_decay.data, self.time_first.data)
        out = Tensor(out.transpose(0, 2, 1, 3).reshape(B, T, D), requires_grad=x.requires_grad)

        r_perm = r_.permute(0, 2, 1, 3).reshape(B, T, D)
        return r_perm * self.output(out)


class ChannelMixing(Module):
    def __init__(self, config):
        super().__init__()
        D = config.embed_dim
        self.key = Linear(D, config.ffn_dim, bias=config.bias)
        self.value = Linear(config.ffn_dim, D, bias=config.bias)
        self.receptance = Linear(D, D, bias=config.bias)

        self.time_mix_r = Tensor(np.ones(D, dtype=np.float32) * 0.5, requires_grad=True)
        self.time_mix_k = Tensor(np.ones(D, dtype=np.float32) * 0.5, requires_grad=True)

    def forward(self, x):
        B, T, D = x.shape
        x_prev = Tensor(np.concatenate([x.data[:, :1, :], x.data[:, :-1, :]], axis=1), requires_grad=x.requires_grad)

        r = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        k = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)

        r = self.receptance(r).sigmoid()
        k = self.key(k)
        k = k.relu() ** 2
        kv = self.value(k)
        return r * kv


class RWKVBlock(Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = LayerNorm(config.embed_dim)
        self.ln2 = LayerNorm(config.embed_dim)
        self.time_mix = TimeMixing(config)
        self.channel_mix = ChannelMixing(config)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.time_mix(h)
        h = self.ln2(x)
        x = x + self.channel_mix(h)
        return x


class RWKVModel(Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or RWKVConfig()
        c = self.config

        self.embed = Embedding(c.vocab_size, c.embed_dim)
        self.layers = ModuleList([RWKVBlock(c, i) for i in range(c.num_layers)])
        self.norm = LayerNorm(c.embed_dim)
        self.head = Linear(c.embed_dim, c.vocab_size, bias=False)

        if c.tie_embeddings:
            self.head.weight = self.embed.weight

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.head(h)
        return logits

    def generate(self, prompt_ids, max_tokens=128, temperature=0.8, top_k=50, top_p=0.9):
        if isinstance(prompt_ids, np.ndarray):
            prompt_ids = Tensor(prompt_ids)
        ids = list(prompt_ids.data.flatten().astype(int))
        generated = []
        with no_grad():
            inp = Tensor(np.array([ids], dtype=np.int32))
            logits = self.forward(inp)
        for _ in range(max_tokens):
            last_logits = logits.data[0, -1].copy()
            if temperature > 0:
                last_logits = last_logits / max(temperature, 0.01)
            if top_k > 0:
                threshold = np.sort(last_logits)[-min(top_k, len(last_logits))]
                last_logits[last_logits < threshold] = -1e9
            if top_p < 1.0:
                sorted_idx = np.argsort(last_logits)[::-1]
                sorted_logits = last_logits[sorted_idx].copy()
                cum_probs = np.cumsum(np.exp(sorted_logits) / (np.exp(sorted_logits).sum() + 1e-8))
                mask = cum_probs > top_p
                mask[1:] = mask[:-1]
                mask[0] = False
                sorted_logits[mask] = -1e9
                last_logits[sorted_idx] = sorted_logits
            probs = np.exp(last_logits - last_logits.max())
            probs = probs / (probs.sum() + 1e-8)
            next_id = int(np.random.choice(len(probs), p=probs))
            generated.append(next_id)
            ids.append(next_id)
            with no_grad():
                inp = Tensor(np.array([ids], dtype=np.int32))
                logits = self.forward(inp)
        return ids
