"""Mamba (S6) structured state space model. Pure Python + NumPy."""
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from qythera.tensor import Tensor, no_grad
from qythera.nn import Module, Linear, Embedding, LayerNorm, SiLU, ModuleList


@dataclass
class MambaConfig:
    vocab_size: int = 32000
    embed_dim: int = 256
    num_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    max_seq_len: int = 2048
    bias: bool = False
    tie_embeddings: bool = True


def parallel_scan(a, b):
    """Parallel prefix scan: h_t = a_t * h_{t-1} + b_t. O(T log T)."""
    T = a.shape[0]
    if T == 0:
        return b * 0.0
    if T == 1:
        return b.copy()
    log_T = int(math.ceil(math.log2(T)))
    pad = (1 << log_T) - T
    a_ = np.concatenate([a, np.ones((pad,) + a.shape[1:], dtype=a.dtype)], axis=0)
    b_ = np.concatenate([b, np.zeros((pad,) + b.shape[1:], dtype=b.dtype)], axis=0)
    n = len(a_)
    out = b_.copy()
    stride = 1
    while stride < n:
        for i in range(0, n, stride * 2):
            r = min(i + stride, n)
            r2 = min(i + stride * 2, n)
            if r < r2:
                out[i:r2] = out[i:r] + a_[i:r] * out[r:r2]
                a_[i:r2] = a_[i:r] * a_[i + stride:r2]
        stride *= 2
    return out[:T]


def causal_conv1d(x, weight, bias=None):
    """Causal 1D convolution. x: (B, C, L), weight: (out, in, K)."""
    B, C, L = x.shape
    out_ch, _, K = weight.shape
    pad = K - 1
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, 0)))
    y = np.zeros((B, out_ch, L), dtype=np.float32)
    for k in range(K):
        y += np.einsum('bcl,oc->bol', x_padded[:, :, k:k + L], weight[:, :, k])
    if bias is not None:
        y += bias.reshape(1, -1, 1)
    return y


class SSMBlock(Module):
    def __init__(self, d_inner, d_state):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.A_log = Tensor(
            np.log(np.arange(1, d_state + 1, dtype=np.float32))[None, :].repeat(d_inner, axis=0),
            requires_grad=True
        )
        self.D = Tensor(np.ones(d_inner, dtype=np.float32), requires_grad=True)

        self.in_proj = Linear(d_inner, d_inner + d_state * 2 + 1, bias=False)
        self.dt_proj = Linear(1, d_inner, bias=True)
        self.out_proj = Linear(d_inner, d_inner, bias=False)

    def forward(self, x, h_prev=None):
        B, T, D = x.shape
        d_inner = self.d_inner
        d_state = self.d_state

        proj = self.in_proj(x)
        p = proj.data

        x_branch = p[:, :, :d_inner]
        B_proj = p[:, :, d_inner:d_inner + d_state]
        C_proj = p[:, :, d_inner + d_state:d_inner + 2 * d_state]
        dt_raw = p[:, :, d_inner + 2 * d_state:]

        dt = self.dt_proj(Tensor(dt_raw, requires_grad=proj.requires_grad))
        dt_np = np.abs(dt.data)

        A = -np.exp(self.A_log.data)

        h = np.zeros((B, d_inner, d_state), dtype=np.float32)
        if h_prev is not None:
            h = h_prev.copy()

        y_data = np.zeros((B, T, d_inner), dtype=np.float32)
        for t in range(T):
            dt_t = dt_np[:, t:t + 1, :].reshape(B, d_inner, 1)
            x_t = x_branch[:, t:t + 1, :].reshape(B, d_inner, 1)
            B_t = B_proj[:, t:t + 1, :].reshape(B, d_state)
            C_t = C_proj[:, t:t + 1, :].reshape(B, d_state)

            A_bar = np.exp(A * dt_t)
            B_bar = dt_t * B_t[:, None, :]

            h = A_bar * h + B_bar * x_t
            y_t = np.einsum('bds,bs->bd', h, C_t)
            y_data[:, t, :] = y_t

        y = Tensor(y_data, requires_grad=proj.requires_grad)
        y = y + Tensor(x_branch, requires_grad=proj.requires_grad) * Tensor(self.D.data.reshape(1, 1, -1), requires_grad=proj.requires_grad)
        y = y.silu()

        out = self.out_proj(y)
        return out


class MambaLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.norm = LayerNorm(config.embed_dim)
        d_inner = config.embed_dim * config.expand

        self.ssm = SSMBlock(d_inner, config.d_state)
        self.conv_weight = Tensor(
            np.random.randn(d_inner, d_inner, config.d_conv).astype(np.float32) * math.sqrt(2.0 / (d_inner * config.d_conv)),
            requires_grad=True
        )
        self.conv_bias = Tensor(np.zeros(d_inner, dtype=np.float32), requires_grad=True)
        self.in_proj = Linear(config.embed_dim, d_inner * 2, bias=config.bias)
        self.out_proj = Linear(d_inner, config.embed_dim, bias=config.bias)

    def forward(self, x, h_prev=None):
        residual = x
        x = self.norm(x)

        proj = self.in_proj(x)
        B, T, D = x.shape
        parts = proj.data
        d_inner = parts.shape[-1] // 2
        x_branch = Tensor(parts[:, :, :d_inner], requires_grad=proj.requires_grad)
        z_branch = Tensor(parts[:, :, d_inner:], requires_grad=proj.requires_grad)

        x_conv = x_branch.transpose(1, 2)
        x_conv_np = causal_conv1d(x_conv.data, self.conv_weight.data, self.conv_bias.data)
        x_conv = Tensor(x_conv_np, requires_grad=x_conv.requires_grad)
        x_conv = x_conv.transpose(1, 2)

        x_ssm = self.ssm(x_conv, h_prev)

        x_ssm = x_ssm.silu()
        z_sig = z_branch.sigmoid()
        out = x_ssm * z_sig

        return residual + self.out_proj(out)


class MambaModel(Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MambaConfig()
        c = self.config

        self.embed = Embedding(c.vocab_size, c.embed_dim)
        self.layers = ModuleList([MambaLayer(c) for _ in range(c.num_layers)])
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
