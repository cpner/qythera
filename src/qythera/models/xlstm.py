"""xLSTM: Extended LSTM with sLSTM and mLSTM. Pure Python + NumPy."""
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from qythera.tensor import Tensor, no_grad
from qythera.nn import Module, Linear, Embedding, LayerNorm, SiLU, ModuleList


@dataclass
class XLSTMConfig:
    vocab_size: int = 32000
    embed_dim: int = 256
    num_layers: int = 6
    ffn_dim: int = 768
    num_heads: int = 4
    max_seq_len: int = 2048
    bias: bool = True
    tie_embeddings: bool = True
    use_slstm: bool = True


def parallel_prefix_scan(f, g, x):
    """Parallel prefix scan for matrix LSTM gates.
    f, g, x have shape (T, D). Returns h of shape (T, D)."""
    T = f.shape[0]
    if T == 0:
        return x * 0.0
    if T == 1:
        return f * 0 + x
    log_T = int(math.ceil(math.log2(T)))
    pad = (1 << log_T) - T
    f_ = np.concatenate([f, np.ones((pad,) + f.shape[1:], dtype=f.dtype)], axis=0)
    g_ = np.concatenate([g, np.zeros((pad,) + g.shape[1:], dtype=g.dtype)], axis=0)
    x_ = np.concatenate([x, np.zeros((pad,) + x.shape[1:], dtype=x.dtype)], axis=0)
    n = len(f_)
    out = x_.copy()
    stride = 1
    while stride < n:
        for i in range(0, n, stride * 2):
            r = min(i + stride, n)
            r2 = min(i + stride * 2, n)
            if r < r2:
                out[i:r2] = f_[i:r] * out[i:r] + g_[i:r] * x_[i:r]
                f_[i:r2] = f_[i:r] * f_[r:r2]
                g_[i:r2] = f_[i:r] * g_[r:r2] + g_[i:r]
        stride *= 2
    return out[:T]


class sLSTM(Module):
    """Scalar LSTM with exponential gating."""
    def __init__(self, d_model, d_state=64, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.W_f = Linear(d_model, d_state, bias=bias)
        self.W_i = Linear(d_model, d_state, bias=bias)
        self.W_g = Linear(d_model, d_state, bias=bias)
        self.W_o = Linear(d_model, d_state, bias=bias)

    def forward(self, x):
        B, T, D = x.shape
        d = self.d_state

        f_gate = self.W_f(x).sigmoid()
        i_gate = self.W_i(x).sigmoid()
        g_gate = self.W_g(x).tanh()
        o_gate = self.W_o(x).sigmoid()

        h = np.zeros((B, T, d), dtype=np.float32)
        c = np.zeros((B, T, d), dtype=np.float32)

        f_np = f_gate.data
        i_np = i_gate.data
        g_np = g_gate.data
        o_np = o_gate.data

        for b in range(B):
            h_b = np.zeros((T, d), dtype=np.float32)
            c_b = np.zeros((T, d), dtype=np.float32)
            for t in range(T):
                if t == 0:
                    c_b[t] = i_np[b, t] * g_np[b, t]
                    h_b[t] = o_np[b, t] * np.tanh(c_b[t])
                else:
                    c_b[t] = f_np[b, t] * c_b[t - 1] + i_np[b, t] * g_np[b, t]
                    h_b[t] = o_np[b, t] * np.tanh(c_b[t])
            h[b] = h_b
            c[b] = c_b

        return Tensor(h, requires_grad=x.requires_grad), Tensor(c, requires_grad=x.requires_grad)


class mLSTM(Module):
    """Matrix LSTM with parallel prefix scan."""
    def __init__(self, d_model, d_state=64, num_heads=4, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_heads = num_heads
        self.head_dim = d_state // num_heads

        self.W_q = Linear(d_model, d_state, bias=bias)
        self.W_k = Linear(d_model, d_state, bias=bias)
        self.W_v = Linear(d_model, d_state, bias=bias)
        self.W_f = Linear(d_model, num_heads, bias=bias)
        self.W_i = Linear(d_model, num_heads, bias=bias)
        self.W_o = Linear(d_model, d_state, bias=bias)

    def forward(self, x):
        B, T, D = x.shape
        NH = self.num_heads
        HS = self.head_dim

        q = self.W_q(x).reshape(B, T, NH, HS)
        k = self.W_k(x).reshape(B, T, NH, HS)
        v = self.W_v(x).reshape(B, T, NH, HS)
        f_gate = self.W_f(x).sigmoid()
        i_gate = self.W_i(x).sigmoid()
        o_gate = self.W_o(x)

        C = np.zeros((B, NH, HS, HS), dtype=np.float32)
        h = np.zeros((B, T, D), dtype=np.float32)

        f_np = f_gate.data
        i_np = i_gate.data

        for b in range(B):
            for t in range(T):
                f_t = f_np[b, t]
                i_t = i_np[b, t]
                q_t = q[b, t]
                k_t = k[b, t]
                v_t = v[b, t]

                C = C * f_t[:, None, None] + i_t[:, None, None] * np.einsum('d,e->de', v_t.reshape(NH, HS), k_t.reshape(NH, HS))

                q_hat = q_t.reshape(NH, 1, HS)
                out_t = np.einsum('bde,bd->be', C, q_hat.squeeze(1))
                h[b, t] = o_gate.data[b, t] * out_t.reshape(D)

        return Tensor(h, requires_grad=x.requires_grad)


class xLSTMBlock(Module):
    def __init__(self, config, use_slstm=True):
        super().__init__()
        self.use_slstm = use_slstm
        d_state = config.embed_dim

        if use_slstm:
            self.lstm = sLSTM(config.embed_dim, d_state, config.bias)
            self.proj = Linear(d_state, config.embed_dim, bias=config.bias)
        else:
            self.lstm = mLSTM(config.embed_dim, config.embed_dim, config.num_heads, config.bias)
            self.proj = Linear(config.embed_dim, config.embed_dim, bias=config.bias)

        self.ffn_norm = LayerNorm(config.embed_dim)
        self.ffn = Linear(config.embed_dim, config.ffn_dim, bias=config.bias)
        self.ffn_out = Linear(config.ffn_dim, config.embed_dim, bias=config.bias)

    def forward(self, x):
        residual = x
        h, _ = self.lstm(x)
        x = residual + self.proj(h)

        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h).silu()
        h = self.ffn_out(h)
        x = residual + h
        return x


class xLSTMModel(Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or XLSTMConfig()
        c = self.config

        self.embed = Embedding(c.vocab_size, c.embed_dim)
        self.layers = ModuleList([
            xLSTMBlock(c, use_slstm=(i % 2 == 0) if not c.use_slstm else True)
            for i in range(c.num_layers)
        ])
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
