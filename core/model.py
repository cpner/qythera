"""Complete transformer model built on qythera's own tensor engine and nn modules."""
import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from core.tensor import Tensor, no_grad, zeros, ones, arange
from core.nn import (
    Module, Linear, Embedding, RMSNorm as NNRMSNorm, Dropout,
    ModuleList, Sequential, LayerNorm, GELU as NNGELU, SiLU as NNSiLU,
    ReLU as NNReLU,
)
from core.positional import RoPE, ALiBi, SinusoidalPE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 4
    ffn_dim: int = 768
    rope_base: float = 10000.0
    rope_dim: Optional[int] = None
    rms_eps: float = 1e-6
    activation: str = "swiglu"
    max_seq_len: int = 2048
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    num_experts: int = 0
    num_experts_per_tok: int = 2
    num_shared_experts: int = 0
    norm_type: str = "rms"
    positional_encoding: str = "rope"
    parallel_attn_ffn: bool = False
    sandwich_norm: bool = False
    logit_softcap: float = 0.0
    sliding_window: Optional[int] = None

    def __post_init__(self):
        if self.rope_dim is None:
            self.rope_dim = self.embed_dim
        if self.num_kv_heads == 0:
            self.num_kv_heads = self.num_heads


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class KVCache:
    def __init__(self, max_seq_len: int, num_layers: int, num_kv_heads: int, head_dim: int,
                 sliding_window: Optional[int] = None):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.k_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.v_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.cur_len = [0] * num_layers

    def update(self, layer: int, new_k: np.ndarray, new_v: np.ndarray, position: int):
        n = new_k.shape[2]
        start = position
        if self.sliding_window is not None:
            start = max(0, position - self.sliding_window + n)
            if start > 0:
                shift = start
                self.k_cache[layer][:, :, :self.sliding_window - shift] = \
                    self.k_cache[layer][:, :, shift:self.sliding_window].copy()
                self.v_cache[layer][:, :, :self.sliding_window - shift] = \
                    self.v_cache[layer][:, :, shift:self.sliding_window].copy()
                self.k_cache[layer][:, :, self.sliding_window - shift:self.sliding_window] = new_k
                self.v_cache[layer][:, :, self.sliding_window - shift:self.sliding_window] = new_v
                end = self.sliding_window
            else:
                self.k_cache[layer][:, :, position:position + n] = new_k
                self.v_cache[layer][:, :, position:position + n] = new_v
                end = position + n
        else:
            self.k_cache[layer][:, :, position:position + n] = new_k
            self.v_cache[layer][:, :, position:position + n] = new_v
            end = position + n
        self.cur_len[layer] = end
        if self.sliding_window is not None:
            return (self.k_cache[layer][:, :, :end].copy(),
                    self.v_cache[layer][:, :, :end].copy())
        return (self.k_cache[layer][:, :, :end].copy(),
                self.v_cache[layer][:, :, :end].copy())

    def get(self, layer: int):
        c = self.cur_len[layer]
        return (self.k_cache[layer][:, :, :c].copy(),
                self.v_cache[layer][:, :, :c].copy())

    def reset(self):
        self.cur_len = [0] * self.num_layers

    def get_seq_len(self, layer: int = 0):
        return self.cur_len[layer]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.embed_dim, bias=config.bias)
        self.attn_dropout = Dropout(config.dropout) if config.dropout > 0 else None

        if config.rope_dim is not None:
            rope_d = min(config.rope_dim, self.head_dim)
        else:
            rope_d = self.head_dim
        self.rope = RoPE(rope_d, max_seq_len=config.max_seq_len, base=config.rope_base)
        self.alibi = None
        if config.positional_encoding == "alibi":
            self.alibi = ALiBi(config.num_heads, max_seq_len=config.max_seq_len)

    def forward(self, x: Tensor, kv_cache: Optional[KVCache] = None,
                layer_idx: int = 0, position: int = 0) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.rope is not None:
            with no_grad():
                q_np = Tensor(q.data)
                k_np = Tensor(k.data)
                q_rot = self.rope.apply_rotary(q_np, offset=position)
                k_rot = self.rope.apply_rotary(k_np, offset=position)
            q = Tensor(q_rot.data, requires_grad=q.requires_grad)
            k = Tensor(k_rot.data, requires_grad=k.requires_grad)

        kv_out = None
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k.data, v.data, position)
            k = Tensor(k)
            v = Tensor(v)
            kv_out = (k, v)

        S_q = L
        S_kv = k.shape[2]

        if self.num_kv_groups > 1:
            k = Tensor(np.repeat(k.data, self.num_kv_groups, axis=1), requires_grad=k.requires_grad)
            v = Tensor(np.repeat(v.data, self.num_kv_groups, axis=1), requires_grad=v.requires_grad)

        attn = self._compute_attention(q, k, v, B, S_q, S_kv, position, layer_idx)

        attn = Tensor(attn.data.transpose(0, 2, 1, 3), requires_grad=attn.requires_grad)
        attn = attn.reshape(B, L, self.num_heads * self.head_dim)
        out = self.o_proj(attn)
        return out, kv_out

    def _compute_attention(self, q, k, v, B, S_q, S_kv, position, layer_idx):
        q_scaled = Tensor(q.data * self.scaling, requires_grad=q.requires_grad)
        k_t = Tensor(k.data.transpose(0, 1, 3, 2), requires_grad=k.requires_grad)
        attn = q_scaled.matmul(k_t)

        causal_mask = self._make_causal_mask(S_q, S_kv, position)
        attn = Tensor(np.where(causal_mask[None, None, :, :], attn.data, -1e9),
                       requires_grad=attn.requires_grad)

        if self.config.sliding_window is not None and self.config.positional_encoding == "rope":
            start = max(0, S_kv - self.config.sliding_window)
            sw_mask = np.ones((S_q, S_kv), dtype=np.bool_)
            for i in range(S_q):
                end = min(start + i + 1, S_kv)
                sw_mask[i, :start] = False
                sw_mask[i, end:] = False
            attn = Tensor(np.where(sw_mask[None, None, :, :], attn.data, -1e9),
                           requires_grad=attn.requires_grad)

        if self.alibi is not None:
            total_len = position + S_q
            full_bias = self.alibi.get_bias(total_len)
            alibi_bias = Tensor(full_bias[:, position:position + S_q, :S_kv].astype(np.float32))
            while alibi_bias.ndim < attn.ndim:
                alibi_bias = alibi_bias.unsqueeze(0)
            attn = attn + alibi_bias

        attn = attn.softmax(axis=-1)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)

        out = attn.matmul(v)
        return out

    def _make_causal_mask(self, S_q, S_kv, offset=0):
        mask = np.zeros((S_q, S_kv), dtype=np.bool_)
        for i in range(S_q):
            for j in range(S_kv):
                if j <= i + offset:
                    mask[i, j] = True
        return mask


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class FeedForward(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        ffn = config.ffn_dim
        act = config.activation

        if act == "swiglu":
            self.w1 = Linear(dim, ffn, bias=config.bias)
            self.w2 = Linear(ffn, dim, bias=config.bias)
            self.w3 = Linear(dim, ffn, bias=config.bias)
        elif act == "geglu":
            self.w1 = Linear(dim, ffn, bias=config.bias)
            self.w2 = Linear(ffn, dim, bias=config.bias)
            self.w3 = Linear(dim, ffn, bias=config.bias)
        elif act == "reglu":
            self.w1 = Linear(dim, ffn, bias=config.bias)
            self.w2 = Linear(ffn, dim, bias=config.bias)
            self.w3 = Linear(dim, ffn, bias=config.bias)
        else:
            self.w1 = Linear(dim, ffn, bias=config.bias)
            self.w2 = Linear(ffn, dim, bias=config.bias)

        self.dropout = Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        act = self.config.activation
        if act == "swiglu":
            gate = self.w1(x).silu()
            up = self.w3(x)
            out = self.w2(Tensor(gate.data * up.data, requires_grad=gate.requires_grad or up.requires_grad))
        elif act == "geglu":
            a = self.w1(x)
            b = self.w3(x)
            c = 0.7978845608028654
            k = 0.044715
            t = Tensor(np.tanh(c * (b.data + k * b.data ** 3)), requires_grad=b.requires_grad)
            gated = Tensor(a.data * 0.5 * (1 + t.data), requires_grad=a.requires_grad or t.requires_grad)
            out = self.w2(gated)
        elif act == "reglu":
            a = self.w1(x)
            b = self.w3(x)
            out = self.w2(Tensor(a.data * np.maximum(0, b.data), requires_grad=a.requires_grad or b.requires_grad))
        elif act == "gelu":
            h = self.w1(x).gelu()
            out = self.w2(h)
        elif act == "relu":
            h = self.w1(x).relu()
            out = self.w2(h)
        else:
            h = self.w1(x).gelu()
            out = self.w2(h)

        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# MoE (Mixture of Experts)
# ---------------------------------------------------------------------------

class MoELayer(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_shared_experts = config.num_shared_experts
        self.expert_dim = config.ffn_dim

        self.gate = Linear(config.embed_dim, self.num_experts, bias=False)
        self.experts = ModuleList([
            FeedForward(config) for _ in range(self.num_experts)
        ])
        if self.num_shared_experts > 0:
            self.shared_experts = ModuleList([
                FeedForward(config) for _ in range(self.num_shared_experts)
            ])
        else:
            self.shared_experts = None
        self.load_balance_loss = Tensor(np.float32(0.0))

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        router_logits = self.gate(x_flat)
        router_probs = router_logits.softmax(axis=-1)

        k = min(self.num_experts_per_tok, self.num_experts)
        topk_vals, topk_idx = router_probs.topk(k, dim=-1)

        out = np.zeros((B * L, D), dtype=np.float32)
        expert_mask = np.zeros((self.num_experts, B * L), dtype=np.bool_)
        for e in range(self.num_experts):
            expert_mask[e] = (topk_idx.data == e).any(axis=-1)

        topk_vals_data = topk_vals.data
        topk_idx_data = topk_idx.data

        for e in range(self.num_experts):
            if not expert_mask[e].any():
                continue
            idx = np.where(expert_mask[e])[0]
            x_e = Tensor(x_flat.data[idx])
            expert_out = self.experts[e](x_e)
            for j in range(k):
                mask = (topk_idx_data[:, j] == e) & expert_mask[e]
                if mask.any():
                    w = topk_vals_data[mask, j:j+1]
                    out[mask] += (expert_out.data[:mask.sum()] * w).astype(np.float32) if expert_out.data[:mask.sum()].shape == w.shape else expert_out.data[:mask.sum()] * w

        if self.shared_experts is not None:
            shared_out = x_flat
            for se in self.shared_experts:
                shared_out = se(shared_out)
            out = out + shared_out.data

        if self.training and self.num_experts > 1:
            self._compute_load_balance_loss(router_probs, B * L)

        return Tensor(out.reshape(B, L, D), requires_grad=x.requires_grad)

    def _compute_load_balance_loss(self, router_probs, total_tokens):
        probs_mean = router_probs.mean(axis=0)
        expert_counts = np.zeros(self.num_experts, dtype=np.float32)
        k = self.num_experts_per_tok
        _, topk_idx = router_probs.topk(k, dim=-1)
        for e in range(self.num_experts):
            expert_counts[e] = (topk_idx.data == e).sum()
        freq = expert_counts / (total_tokens * k)
        aux_loss = self.num_experts * (probs_mean.data * freq).sum()
        self.load_balance_loss = Tensor(np.float32(aux_loss))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if config.norm_type == "rms":
            self.attn_norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
            self.ffn_norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
        else:
            self.attn_norm = LayerNorm(config.embed_dim)
            self.ffn_norm = LayerNorm(config.embed_dim)

        self.attn = Attention(config)

        if config.num_experts > 0:
            self.ffn = MoELayer(config)
        else:
            self.ffn = FeedForward(config)

        if config.sandwich_norm:
            if config.norm_type == "rms":
                self.post_attn_norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
                self.post_ffn_norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
            else:
                self.post_attn_norm = LayerNorm(config.embed_dim)
                self.post_ffn_norm = LayerNorm(config.embed_dim)

        self.drop = Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, x: Tensor, kv_cache: Optional[KVCache] = None,
                position: int = 0) -> Tuple[Tensor, Optional[Tuple]]:
        if self.config.parallel_attn_ffn:
            normed = self.attn_norm(x)
            attn_out, kv_out = self.attn(normed, kv_cache, self.layer_idx, position)
            ffn_out = self.ffn(self.ffn_norm(x))
            if self.config.sandwich_norm:
                attn_out = self.post_attn_norm(attn_out)
                ffn_out = self.post_ffn_norm(ffn_out)
            if self.drop is not None:
                attn_out = self.drop(attn_out)
                ffn_out = self.drop(ffn_out)
            return x + attn_out + ffn_out, kv_out
        else:
            normed = self.attn_norm(x)
            attn_out, kv_out = self.attn(normed, kv_cache, self.layer_idx, position)
            if self.config.sandwich_norm:
                attn_out = self.post_attn_norm(attn_out)
            if self.drop is not None:
                attn_out = self.drop(attn_out)
            h = x + attn_out

            normed2 = self.ffn_norm(h)
            ffn_out = self.ffn(normed2)
            if self.config.sandwich_norm:
                ffn_out = self.post_ffn_norm(ffn_out)
            if self.drop is not None:
                ffn_out = self.drop(ffn_out)
            return h + ffn_out, kv_out


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class Transformer(Module):
    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        c = self.config

        self.embed = Embedding(c.vocab_size, c.embed_dim)

        if c.positional_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPE(c.max_seq_len, c.embed_dim)
        else:
            self.pos_encoding = None

        self.layers = ModuleList([
            TransformerBlock(c, i) for i in range(c.num_layers)
        ])

        if c.norm_type == "rms":
            self.norm = NNRMSNorm(c.embed_dim, eps=c.rms_eps)
        else:
            self.norm = LayerNorm(c.embed_dim)

        self.head = Linear(c.embed_dim, c.vocab_size, bias=False)
        self.embed_drop = Dropout(c.dropout) if c.dropout > 0 else None

        self._kv_cache = None

    def forward(self, x: Tensor, kv_cache: Optional[KVCache] = None,
                position: int = 0) -> Tensor:
        if isinstance(x, np.ndarray):
            x = Tensor(x)

        if x.dtype and x.dtype.name == "INT8":
            x = x.float()

        B, L = x.shape
        h = self.embed(x)

        if self.pos_encoding is not None:
            with no_grad():
                pe = Tensor(self.pos_encoding.forward(L + position), requires_grad=False)
                if position > 0:
                    pe = Tensor(pe.data[position:position + L], requires_grad=False)
                else:
                    pe = Tensor(pe.data[:L], requires_grad=False)
            h = h + pe

        if self.embed_drop is not None:
            h = self.embed_drop(h)

        use_cache = kv_cache is not None
        for i, layer in enumerate(self.layers):
            h, _ = layer(h, kv_cache=kv_cache, position=position)

        h = self.norm(h)
        logits = self.head(h)

        if self.config.logit_softcap > 0:
            logits = Tensor(np.tanh(logits.data / self.config.logit_softcap) * self.config.logit_softcap,
                            requires_grad=logits.requires_grad)

        return logits

    def generate(self, prompt_ids: Tensor, max_tokens: int = 128,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> List[int]:
        if isinstance(prompt_ids, np.ndarray):
            prompt_ids = Tensor(prompt_ids)
        ids = list(prompt_ids.data.flatten().astype(int))
        kv_cache = KVCache(
            self.config.max_seq_len, self.config.num_layers,
            self.config.num_kv_heads, self.config.embed_dim // self.config.num_heads,
            sliding_window=self.config.sliding_window
        )

        with no_grad():
            inp = Tensor(np.array([ids], dtype=np.int32))
            logits = self.forward(inp, kv_cache=kv_cache, position=0)

        generated = []
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

            pos = kv_cache.get_seq_len() if kv_cache else len(ids) - 1
            with no_grad():
                inp = Tensor(np.array([[next_id]], dtype=np.int32))
                logits = self.forward(inp, kv_cache=kv_cache, position=pos)

        kv_cache.reset()
        return ids

    def init_kv_cache(self):
        self._kv_cache = KVCache(
            self.config.max_seq_len, self.config.num_layers,
            self.config.num_kv_heads, self.config.embed_dim // self.config.num_heads,
            sliding_window=self.config.sliding_window
        )
        return self._kv_cache

    def clear_kv_cache(self):
        if self._kv_cache is not None:
            self._kv_cache.reset()

    def count_parameters(self) -> int:
        total = 0
        for p in self.parameters():
            total += p.size
        return total

    def estimate_memory(self) -> dict:
        param_bytes = self.count_parameters() * 4
        grad_bytes = param_bytes
        buf_bytes = self.config.num_layers * 2 * self.config.max_seq_len * \
                    self.config.num_kv_heads * (self.config.embed_dim // self.config.num_heads) * 4
        total = param_bytes + grad_bytes + buf_bytes
        return {
            "parameters": self.count_parameters(),
            "param_mb": param_bytes / (1024 * 1024),
            "grad_mb": grad_bytes / (1024 * 1024),
            "kv_cache_mb": buf_bytes / (1024 * 1024),
            "total_mb": total / (1024 * 1024),
        }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        state = self.state_dict()
        flat = {}
        for k, v in state.items():
            if isinstance(v, Tensor):
                flat[k] = v.data.copy()
            else:
                flat[k] = v
        np.savez(os.path.join(path, "model.npz"), **flat)
        import json
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({
                "vocab_size": self.config.vocab_size,
                "embed_dim": self.config.embed_dim,
                "num_layers": self.config.num_layers,
                "num_heads": self.config.num_heads,
                "num_kv_heads": self.config.num_kv_heads,
                "ffn_dim": self.config.ffn_dim,
                "rope_base": self.config.rope_base,
                "rope_dim": self.config.rope_dim,
                "rms_eps": self.config.rms_eps,
                "activation": self.config.activation,
                "max_seq_len": self.config.max_seq_len,
                "dropout": self.config.dropout,
                "bias": self.config.bias,
                "tie_embeddings": self.config.tie_embeddings,
                "num_experts": self.config.num_experts,
                "num_experts_per_tok": self.config.num_experts_per_tok,
                "num_shared_experts": self.config.num_shared_experts,
                "norm_type": self.config.norm_type,
                "positional_encoding": self.config.positional_encoding,
                "parallel_attn_ffn": self.config.parallel_attn_ffn,
                "sandwich_norm": self.config.sandwich_norm,
                "logit_softcap": self.config.logit_softcap,
                "sliding_window": self.config.sliding_window,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Transformer":
        import json
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        config = TransformerConfig(**cfg)
        model = cls(config)
        data = np.load(os.path.join(path, "model.npz"))
        state = {}
        for k in data.files:
            state[k] = Tensor(data[k])
        model.load_state_dict(state)
        return model
