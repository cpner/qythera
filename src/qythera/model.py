"""Complete transformer model built on qythera's own tensor engine and nn modules."""
import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from qythera.tensor import Tensor, no_grad, zeros, ones, arange
from qythera.nn import (
    Module, Linear, Embedding, RMSNorm as NNRMSNorm, Dropout,
    ModuleList, Sequential, LayerNorm, GELU as NNGELU, SiLU as NNSiLU,
    ReLU as NNReLU,
)
from qythera.positional import RoPE, ALiBi, SinusoidalPE


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
    use_flash: bool = True
    flash_block_size: int = 64
    use_mla: bool = False
    mla_latent_dim: int = 64
    moe_routing: str = "topk"
    switch_capacity_factor: float = 1.25

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


class PagedKVCache:
    def __init__(self, max_seq_len: int, num_layers: int, num_kv_heads: int,
                 head_dim: int, page_size: int = 16):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_pages = max_seq_len // page_size
        self.k_pages = [np.zeros((self.num_pages, page_size, num_kv_heads, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.v_pages = [np.zeros((self.num_pages, page_size, num_kv_heads, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.free_list = [list(range(self.num_pages)) for _ in range(num_layers)]
        self.page_tables = [{} for _ in range(num_layers)]
        self.cur_len = [0] * num_layers
        self.attention_scores = [np.zeros(self.num_pages * page_size, dtype=np.float32)
                                 for _ in range(num_layers)]

    def _allocate_page(self, layer: int) -> int:
        if not self.free_list[layer]:
            return -1
        return self.free_list[layer].pop(0)

    def _free_page(self, layer: int, page_idx: int):
        self.free_list[layer].append(page_idx)
        self.page_tables[layer].pop(page_idx, None)

    def update(self, layer: int, new_k: np.ndarray, new_v: np.ndarray, position: int):
        n = new_k.shape[2]
        for i in range(n):
            pos = position + i
            page_idx = pos // self.page_size
            offset = pos % self.page_size
            if page_idx not in self.page_tables[layer]:
                alloc = self._allocate_page(layer)
                if alloc == -1:
                    continue
                self.page_tables[layer][page_idx] = alloc
            physical = self.page_tables[layer][page_idx]
            self.k_pages[layer][physical, offset] = new_k[0, :, i, :]
            self.v_pages[layer][physical, offset] = new_v[0, :, i, :]
        self.cur_len[layer] = position + n
        return self._gather(layer)

    def _gather(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        length = self.cur_len[layer]
        if length == 0:
            return (np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32),
                    np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32))
        k_out = np.zeros((1, self.num_kv_heads, length, self.head_dim), dtype=np.float32)
        v_out = np.zeros((1, self.num_kv_heads, length, self.head_dim), dtype=np.float32)
        for pos in range(length):
            p = pos // self.page_size
            o = pos % self.page_size
            if p in self.page_tables[layer]:
                phys = self.page_tables[layer][p]
                k_out[0, :, pos, :] = self.k_pages[layer][phys, o]
                v_out[0, :, pos, :] = self.v_pages[layer][phys, o]
        return k_out, v_out

    def get(self, layer: int):
        return self._gather(layer)

    def reset(self):
        self.cur_len = [0] * self.num_layers
        for l in range(self.num_layers):
            self.free_list[l] = list(range(self.num_pages))
            self.page_tables[l].clear()

    def get_seq_len(self, layer: int = 0):
        return self.cur_len[layer]

    def evict_page(self, layer: int) -> bool:
        if not self.page_tables[layer]:
            return False
        target_page = max(self.page_tables[layer].keys())
        physical = self.page_tables[layer][target_page]
        self._free_page(layer, physical)
        return True


class H2OKVCache:
    def __init__(self, max_seq_len: int, num_layers: int, num_kv_heads: int,
                 head_dim: int, sink_size: int = 4, reserve_size: int = 4):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sink_size = sink_size
        self.reserve_size = reserve_size
        self.k_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.v_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.cur_len = [0] * num_layers
        self.attention_accum = [np.zeros(max_seq_len, dtype=np.float32) for _ in range(num_layers)]
        self.window_len = max_seq_len - sink_size - reserve_size

    def update(self, layer: int, new_k: np.ndarray, new_v: np.ndarray,
               position: int, attn_scores: Optional[np.ndarray] = None):
        n = new_k.shape[2]
        end = position + n

        if end <= self.max_seq_len:
            self.k_cache[layer][:, :, position:end] = new_k
            self.v_cache[layer][:, :, position:end] = new_v
            if attn_scores is not None:
                self.attention_accum[layer][position:end] += attn_scores.flatten()[:n]
        else:
            self._evict_h2o(layer)

        if end <= self.max_seq_len:
            self.k_cache[layer][:, :, position:end] = new_k
            self.v_cache[layer][:, :, position:end] = new_v
            if attn_scores is not None:
                self.attention_accum[layer][position:end] += attn_scores.flatten()[:n]

        self.cur_len[layer] = min(end, self.max_seq_len)
        return self._gather(layer)

    def _evict_h2o(self, layer: int):
        k = self.sink_size
        total = self.cur_len[layer]
        window = self.attention_accum[layer][k:total]

        if len(window) == 0:
            return

        keep_count = min(self.reserve_size, len(window))
        top_indices = np.argpartition(window, -keep_count)[-keep_count:]
        top_indices = np.sort(top_indices) + k

        sink_k = self.k_cache[layer][:, :, :k].copy()
        sink_v = self.v_cache[layer][:, :, :k].copy()
        top_k = self.k_cache[layer][:, :, top_indices].copy()
        top_v = self.v_cache[layer][:, :, top_indices].copy()

        self.k_cache[layer] *= 0
        self.v_cache[layer] *= 0

        new_len = k + keep_count
        self.k_cache[layer][:, :, :k] = sink_k
        self.v_cache[layer][:, :, :k] = sink_v
        self.k_cache[layer][:, :, k:k + keep_count] = top_k
        self.v_cache[layer][:, :, k:k + keep_count] = top_v

        self.attention_accum[layer][:k] = 0
        self.attention_accum[layer][k:k + keep_count] = self.attention_accum[layer][top_indices]
        self.attention_accum[layer][k + keep_count:] = 0
        self.cur_len[layer] = new_len

    def _gather(self, layer: int):
        c = self.cur_len[layer]
        return (self.k_cache[layer][:, :, :c].copy(),
                self.v_cache[layer][:, :, :c].copy())

    def get(self, layer: int):
        return self._gather(layer)

    def reset(self):
        self.cur_len = [0] * self.num_layers
        for l in range(self.num_layers):
            self.attention_accum[l][:] = 0

    def get_seq_len(self, layer: int = 0):
        return self.cur_len[layer]


class AttentionSinkKVCache:
    def __init__(self, max_seq_len: int, num_layers: int, num_kv_heads: int,
                 head_dim: int, sink_size: int = 4, sliding_window: Optional[int] = None):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sink_size = sink_size
        self.sliding_window = sliding_window
        self.k_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.v_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.cur_len = [0] * num_layers
        self.k_sink = [np.zeros((1, num_kv_heads, sink_size, head_dim), dtype=np.float32)
                       for _ in range(num_layers)]
        self.v_sink = [np.zeros((1, num_kv_heads, sink_size, head_dim), dtype=np.float32)
                       for _ in range(num_layers)]

    def update(self, layer: int, new_k: np.ndarray, new_v: np.ndarray, position: int):
        n = new_k.shape[2]
        total = position + n

        if self.sliding_window is not None:
            window_start = max(self.sink_size, total - self.sliding_window)
            slot = window_start + (total - window_start) % self.sliding_window
        else:
            slot = position

        sink_end = min(n, self.sink_size)
        if sink_end > 0:
            self.k_sink[layer][:, :, :sink_end] = new_k[:, :, :sink_end]
            self.v_sink[layer][:, :, :sink_end] = new_v[:, :, :sink_end]

        remaining = n - sink_end
        if remaining > 0:
            start = max(self.sink_size, slot)
            self.k_cache[layer][:, :, start:start + remaining] = new_k[:, :, sink_end:]
            self.v_cache[layer][:, :, start:start + remaining] = new_v[:, :, sink_end:]

        self.cur_len[layer] = total
        return self._gather(layer)

    def _gather(self, layer: int):
        length = self.cur_len[layer]
        k = self.k_sink[layer][:, :, :min(self.sink_size, length)].copy()
        v = self.v_sink[layer][:, :, :min(self.sink_size, length)].copy()

        window_len = max(0, length - self.sink_size)
        if self.sliding_window is not None:
            window_len = min(window_len, self.sliding_window)

        if window_len > 0:
            w_start = length - window_len
            w_k = self.k_cache[layer][:, :, w_start:length].copy()
            w_v = self.v_cache[layer][:, :, w_start:length].copy()
            full_k = np.concatenate([k, w_k], axis=2)
            full_v = np.concatenate([v, w_v], axis=2)
        else:
            full_k = k
            full_v = v
        return full_k, full_v

    def get(self, layer: int):
        return self._gather(layer)

    def reset(self):
        self.cur_len = [0] * self.num_layers

    def get_seq_len(self, layer: int = 0):
        return self.cur_len[layer]


class MLACache:
    def __init__(self, max_seq_len: int, num_layers: int, latent_dim: int,
                 num_heads: int, head_dim: int):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_cache = [np.zeros((1, max_seq_len, latent_dim), dtype=np.float32)
                             for _ in range(num_layers)]
        self.cur_len = [0] * num_layers

    def update(self, layer: int, new_latent: np.ndarray, position: int):
        n = new_latent.shape[1]
        self.latent_cache[layer][:, position:position + n] = new_latent
        self.cur_len[layer] = position + n
        return self.latent_cache[layer][:, :self.cur_len[layer]].copy()

    def get(self, layer: int):
        return self.latent_cache[layer][:, :self.cur_len[layer]].copy()

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
        self.use_flash = config.use_flash
        self.flash_block_size = config.flash_block_size
        self.use_mla = config.use_mla
        self.mla_latent_dim = config.mla_latent_dim

        self.q_proj = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)

        if self.use_mla:
            self.w_dkv = Linear(self.embed_dim, self.mla_latent_dim, bias=config.bias)
            self.w_uk = Linear(self.mla_latent_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
            self.w_uv = Linear(self.mla_latent_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
            self.k_proj = None
            self.v_proj = None
        else:
            self.k_proj = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
            self.v_proj = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
            self.w_dkv = None
            self.w_uk = None
            self.w_uv = None

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
                mla_cache: Optional[MLACache] = None,
                layer_idx: int = 0, position: int = 0) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv_out = None
        if self.use_mla:
            c_kv = self.w_dkv(x)
            k = self.w_uk(c_kv).reshape(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.w_uv(c_kv).reshape(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

            if mla_cache is not None:
                latent = mla_cache.update(layer_idx, c_kv.data, position)
                kv_out = (Tensor(latent), None)
        else:
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

        if not self.use_mla and kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k.data, v.data, position)
            k = Tensor(k)
            v = Tensor(v)
            kv_out = (k, v)

        S_q = L
        S_kv = k.shape[2]

        if self.num_kv_groups > 1:
            k = Tensor(np.repeat(k.data, self.num_kv_groups, axis=1), requires_grad=k.requires_grad)
            v = Tensor(np.repeat(v.data, self.num_kv_groups, axis=1), requires_grad=v.requires_grad)

        if self.use_flash:
            attn = self._flash_attention_tiled(q, k, v, B, S_q, S_kv, position, layer_idx)
        else:
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

    def _flash_attention_tiled(self, q, k, v, B, S_q, S_kv, position, layer_idx):
        block_size = self.flash_block_size
        n_heads = self.num_heads
        d = self.head_dim

        q_np = q.data
        k_np = k.data
        v_np = v.data

        output = np.zeros((B, n_heads, S_q, d), dtype=np.float32)
        m_prev = np.full((B, n_heads, S_q, 1), -1e9, dtype=np.float32)
        l_prev = np.zeros((B, n_heads, S_q, 1), dtype=np.float32)

        for j_start in range(0, S_kv, block_size):
            j_end = min(j_start + block_size, S_kv)
            k_block = k_np[:, :, j_start:j_end, :]
            v_block = v_np[:, :, j_start:j_end, :]

            scale = self.scaling
            scores = np.einsum('bhid,bhjd->bhij', q_np * scale, k_block)

            causal_mask = np.ones((S_q, j_end - j_start), dtype=np.bool_)
            for i in range(S_q):
                for jj in range(j_start, j_end):
                    if jj <= i + position:
                        causal_mask[i, jj - j_start] = True
                    else:
                        causal_mask[i, jj - j_start] = False
            scores = np.where(causal_mask[None, None, :, :], scores, -1e9)

            if self.config.sliding_window is not None:
                sw_start = max(0, S_kv - self.config.sliding_window)
                sw_mask = np.ones((S_q, j_end - j_start), dtype=np.bool_)
                for i in range(S_q):
                    end_pos = min(sw_start + i + 1, S_kv)
                    for jj in range(j_start, j_end):
                        if jj < sw_start or jj >= end_pos:
                            sw_mask[i, jj - j_start] = False
                scores = np.where(sw_mask[None, None, :, :], scores, -1e9)

            m_curr = scores.max(axis=-1, keepdims=True)
            m_new = np.maximum(m_prev, m_curr)
            exp_scores = np.exp(scores - m_new)
            l_new = l_prev * np.exp(m_prev - m_new) + exp_scores.sum(axis=-1, keepdims=True)
            output = output * (l_prev * np.exp(m_prev - m_new) / l_new) + \
                     np.einsum('bhij,bhjd->bhid', exp_scores, v_block) / l_new
            m_prev = m_new
            l_prev = l_new

        return Tensor(output, requires_grad=q.requires_grad)


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
        self.moe_routing = config.moe_routing
        self.switch_capacity_factor = config.switch_capacity_factor
        self.expert_capacity = None

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

        if self.moe_routing == "switch":
            out = self._switch_routing(x_flat, router_probs, B, L, D)
        elif self.moe_routing == "expert_choice":
            out = self._expert_choice_routing(x_flat, router_probs, B, L, D)
        else:
            out = self._topk_routing(x_flat, router_probs, B, L, D)

        if self.shared_experts is not None:
            shared_out = x_flat
            for se in self.shared_experts:
                shared_out = se(shared_out)
            out = out + shared_out.data

        if self.training and self.num_experts > 1:
            self._compute_load_balance_loss(router_probs, B * L)

        return Tensor(out.reshape(B, L, D), requires_grad=x.requires_grad)

    def _topk_routing(self, x_flat, router_probs, B, L, D):
        num_experts = self.num_experts
        k = min(self.num_experts_per_tok, num_experts)

        out = np.zeros((x_flat.shape[0], D), dtype=np.float32)

        if num_experts <= 1:
            expert_out = self.experts[0](Tensor(x_flat.data))
            out = expert_out.data
        else:
            topk_vals, topk_idx = router_probs.topk(k, dim=-1)
            topk_vals_data = topk_vals.data
            topk_idx_data = topk_idx.data

            for e in range(num_experts):
                mask_e = (topk_idx_data == e).any(axis=-1)
                if not mask_e.any():
                    continue
                idx = np.where(mask_e)[0]
                x_e = Tensor(x_flat.data[idx])
                expert_out = self.experts[e](x_e)
                for j in range(k):
                    mask = (topk_idx_data[:, j] == e) & mask_e
                    if mask.any():
                        w = topk_vals_data[mask, j:j+1]
                        n = mask.sum()
                        out[mask] += (expert_out.data[:n] * w).astype(np.float32)
        return out

    def _switch_routing(self, x_flat, router_probs, B, L, D):
        total_tokens = x_flat.shape[0]
        num_experts = self.num_experts
        k = min(1, num_experts)
        topk_vals, topk_idx = router_probs.topk(k, dim=-1)

        capacity = int(total_tokens * self.switch_capacity_factor / self.num_experts)
        self.expert_capacity = capacity

        out = np.zeros((total_tokens, D), dtype=np.float32)
        expert_counts = np.zeros(self.num_experts, dtype=np.int32)

        for t in range(total_tokens):
            e = int(topk_idx.data[t, 0])
            if expert_counts[e] < capacity:
                expert_counts[e] += 1
                x_t = Tensor(x_flat.data[t:t+1])
                expert_out = self.experts[e](x_t)
                w = topk_vals.data[t, 0]
                out[t] = expert_out.data[0] * w
        return out

    def _expert_choice_routing(self, x_flat, router_probs, B, L, D):
        total_tokens = x_flat.shape[0]
        capacity_per_expert = int(total_tokens * self.switch_capacity_factor / self.num_experts)
        self.expert_capacity = capacity_per_expert

        out = np.zeros((total_tokens, D), dtype=np.float32)
        token_weight_sum = np.zeros((total_tokens, 1), dtype=np.float32)

        for e in range(self.num_experts):
            probs_e = router_probs.data[:, e]
            top_indices = np.argsort(probs_e)[::-1][:capacity_per_expert]
            if len(top_indices) == 0:
                continue
            x_e = Tensor(x_flat.data[top_indices])
            expert_out = self.experts[e](x_e)
            weights = probs_e[top_indices]
            weights = weights / (weights.sum() + 1e-8)
            for i, t_idx in enumerate(top_indices):
                out[t_idx] += expert_out.data[i] * weights[i]
                token_weight_sum[t_idx] += weights[i]

        return out

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
                mla_cache: Optional[MLACache] = None,
                position: int = 0) -> Tuple[Tensor, Optional[Tuple]]:
        if self.config.parallel_attn_ffn:
            normed = self.attn_norm(x)
            attn_out, kv_out = self.attn(normed, kv_cache, mla_cache, self.layer_idx, position)
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
            attn_out, kv_out = self.attn(normed, kv_cache, mla_cache, self.layer_idx, position)
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
        self._mla_cache = None

    def forward(self, x: Tensor, kv_cache: Optional[KVCache] = None,
                mla_cache: Optional[MLACache] = None, position: int = 0) -> Tensor:
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

        use_cache = kv_cache is not None or mla_cache is not None
        for i, layer in enumerate(self.layers):
            h, _ = layer(h, kv_cache=kv_cache, mla_cache=mla_cache, position=position)

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

        if self.config.use_mla:
            cache = MLACache(
                self.config.max_seq_len, self.config.num_layers,
                self.config.mla_latent_dim,
                self.config.num_heads, self.config.embed_dim // self.config.num_heads
            )
        else:
            cache = KVCache(
                self.config.max_seq_len, self.config.num_layers,
                self.config.num_kv_heads, self.config.embed_dim // self.config.num_heads,
                sliding_window=self.config.sliding_window
            )

        with no_grad():
            inp = Tensor(np.array([ids], dtype=np.int32))
            logits = self.forward(inp, kv_cache=cache if not self.config.use_mla else None,
                                  mla_cache=cache if self.config.use_mla else None, position=0)

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

            pos = cache.get_seq_len() if cache else len(ids) - 1
            with no_grad():
                inp = Tensor(np.array([[next_id]], dtype=np.int32))
                logits = self.forward(inp, kv_cache=cache if not self.config.use_mla else None,
                                      mla_cache=cache if self.config.use_mla else None, position=pos)

        cache.reset()
        return ids

    def init_kv_cache(self):
        if self.config.use_mla:
            self._mla_cache = MLACache(
                self.config.max_seq_len, self.config.num_layers,
                self.config.mla_latent_dim,
                self.config.num_heads, self.config.embed_dim // self.config.num_heads
            )
            return self._mla_cache
        else:
            self._kv_cache = KVCache(
                self.config.max_seq_len, self.config.num_layers,
                self.config.num_kv_heads, self.config.embed_dim // self.config.num_heads,
                sliding_window=self.config.sliding_window
            )
            return self._kv_cache

    def clear_kv_cache(self):
        if self._kv_cache is not None:
            self._kv_cache.reset()
        if self._mla_cache is not None:
            self._mla_cache.reset()

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
