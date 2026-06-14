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

    def get_seq_len(self, layer: int = 0):
        return self.cur_len[layer]


class PrefixKVCache:
    def __init__(self, max_seq_len: int, num_layers: int, num_kv_heads: int,
                 head_dim: int, prefix_len: int = 0):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.prefix_len = prefix_len
        self.k_prefix = [np.zeros((1, num_kv_heads, prefix_len, head_dim), dtype=np.float32)
                         for _ in range(num_layers)]
        self.v_prefix = [np.zeros((1, num_kv_heads, prefix_len, head_dim), dtype=np.float32)
                         for _ in range(num_layers)]
        self.k_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.v_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.cur_len = [0] * num_layers
        self.prefix_set = [False] * num_layers

    def set_prefix(self, layer: int, k_prefix: np.ndarray, v_prefix: np.ndarray):
        n = k_prefix.shape[2]
        self.k_prefix[layer][:, :, :n] = k_prefix[:, :, :self.prefix_len]
        self.v_prefix[layer][:, :, :n] = v_prefix[:, :, :self.prefix_len]
        self.prefix_set[layer] = True

    def update(self, layer: int, new_k: np.ndarray, new_v: np.ndarray, position: int):
        n = new_k.shape[2]
        self.k_cache[layer][:, :, position:position + n] = new_k
        self.v_cache[layer][:, :, position:position + n] = new_v
        self.cur_len[layer] = position + n
        return self._gather(layer)

    def _gather(self, layer: int):
        c = self.cur_len[layer]
        if self.prefix_set[layer]:
            full_k = np.concatenate([self.k_prefix[layer], self.k_cache[layer][:, :, :c]], axis=2)
            full_v = np.concatenate([self.v_prefix[layer], self.v_cache[layer][:, :, :c]], axis=2)
        else:
            full_k = self.k_cache[layer][:, :, :c].copy()
            full_v = self.v_cache[layer][:, :, :c].copy()
        return full_k, full_v

    def get(self, layer: int):
        return self._gather(layer)

    def reset(self):
        self.cur_len = [0] * self.num_layers
        self.prefix_set = [False] * self.num_layers

    def get_seq_len(self, layer: int = 0):
        return self.cur_len[layer] + (self.prefix_len if self.prefix_set[layer] else 0)


class RadixTreeKVCache:
    def __init__(self, max_seq_len: int, num_layers: int, num_kv_heads: int,
                 head_dim: int, num_children: int = 16):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_children = num_children
        self.nodes = [{} for _ in range(num_layers)]
        self.node_counter = [0] * num_layers
        self.cur_len = [0] * num_layers
        self.cur_path = [[] for _ in range(num_layers)]
        self.k_data = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                       for _ in range(num_layers)]
        self.v_data = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                       for _ in range(num_layers)]

    def _get_or_create_node(self, layer: int, parent: int, token_id: int) -> int:
        key = (parent, token_id)
        if key in self.nodes[layer]:
            return self.nodes[layer][key]
        node_id = self.node_counter[layer]
        self.node_counter[layer] += 1
        self.nodes[layer][key] = node_id
        return node_id

    def update(self, layer: int, new_k: np.ndarray, new_v: np.ndarray, position: int):
        n = new_k.shape[2]
        self.k_data[layer][:, :, position:position + n] = new_k
        self.v_data[layer][:, :, position:position + n] = new_v
        self.cur_len[layer] = position + n
        return self._gather(layer)

    def insert_token(self, layer: int, token_id: int):
        parent = self.cur_path[layer][-1] if self.cur_path[layer] else -1
        node_id = self._get_or_create_node(layer, parent, token_id)
        self.cur_path[layer].append(node_id)

    def lookup_prefix_length(self, layer: int, token_ids) -> int:
        match_len = 0
        parent = -1
        for i, tid in enumerate(token_ids):
            key = (parent, int(tid))
            if key not in self.nodes[layer]:
                break
            parent = self.nodes[layer][key]
            match_len += 1
        return match_len

    def _gather(self, layer: int):
        c = self.cur_len[layer]
        return (self.k_data[layer][:, :, :c].copy(),
                self.v_data[layer][:, :, :c].copy())

    def get(self, layer: int):
        return self._gather(layer)

    def reset(self):
        self.cur_len = [0] * self.num_layers
        self.cur_path = [[] for _ in range(self.num_layers)]

    def get_seq_len(self, layer: int = 0):
        return self.cur_len[layer]


class KVQuantizedCache:
    def __init__(self, max_seq_len: int, num_layers: int, num_kv_heads: int,
                 head_dim: int, bits: int = 8, group_size: int = 64):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.bits = bits
        self.group_size = group_size
        self.k_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.v_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.cur_len = [0] * num_layers
        self.quantized_k = [None] * num_layers
        self.quantized_v = [None] * num_layers
        self.scales_k = [None] * num_layers
        self.scales_v = [None] * num_layers

    def _quantize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        original_shape = x.shape
        flat = x.reshape(-1)
        num_groups = max(1, len(flat) // self.group_size)
        padded = len(flat) - (len(flat) % self.group_size) if len(flat) % self.group_size != 0 else len(flat)
        if padded > len(flat):
            flat = np.pad(flat, (0, padded - len(flat)))
        groups = flat.reshape(num_groups, -1)
        scales = np.max(np.abs(groups), axis=1, keepdims=True)
        scales = np.where(scales == 0, 1.0, scales)
        quantized = np.clip(np.round(groups / scales * (2 ** (self.bits - 1) - 1)),
                           -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
        return quantized.astype(np.float32), scales.astype(np.float32)

    def _dequantize(self, quantized: np.ndarray, scales: np.ndarray, original_len: int) -> np.ndarray:
        flat = (quantized / (2 ** (self.bits - 1) - 1) * scales).flatten()
        return flat[:original_len]

    def _quantize_full(self, layer: int):
        k = self.k_cache[layer][:, :, :self.cur_len[layer]]
        v = self.v_cache[layer][:, :, :self.cur_len[layer]]
        self.quantized_k[layer], self.scales_k[layer] = self._quantize(k)
        self.quantized_v[layer], self.scales_v[layer] = self._quantize(v)

    def update(self, layer: int, new_k: np.ndarray, new_v: np.ndarray, position: int):
        n = new_k.shape[2]
        self.k_cache[layer][:, :, position:position + n] = new_k
        self.v_cache[layer][:, :, position:position + n] = new_v
        self.cur_len[layer] = position + n
        self._quantize_full(layer)
        return self._gather(layer)

    def _gather(self, layer: int):
        c = self.cur_len[layer]
        if c == 0:
            return (np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32),
                    np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=np.float32))
        return (self.k_cache[layer][:, :, :c].copy(),
                self.v_cache[layer][:, :, :c].copy())

    def get(self, layer: int):
        return self._gather(layer)

    def reset(self):
        self.cur_len = [0] * self.num_layers
        self.quantized_k = [None] * self.num_layers
        self.quantized_v = [None] * self.num_layers

    def get_seq_len(self, layer: int = 0):
        return self.cur_len[layer]

    def get_memory_savings(self) -> float:
        full_bytes = self.num_layers * 2 * self.max_seq_len * self.num_kv_heads * self.head_dim * 4
        quant_bytes = self.num_layers * 2 * self.max_seq_len * self.num_kv_heads * self.head_dim * (self.bits / 8)
        return 1.0 - (quant_bytes / full_bytes)


class SnapKVCache:
    def __init__(self, max_seq_len: int, num_layers: int, num_kv_heads: int,
                 head_dim: int, window_size: int = 64, num_snapshots: int = 16):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.num_snapshots = num_snapshots
        self.k_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.v_cache = [np.zeros((1, num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
                        for _ in range(num_layers)]
        self.snap_k = [np.zeros((1, num_kv_heads, num_snapshots, head_dim), dtype=np.float32)
                       for _ in range(num_layers)]
        self.snap_v = [np.zeros((1, num_kv_heads, num_snapshots, head_dim), dtype=np.float32)
                       for _ in range(num_layers)]
        self.snap_count = [0] * num_layers
        self.cur_len = [0] * num_layers

    def _compress(self, layer: int):
        c = self.cur_len[layer]
        if c <= self.window_size:
            return
        start = max(0, c - self.window_size - self.num_snapshots)
        k_window = self.k_cache[layer][:, :, start:c - self.window_size]
        v_window = self.v_cache[layer][:, :, start:c - self.window_size]
        attn_importance = np.abs(k_window).mean(axis=(0, 1, 3))
        top_indices = np.argsort(attn_importance)[::-1][:self.num_snapshots]
        top_indices = np.sort(top_indices)

        n_new = min(self.num_snapshots, len(top_indices))
        self.snap_k[layer][:, :, :n_new] = k_window[:, :, top_indices]
        self.snap_v[layer][:, :, :n_new] = v_window[:, :, top_indices]
        self.snap_count[layer] = n_new

    def update(self, layer: int, new_k: np.ndarray, new_v: np.ndarray, position: int):
        n = new_k.shape[2]
        self.k_cache[layer][:, :, position:position + n] = new_k
        self.v_cache[layer][:, :, position:position + n] = new_v
        self.cur_len[layer] = position + n

        if self.cur_len[layer] > self.window_size + self.num_snapshots:
            self._compress(layer)

        return self._gather(layer)

    def _gather(self, layer: int):
        c = self.cur_len[layer]
        if self.snap_count[layer] > 0:
            snap_k = self.snap_k[layer][:, :, :self.snap_count[layer]].copy()
            snap_v = self.snap_v[layer][:, :, :self.snap_count[layer]].copy()
            window_start = max(0, c - self.window_size)
            win_k = self.k_cache[layer][:, :, window_start:c].copy()
            win_v = self.v_cache[layer][:, :, window_start:c].copy()
            full_k = np.concatenate([snap_k, win_k], axis=2)
            full_v = np.concatenate([snap_v, win_v], axis=2)
        else:
            full_k = self.k_cache[layer][:, :, :c].copy()
            full_v = self.v_cache[layer][:, :, :c].copy()
        return full_k, full_v

    def get(self, layer: int):
        return self._gather(layer)

    def reset(self):
        self.cur_len = [0] * self.num_layers
        self.snap_count = [0] * self.num_layers

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
# Cross Attention (Q from decoder, KV from encoder)
# ---------------------------------------------------------------------------

class CrossAttention(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.wq = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wk = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wv = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wo = Linear(self.num_heads * self.head_dim, self.embed_dim, bias=config.bias)
        self.attn_dropout = Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, x: Tensor, context: Tensor, mask=None,
                kv_cache=None) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, L_q, D = x.shape
        _, L_kv, _ = context.shape

        q = self.wq(x).reshape(B, L_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.wk(context).reshape(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(context).reshape(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        kv_out = None
        if kv_cache is not None:
            k, v = kv_cache.update(0, k.data, v.data, 0)
            k = Tensor(k)
            v = Tensor(v)
            kv_out = (k, v)
            L_kv = k.shape[2]

        if self.num_kv_groups > 1:
            k = Tensor(np.repeat(k.data, self.num_kv_groups, axis=1), requires_grad=k.requires_grad)
            v = Tensor(np.repeat(v.data, self.num_kv_groups, axis=1), requires_grad=v.requires_grad)

        q_scaled = Tensor(q.data * self.scaling, requires_grad=q.requires_grad)
        k_t = Tensor(k.data.transpose(0, 1, 3, 2), requires_grad=k.requires_grad)
        attn = q_scaled.matmul(k_t)

        if mask is not None:
            attn = Tensor(np.where(mask[None, None, :, :] if mask.ndim == 2 else mask, attn.data, -1e9),
                           requires_grad=attn.requires_grad)

        attn = attn.softmax(axis=-1)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)

        out = attn.matmul(v)
        out = Tensor(out.data.transpose(0, 2, 1, 3), requires_grad=out.requires_grad)
        out = out.reshape(B, L_q, self.num_heads * self.head_dim)
        return self.wo(out), kv_out


# ---------------------------------------------------------------------------
# Linear Attention (Performer FAVOR+)
# ---------------------------------------------------------------------------

class LinearAttention(Module):
    def __init__(self, config: TransformerConfig, num_features: int = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_features = num_features or self.head_dim

        self.wq = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wk = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wv = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wo = Linear(self.num_heads * self.head_dim, self.embed_dim, bias=config.bias)

        self.register_parameter("projection_matrix", None)

    def _get_projection_matrix(self, seq_len: int, device_dtype=np.float32):
        if self.projection_matrix is not None:
            return Tensor(self.projection_matrix)
        dim = self.head_dim
        num_features = self.num_features
        projection = np.random.randn(dim, num_features).astype(device_dtype) * (1.0 / np.sqrt(num_features))
        self.projection_matrix = projection
        return Tensor(projection)

    def _feature_map(self, x: Tensor) -> Tensor:
        projection = self._get_projection_matrix(x.shape[-1], x.data.dtype)
        projected = x.matmul(Tensor(projection.data.T))
        return projected.elu(alpha=1.0) + 1.0

    def forward(self, x: Tensor, mask=None, kv_cache=None) -> Tuple[Tensor, Optional[Tuple]]:
        B, L, D = x.shape

        q = self.wq(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        B_h, H, S, D_h = q.shape

        q_phi = self._feature_map(q)
        k_phi = self._feature_map(k)

        kv_out = None
        if kv_cache is not None:
            k_data, v_data = kv_cache.update(0, k.data, v.data, 0)
            k = Tensor(k_data)
            v = Tensor(v_data)
            k_phi = self._feature_map(k)
            kv_out = (k, v)

        k_phi_t = Tensor(k_phi.data.transpose(0, 1, 3, 2), requires_grad=k_phi.requires_grad)

        kv = k_phi_t.matmul(v)
        numerator = q_phi.matmul(kv)

        k_sum = k_phi.sum(axis=2, keepdims=True).permute(0, 1, 3, 2)
        denominator = q_phi.matmul(k_sum) + 1e-6

        out = Tensor(numerator.data / denominator.data, requires_grad=numerator.requires_grad)

        if mask is not None:
            mask_expanded = mask[None, None, :, :] if mask.ndim == 2 else mask
            out = Tensor(out.data * mask_expanded.astype(out.data.dtype), requires_grad=out.requires_grad)

        out = Tensor(out.data.transpose(0, 2, 1, 3), requires_grad=out.requires_grad)
        out = out.reshape(B, L, self.num_heads * self.head_dim)
        return self.wo(out), kv_out


# ---------------------------------------------------------------------------
# Cosformer (cosine similarity attention)
# ---------------------------------------------------------------------------

class Cosformer(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scaling = self.head_dim ** -0.5

        self.wq = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wk = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wv = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wo = Linear(self.num_heads * self.head_dim, self.embed_dim, bias=config.bias)
        self.attn_dropout = Dropout(config.dropout) if config.dropout > 0 else None

    def _cosine_reweight(self, x: Tensor) -> Tensor:
        cos_val = x.cos()
        sin_val = x.sin()
        return Tensor(cos_val.data + sin_val.data, requires_grad=x.requires_grad)

    def _causal_mask(self, S: int) -> np.ndarray:
        mask = np.triu(np.ones((S, S), dtype=np.bool_), k=1)
        return ~mask

    def forward(self, x: Tensor, mask=None, kv_cache=None) -> Tuple[Tensor, Optional[Tuple]]:
        B, L, D = x.shape

        q = self.wq(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        S_q = L
        S_kv = k.shape[2]

        kv_out = None
        if kv_cache is not None:
            k_data, v_data = kv_cache.update(0, k.data, v.data, 0)
            k = Tensor(k_data)
            v = Tensor(v_data)
            kv_out = (k, v)
            S_kv = k.shape[2]

        q_rew = self._cosine_reweight(q)
        k_rew = self._cosine_reweight(k)

        k_rew_t = Tensor(k_rew.data.transpose(0, 1, 3, 2), requires_grad=k_rew.requires_grad)
        attn = q_rew.matmul(k_rew_t)
        attn = Tensor(attn.data * self.scaling, requires_grad=attn.requires_grad)

        causal = self._causal_mask(max(S_q, S_kv))[:S_q, :S_kv]
        attn = Tensor(np.where(causal[None, None, :, :], attn.data, -1e9),
                       requires_grad=attn.requires_grad)

        if mask is not None:
            mask_exp = mask[None, None, :, :] if mask.ndim == 2 else mask
            attn = Tensor(np.where(mask_exp, attn.data, -1e9),
                           requires_grad=attn.requires_grad)

        attn = attn.softmax(axis=-1)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)

        out = attn.matmul(v)
        out = Tensor(out.data.transpose(0, 2, 1, 3), requires_grad=out.requires_grad)
        out = out.reshape(B, L, self.num_heads * self.head_dim)
        return self.wo(out), kv_out


# ---------------------------------------------------------------------------
# AFT (Attention Free Transformer)
# ---------------------------------------------------------------------------

class AFT(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        self.wq = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wv = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wo = Linear(self.num_heads * self.head_dim, self.embed_dim, bias=config.bias)

        self.pos_bias = Embedding(config.max_seq_len, self.num_heads)
        self.alpha = Embedding(config.max_seq_len, self.num_heads)

    def forward(self, x: Tensor, mask=None, kv_cache=None) -> Tuple[Tensor, Optional[Tuple]]:
        B, L, D = x.shape

        q = self.wq(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        B_h, H, S, D_h = q.shape

        with no_grad():
            positions = Tensor(np.arange(S, dtype=np.int32))
            pos_bias = self.pos_bias(positions).data
            alpha_vals = self.alpha(positions).data

        exp_pos_bias = np.zeros((S, S), dtype=np.float32)
        for i in range(S):
            for j in range(S):
                if i >= j:
                    exp_pos_bias[i, j] = np.exp(-float(pos_bias[i - j, 0]))
                else:
                    exp_pos_bias[i, j] = np.exp(-float(pos_bias[j - i, 0]))

        q_np = q.data
        v_np = v.data

        pos_weighted_v = np.einsum('bhjd,ij->bhid', v_np, exp_pos_bias)
        denom = exp_pos_bias.sum(axis=1, keepdims=True)
        pos_weighted_v = pos_weighted_v / (denom[np.newaxis, np.newaxis, :, :] + 1e-6)

        out = Tensor(pos_weighted_v, requires_grad=q.requires_grad)

        out = Tensor(out.data.transpose(0, 2, 1, 3), requires_grad=out.requires_grad)
        out = out.reshape(B, L, self.num_heads * self.head_dim)
        return self.wo(out), None


# ---------------------------------------------------------------------------
# Infini-Attention with compressive memory
# ---------------------------------------------------------------------------

class InfiniAttention(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scaling = self.head_dim ** -0.5

        self.wq = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wk = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wv = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wo = Linear(self.num_heads * self.head_dim, self.embed_dim, bias=config.bias)

        self.norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
        self.gate_linear = Linear(config.embed_dim, 1, bias=config.bias)

        self.memory = None
        self.memory_normalizer = None

    def _init_memory(self, B, H, D):
        self.memory = np.zeros((B, H, D, D), dtype=np.float32)
        self.memory_normalizer = np.zeros((B, H, D), dtype=np.float32)

    def _get_initial_state(self, B, H, D):
        if self.memory is None:
            self._init_memory(B, H, D)
        return self.memory.copy(), self.memory_normalizer.copy()

    def _update_memory(self, k: np.ndarray, v: np.ndarray):
        new_memory = np.einsum('bhsi,bhsj->bhij', v, k)
        new_normalizer = k.sum(axis=2)
        self.memory = self.memory + new_memory
        self.memory_normalizer = self.memory_normalizer + new_normalizer

    def _memory_attention(self, q: np.ndarray):
        B, H, S, D = q.shape
        norm_factor = np.einsum('bhid,bhd->bhi', q, self.memory_normalizer) + 1e-6
        memory_output = np.einsum('bhde,bhse->bhsd', self.memory, q)
        memory_output = memory_output / norm_factor[:, :, :, np.newaxis]
        return memory_output

    def forward(self, x: Tensor, mask=None, kv_cache=None) -> Tuple[Tensor, Optional[Tuple]]:
        B, L, D = x.shape
        normed = self.norm(x)

        q = self.wq(normed).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.wk(normed).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(normed).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        gate_input = x.mean(axis=1, keepdims=True)
        gate = Tensor(np.clip(1.0 / (1.0 + np.exp(-self.gate_linear(gate_input).data)), 0.01, 0.99),
                       requires_grad=x.requires_grad)
        beta = float(gate.data.flatten()[0])

        B_h, H, S, D_h = q.shape

        if self.memory is None:
            self._init_memory(B_h, H, D_h)

        memory_out = self._memory_attention(q.data)
        self._update_memory(k.data, v.data)

        q_scaled = Tensor(q.data * self.scaling, requires_grad=q.requires_grad)
        k_t = Tensor(k.data.transpose(0, 1, 3, 2), requires_grad=k.requires_grad)
        local_attn = q_scaled.matmul(k_t)

        causal_mask = np.zeros((S, S), dtype=np.bool_)
        for i in range(S):
            for j in range(S):
                if j <= i:
                    causal_mask[i, j] = True
        local_attn = Tensor(np.where(causal_mask[None, None, :, :], local_attn.data, -1e9),
                             requires_grad=local_attn.requires_grad)

        local_attn = local_attn.softmax(axis=-1)
        local_out = local_attn.matmul(v)

        mem_out = Tensor(memory_out, requires_grad=q.requires_grad)

        combined = Tensor(local_out.data * beta + mem_out.data * (1.0 - beta),
                           requires_grad=local_out.requires_grad)

        out = Tensor(combined.data.transpose(0, 2, 1, 3), requires_grad=combined.requires_grad)
        out = out.reshape(B, L, self.num_heads * self.head_dim)
        return self.wo(out), None

    def reset_memory(self):
        self.memory = None
        self.memory_normalizer = None


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
# Post-Norm Transformer Block
# ---------------------------------------------------------------------------

class PostNormTransformerBlock(Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if config.norm_type == "rms":
            self.norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
        else:
            self.norm = LayerNorm(config.embed_dim)

        self.attn = Attention(config)

        if config.num_experts > 0:
            self.ffn = MoELayer(config)
        else:
            self.ffn = FeedForward(config)

        self.drop = Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, x: Tensor, mask=None, kv_cache: Optional[KVCache] = None,
                mla_cache: Optional[MLACache] = None,
                position: int = 0) -> Tuple[Tensor, Optional[Tuple]]:
        attn_out, kv_out = self.attn(x, kv_cache, mla_cache, self.layer_idx, position)
        if self.drop is not None:
            attn_out = self.drop(attn_out)
        h = x + attn_out

        ffn_out = self.ffn(h)
        if self.drop is not None:
            ffn_out = self.drop(ffn_out)
        return self.norm(h + ffn_out), kv_out


# ---------------------------------------------------------------------------
# Encoder-Decoder Transformer
# ---------------------------------------------------------------------------

class EncoderBlock(Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if config.norm_type == "rms":
            self.norm1 = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
            self.norm2 = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
        else:
            self.norm1 = LayerNorm(config.embed_dim)
            self.norm2 = LayerNorm(config.embed_dim)

        self.attn = Attention(config)
        self.ffn = FeedForward(config)
        self.drop = Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, x: Tensor, mask=None) -> Tuple[Tensor, None]:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed)
        if self.drop is not None:
            attn_out = self.drop(attn_out)
        h = x + attn_out

        normed2 = self.norm2(h)
        ffn_out = self.ffn(normed2)
        if self.drop is not None:
            ffn_out = self.drop(ffn_out)
        return h + ffn_out, None


class DecoderBlock(Module):
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if config.norm_type == "rms":
            self.self_norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
            self.cross_norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
            self.ffn_norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
        else:
            self.self_norm = LayerNorm(config.embed_dim)
            self.cross_norm = LayerNorm(config.embed_dim)
            self.ffn_norm = LayerNorm(config.embed_dim)

        self.self_attn = Attention(config)
        self.cross_attn = CrossAttention(config)
        self.ffn = FeedForward(config)
        self.drop = Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, x: Tensor, encoder_out: Tensor, mask=None, cross_mask=None,
                kv_cache: Optional[KVCache] = None,
                mla_cache: Optional[MLACache] = None,
                position: int = 0) -> Tuple[Tensor, Optional[Tuple]]:
        normed = self.self_norm(x)
        self_out, kv_out = self.self_attn(normed, kv_cache, mla_cache, self.layer_idx, position)
        if self.drop is not None:
            self_out = self.drop(self_out)
        h = x + self_out

        normed2 = self.cross_norm(h)
        cross_out, _ = self.cross_attn(normed2, encoder_out, mask=cross_mask)
        if self.drop is not None:
            cross_out = self.drop(cross_out)
        h = h + cross_out

        normed3 = self.ffn_norm(h)
        ffn_out = self.ffn(normed3)
        if self.drop is not None:
            ffn_out = self.drop(ffn_out)
        return h + ffn_out, kv_out


class EncoderDecoderTransformer(Module):
    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        c = self.config

        self.embed = Embedding(c.vocab_size, c.embed_dim)

        if c.positional_encoding == "sinusoidal":
            self.pos_encoding = SinusoidalPE(c.max_seq_len, c.embed_dim)
        else:
            self.pos_encoding = None

        self.encoder_layers = ModuleList([
            EncoderBlock(c, i) for i in range(c.num_layers)
        ])
        self.decoder_layers = ModuleList([
            DecoderBlock(c, i) for i in range(c.num_layers)
        ])

        if c.norm_type == "rms":
            self.encoder_norm = NNRMSNorm(c.embed_dim, eps=c.rms_eps)
            self.decoder_norm = NNRMSNorm(c.embed_dim, eps=c.rms_eps)
        else:
            self.encoder_norm = LayerNorm(c.embed_dim)
            self.decoder_norm = LayerNorm(c.embed_dim)

        self.head = Linear(c.embed_dim, c.vocab_size, bias=False)
        self.embed_drop = Dropout(c.dropout) if c.dropout > 0 else None

    def _add_pos_encoding(self, x: Tensor, offset: int = 0) -> Tensor:
        B, L, D = x.shape
        if self.pos_encoding is not None:
            with no_grad():
                pe = Tensor(self.pos_encoding.forward(L + offset), requires_grad=False)
                pe = Tensor(pe.data[offset:offset + L], requires_grad=False)
            x = x + pe
        return x

    def forward(self, src: Tensor, tgt: Tensor, src_mask=None, tgt_mask=None) -> Tensor:
        if isinstance(src, np.ndarray):
            src = Tensor(src)
        if isinstance(tgt, np.ndarray):
            tgt = Tensor(tgt)

        if src.dtype and src.dtype.name == "INT8":
            src = src.float()
        if tgt.dtype and tgt.dtype.name == "INT8":
            tgt = tgt.float()

        enc_h = self.embed(src)
        enc_h = self._add_pos_encoding(enc_h)

        if self.embed_drop is not None:
            enc_h = self.embed_drop(enc_h)

        for layer in self.encoder_layers:
            enc_h, _ = layer(enc_h, mask=src_mask)
        enc_h = self.encoder_norm(enc_h)

        dec_h = self.embed(tgt)
        dec_h = self._add_pos_encoding(dec_h)

        if self.embed_drop is not None:
            dec_h = self.embed_drop(dec_h)

        for layer in self.decoder_layers:
            dec_h, _ = layer(dec_h, enc_h, mask=tgt_mask, cross_mask=src_mask)
        dec_h = self.decoder_norm(dec_h)

        logits = self.head(dec_h)

        if self.config.logit_softcap > 0:
            logits = Tensor(np.tanh(logits.data / self.config.logit_softcap) * self.config.logit_softcap,
                            requires_grad=logits.requires_grad)

        return logits


# ---------------------------------------------------------------------------
# DeepSeekMoE (fine-grained experts with shared expert)
# ---------------------------------------------------------------------------

class DeepSeekMoE(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_shared = max(1, config.num_shared_experts)
        self.num_routed = max(1, config.num_experts)
        self.num_experts_per_tok = config.num_experts_per_tok

        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.head = Linear(config.embed_dim, config.vocab_size, bias=False)

        self.gate = Linear(config.embed_dim, self.num_routed, bias=False)

        self.shared_experts = ModuleList([
            FeedForward(config) for _ in range(self.num_shared)
        ])
        self.routed_experts = ModuleList([
            FeedForward(config) for _ in range(self.num_routed)
        ])

        self.norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
        self.load_balance_loss = Tensor(np.float32(0.0))

    def forward(self, x: Tensor, mask=None) -> Tensor:
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        residual = x

        normed = self.norm(x)
        x_flat = normed.reshape(B * L, D)

        router_logits = self.gate(x_flat)
        router_probs = router_logits.softmax(axis=-1)

        out = np.zeros((x_flat.shape[0], D), dtype=np.float32)

        if self.num_routed <= 1:
            for e in range(self.num_routed):
                expert_out = self.routed_experts[e](x_flat)
                out = expert_out.data
        else:
            k = min(self.num_experts_per_tok, self.num_routed)
            topk_vals, topk_idx = router_probs.topk(k, dim=-1)
            for e in range(self.num_routed):
                mask_e = (topk_idx.data == e).any(axis=-1)
                if not mask_e.any():
                    continue
                idx = np.where(mask_e)[0]
                x_e = Tensor(x_flat.data[idx])
                expert_out = self.routed_experts[e](x_e)
                for j in range(k):
                    mask = (topk_idx.data[:, j] == e) & mask_e
                    if mask.any():
                        w = topk_vals.data[mask, j:j+1]
                        n = mask.sum()
                        out[mask] += (expert_out.data[:n] * w).astype(np.float32)

        shared_out = x_flat.data.copy()
        for se in self.shared_experts:
            shared_out = se(Tensor(shared_out)).data

        combined = out + shared_out
        result = Tensor(combined.reshape(B, L, D), requires_grad=x.requires_grad)
        return self.head(self.norm(residual + result))

    def compute_load_balance_loss(self, router_probs: Tensor, total_tokens: int):
        probs_mean = router_probs.mean(axis=0)
        k = min(self.num_experts_per_tok, self.num_routed)
        _, topk_idx = router_probs.topk(k, dim=-1)
        expert_counts = np.zeros(self.num_routed, dtype=np.float32)
        for e in range(self.num_routed):
            expert_counts[e] = (topk_idx.data == e).sum()
        freq = expert_counts / (total_tokens * k)
        aux_loss = self.num_routed * (probs_mean.data * freq).sum()
        self.load_balance_loss = Tensor(np.float32(aux_loss))


# ---------------------------------------------------------------------------
# Mixture of Depths (MoD)
# ---------------------------------------------------------------------------

class MixtureOfDepths(Module):
    def __init__(self, config: TransformerConfig, threshold: float = 0.5):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.threshold = threshold

        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.head = Linear(config.embed_dim, config.vocab_size, bias=False)

        self.router = Linear(config.embed_dim, 1, bias=False)

        if config.norm_type == "rms":
            self.norm1 = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
            self.norm2 = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
            self.final_norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
        else:
            self.norm1 = LayerNorm(config.embed_dim)
            self.norm2 = LayerNorm(config.embed_dim)
            self.final_norm = LayerNorm(config.embed_dim)

        self.attn = Attention(config)
        self.ffn = FeedForward(config)
        self.drop = Dropout(config.dropout) if config.dropout > 0 else None

        self.total_tokens = 0
        self.active_tokens = 0

    def _compute_routing(self, x: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        router_logits = self.router(x)
        router_probs = 1.0 / (1.0 + np.exp(-router_logits.data))
        selected = (router_probs >= self.threshold).astype(np.float32)
        return router_probs, selected

    def forward(self, x: Tensor, mask=None, kv_cache: Optional[KVCache] = None,
                mla_cache: Optional[MLACache] = None,
                position: int = 0) -> Tensor:
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        residual = x

        normed = self.norm1(x)
        routing_probs, selected = self._compute_routing(normed)

        self.total_tokens += B * L
        self.active_tokens += selected.sum()

        attn_out, kv_out = self.attn(normed, kv_cache, mla_cache, 0, position)
        if self.drop is not None:
            attn_out = self.drop(attn_out)

        selected_expanded = Tensor(np.broadcast_to(selected, (B, L, D)), requires_grad=x.requires_grad)
        h = x + attn_out * selected_expanded

        normed2 = self.norm2(h)
        ffn_out = self.ffn(normed2)
        if self.drop is not None:
            ffn_out = self.drop(ffn_out)
        result = h + ffn_out * selected_expanded
        return self.head(self.final_norm(result))

    def get_utilization(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.active_tokens / self.total_tokens

    def reset_counters(self):
        self.total_tokens = 0
        self.active_tokens = 0


# ---------------------------------------------------------------------------
# SparseAttention (Longformer-style)
# ---------------------------------------------------------------------------

class SparseAttention(Module):
    def __init__(self, config: TransformerConfig, window_size: int = 256, num_global_tokens: int = 1):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens

        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.wq = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wk = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wv = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wo = Linear(self.num_heads * self.head_dim, self.embed_dim, bias=config.bias)

    def _sparse_mask(self, S: int) -> np.ndarray:
        mask = np.zeros((S, S), dtype=np.bool_)
        for i in range(S):
            start = max(0, i - self.window_size)
            end = min(S, i + self.window_size + 1)
            mask[i, start:end] = True
            mask[:self.num_global_tokens, i] = True
            mask[i, :self.num_global_tokens] = True
        return mask

    def forward(self, x: Tensor, mask=None) -> Tensor:
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        q = self.wq(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.num_kv_groups > 1:
            k = Tensor(np.repeat(k.data, self.num_kv_groups, axis=1), requires_grad=k.requires_grad)
            v = Tensor(np.repeat(v.data, self.num_kv_groups, axis=1), requires_grad=v.requires_grad)

        q_scaled = Tensor(q.data * self.scaling, requires_grad=q.requires_grad)
        k_t = Tensor(k.data.transpose(0, 1, 3, 2), requires_grad=k.requires_grad)
        attn = q_scaled.matmul(k_t)

        sparse_mask = self._sparse_mask(L)
        attn = Tensor(np.where(sparse_mask[None, None, :, :], attn.data, -1e9),
                       requires_grad=attn.requires_grad)

        if mask is not None:
            mask_exp = mask[None, None, :, :] if mask.ndim == 2 else mask
            attn = Tensor(np.where(mask_exp, attn.data, -1e9),
                           requires_grad=attn.requires_grad)

        attn = attn.softmax(axis=-1)
        out = attn.matmul(v)
        out = Tensor(out.data.transpose(0, 2, 1, 3), requires_grad=out.requires_grad)
        out = out.reshape(B, L, self.num_heads * self.head_dim)
        return self.wo(out)


# ---------------------------------------------------------------------------
# BigBirdAttention (random + window + global)
# ---------------------------------------------------------------------------

class BigBirdAttention(Module):
    def __init__(self, config: TransformerConfig, window_size: int = 64, num_random: int = 64, num_global: int = 2):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.window_size = window_size
        self.num_random = num_random
        self.num_global = num_global

        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.wq = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wk = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wv = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wo = Linear(self.num_heads * self.head_dim, self.embed_dim, bias=config.bias)

        self._random_mask_cache = {}

    def _bigbird_mask(self, S: int) -> np.ndarray:
        if S in self._random_mask_cache:
            return self._random_mask_cache[S]
        mask = np.zeros((S, S), dtype=np.bool_)
        for i in range(S):
            start = max(0, i - self.window_size)
            end = min(S, i + self.window_size + 1)
            mask[i, start:end] = True
            mask[:self.num_global, :] = True
            mask[:, :self.num_global] = True
            n_rand = min(self.num_random, S)
            rand_idx = np.random.choice(S, size=n_rand, replace=False)
            mask[i, rand_idx] = True
        self._random_mask_cache[S] = mask
        return mask

    def forward(self, x: Tensor, mask=None) -> Tensor:
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        q = self.wq(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.num_kv_groups > 1:
            k = Tensor(np.repeat(k.data, self.num_kv_groups, axis=1), requires_grad=k.requires_grad)
            v = Tensor(np.repeat(v.data, self.num_kv_groups, axis=1), requires_grad=v.requires_grad)

        q_scaled = Tensor(q.data * self.scaling, requires_grad=q.requires_grad)
        k_t = Tensor(k.data.transpose(0, 1, 3, 2), requires_grad=k.requires_grad)
        attn = q_scaled.matmul(k_t)

        bb_mask = self._bigbird_mask(L)
        attn = Tensor(np.where(bb_mask[None, None, :, :], attn.data, -1e9),
                       requires_grad=attn.requires_grad)

        if mask is not None:
            mask_exp = mask[None, None, :, :] if mask.ndim == 2 else mask
            attn = Tensor(np.where(mask_exp, attn.data, -1e9),
                           requires_grad=attn.requires_grad)

        attn = attn.softmax(axis=-1)
        out = attn.matmul(v)
        out = Tensor(out.data.transpose(0, 2, 1, 3), requires_grad=out.requires_grad)
        out = out.reshape(B, L, self.num_heads * self.head_dim)
        return self.wo(out)


# ---------------------------------------------------------------------------
# RingAttention (simulated ring exchange with online softmax)
# ---------------------------------------------------------------------------

class RingAttention(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.num_workers = 2

        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.wq = Linear(self.embed_dim, self.num_heads * self.head_dim, bias=config.bias)
        self.wk = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wv = Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.bias)
        self.wo = Linear(self.num_heads * self.head_dim, self.embed_dim, bias=config.bias)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        q = self.wq(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.num_kv_groups > 1:
            k = Tensor(np.repeat(k.data, self.num_kv_groups, axis=1), requires_grad=k.requires_grad)
            v = Tensor(np.repeat(v.data, self.num_kv_groups, axis=1), requires_grad=v.requires_grad)

        n_workers = min(self.num_workers, L)
        chunk_size = L // n_workers
        if chunk_size == 0:
            chunk_size = 1
            n_workers = L

        output = np.zeros_like(q.data)
        m_prev = np.full((B, self.num_heads, L, 1), -np.inf, dtype=np.float32)
        l_prev = np.zeros((B, self.num_heads, L, 1), dtype=np.float32)

        for ring_step in range(n_workers):
            k_chunk = k.data[:, :, ring_step * chunk_size:(ring_step + 1) * chunk_size, :]
            v_chunk = v.data[:, :, ring_step * chunk_size:(ring_step + 1) * chunk_size, :]
            if k_chunk.shape[2] == 0:
                continue

            scores = np.einsum('bhid,bhjd->bhij', q.data * self.scaling, k_chunk)

            if mask is not None:
                mask_exp = mask[None, None, :, :] if mask.ndim == 2 else mask
                chunk_mask = mask_exp[:, :, :, ring_step * chunk_size:(ring_step + 1) * chunk_size]
                if chunk_mask.shape[-1] == scores.shape[-1]:
                    scores = np.where(chunk_mask, scores, -1e9)

            m_curr = scores.max(axis=-1, keepdims=True)
            m_new = np.maximum(m_prev, m_curr)
            exp_scores = np.exp(scores - m_new)
            l_new = l_prev * np.exp(m_prev - m_new) + exp_scores.sum(axis=-1, keepdims=True)
            output = output * (l_prev * np.exp(m_prev - m_new) / l_new) + \
                     np.einsum('bhij,bhjd->bhid', exp_scores, v_chunk) / l_new
            m_prev = m_new
            l_prev = l_new

        out = Tensor(output.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim),
                      requires_grad=q.requires_grad)
        return self.wo(out)


# ---------------------------------------------------------------------------
# GShard MoE (top-1 routing with capacity factor)
# ---------------------------------------------------------------------------

class GShardMoE(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_experts = config.num_experts if config.num_experts > 0 else 8
        self.capacity_factor = config.switch_capacity_factor

        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.gate = Linear(self.embed_dim, self.num_experts, bias=False)
        self.experts = ModuleList([
            FeedForward(config) for _ in range(self.num_experts)
        ])

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        total_tokens = x_flat.shape[0]

        router_logits = self.gate(x_flat)
        router_probs = router_logits.softmax(axis=-1)
        topk_vals, topk_idx = router_probs.topk(1, dim=-1)

        capacity = int(total_tokens * self.capacity_factor / self.num_experts)
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

        return Tensor(out.reshape(B, L, D), requires_grad=x.requires_grad)


# ---------------------------------------------------------------------------
# GLaM MoE (64 experts, 2 activated)
# ---------------------------------------------------------------------------

class GLaMMoE(Module):
    def __init__(self, config: TransformerConfig, num_experts: int = 64, num_activated: int = 2):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_experts = num_experts
        self.num_activated = num_activated

        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.gate = Linear(self.embed_dim, self.num_experts, bias=False)
        self.experts = ModuleList([
            FeedForward(config) for _ in range(self.num_experts)
        ])

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)

        router_logits = self.gate(x_flat)
        router_probs = router_logits.softmax(axis=-1)

        k = min(self.num_activated, self.num_experts)
        topk_vals, topk_idx = router_probs.topk(k, dim=-1)

        out = np.zeros((x_flat.shape[0], D), dtype=np.float32)

        for e in range(self.num_experts):
            mask_e = (topk_idx.data == e).any(axis=-1)
            if not mask_e.any():
                continue
            idx = np.where(mask_e)[0]
            x_e = Tensor(x_flat.data[idx])
            expert_out = self.experts[e](x_e)
            for j in range(k):
                mask = (topk_idx.data[:, j] == e) & mask_e
                if mask.any():
                    w = topk_vals.data[mask, j:j+1]
                    n = mask.sum()
                    out[mask] += (expert_out.data[:n] * w).astype(np.float32)

        return Tensor(out.reshape(B, L, D), requires_grad=x.requires_grad)


# ---------------------------------------------------------------------------
# SoftMoE (soft routing via weighted average)
# ---------------------------------------------------------------------------

class SoftMoE(Module):
    def __init__(self, config: TransformerConfig, num_experts: int = 8):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_experts = num_experts

        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.gate = Linear(self.embed_dim, self.num_experts, bias=False)
        self.experts = ModuleList([
            FeedForward(config) for _ in range(self.num_experts)
        ])

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)

        router_logits = self.gate(x_flat)
        router_probs = router_logits.softmax(axis=-1)

        out = np.zeros((x_flat.shape[0], D), dtype=np.float32)
        for e in range(self.num_experts):
            weights = router_probs.data[:, e:e+1]
            x_e = Tensor(x_flat.data * weights)
            expert_out = self.experts[e](x_e)
            out += expert_out.data * weights

        return Tensor(out.reshape(B, L, D), requires_grad=x.requires_grad)


# ---------------------------------------------------------------------------
# Adaptive Computation Time (ACT)
# ---------------------------------------------------------------------------

class AdaptiveComputationTime(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim

        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.halting_prob = Linear(config.embed_dim, 1, bias=False)

        if config.norm_type == "rms":
            self.norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
        else:
            self.norm = LayerNorm(config.embed_dim)

        self.ffn = FeedForward(config)
        self.attn_norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps) if config.norm_type == "rms" else LayerNorm(config.embed_dim)

    def forward(self, x: Tensor, max_steps: int = 10) -> Tensor:
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        output = np.zeros_like(x.data)
        remainder = np.ones((B, L, 1), dtype=np.float32)
        accumulated = np.zeros((B, L, 1), dtype=np.float32)

        for step in range(max_steps):
            normed = self.norm(Tensor(x.data + output))
            halt_logits = self.halting_prob(normed)
            halt_prob = 1.0 / (1.0 + np.exp(-halt_logits.data))

            p = halt_prob * remainder
            remainder = remainder * (1.0 - halt_prob)

            normed2 = self.attn_norm(Tensor(x.data + output))
            step_out = self.ffn(normed2)

            output = output + step_out.data * p

            if remainder.sum() < 1e-6:
                break

        output = output + x.data * remainder
        return Tensor(output, requires_grad=x.requires_grad)


# ---------------------------------------------------------------------------
# Universal Transformer (shared weights across depth)
# ---------------------------------------------------------------------------

class UniversalTransformer(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.shared_block = TransformerBlock(config, 0)
        self.act = AdaptiveComputationTime(config)

        if config.norm_type == "rms":
            self.norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
        else:
            self.norm = LayerNorm(config.embed_dim)
        self.head = Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, x: Tensor, mask=None, max_steps: int = 6) -> Tensor:
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        h = x

        for _ in range(max_steps):
            block_out, _ = self.shared_block(h)
            h = block_out

        h = self.act(h)
        h = self.norm(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# MixtralMoE (interleaved every other layer)
# ---------------------------------------------------------------------------

class MixtralMoE(Module):
    def __init__(self, config: TransformerConfig, num_experts: int = 8, num_activated: int = 2):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_experts = num_experts
        self.num_activated = num_activated
        self.num_layers = config.num_layers

        self.embed = Embedding(config.vocab_size, config.embed_dim)
        self.head = Linear(config.embed_dim, config.vocab_size, bias=False)

        self.attn_blocks = ModuleList([
            TransformerBlock(config, i) for i in range(self.num_layers)
        ])

        self.moe_layers = ModuleList()
        for _ in range(self.num_layers // 2):
            self.moe_layers.append(
                GLaMMoE(config, num_experts=num_experts, num_activated=num_activated)
            )

        if config.norm_type == "rms":
            self.norm = NNRMSNorm(config.embed_dim, eps=config.rms_eps)
        else:
            self.norm = LayerNorm(config.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(x, np.ndarray):
            x = Tensor(x)
        if x.ndim == 2:
            x = self.embed(x)
        B, L, D = x.shape
        h = x

        moe_idx = 0
        for i in range(self.num_layers):
            h, _ = self.attn_blocks[i](h)
            if i % 2 == 1 and moe_idx < len(self.moe_layers):
                h = self.moe_layers[moe_idx](h)
                moe_idx += 1

        h = self.norm(h)
        return self.head(h)


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

        if c.tie_embeddings:
            self.head.weight = self.embed.weight

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
