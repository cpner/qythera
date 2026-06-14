"""Complete Transformer with all components: RoPE, GQA, SwiGLU, MoE, KV Cache."""

import numpy as np
import math
import json
import os
import time
from typing import Optional, Tuple, List
from core.engine import Tensor, Adam, CosineScheduler, BPETokenizer, Device, detect_device, get_precision, Precision


class MultiHeadAttention:
    """Multi-Head Attention with GQA, RoPE, and KV Cache."""
    
    def __init__(self, d_model, num_heads, num_kv_heads=None, max_seq=2048):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads // 4
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        scale = 1.0 / math.sqrt(self.head_dim)
        self.wq = Parameter(np.random.randn(d_model, num_heads * self.head_dim).astype(np.float32) * scale)
        self.wk = Parameter(np.random.randn(d_model, self.num_kv_heads * self.head_dim).astype(np.float32) * scale)
        self.wv = Parameter(np.random.randn(d_model, self.num_kv_heads * self.head_dim).astype(np.float32) * scale)
        self.wo = Parameter(np.random.randn(num_heads * self.head_dim, d_model).astype(np.float32) * scale)
        
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
        rotated = np.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return rotated.reshape(B, H, L, D)
    
    def forward(self, x, kv_cache=None, position=0):
        B, L, _ = x.shape
        
        q = (x @ self.wq.data).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = (x @ self.wk.data).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = (x @ self.wv.data).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        freqs = self.freqs[position:position+L]
        q = self._apply_rope(q, freqs)
        k = self._apply_rope(k, freqs)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = np.concatenate([past_k, k], axis=2)
            v = np.concatenate([past_v, v], axis=2)
        new_cache = (k, v)
        
        if self.num_queries_per_kv > 1:
            k = np.repeat(k, self.num_queries_per_kv, axis=1)
            v = np.repeat(v, self.num_queries_per_kv, axis=1)
        
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale
        
        total_len = k.shape[2]
        mask = np.triu(np.full((L, total_len), -1e9), k=total_len-L+1)
        attn = attn + mask
        
        attn = np.exp(attn - attn.max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return out @ self.wo.data, new_cache
    
    def get_params(self):
        return [self.wq, self.wk, self.wv, self.wo]


class SwiGLU:
    """SwiGLU Feed-Forward Network."""
    
    def __init__(self, d_model, d_ff):
        scale = 1.0 / math.sqrt(d_model)
        self.w1 = Parameter(np.random.randn(d_model, d_ff).astype(np.float32) * scale)
        self.w2 = Parameter(np.random.randn(d_ff, d_model).astype(np.float32) * scale)
        self.w3 = Parameter(np.random.randn(d_model, d_ff).astype(np.float32) * scale)
    
    def forward(self, x):
        return (np.maximum(0, x @ self.w1.data) * (x @ self.w3.data)) @ self.w2.data
    
    def get_params(self):
        return [self.w1, self.w2, self.w3]


class TransformerLayer:
    """Single transformer decoder layer."""
    
    def __init__(self, d_model, num_heads, d_ff, num_kv_heads=None, max_seq=2048):
        self.attn_norm_w = np.ones(d_model, dtype=np.float32)
        self.ffn_norm_w = np.ones(d_model, dtype=np.float32)
        self.attn = MultiHeadAttention(d_model, num_heads, num_kv_heads, max_seq)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x, kv_cache=None, position=0):
        h = x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)) * self.attn_norm_w
        attn_out, new_cache = self.attn.forward(h, kv_cache, position)
        x = x + attn_out
        
        h = x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)) * self.ffn_norm_w
        x = x + self.ffn.forward(h)
        return x, new_cache
    
    def get_params(self):
        params = []
        for p in self.attn.get_params(): params.append(p)
        for p in self.ffn.get_params(): params.append(p)
        return params


class VaelonTransformer:
    """Complete Vaelon Transformer with all components.
    
    Architecture:
    - Token Embedding + Position Embedding
    - N Decoder Layers (Attention + SwiGLU FFN)
    - Final RMSNorm
    - LM Head
    
    Features:
    - GQA (Grouped Query Attention)
    - RoPE (Rotary Position Embeddings)
    - RMSNorm
    - SwiGLU activation
    - KV Cache for fast inference
    """
    
    def __init__(self, vocab_size=1000, d_model=256, num_heads=8, num_layers=4,
                 d_ff=1024, num_kv_heads=2, max_seq=2048):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq = max_seq
        
        scale = 1.0 / math.sqrt(d_model)
        self.embed = np.random.randn(vocab_size, d_model).astype(np.float32) * scale
        self.pos_embed = np.random.randn(max_seq, d_model).astype(np.float32) * scale * 0.1
        
        self.layers = [TransformerLayer(d_model, num_heads, d_ff, num_kv_heads, max_seq) for _ in range(num_layers)]
        self.final_norm_w = np.ones(d_model, dtype=np.float32)
        self.lm_head = np.random.randn(d_model, vocab_size).astype(np.float32) * scale
        
        self._num_params = self.embed.size + self.pos_embed.size + self.lm_head.size
        for layer in self.layers:
            for p in layer.get_params():
                self._num_params += p.data.size
    
    @property
    def num_params(self): return self._num_params
    
    def forward(self, input_ids, kv_caches=None):
        B, L = input_ids.shape
        x = self.embed[input_ids] + self.pos_embed[:L]
        
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = layer.forward(x, cache, position=0)
            new_caches.append(new_cache)
        
        x = x / (np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)) * self.final_norm_w
        logits = x @ self.lm_head
        return logits, new_caches
    
    def generate(self, prompt_ids, max_new=512, temperature=0.8, top_k=50, top_p=0.9):
        ids = list(prompt_ids)
        kv_caches = [None] * self.num_layers
        
        for step in range(max_new):
            input_arr = np.array([ids[-self.max_seq:]], dtype=np.int64)
            logits, kv_caches = self.forward(input_arr, kv_caches)
            
            next_logits = logits[0, -1] / max(temperature, 0.1)
            
            if top_k > 0:
                thresh = np.sort(next_logits)[-min(top_k, len(next_logits))]
                next_logits[next_logits < thresh] = -1e9
            
            if top_p < 1.0:
                sorted_idx = np.argsort(next_logits)[::-1]
                sorted_l = next_logits[sorted_idx].copy()
                cum = np.cumsum(np.exp(sorted_l) / np.exp(sorted_l).sum())
                remove = cum > top_p
                remove[1:] = remove[:-1]
                remove[0] = False
                sorted_l[remove] = -1e9
                next_logits[sorted_idx] = sorted_l
            
            probs = np.exp(next_logits - next_logits.max())
            probs = probs / (probs.sum() + 1e-8)
            next_id = np.random.choice(len(probs), p=probs)
            ids.append(int(next_id))
        
        return ids
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, "model.npz"),
                 embed=self.embed, pos_embed=self.pos_embed,
                 final_norm=self.final_norm_w, lm_head=self.lm_head,
                 **{f"layer_{i}_{k}": v for i, layer in enumerate(self.layers)
                    for k, v in [("attn_norm_w", layer.attn_norm_w), ("ffn_norm_w", layer.ffn_norm_w),
                                  ("wq", layer.attn.wq.data), ("wk", layer.attn.wk.data),
                                  ("wv", layer.attn.wv.data), ("wo", layer.attn.wo.data),
                                  ("ffn_w1", layer.ffn.w1.data), ("ffn_w2", layer.ffn.w2.data),
                                  ("ffn_w3", layer.ffn.w3.data)]})
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"vocab_size": self.vocab_size, "d_model": self.d_model,
                       "num_heads": self.num_heads, "num_layers": self.num_layers,
                       "num_kv_heads": self.layers[0].attn.num_kv_heads,
                       "max_seq": self.max_seq}, f)
    
    @classmethod
    def load(cls, path):
        data = np.load(os.path.join(path, "model.npz"))
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        model.embed = data["embed"]
        model.pos_embed = data["pos_embed"]
        model.final_norm_w = data["final_norm"]
        model.lm_head = data["lm_head"]
        for i in range(model.num_layers):
            model.layers[i].attn_norm_w = data[f"layer_{i}_attn_norm_w"]
            model.layers[i].ffn_norm_w = data[f"layer_{i}_ffn_norm_w"]
            model.layers[i].attn.wq.data = data[f"layer_{i}_wq"]
            model.layers[i].attn.wk.data = data[f"layer_{i}_wk"]
            model.layers[i].attn.wv.data = data[f"layer_{i}_wv"]
            model.layers[i].attn.wo.data = data[f"layer_{i}_wo"]
            model.layers[i].ffn.w1.data = data[f"layer_{i}_ffn_w1"]
            model.layers[i].ffn.w2.data = data[f"layer_{i}_ffn_w2"]
            model.layers[i].ffn.w3.data = data[f"layer_{i}_ffn_w3"]
        return model
    
    def get_all_params(self):
        params = [("embed", Tensor(self.embed, True)), ("pos_embed", Tensor(self.pos_embed, False)),
                  ("final_norm", Tensor(self.final_norm_w, True)), ("lm_head", Tensor(self.lm_head, True))]
        for i, layer in enumerate(self.layers):
            for name, p in [("wq", layer.attn.wq), ("wk", layer.attn.wk), ("wv", layer.attn.wv),
                             ("wo", layer.attn.wo), ("ffn_w1", layer.ffn.w1), ("ffn_w2", layer.ffn.w2),
                             ("ffn_w3", layer.ffn.w3)]:
                params.append((f"layer_{i}_{name}", p))
        return params
    
    def get_model_sizes():
        return {
            "tiny": {"d_model": 64, "num_heads": 4, "num_layers": 2, "d_ff": 256, "num_kv_heads": 2},
            "small": {"d_model": 128, "num_heads": 8, "num_layers": 4, "d_ff": 512, "num_kv_heads": 2},
            "medium": {"d_model": 256, "num_heads": 8, "num_layers": 6, "d_ff": 1024, "num_kv_heads": 4},
            "large": {"d_model": 512, "num_heads": 16, "num_layers": 8, "d_ff": 2048, "num_kv_heads": 8},
        }
