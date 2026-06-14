"""Complete Vaelon Transformer with all components."""

import numpy as np
import math
import json
import os
from typing import Optional, List, Dict, Tuple
from core.layers.attention import MultiHeadAttention
from core.layers.norm import RMSNorm
from core.layers.ffn import SwiGLU
from core.layers.embedding import TokenEmbedding, PositionalEncoding
from core.layers.moe import MoELayer


class ModelConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.d_model = kwargs.get("d_model", 256)
        self.num_heads = kwargs.get("num_heads", 8)
        self.num_layers = kwargs.get("num_layers", 4)
        self.d_ff = kwargs.get("d_ff", 1024)
        self.num_kv_heads = kwargs.get("num_kv_heads", 2)
        self.max_seq = kwargs.get("max_seq", 2048)
        self.num_experts = kwargs.get("num_experts", 0)
        self.dropout = kwargs.get("dropout", 0.0)
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    @classmethod
    def tiny(cls):
        return cls(d_model=64, num_heads=4, num_layers=2, d_ff=256, num_kv_heads=2, vocab_size=1000)
    
    @classmethod
    def small(cls):
        return cls(d_model=128, num_heads=8, num_layers=4, d_ff=512, num_kv_heads=2, vocab_size=32000)
    
    @classmethod
    def medium(cls):
        return cls(d_model=256, num_heads=8, num_layers=6, d_ff=1024, num_kv_heads=4, vocab_size=32000)
    
    @classmethod
    def large(cls):
        return cls(d_model=512, num_heads=16, num_layers=8, d_ff=2048, num_kv_heads=8, vocab_size=32000)
    
    @classmethod
    def xlarge(cls):
        return cls(d_model=1024, num_heads=32, num_layers=12, d_ff=4096, num_kv_heads=8, vocab_size=32000, num_experts=8)


class TransformerLayer:
    def __init__(self, config):
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config.d_model, config.num_heads, config.num_kv_heads, config.max_seq)
        if config.num_experts > 1:
            self.ffn = MoELayer(config.d_model, config.d_ff, config.num_experts)
        else:
            self.ffn = SwiGLU(config.d_model, config.d_ff)
    
    def forward(self, x, kv_cache=None, position=0):
        h = self.attn_norm.forward(x)
        attn_out, new_cache = self.attn.forward(h, kv_cache, position)
        x = x + attn_out
        h = self.ffn_norm.forward(x)
        ffn_out = self.ffn.forward(h)
        x = x + ffn_out
        return x, new_cache


class VaelonTransformer:
    def __init__(self, config=None):
        if config is None:
            config = ModelConfig()
        self.config = config
        
        scale = 1.0 / math.sqrt(config.d_model)
        self.embed = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_embed = PositionalEncoding(config.d_model, config.max_seq)
        self.layers = [TransformerLayer(config) for _ in range(config.num_layers)]
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = np.random.randn(config.d_model, config.vocab_size).astype(np.float32) * scale
        
        self._num_params = sum(np.prod(p.shape) for layer in self.layers
                              for p in [layer.attn.wq, layer.attn.wk, layer.attn.wv, layer.attn.wo,
                                        layer.ffn.w1 if hasattr(layer.ffn, 'w1') else np.array([]),
                                        layer.ffn.w2 if hasattr(layer.ffn, 'w2') else np.array([]),
                                        layer.ffn.w3 if hasattr(layer.ffn, 'w3') else np.array([])])
        self._num_params += self.embed.weight.size + self.lm_head.size
    
    @property
    def num_params(self): return int(self._num_params)
    
    def forward(self, input_ids, kv_caches=None):
        B, L = input_ids.shape
        x = self.embed.forward(input_ids) + self.pos_embed.forward(L)
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = layer.forward(x, cache, position=0)
            new_caches.append(new_cache)
        x = self.final_norm.forward(x)
        logits = x @ self.lm_head
        return logits, new_caches
    
    def generate(self, prompt_ids, max_new=512, temperature=0.8, top_k=50, top_p=0.9):
        ids = list(prompt_ids)
        kv_caches = [None] * self.config.num_layers
        for _ in range(max_new):
            input_arr = np.array([ids[-self.config.max_seq:]], dtype=np.int64)
            logits, kv_caches = self.forward(input_arr, kv_caches)
            next_logits = logits[0, -1] / max(temperature, 0.1)
            if top_k > 0:
                thresh = np.sort(next_logits)[-min(top_k, len(next_logits))]
                next_logits[next_logits < thresh] = -1e9
            probs = np.exp(next_logits - next_logits.max())
            probs = probs / (probs.sum() + 1e-8)
            ids.append(int(np.random.choice(len(probs), p=probs)))
        return ids
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, "model.npz"), embed=self.embed.weight, lm_head=self.lm_head,
                 **{f"L{i}_{k}": v for i, l in enumerate(self.layers)
                    for k, v in [("norm1", l.attn_norm.weight), ("norm2", l.ffn_norm.weight),
                                  ("wq", l.attn.wq), ("wk", l.attn.wk), ("wv", l.attn.wv), ("wo", l.attn.wo),
                                  ("w1", l.ffn.w1), ("w2", l.ffn.w2), ("w3", l.ffn.w3)]})
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f)
    
    @classmethod
    def load(cls, path):
        data = np.load(os.path.join(path, "model.npz"))
        with open(os.path.join(path, "config.json")) as f:
            config = ModelConfig.from_dict(json.load(f))
        model = cls(config)
        model.embed.weight = data["embed"]
        model.lm_head = data["lm_head"]
        for i in range(config.num_layers):
            model.layers[i].attn_norm.weight = data[f"L{i}_norm1"]
            model.layers[i].ffn_norm.weight = data[f"L{i}_norm2"]
            model.layers[i].attn.wq = data[f"L{i}_wq"]
            model.layers[i].attn.wk = data[f"L{i}_wk"]
            model.layers[i].attn.wv = data[f"L{i}_wv"]
            model.layers[i].attn.wo = data[f"L{i}_wo"]
            model.layers[i].ffn.w1 = data[f"L{i}_w1"]
            model.layers[i].ffn.w2 = data[f"L{i}_w2"]
            model.layers[i].ffn.w3 = data[f"L{i}_w3"]
        return model
