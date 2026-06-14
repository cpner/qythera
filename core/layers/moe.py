import math
"""Mixture of Experts layer."""

import numpy as np


class Expert:
    def __init__(self, d_model, d_ff):
        scale = 1.0 / math.sqrt(d_model)
        self.w1 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.w2 = np.random.randn(d_ff, d_model).astype(np.float32) * scale
        self.w3 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
    
    def forward(self, x):
        return (np.maximum(0, x @ self.w1) * (x @ self.w3)) @ self.w2


class MoELayer:
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = np.random.randn(d_model, num_experts).astype(np.float32) * 0.02
        self.experts = [Expert(d_model, d_ff) for _ in range(num_experts)]
    
    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        logits = x_flat @ self.gate
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-8)
        
        topk_indices = np.argsort(probs, axis=-1)[:, -self.top_k:]
        topk_weights = np.take_along_axis(probs, topk_indices, axis=-1)
        topk_weights = topk_weights / (topk_weights.sum(axis=-1, keepdims=True) + 1e-8)
        
        out = np.zeros_like(x_flat)
        for e_idx in range(self.num_experts):
            mask = (topk_indices == e_idx).any(axis=-1)
            if mask.any():
                expert_out = self.experts[e_idx].forward(x_flat[mask])
                for k in range(self.top_k):
                    k_mask = (topk_indices[:, k] == e_idx) & mask
                    if k_mask.any():
                        out[k_mask] += topk_weights[k_mask, k:k+1] * expert_out[:k_mask.sum()]
        
        return out.reshape(B, L, D)
