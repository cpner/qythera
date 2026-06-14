import numpy as np
from core.nn.module import Module, Parameter
from core.nn.linear import Linear
from core.nn.ffn import SwiGLU
from core.autodiff.tensor import Tensor


class Expert(SwiGLU):
    """Single expert network (SwiGLU FFN)."""
    pass


class MoELayer(Module):
    """Mixture of Experts layer with top-k routing."""

    def __init__(self, dim, intermediate, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = Linear(dim, num_experts, bias=False)
        self.expert_list = [Expert(dim, intermediate) for _ in range(num_experts)]

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)

        logits = self.gate(x_flat)
        weights = logits.softmax(axis=-1)

        topk_w = Tensor(np.sort(weights.data, axis=-1)[:, -self.top_k:])
        topk_idx = Tensor(np.argsort(weights.data, axis=-1)[:, -self.top_k:])

        topk_w = topk_w / (topk_w.sum(axis=-1, keepdims=True) + 1e-8)

        out = Tensor(np.zeros_like(x_flat.data))
        aux_loss = 0.0

        for e_idx in range(self.num_experts):
            mask = (topk_idx.data == e_idx).any(axis=-1)
            if mask.any():
                expert_in = Tensor(x_flat.data[mask].copy())
                expert_out = self.expert_list[e_idx](expert_in)
                for k in range(self.top_k):
                    k_mask = (topk_idx.data[:, k] == e_idx) & mask
                    if k_mask.any():
                        w = Tensor(topk_w.data[k_mask, k:k+1].copy())
                        e_out = Tensor(expert_out.data[:k_mask.sum()].copy())
                        out.data[k_mask] = (w * e_out).data

        router_probs = weights.mean(axis=0)
        aux_loss = float(self.num_experts * (router_probs.data ** 2).sum()) * 0.01

        return out.reshape(B, L, D), aux_loss
