"""Expert pool for Mixture of Experts."""

import torch
import torch.nn as nn
from vaelon.layers.ffn import SwiGLU


class ExpertPool(nn.Module):
    """Pool of expert FFN modules."""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int, bias: bool = False):
        super().__init__()
        self.experts = nn.ModuleList([
            SwiGLU(hidden_size, intermediate_size, bias=bias)
            for _ in range(num_experts)
        ])
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor,
                expert_weights: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(x)
        num_experts_per_tok = expert_indices.shape[1]
        for e_idx in range(self.num_experts):
            mask = (expert_indices == e_idx).any(dim=-1)
            if mask.any():
                expert_output = self.experts[e_idx](x[mask])
                for k in range(num_experts_per_tok):
                    k_mask = (expert_indices[:, k] == e_idx) & mask
                    if k_mask.any():
                        output[k_mask] += expert_weights[k_mask, k].unsqueeze(-1) * expert_output[:k_mask.sum()]
        return output
