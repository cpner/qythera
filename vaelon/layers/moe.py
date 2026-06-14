"""Mixture of Experts layer."""

import torch
import torch.nn as nn
from vaelon.moe.router import ExpertRouter
from vaelon.moe.experts import ExpertPool
from vaelon.moe.load_balancing import load_balancing_loss


class VaelonMoE(nn.Module):
    """Mixture of Experts feed-forward layer."""

    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int = 8, num_experts_per_tok: int = 2, bias: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router = ExpertRouter(hidden_size, num_experts)
        self.experts = ExpertPool(hidden_size, intermediate_size, num_experts, bias=bias)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        router_logits = self.router(x_flat)
        expert_weights, expert_indices = torch.topk(
            router_logits.softmax(dim=-1), self.num_experts_per_tok, dim=-1
        )
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        y = self.experts(x_flat, expert_indices, expert_weights)
        y = self.norm(y.view(batch_size, seq_len, hidden))
        aux_loss = load_balancing_loss(router_logits, expert_indices, self.num_experts)
        return y, {"aux_loss": aux_loss, "router_logits": router_logits}
