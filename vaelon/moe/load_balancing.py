"""Load balancing loss for Mixture of Experts."""

import torch


def load_balancing_loss(router_logits: torch.Tensor, expert_indices: torch.Tensor,
                        num_experts: int) -> torch.Tensor:
    """Compute auxiliary load balancing loss."""
    routing_weights = torch.softmax(router_logits, dim=-1)
    expert_mask = torch.zeros_like(routing_weights)
    expert_mask.scatter_(1, expert_indices, 1.0)
    tokens_per_expert = expert_mask.float().mean(dim=0)
    router_prob_per_expert = routing_weights.mean(dim=0)
    return num_experts * (tokens_per_expert * router_prob_per_expert).sum()
