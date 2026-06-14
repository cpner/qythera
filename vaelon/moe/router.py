"""Expert routing for Mixture of Experts."""

import torch
import torch.nn as nn


class ExpertRouter(nn.Module):
    """Routes tokens to experts using learned gating."""

    def __init__(self, hidden_size: int, num_experts: int, noise_std: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        return logits
