"""SwiGLU Feed-Forward Network."""

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """SwiGLU: Swish-gated Linear Unit.

    SwiGLU(x) = (Swish(xW_gate) * (xW_up)) W_down
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
