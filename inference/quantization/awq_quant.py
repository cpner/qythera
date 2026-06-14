import torch
import torch.nn as nn
from typing import Optional


class AWQQuantizer:
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size

    def quantize_layer(self, weight: torch.Tensor) -> dict:
        scales = weight.abs().amax(dim=-1, keepdim=True) / (2 ** (self.bits - 1) - 1)
        quantized = torch.clamp(torch.round(weight / scales), -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
        return {"weight": quantized, "scale": scales}

    def quantize_model(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                q = self.quantize_layer(module.weight.data)
                module.weight = nn.Parameter(q["weight"].float(), requires_grad=False)
        return model

    def dequantize(self, quantized_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return quantized_weight * scale
