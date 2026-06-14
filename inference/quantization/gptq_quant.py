import torch
import torch.nn as nn


class GPTQQuantizer:
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size

    def quantize_weight(self, weight: torch.Tensor) -> dict:
        rows, cols = weight.shape
        scaled_weight = weight.clone()
        for col in range(0, cols, self.group_size):
            end = min(col + self.group_size, cols)
            group = scaled_weight[:, col:end]
            max_val = group.abs().amax(dim=-1, keepdim=True)
            scale = max_val / (2 ** (self.bits - 1) - 1)
            scale = torch.clamp(scale, min=1e-6)
            quantized = torch.clamp(torch.round(group / scale), -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
            scaled_weight[:, col:end] = quantized
            dequant = quantized * scale
            scaled_weight[:, col:end] = dequant
        return {"weight": scaled_weight}

    def quantize_model(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                q = self.quantize_weight(module.weight.data)
                module.weight = nn.Parameter(q["weight"].float(), requires_grad=False)
        return model
