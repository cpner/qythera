"""Weight quantization utilities for Vaelon model."""

import torch
import torch.nn as nn


def quantize_weight_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to INT8 with per-channel scaling."""
    abs_max = weight.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / 127.0
    quantized = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
    return quantized, scale


def quantize_weight_int4(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weight tensor to INT4 with group-wise scaling."""
    original_shape = weight.shape
    if weight.dim() == 2:
        rows, cols = weight.shape
        padded_cols = ((cols + group_size - 1) // group_size) * group_size
        if padded_cols != cols:
            weight = torch.nn.functional.pad(weight, (0, padded_cols - cols))
        weight_grouped = weight.view(rows, -1, group_size)
        abs_max = weight_grouped.abs().amax(dim=-1, keepdim=True)
        scale = abs_max / 7.0
        quantized = torch.clamp(torch.round(weight_grouped / scale), -8, 7).to(torch.int8)
        return quantized, scale.squeeze(-1), torch.tensor(original_shape)
    abs_max = weight.abs().amax()
    scale = abs_max / 7.0
    quantized = torch.clamp(torch.round(weight / scale), -8, 7).to(torch.int8)
    return quantized, scale, torch.tensor(original_shape)


def quantize_model(model: nn.Module, bits: int = 8, modules_to_quantize: list = None):
    """Apply weight-only quantization to model linear layers."""
    target_classes = {nn.Linear}
    if modules_to_quantize:
        target_classes = set(modules_to_quantize)
    quantized_count = 0
    for name, module in model.named_modules():
        if type(module) in target_classes:
            if bits == 8:
                q_weight, scale = quantize_weight_int8(module.weight.data)
                module.weight = nn.Parameter(q_weight.float() * scale, requires_grad=False)
            elif bits == 4:
                q_weight, scale, shape = quantize_weight_int4(module.weight.data)
                module.weight = nn.Parameter(q_weight.float(), requires_grad=False)
            quantized_count += 1
    print(f"Quantized {quantized_count} modules to INT{bits}")
    return model
