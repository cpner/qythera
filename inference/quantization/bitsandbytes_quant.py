import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


class BnBQuantizer:
    def __init__(self, bits: int = 4):
        self.bits = bits

    def quantize_linear(self, module: nn.Linear) -> nn.Linear:
        if not HAS_BNB:
            print("bitsandbytes not installed, skipping quantization")
            return module
        if self.bits == 4:
            return bnb.nn.Linear4bit(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
            )
        elif self.bits == 8:
            return bnb.nn.Linear8bitLt(
                module.in_features, module.out_features,
                bias=module.bias is not None,
            )
        return module

    def quantize_model(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quantized = self.quantize_linear(module)
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                    setattr(parent, attr_name, quantized)
                else:
                    setattr(model, attr_name, quantized)
        return model
