import numpy as np
from core.nn.module import Module, Parameter
from core.autodiff.tensor import Tensor


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        std = 1.0 / np.sqrt(in_features)
        self.weight = Parameter(np.random.randn(in_features, out_features).astype(np.float32) * std)
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
