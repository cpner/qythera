import numpy as np
from core.nn.module import Module, Parameter
from core.autodiff.tensor import Tensor


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(dim, dtype=np.float32))
        self.eps = eps

    def forward(self, x):
        return x.rmsnorm(eps=self.eps) * self.weight


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = Parameter(np.ones(dim, dtype=np.float32))
        self.bias = Parameter(np.zeros(dim, dtype=np.float32))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_norm = (x - mean) / (var + self.eps).pow(0.5)
        return x_norm * self.weight + self.bias
