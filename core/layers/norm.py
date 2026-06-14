"""RMSNorm normalization layer."""

import numpy as np


class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        self.weight = np.ones(dim, dtype=np.float32)
        self.eps = eps
    
    def forward(self, x):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight
