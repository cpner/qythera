import math
"""Feed-Forward Networks: SwiGLU and GeGLU."""

import numpy as np


class SwiGLU:
    def __init__(self, d_model, d_ff):
        scale = 1.0 / math.sqrt(d_model)
        self.w1 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.w2 = np.random.randn(d_ff, d_model).astype(np.float32) * scale
        self.w3 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
    
    def forward(self, x):
        return (np.maximum(0, x @ self.w1) * (x @ self.w3)) @ self.w2


class GeGLU:
    def __init__(self, d_model, d_ff):
        scale = 1.0 / math.sqrt(d_model)
        self.w1 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.w2 = np.random.randn(d_ff, d_model).astype(np.float32) * scale
        self.w3 = np.random.randn(d_model, d_ff).astype(np.float32) * scale
    
    def forward(self, x):
        return (0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x @ self.w1 + 0.044715 * (x @ self.w1)**3))) * (x @ self.w3)) @ self.w2
