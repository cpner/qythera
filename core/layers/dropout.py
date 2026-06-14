"""Dropout layer."""

import numpy as np


class Dropout:
    def __init__(self, p=0.1):
        self.p = p
    
    def forward(self, x, training=True):
        if not training or self.p == 0:
            return x
        mask = (np.random.random(x.shape) > self.p).astype(np.float32)
        return x * mask / (1 - self.p)
