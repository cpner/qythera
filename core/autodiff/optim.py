"""Optimizers for gradient descent: SGD, Adam, AdamW."""

import numpy as np
from typing import List, Optional
from core.autodiff.tensor import Tensor


class Optimizer:
    """Base optimizer class."""
    
    def __init__(self, params: List[Tensor], lr: float = 0.001):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum."""
    
    def __init__(self, params: List[Tensor], lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data.copy()
            if self.weight_decay > 0:
                g = g + self.weight_decay * p.data
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + g
                g = self.velocities[i]
            p.data = p.data - self.lr * g


class Adam(Optimizer):
    """Adam optimizer: adaptive learning rates with momentum."""
    
    def __init__(self, params: List[Tensor], lr=0.001, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data.copy()
            if self.weight_decay > 0:
                g = g + self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """AdamW optimizer: Adam with decoupled weight decay."""
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data.copy()
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0:
                p.data = p.data - self.lr * self.weight_decay * p.data
