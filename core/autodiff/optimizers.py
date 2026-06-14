"""Optimizers: Adam, AdamW, SGD, Lion."""

import numpy as np
from typing import List


class Optimizer:
    def __init__(self, params, lr=0.001):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None
    
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in self.params]
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad.copy()
            if self.weight_decay > 0: g += self.weight_decay * p.data
            self.velocities[i] = self.momentum * self.velocities[i] + g
            p.data -= self.lr * self.velocities[i]


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
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
            if p.grad is None: continue
            g = p.grad.copy()
            if self.weight_decay > 0: g += self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


class AdamW(Adam):
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad.copy()
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad.copy()
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            update = torch.sign(self.m[i]) if hasattr(torch, 'sign') else np.sign(self.m[i])
            p.data -= self.lr * (update + self.weight_decay * p.data)
