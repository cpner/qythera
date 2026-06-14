import numpy as np

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2, self.eps, self.wd = betas[0], betas[1], eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
    def zero_grad(self):
        for p in self.params: p.grad = None
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad.data.copy() + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.momentum = lr, momentum
        self.v = [np.zeros_like(p.data) for p in self.params]
    def zero_grad(self):
        for p in self.params: p.grad = None
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            self.v[i] = self.momentum * self.v[i] + p.grad.data
            p.data -= self.lr * self.v[i]
