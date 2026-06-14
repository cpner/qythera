"""Learning rate schedulers."""

import math


class CosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        self.optimizer.lr = lr
        return lr


class LinearScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.base_lr * (1 - progress) + self.min_lr * progress
        self.optimizer.lr = lr
        return lr


class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        lr = self.base_lr * min(1.0, self.step_count / self.warmup_steps)
        self.optimizer.lr = lr
        return lr
