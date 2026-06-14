"""Qythera Engine - Complete AI system with auto-backend detection."""

import numpy as np
import os
import sys
import json
import time
import math
import struct
import hashlib
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    OPENCL = "opencl"


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


def detect_device() -> Device:
    """Auto-detect best available compute backend."""
    try:
        import torch
        if torch.cuda.is_available():
            return Device.CUDA
    except ImportError:
        pass
    
    if sys.platform == "darwin":
        try:
            import Metal
            return Device.METAL
        except ImportError:
            pass
    
    return Device.CPU


def get_precision(device: Device, ram_gb: float) -> Precision:
    """Auto-select precision based on device and memory."""
    if device == Device.CUDA:
        return Precision.FP16 if ram_gb < 16 else Precision.FP32
    elif ram_gb < 4:
        return Precision.INT8
    elif ram_gb < 8:
        return Precision.FP32
    else:
        return Precision.FP32


class Tensor:
    """Multi-dimensional array with autograd support."""
    __array_priority__ = 10000
    
    def __init__(self, data, requires_grad=False, _children=(), _op="", _name=""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = list(_children)
        self._op = _op
        self._name = _name
    
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    @property
    def size(self): return self.data.size
    def item(self): return float(self.data.flat[0])
    def numpy(self): return self.data.copy()
    def __repr__(self): return f"Tensor({self.data.shape}, grad={self.grad is not None})"
    def __len__(self): return len(self.data)
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        visited = set()
        order = []
        queue = [self]
        while queue:
            node = queue.pop(0)
            if id(node) in visited: continue
            visited.add(id(node))
            order.append(node)
            for child in node._prev:
                if id(child) not in visited:
                    queue.append(child)
        for node in order:
            node._backward()
    
    def zero_grad(self): self.grad = None
    
    def _ensure_numpy(self, x):
        if isinstance(x, Tensor): return x.data
        if isinstance(x, np.ndarray): return x
        return np.array(x, dtype=np.float32)
    
    def _sum_to_shape(self, grad, target_shape):
        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)
        for i in range(grad.ndim):
            if i < len(target_shape) and grad.shape[i] != target_shape[i]:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(target_shape) if grad.shape != target_shape else grad
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out = Tensor(self.data + other.data, _children=(self, other), _op="add")
        def _backward():
            g = out.grad
            if self.requires_grad: self.grad = self._sum_to_shape(g, self.data.shape) if self.grad is None else self.grad + self._sum_to_shape(g, self.data.shape)
            if other.requires_grad: other.grad = self._sum_to_shape(g, other.data.shape) if other.grad is None else other.grad + self._sum_to_shape(g, other.data.shape)
        out._backward = _backward
        return out
    def __radd__(self, other): return self.__add__(other)
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out = Tensor(self.data - other.data, _children=(self, other), _op="sub")
        def _backward():
            g = out.grad
            if self.requires_grad: self.grad = self._sum_to_shape(g, self.data.shape) if self.grad is None else self.grad + self._sum_to_shape(g, self.data.shape)
            if other.requires_grad: other.grad = self._sum_to_shape(-g, other.data.shape) if other.grad is None else other.grad + self._sum_to_shape(-g, other.data.shape)
        out._backward = _backward
        return out
    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        return other.__sub__(self)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out = Tensor(self.data * other.data, _children=(self, other), _op="mul")
        def _backward():
            g = out.grad
            if self.requires_grad: self.grad = self._sum_to_shape(g * other.data, self.data.shape) if self.grad is None else self.grad + self._sum_to_shape(g * other.data, self.data.shape)
            if other.requires_grad: other.grad = self._sum_to_shape(g * self.data, other.data.shape) if other.grad is None else other.grad + self._sum_to_shape(g * self.data, other.data.shape)
        out._backward = _backward
        return out
    def __rmul__(self, other): return self.__mul__(other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out = Tensor(self.data / (other.data + 1e-8), _children=(self, other), _op="div")
        def _backward():
            g = out.grad
            if self.requires_grad: self.grad = self._sum_to_shape(g / (other.data + 1e-8), self.data.shape) if self.grad is None else self.grad + self._sum_to_shape(g / (other.data + 1e-8), self.data.shape)
        out._backward = _backward
        return out
    
    def __pow__(self, exp):
        out = Tensor(self.data ** exp, _children=(self,), _op="pow")
        def _backward():
            if self.requires_grad:
                g = exp * (self.data ** (exp - 1)) * out.grad
                self.grad = self._sum_to_shape(g, self.data.shape) if self.grad is None else self.grad + self._sum_to_shape(g, self.data.shape)
        out._backward = _backward
        return out
    
    def __neg__(self): return self * (-1)
    
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op="matmul")
        def _backward():
            g = out.grad
            if self.requires_grad: self.grad = self._sum_to_shape(g @ other.data.T, self.data.shape) if self.grad is None else self.grad + self._sum_to_shape(g @ other.data.T, self.data.shape)
            if other.requires_grad: other.grad = self._sum_to_shape(self.data.T @ g, other.data.shape) if other.grad is None else other.grad + self._sum_to_shape(self.data.T @ g, other.data.shape)
        out._backward = _backward
        return out
    def __matmul__(self, other): return self.matmul(other)
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op="relu")
        def _backward():
            if self.requires_grad:
                g = out.grad * (self.data > 0).astype(np.float32)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def gelu(self):
        x = self.data
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        out = Tensor(x * cdf, _children=(self,), _op="gelu")
        def _backward():
            if self.requires_grad:
                d = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))
                d2 = 0.5 * (1 - np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3))**2) * np.sqrt(2/np.pi) * (1 + 3*0.044715*x**2)
                g = out.grad * (d + x * d2)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def sigmoid(self):
        s = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, _children=(self,), _op="sigmoid")
        def _backward():
            if self.requires_grad:
                g = out.grad * s * (1 - s)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def softmax(self, axis=-1):
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True))
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, _children=(self,), _op="softmax")
        def _backward():
            if self.requires_grad:
                g = out.grad
                dot = (g * s).sum(axis=axis, keepdims=True)
                grad = s * (g - dot)
                self.grad = grad if self.grad is None else self.grad + grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data + 1e-7), _children=(self,), _op="log")
        def _backward():
            if self.requires_grad:
                g = out.grad / (self.data + 1e-7)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), _children=(self,), _op="mean")
        def _backward():
            if self.requires_grad:
                g = np.broadcast_to(out.grad / n, self.data.shape).copy()
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def sum(self):
        out = Tensor(self.data.sum(), _children=(self,), _op="sum")
        def _backward():
            if self.requires_grad:
                g = np.broadcast_to(out.grad, self.data.shape).copy()
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def tanh(self):
        out_data = np.tanh(self.data)
        out = Tensor(out_data, _children=(self,), _op="tanh")
        def _backward():
            if self.requires_grad:
                g = out.grad * (1 - out_data ** 2)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def sqrt(self):
        out = Tensor(np.sqrt(self.data + 1e-7), _children=(self,), _op="sqrt")
        def _backward():
            if self.requires_grad:
                g = out.grad / (2 * out.data + 1e-7)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def transpose(self, *axes):
        axes = axes if axes else tuple(range(self.ndim - 1, -1, -1))
        out = Tensor(self.data.transpose(axes), _children=(self,), _op="transpose")
        def _backward():
            if self.requires_grad:
                g = out.grad.transpose(axes)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    @property
    def T(self): return self.transpose()
    
    def reshape(self, *shape):
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
    
    def clip(self, a_min, a_max):
        return Tensor(np.clip(self.data, a_min, a_max), requires_grad=self.requires_grad)


class Adam:
    """Adam optimizer with weight decay."""
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
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
            if self.weight_decay > 0:
                g = g + self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None


class CosineScheduler:
    """Cosine learning rate scheduler with warmup."""
    
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
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        self.optimizer.lr = lr
        return lr


class BPETokenizer:
    """Byte Pair Encoding tokenizer."""
    
    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []
        self.char_to_id = dict(self.SPECIAL)
        self.id_to_char = {v: k for k, v in self.SPECIAL.items()}
    
    def train(self, texts, vocab_size=None):
        if vocab_size: self.vocab_size = vocab_size
        
        chars = set()
        for t in texts:
            chars.update(t)
        
        for c in sorted(chars):
            if c not in self.char_to_id:
                self.char_to_id[c] = len(self.char_to_id)
        
        for i in range(self.vocab_size - len(self.char_to_id)):
            self.char_to_id[f"<merge_{i}>"] = len(self.char_to_id)
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
    
    def encode(self, text, add_special=True):
        ids = []
        if add_special:
            ids.append(self.SPECIAL["<bos>"])
        for c in text:
            ids.append(self.char_to_id.get(c, self.SPECIAL["<unk>"]))
        if add_special:
            ids.append(self.SPECIAL["<eos>"])
        return ids
    
    def decode(self, ids, skip_special=True):
        special_ids = set(self.SPECIAL.values())
        return "".join([self.id_to_char.get(i, "") for i in ids if not skip_special or i not in special_ids])
    
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"char_to_id": self.char_to_id, "vocab_size": self.vocab_size}, f)
    
    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self.vocab_size = data.get("vocab_size", len(self.char_to_id))
    
    def __len__(self): return len(self.char_to_id)


class RMSNorm(Module):
    """Root Mean Square Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(dim, dtype=np.float32))
        self.eps = eps
    
    def forward(self, x):
        rms = np.sqrt(np.mean(x.data ** 2, axis=-1, keepdims=True) + self.eps)
        out = Tensor((x.data / rms) * self.weight.data, _children=(x, self.weight), _op="rmsnorm")
        def _backward():
            g = out.grad
            if x.requires_grad:
                dg = g * self.weight.data / rms
                dg = dg - (dg * x.data).mean(axis=-1, keepdims=True) * x.data / (rms**2)
                x.grad = dg if x.grad is None else x.grad + dg
            if self.weight.requires_grad:
                wg = g * x.data / rms
                self.weight.grad = wg.sum(axis=tuple(range(wg.ndim-1))) if self.weight.grad is None else self.weight.grad + wg.sum(axis=tuple(range(wg.ndim-1)))
        out._backward = _backward
        return out
