"""Tensor with automatic differentiation support.

Core building block for all neural network computations.
Supports: add, sub, mul, div, matmul, activations, reductions.
All operations track computation history for backward pass.
"""

import numpy as np
from typing import Optional, Tuple, List


class Tensor:
    """Multi-dimensional tensor with autograd support."""
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
    @property
    def dtype(self): return self.data.dtype
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
        s = e / (e.sum(axis=axis, keepdims=True) + 1e-8)
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
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,), _op="tanh")
        def _backward():
            if self.requires_grad:
                g = out.grad * (1 - t ** 2)
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
    
    def float(self):
        return Tensor(self.data.astype(np.float32), requires_grad=self.requires_grad)
    
    def half(self):
        return Tensor(self.data.astype(np.float16), requires_grad=self.requires_grad)
    
    def to(self, dtype):
        if dtype == "fp32": return self.float()
        elif dtype == "fp16": return self.half()
        return self
    
    @staticmethod
    def zeros(*shape, requires_grad=False):
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)
    
    @staticmethod
    def ones(*shape, requires_grad=False):
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)
    
    @staticmethod
    def randn(*shape, requires_grad=False, std=0.02):
        return Tensor(np.random.randn(*shape).astype(np.float32) * std, requires_grad=requires_grad)
    
    @staticmethod
    def cat(tensors, dim=0):
        return Tensor(np.concatenate([t.data for t in tensors], axis=dim))
