"""Tensor with automatic differentiation."""

import numpy as np


class Tensor:
    __array_priority__ = 10000
    
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, Tensor): data = data.data
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = list(_children)
        self._op = _op
    
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    @property
    def size(self): return self.data.size
    def item(self): return float(self.data.flat[0])
    def numpy(self): return self.data.copy()
    def __repr__(self): return f"Tensor({self.data.shape})"
    def __len__(self): return len(self.data)
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        visited = set()
        order = []
        def build(node):
            if id(node) in visited: return
            visited.add(id(node))
            for c in node._prev: build(c)
            order.append(node)
        build(self)
        for node in reversed(order):
            node._backward()
    
    def zero_grad(self): self.grad = None
    
    def _sum_to(self, grad, shape):
        while grad.ndim > len(shape): grad = grad.sum(axis=0)
        for i in range(grad.ndim):
            if i < len(shape) and grad.shape[i] != shape[i]:
                grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(shape) if grad.shape != shape else grad
    
    def _get_grad(self, out):
        return out.grad if out.grad is not None else np.zeros_like(out.data)
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        rg = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, requires_grad=rg, _children=(self, other), _op="add")
        def _backward():
            g = self._get_grad(out)
            if self.requires_grad: self.grad = self._sum_to(g, self.data.shape) if self.grad is None else self.grad + self._sum_to(g, self.data.shape)
            if other.requires_grad: other.grad = self._sum_to(g, other.data.shape) if other.grad is None else other.grad + self._sum_to(g, other.data.shape)
        out._backward = _backward
        return out
    def __radd__(self, other): return self.__add__(other)
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        rg = self.requires_grad or other.requires_grad
        out = Tensor(self.data - other.data, requires_grad=rg, _children=(self, other), _op="sub")
        def _backward():
            g = self._get_grad(out)
            if self.requires_grad: self.grad = self._sum_to(g, self.data.shape) if self.grad is None else self.grad + self._sum_to(g, self.data.shape)
            if other.requires_grad: other.grad = self._sum_to(-g, other.data.shape) if other.grad is None else other.grad + self._sum_to(-g, other.data.shape)
        out._backward = _backward
        return out
    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        return other.__sub__(self)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        rg = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, requires_grad=rg, _children=(self, other), _op="mul")
        def _backward():
            g = self._get_grad(out)
            if self.requires_grad: self.grad = self._sum_to(g * other.data, self.data.shape) if self.grad is None else self.grad + self._sum_to(g * other.data, self.data.shape)
            if other.requires_grad: other.grad = self._sum_to(g * self.data, other.data.shape) if other.grad is None else other.grad + self._sum_to(g * self.data, other.data.shape)
        out._backward = _backward
        return out
    def __rmul__(self, other): return self.__mul__(other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out = Tensor(self.data / (other.data + 1e-8), requires_grad=self.requires_grad, _children=(self, other), _op="div")
        def _backward():
            g = self._get_grad(out)
            if self.requires_grad: self.grad = self._sum_to(g / (other.data + 1e-8), self.data.shape) if self.grad is None else self.grad + self._sum_to(g / (other.data + 1e-8), self.data.shape)
        out._backward = _backward
        return out
    
    def __pow__(self, exp):
        out = Tensor(self.data ** exp, requires_grad=self.requires_grad, _children=(self,), _op="pow")
        def _backward():
            if self.requires_grad:
                g = exp * (self.data ** (exp - 1)) * self._get_grad(out)
                self.grad = self._sum_to(g, self.data.shape) if self.grad is None else self.grad + self._sum_to(g, self.data.shape)
        out._backward = _backward
        return out
    
    def __neg__(self): return self * (-1)
    
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        rg = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, requires_grad=rg, _children=(self, other), _op="matmul")
        def _backward():
            g = self._get_grad(out)
            if self.requires_grad: self.grad = self._sum_to(g @ other.data.T, self.data.shape) if self.grad is None else self.grad + self._sum_to(g @ other.data.T, self.data.shape)
            if other.requires_grad: other.grad = self._sum_to(self.data.T @ g, other.data.shape) if other.grad is None else other.grad + self._sum_to(self.data.T @ g, other.data.shape)
        out._backward = _backward
        return out
    def __matmul__(self, other): return self.matmul(other)
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,), _op="relu")
        def _backward():
            if self.requires_grad:
                g = self._get_grad(out) * (self.data > 0).astype(np.float32)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def gelu(self):
        x = self.data
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        out = Tensor(x * cdf, requires_grad=self.requires_grad, _children=(self,), _op="gelu")
        def _backward():
            if self.requires_grad:
                d = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))
                d2 = 0.5 * (1 - np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3))**2) * np.sqrt(2/np.pi) * (1 + 3*0.044715*x**2)
                g = self._get_grad(out) * (d + x * d2)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def sigmoid(self):
        s = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")
        def _backward():
            if self.requires_grad:
                g = self._get_grad(out) * s * (1 - s)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def softmax(self, axis=-1):
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True))
        s = e / (e.sum(axis=axis, keepdims=True) + 1e-8)
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="softmax")
        def _backward():
            if self.requires_grad:
                g = self._get_grad(out)
                dot = (g * s).sum(axis=axis, keepdims=True)
                grad = s * (g - dot)
                self.grad = grad if self.grad is None else self.grad + grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data + 1e-7), requires_grad=self.requires_grad, _children=(self,), _op="log")
        def _backward():
            if self.requires_grad:
                g = self._get_grad(out) / (self.data + 1e-7)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad, _children=(self,), _op="mean")
        def _backward():
            if self.requires_grad:
                g = np.broadcast_to(self._get_grad(out) / n, self.data.shape).copy()
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad, _children=(self,), _op="sum")
        def _backward():
            if self.requires_grad:
                g = np.broadcast_to(self._get_grad(out), self.data.shape).copy()
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op="tanh")
        def _backward():
            if self.requires_grad:
                g = self._get_grad(out) * (1 - t ** 2)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def sqrt(self):
        out = Tensor(np.sqrt(self.data + 1e-7), requires_grad=self.requires_grad, _children=(self,), _op="sqrt")
        def _backward():
            if self.requires_grad:
                g = self._get_grad(out) / (2 * out.data + 1e-7)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    def transpose(self, *axes):
        axes = axes if axes else tuple(range(self.ndim - 1, -1, -1))
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad, _children=(self,), _op="transpose")
        def _backward():
            if self.requires_grad:
                g = self._get_grad(out).transpose(axes)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out
    
    @property
    def T(self): return self.transpose()
    def reshape(self, *shape): return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
    def clip(self, a, b): return Tensor(np.clip(self.data, a, b), requires_grad=self.requires_grad)
    def float(self): return Tensor(self.data.astype(np.float32), requires_grad=self.requires_grad)
    def half(self): return Tensor(self.data.astype(np.float16), requires_grad=self.requires_grad)
    
    @staticmethod
    def zeros(*shape, requires_grad=False): return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)
    @staticmethod
    def ones(*shape, requires_grad=False): return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)
    @staticmethod
    def randn(*shape, requires_grad=False, std=0.02): return Tensor(np.random.randn(*shape).astype(np.float32) * std, requires_grad=requires_grad)
