"""Custom Tensor with automatic differentiation support.

All computations are tracked for backward pass.
Uses numpy for efficient numerical operations.
"""

import numpy as np
from typing import Optional, Tuple, List
import weakref


class Tensor:
    __array_priority__ = 10000
    """Multi-dimensional tensor with automatic differentiation.
    
    Tracks computation history for gradient computation via reverse-mode AD.
    """
    
    def __init__(self, data, requires_grad=False, _children=(), _op="", name=""):
        if isinstance(data, Tensor):
            data = data.data
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = list(_children)
        self._op = _op
        self.name = name
        self._ctx = None

    @classmethod
    def zeros(cls, *shape, requires_grad=False):
        return cls(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, requires_grad=False):
        return cls(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape, requires_grad=False, std=0.02):
        return cls(np.random.randn(*shape).astype(np.float32) * std, requires_grad=requires_grad)

    @classmethod
    def randint(cls, low, high, shape, requires_grad=False):
        return cls(np.random.randint(low, high, shape).astype(np.float32), requires_grad=requires_grad)

    @property
    def shape(self): return self.data.shape

    @property
    def ndim(self): return self.data.ndim

    @property
    def size(self): return self.data.size

    @property
    def dtype(self): return self.data.dtype

    def numpy(self): return self.data.copy()

    def item(self): return float(self.data.flat[0])

    def clone(self):
        out = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        return out

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Tensor({self.data.shape}, grad={self.grad is not None}, op={self._op})"

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        return Tensor(self.data[idx], requires_grad=self.requires_grad)

    # ===== Arithmetic Operations =====

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out_data = self.data + other.data
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="add")
        
        def _backward():
            if self.requires_grad:
                g = out.grad
                while g.ndim > self.data.ndim:
                    g = g.sum(axis=0)
                for i in range(g.ndim - self.data.ndim):
                    g = g.sum(axis=0)
                self.grad = (self.grad + g) if self.grad is not None else g
            if other.requires_grad:
                g = out.grad
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                for i in range(g.ndim - other.data.ndim):
                    g = g.sum(axis=0)
                other.grad = (other.grad + g) if other.grad is not None else g
        out._backward = _backward
        return out

    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out_data = self.data - other.data
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="sub")
        def _backward():
            if self.requires_grad:
                g = out.grad
                while g.ndim > self.data.ndim: g = g.sum(axis=0)
                self.grad = (self.grad + g) if self.grad is not None else g
            if other.requires_grad:
                g = -out.grad
                while g.ndim > other.data.ndim: g = g.sum(axis=0)
                other.grad = (other.grad + g) if other.grad is not None else g
        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        return other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out_data = self.data * other.data
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="mul")
        def _backward():
            if self.requires_grad:
                g = out.grad * other.data
                while g.ndim > self.data.ndim: g = g.sum(axis=0)
                self.grad = (self.grad + g) if self.grad is not None else g
            if other.requires_grad:
                g = out.grad * self.data
                while g.ndim > other.data.ndim: g = g.sum(axis=0)
                other.grad = (other.grad + g) if other.grad is not None else g
        out._backward = _backward
        return out

    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out_data = self.data / other.data
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="div")
        def _backward():
            if self.requires_grad:
                g = out.grad / other.data
                self.grad = (self.grad + g) if self.grad is not None else g
            if other.requires_grad:
                g = -(out.grad * self.data) / (other.data ** 2)
                other.grad = (other.grad + g) if other.grad is not None else g
        out._backward = _backward
        return out

    def __pow__(self, exp):
        assert isinstance(exp, (int, float))
        out_data = self.data ** exp
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="pow")
        def _backward():
            if self.requires_grad:
                g = exp * (self.data ** (exp - 1)) * out.grad
                self.grad = (self.grad + g) if self.grad is not None else g
        out._backward = _backward
        return out

    def __neg__(self):
        return self * (-1)

    def __abs__(self):
        return Tensor(np.abs(self.data), requires_grad=self.requires_grad)

    # ===== Matrix Operations =====

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data @ other.data
        out = Tensor(out_data, requires_grad=(self.requires_grad or other.requires_grad),
                      _children=(self, other), _op="matmul")
        def _backward():
            if self.requires_grad:
                g = out.grad @ other.data.T
                self.grad = (self.grad + g) if self.grad is not None else g
            if other.requires_grad:
                g = self.data.T @ out.grad
                other.grad = (other.grad + g) if other.grad is not None else g
        out._backward = _backward
        return out

    def __matmul__(self, other): return self.matmul(other)

    def transpose(self, *axes):
        axes = axes if axes else tuple(range(self.ndim - 1, -1, -1))
        out_data = self.data.transpose(axes)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="transpose")
        def _backward():
            if self.requires_grad:
                g = out.grad.transpose(axes)
                self.grad = (self.grad + g) if self.grad is not None else g
        out._backward = _backward
        return out

    @property
    def T(self): return self.transpose()

    # ===== Reduction Operations =====

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="sum")
        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                g = np.broadcast_to(g, self.data.shape)
                self.grad = (self.grad + g) if self.grad is not None else g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="max")
        def _backward():
            if self.requires_grad:
                mask = (self.data == out.data)
                g = out.grad * mask / mask.sum(axis=axis, keepdims=True)
                self.grad = (self.grad + g) if self.grad is not None else g
        out._backward = _backward
        return out

    def argmax(self, axis=None):
        return Tensor(self.data.argmax(axis=axis))

    def clip(self, a_min, a_max):
        return Tensor(np.clip(self.data, a_min, a_max), requires_grad=self.requires_grad)

    # ===== Shape Operations =====

    def reshape(self, *shape):
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

    def view(self, *shape): return self.reshape(*shape)

    def expand(self, *shape):
        return Tensor(np.broadcast_to(self.data, shape), requires_grad=self.requires_grad)

    def unsqueeze(self, axis=0):
        return Tensor(np.expand_dims(self.data, axis=axis), requires_grad=self.requires_grad)

    def squeeze(self, axis=None):
        return Tensor(np.squeeze(self.data, axis=axis), requires_grad=self.requires_grad)

    def cat(self, other, dim=0):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(np.concatenate([self.data, other.data], axis=dim),
                      requires_grad=(self.requires_grad or other.requires_grad))

    def repeat_interleave(self, repeats, dim=0):
        return Tensor(np.repeat(self.data, repeats, axis=dim), requires_grad=self.requires_grad)

    # ===== Activation Functions =====

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="relu")
        def _backward():
            if self.requires_grad:
                g = out.grad * (self.data > 0).astype(np.float32)
                self.grad = (self.grad + g) if self.grad is not None else g
        out._backward = _backward
        return out

    def gelu(self):
        x = self.data
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        out_data = x * cdf
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="gelu")
        def _backward():
            if self.requires_grad:
                d = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))
                d2 = 0.5 * (1 - np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3))**2) * np.sqrt(2/np.pi) * (1 + 3*0.044715*x**2)
                g = out.grad * (d + x * d2)
                self.grad = (self.grad + g) if self.grad is not None else g
        out._backward = _backward
        return out

    def silu(self):
        return self * self.sigmoid()

    def sigmoid(self):
        s = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")
        def _backward():
            if self.requires_grad:
                g = out.grad * s * (1 - s)
                self.grad = (self.grad + g) if self.grad is not None else g
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True))
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="softmax")
        def _backward():
            if self.requires_grad:
                g = out.grad
                dot = (g * s).sum(axis=axis, keepdims=True)
                grad = s * (g - dot)
                self.grad = (self.grad + grad) if self.grad is not None else grad
        out._backward = _backward
        return out

    # ===== Normalization =====

    def layernorm(self, eps=1e-5):
        x = self.data
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        out = Tensor(x_norm, requires_grad=self.requires_grad, _children=(self,), _op="layernorm")
        def _backward():
            if self.requires_grad:
                N = x.shape[-1]
                dx = (1/N) * out.grad * np.sqrt(var + eps) * (N - (out.grad * (x - mean)).sum(axis=-1, keepdims=True) * (1/(var + eps)))
                self.grad = (self.grad + dx) if self.grad is not None else dx
        out._backward = _backward
        return out

    def rmsnorm(self, eps=1e-6):
        x = self.data
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        out = Tensor(x / rms, requires_grad=self.requires_grad, _children=(self,), _op="rmsnorm")
        def _backward():
            if self.requires_grad:
                g = out.grad
                g = g / rms
                g = g - (g * x).mean(axis=-1, keepdims=True) * x / (rms ** 2)
                self.grad = (self.grad + g) if self.grad is not None else g
        out._backward = _backward
        return out

    # ===== Dropout =====

    def dropout(self, p=0.1, training=True):
        if not training or p == 0:
            return self
        mask = (np.random.random(self.data.shape) > p).astype(np.float32)
        out = Tensor(self.data * mask / (1 - p), requires_grad=self.requires_grad,
                      _children=(self,), _op="dropout")
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad + out.grad * mask / (1 - p)) if self.grad is not None else out.grad * mask / (1 - p)
        out._backward = _backward
        return out

    # ===== Comparison =====

    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return Tensor((self.data == other.data).astype(np.float32))

    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return Tensor((self.data > other.data).astype(np.float32))

    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return Tensor((self.data < other.data).astype(np.float32))

    def __ge__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return Tensor((self.data >= other.data).astype(np.float32))

    def __le__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return Tensor((self.data <= other.data).astype(np.float32))


    def backward(self):
        """Run backward pass from this tensor."""
        if self.grad is None:
            self.grad = Tensor(np.ones_like(self.data))
        visited = set()
        order = []
        queue = [self]
        while queue:
            node = queue.pop(0)
            if id(node) in visited:
                continue
            visited.add(id(node))
            order.append(node)
            for child in node._prev:
                if id(child) not in visited:
                    queue.append(child)
        for node in order:
            node._backward()
