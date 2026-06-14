"""Minimal but correct autodiff tensor engine."""

import numpy as np


class Tensor:
    """Tensor with automatic differentiation."""
    __array_priority__ = 10000

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, Tensor):
            data = data.data
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
    def item(self): return float(self.data.flat[0])
    def numpy(self): return self.data.copy()
    def __repr__(self): return f"Tensor({self.data.shape}, grad={self.grad is not None})"

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

    # ===== Arithmetic =====
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out = Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad),
                     _children=(self, other), _op="add")
        def _backward():
            g = out.grad
            if self.requires_grad:
                sg = g
                while sg.ndim > self.data.ndim: sg = sg.sum(axis=0)
                for i in range(sg.ndim):
                    if i < self.data.ndim and sg.shape[i] != self.data.shape[i]:
                        sg = sg.sum(axis=i, keepdims=True)
                self.grad = sg.reshape(self.data.shape) if self.grad is None else self.grad + sg.reshape(self.data.shape)
            if other.requires_grad:
                og = g
                while og.ndim > other.data.ndim: og = og.sum(axis=0)
                for i in range(og.ndim):
                    if i < other.data.ndim and og.shape[i] != other.data.shape[i]:
                        og = og.sum(axis=i, keepdims=True)
                other.grad = og.reshape(other.data.shape) if other.grad is None else other.grad + og.reshape(other.data.shape)
        out._backward = _backward
        return out
    def __radd__(self, other): return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out = Tensor(self.data - other.data, requires_grad=(self.requires_grad or other.requires_grad),
                     _children=(self, other), _op="sub")
        def _backward():
            g = out.grad
            if self.requires_grad:
                sg = g
                while sg.ndim > self.data.ndim: sg = sg.sum(axis=0)
                for i in range(sg.ndim):
                    if i < self.data.ndim and sg.shape[i] != self.data.shape[i]:
                        sg = sg.sum(axis=i, keepdims=True)
                self.grad = sg.reshape(self.data.shape) if self.grad is None else self.grad + sg.reshape(self.data.shape)
            if other.requires_grad:
                og = -g
                while og.ndim > other.data.ndim: og = og.sum(axis=0)
                for i in range(og.ndim):
                    if i < other.data.ndim and og.shape[i] != other.data.shape[i]:
                        og = og.sum(axis=i, keepdims=True)
                other.grad = og.reshape(other.data.shape) if other.grad is None else other.grad + og.reshape(other.data.shape)
        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        return other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad),
                     _children=(self, other), _op="mul")
        def _backward():
            g = out.grad
            if self.requires_grad:
                sg = g * other.data
                while sg.ndim > self.data.ndim: sg = sg.sum(axis=0)
                for i in range(sg.ndim):
                    if i < self.data.ndim and sg.shape[i] != self.data.shape[i]:
                        sg = sg.sum(axis=i, keepdims=True)
                self.grad = sg.reshape(self.data.shape) if self.grad is None else self.grad + sg.reshape(self.data.shape)
            if other.requires_grad:
                og = g * self.data
                while og.ndim > other.data.ndim: og = og.sum(axis=0)
                for i in range(og.ndim):
                    if i < other.data.ndim and og.shape[i] != other.data.shape[i]:
                        og = og.sum(axis=i, keepdims=True)
                other.grad = og.reshape(other.data.shape) if other.grad is None else other.grad + og.reshape(other.data.shape)
        out._backward = _backward
        return out
    def __rmul__(self, other): return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=np.float32))
        out = Tensor(self.data / (other.data + 1e-8), requires_grad=(self.requires_grad or other.requires_grad),
                     _children=(self, other), _op="div")
        def _backward():
            g = out.grad
            if self.requires_grad:
                sg = g / (other.data + 1e-8)
                while sg.ndim > self.data.ndim: sg = sg.sum(axis=0)
                for i in range(sg.ndim):
                    if i < self.data.ndim and sg.shape[i] != self.data.shape[i]:
                        sg = sg.sum(axis=i, keepdims=True)
                self.grad = sg.reshape(self.data.shape) if self.grad is None else self.grad + sg.reshape(self.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, exp):
        out = Tensor(self.data ** exp, requires_grad=self.requires_grad, _children=(self,), _op="pow")
        def _backward():
            if self.requires_grad:
                g = exp * (self.data ** (exp - 1)) * out.grad
                while g.ndim > self.data.ndim: g = g.sum(axis=0)
                self.grad = g.reshape(self.data.shape) if self.grad is None else self.grad + g.reshape(self.data.shape)
        out._backward = _backward
        return out

    def __neg__(self): return self * (-1)

    # ===== Matrix =====
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=(self.requires_grad or other.requires_grad),
                     _children=(self, other), _op="matmul")
        def _backward():
            g = out.grad
            if self.requires_grad:
                sg = g @ other.data.T
                while sg.ndim > self.data.ndim: sg = sg.sum(axis=0)
                for i in range(sg.ndim):
                    if i < self.data.ndim and sg.shape[i] != self.data.shape[i]:
                        sg = sg.sum(axis=i, keepdims=True)
                self.grad = sg.reshape(self.data.shape) if self.grad is None else self.grad + sg.reshape(self.data.shape)
            if other.requires_grad:
                og = self.data.T @ g
                while og.ndim > other.data.ndim: og = og.sum(axis=0)
                for i in range(og.ndim):
                    if i < other.data.ndim and og.shape[i] != other.data.shape[i]:
                        og = og.sum(axis=i, keepdims=True)
                other.grad = og.reshape(other.data.shape) if other.grad is None else other.grad + og.reshape(other.data.shape)
        out._backward = _backward
        return out
    def __matmul__(self, other): return self.matmul(other)

    # ===== Activation =====
    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,), _op="relu")
        def _backward():
            if self.requires_grad:
                g = out.grad * (self.data > 0).astype(np.float32)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")
        def _backward():
            if self.requires_grad:
                g = out.grad * s * (1 - s)
                self.grad = g if self.grad is None else self.grad + g
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
                self.grad = grad if self.grad is None else self.grad + grad
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-7), requires_grad=self.requires_grad, _children=(self,), _op="log")
        def _backward():
            if self.requires_grad:
                g = out.grad / (self.data + 1e-7)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad, _children=(self,), _op="mean")
        def _backward():
            if self.requires_grad:
                g = out.grad / n
                g = np.broadcast_to(g, self.data.shape).copy()
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad, _children=(self,), _op="sum")
        def _backward():
            if self.requires_grad:
                g = np.broadcast_to(out.grad, self.data.shape).copy()
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes if axes else tuple(range(self.ndim - 1, -1, -1))
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad, _children=(self,), _op="transpose")
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

    def item(self): return float(self.data.flat[0])

    def __len__(self): return len(self.data)
