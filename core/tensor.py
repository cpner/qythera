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
    def __repr__(self): return f"Tensor({self.data.shape})"
    def __len__(self): return len(self.data)

    def backward(self):
        if self.grad is None: self.grad = np.ones_like(self.data)
        visited, order = set(), []
        def build(n):
            if id(n) in visited: return
            visited.add(id(n))
            for c in n._prev: build(c)
            order.append(n)
        build(self)
        for n in reversed(order): n._backward()

    def zero_grad(self): self.grad = None

    def _sg(self, out): return out.grad if out.grad is not None else np.zeros_like(out.data)

    def _st(self, g, s):
        while g.ndim > len(s): g = g.sum(axis=0)
        for i in range(g.ndim):
            if i < len(s) and g.shape[i] != s[i]: g = g.sum(axis=i, keepdims=True)
        return g.reshape(s) if g.shape != s else g

    def __add__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(np.array(o, dtype=np.float32))
        out = Tensor(self.data + o.data, requires_grad=(self.requires_grad or o.requires_grad), _children=(self, o))
        def _b():
            g = self._sg(out)
            if self.requires_grad: self.grad = self._st(g, self.data.shape) if self.grad is None else self.grad + self._st(g, self.data.shape)
            if o.requires_grad: o.grad = self._st(g, o.data.shape) if o.grad is None else o.grad + self._st(g, o.data.shape)
        out._backward = _b; return out
    def __radd__(s, o): return s.__add__(o)

    def __sub__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(np.array(o, dtype=np.float32))
        out = Tensor(self.data - o.data, requires_grad=(self.requires_grad or o.requires_grad), _children=(self, o))
        def _b():
            g = self._sg(out)
            if self.requires_grad: self.grad = self._st(g, self.data.shape) if self.grad is None else self.grad + self._st(g, self.data.shape)
            if o.requires_grad: o.grad = self._st(-g, o.data.shape) if o.grad is None else o.grad + self._st(-g, o.data.shape)
        out._backward = _b; return out
    def __rsub__(s, o):
        o = o if isinstance(o, Tensor) else Tensor(np.array(o, dtype=np.float32))
        return o.__sub__(s)

    def __mul__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(np.array(o, dtype=np.float32))
        out = Tensor(self.data * o.data, requires_grad=(self.requires_grad or o.requires_grad), _children=(self, o))
        def _b():
            g = self._sg(out)
            if self.requires_grad: self.grad = self._st(g * o.data, self.data.shape) if self.grad is None else self.grad + self._st(g * o.data, self.data.shape)
            if o.requires_grad: o.grad = self._st(g * self.data, o.data.shape) if o.grad is None else o.grad + self._st(g * self.data, o.data.shape)
        out._backward = _b; return out
    def __rmul__(s, o): return s.__mul__(o)

    def __truediv__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(np.array(o, dtype=np.float32))
        out = Tensor(self.data / (o.data + 1e-8), requires_grad=self.requires_grad, _children=(self, o))
        def _b():
            g = self._sg(out)
            if self.requires_grad: self.grad = self._st(g / (o.data + 1e-8), self.data.shape) if self.grad is None else self.grad + self._st(g / (o.data + 1e-8), self.data.shape)
        out._backward = _b; return out

    def __pow__(self, e):
        out = Tensor(self.data ** e, requires_grad=self.requires_grad, _children=(self,))
        def _b():
            if self.requires_grad:
                g = e * (self.data ** (e - 1)) * self._sg(out)
                self.grad = self._st(g, self.data.shape) if self.grad is None else self.grad + self._st(g, self.data.shape)
        out._backward = _b; return out

    def __neg__(self): return self * (-1)

    def matmul(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        out = Tensor(self.data @ o.data, requires_grad=(self.requires_grad or o.requires_grad), _children=(self, o))
        def _b():
            g = self._sg(out)
            if self.requires_grad: self.grad = self._st(g @ o.data.T, self.data.shape) if self.grad is None else self.grad + self._st(g @ o.data.T, self.data.shape)
            if o.requires_grad: o.grad = self._st(self.data.T @ g, o.data.shape) if o.grad is None else o.grad + self._st(self.data.T @ g, o.data.shape)
        out._backward = _b; return out
    def __matmul__(s, o): return s.matmul(o)

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,))
        def _b():
            if self.requires_grad: self.grad = self._sg(out) * (self.data > 0).astype(np.float32)
        out._backward = _b; return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,))
        def _b():
            if self.requires_grad: self.grad = self._sg(out) * s * (1 - s)
        out._backward = _b; return out

    def softmax(self, axis=-1):
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True))
        s = e / (e.sum(axis=axis, keepdims=True) + 1e-8)
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,))
        def _b():
            if self.requires_grad:
                g = self._sg(out)
                dot = (g * s).sum(axis=axis, keepdims=True)
                self.grad = s * (g - dot)
        out._backward = _b; return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-7), requires_grad=self.requires_grad, _children=(self,))
        def _b():
            if self.requires_grad: self.grad = self._sg(out) / (self.data + 1e-7)
        out._backward = _b; return out

    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad, _children=(self,))
        def _b():
            if self.requires_grad:
                g = np.broadcast_to(self._sg(out) / n, self.data.shape).copy()
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _b; return out

    def sum(self):
        out = Tensor(self.data.sum(), requires_grad=self.requires_grad, _children=(self,))
        def _b():
            if self.requires_grad:
                g = np.broadcast_to(self._sg(out), self.data.shape).copy()
                self.grad = g if self.grad is None else self.grad + g
        out._backward = _b; return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data + 1e-7), requires_grad=self.requires_grad, _children=(self,))
        def _b():
            if self.requires_grad: self.grad = self._sg(out) / (2 * out.data + 1e-7)
        out._backward = _b; return out

    def reshape(self, *s): return Tensor(self.data.reshape(s), requires_grad=self.requires_grad)
    def clip(self, a, b): return Tensor(np.clip(self.data, a, b), requires_grad=self.requires_grad)
