import numpy as np
from core.autodiff.tensor import Tensor


class Optimizer:
    def __init__(self, params, lr=0.001):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, Tensor):
            data = x.data
            if isinstance(data, np.ndarray) and data.dtype == object:
                # Extract numeric values from object arrays
                result = np.zeros_like(data, dtype=np.float32)
                for idx in np.ndindex(data.shape):
                    val = data[idx]
                    if isinstance(val, Tensor):
                        result[idx] = float(val.data.flat[0]) if val.data.size > 0 else 0.0
                    elif isinstance(val, (int, float)):
                        result[idx] = float(val)
                return result
            return data.astype(np.float32).copy() if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)
        if isinstance(x, np.ndarray):
            if x.dtype == object:
                result = np.zeros_like(x, dtype=np.float32)
                for idx in np.ndindex(x.shape):
                    val = x[idx]
                    if isinstance(val, Tensor):
                        result[idx] = float(val.data.flat[0]) if val.data.size > 0 else 0.0
                    elif isinstance(val, (int, float)):
                        result[idx] = float(val)
                return result
            return x.astype(np.float32).copy()
        return np.array(x, dtype=np.float32)


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = self._to_numpy(p.grad)
            if self.weight_decay > 0:
                g = g + self.weight_decay * self._to_numpy(p.data)
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + g
                g = self.velocities[i]
            p.data = self._to_numpy(p.data) - self.lr * g


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
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
            g = self._to_numpy(p.grad)
            if self.weight_decay > 0:
                g = g + self.weight_decay * self._to_numpy(p.data)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            p.data = self._to_numpy(p.data) - self.lr * m_hat / (np.sqrt(v_hat + self.eps))


class AdamW(Adam):
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = self._to_numpy(p.grad)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            p.data = self._to_numpy(p.data) - self.lr * m_hat / (np.sqrt(v_hat + self.eps))

            if self.weight_decay > 0:
                p.data = self._to_numpy(p.data) - self.lr * self.weight_decay * self._to_numpy(p.data)
