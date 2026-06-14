"""Optimizers and LR schedulers. Pure Python + NumPy, depends on core.tensor."""
import math
import copy
import numpy as np
from collections import OrderedDict
from qythera.tensor import Tensor, no_grad

# ---------------------------------------------------------------------------
# Base Optimizer
# ---------------------------------------------------------------------------

class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.param_groups = []
        if isinstance(params, (list, tuple)):
            params = list(params)
        else:
            params = list(params.values()) if isinstance(params, dict) else [params]
        self.param_groups.append({'params': [p for p in params if p.requires_grad], **defaults})
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.data = np.zeros_like(p.grad.data)

    def step(self):
        raise NotImplementedError

    def state_dict(self):
        return {
            'state': {i: {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in s.items()} for i, s in self.state.items()},
            'param_groups': [{k: v for k, v in g.items() if k != 'params'} for g in self.param_groups]
        }

    def load_state_dict(self, state_dict):
        for i, s in state_dict['state'].items():
            self.state[int(i)] = {k: np.array(v) if not isinstance(v, np.ndarray) else v for k, v in s.items()}

    def add_param_group(self, param_group):
        self.param_groups.append(param_group)


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, wd, momentum, dampening = group['lr'], group['weight_decay'], group['momentum'], group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data.copy()
                if wd != 0:
                    g += wd * p.data
                if momentum != 0:
                    state = self.state.setdefault(id(p), {'momentum_buffer': np.zeros_like(p.data)})
                    buf = state['momentum_buffer']
                    buf = momentum * buf + g
                    if dampening != 0:
                        buf *= (1 - dampening)
                    if nesterov:
                        g = g + momentum * buf
                    else:
                        g = buf
                    state['momentum_buffer'] = buf
                p.data -= lr * g


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            amsgrad = group['amsgrad']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'step': 0, 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data), 'max_exp_avg_sq': np.zeros_like(p.data)})
                state['step'] += 1
                if wd != 0:
                    g = g + wd * p.data
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                if amsgrad:
                    np.maximum(state['max_exp_avg_sq'], state['exp_avg_sq'], out=state['max_exp_avg_sq'])
                    denom = np.sqrt(state['max_exp_avg_sq']) + eps
                else:
                    denom = np.sqrt(state['exp_avg_sq']) + eps
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                p.data -= step_size * state['exp_avg'] / (denom / bias_correction2_sqrt + eps)


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------

class AdamW(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            amsgrad = group['amsgrad']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'step': 0, 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data), 'max_exp_avg_sq': np.zeros_like(p.data)})
                state['step'] += 1
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                if amsgrad:
                    np.maximum(state['max_exp_avg_sq'], state['exp_avg_sq'], out=state['max_exp_avg_sq'])
                    denom = np.sqrt(state['max_exp_avg_sq']) + eps
                else:
                    denom = np.sqrt(state['exp_avg_sq']) + eps
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                p.data -= step_size * state['exp_avg'] / (denom / bias_correction2_sqrt + eps)
                if wd != 0:
                    p.data *= (1 - lr * wd)


# ---------------------------------------------------------------------------
# RAdam
# ---------------------------------------------------------------------------

class RAdam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'step': 0, 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data)})
                state['step'] += 1
                if wd != 0:
                    g = g + wd * p.data
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                rho_max = 2.0 / (1 - b2) - 1.0
                rho_t = rho_max - 2 * state['step'] * b2 ** state['step'] / bias_correction2
                if rho_t > 5:
                    rect = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_max / ((rho_max - 4) * (rho_max - 2) * rho_t))
                    p.data -= lr * rect * state['exp_avg'] / (bias_correction1 * (np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps))
                else:
                    p.data -= lr * state['exp_avg'] / bias_correction1


# ---------------------------------------------------------------------------
# NAdam
# ---------------------------------------------------------------------------

class NAdam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum_decay=0.004):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd, mu_decay = group['lr'], group['eps'], group['weight_decay'], group['momentum_decay']
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'step': 0, 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data)})
                state['step'] += 1
                if wd != 0:
                    g = g + wd * p.data
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                mu_t = b1 * (1 - mu_decay) / (1 - mu_decay * state['step'])
                mu_next = b1 * (1 - mu_decay) / (1 - mu_decay * (state['step'] + 1))
                mu_hat = mu_t * bias_correction1
                mu_hat_next = mu_next * bias_correction1
                denom = np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps
                p.data -= lr * (mu_hat_next * state['exp_avg'] / denom + (1 - mu_hat) * g / denom)


# ---------------------------------------------------------------------------
# AdaFactor
# ---------------------------------------------------------------------------

class AdaFactor(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-30, weight_decay=0, scale_parameter=True, relative_step=True, warmup_init=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, scale_parameter=scale_parameter, relative_step=relative_step, warmup_init=warmup_init)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'step': 0, 'exp_avg': np.zeros_like(p.data), 'row_sum': np.zeros(p.data.shape[0]) if p.data.ndim >= 2 else None, 'col_sum': np.zeros(p.data.shape[1]) if p.data.ndim >= 2 else None, 'exp_avg_sq': np.zeros_like(p.data)})
                state['step'] += 1
                if wd != 0:
                    g = g + wd * p.data
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                if p.data.ndim >= 2:
                    R = state['exp_avg_sq'][:, :1].copy()
                    C = state['exp_avg_sq'][:1, :].copy()
                    R += np.mean(g ** 2, axis=1, keepdims=True)
                    C += np.mean(g ** 2, axis=0, keepdims=True)
                    state['exp_avg_sq'] = R @ C / max(R.shape[0], C.shape[1])
                else:
                    state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                denom = np.sqrt(state['exp_avg_sq']) + eps
                p.data -= lr * state['exp_avg'] / denom


# ---------------------------------------------------------------------------
# AdaBelief
# ---------------------------------------------------------------------------

class AdaBelief(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, weight_decouple=True, fixed_decay=False, amsgrad=False, degenerated_to_sgd=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, weight_decouple=weight_decouple, fixed_decay=fixed_decay, amsgrad=amsgrad, degenerated_to_sgd=degenerated_to_sgd)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'step': 0, 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data)})
                state['step'] += 1
                if group['weight_decouple']:
                    p.data *= (1 - lr * wd)
                else:
                    g = g + wd * p.data
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                grad_residual = g - state['exp_avg']
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * grad_residual ** 2
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                denom = np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps
                p.data -= lr * state['exp_avg'] / (bias_correction1 * denom)


# ---------------------------------------------------------------------------
# Adadelta
# ---------------------------------------------------------------------------

class Adadelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, rho, eps, wd = group['lr'], group['rho'], group['eps'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'square_avg': np.zeros_like(p.data), 'acc_delta': np.zeros_like(p.data)})
                if wd != 0:
                    g = g + wd * p.data
                state['square_avg'] = rho * state['square_avg'] + (1 - rho) * g ** 2
                delta = np.sqrt(state['acc_delta'] + eps) / np.sqrt(state['square_avg'] + eps) * g
                state['acc_delta'] = rho * state['acc_delta'] + (1 - rho) * delta ** 2
                p.data -= lr * delta


# ---------------------------------------------------------------------------
# Adagrad
# ---------------------------------------------------------------------------

class Adagrad(Optimizer):
    def __init__(self, params, lr=0.01, lr_decay=0, weight_decay=0, eps=1e-10):
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, wd, eps, lr_decay = group['lr'], group['weight_decay'], group['eps'], group['lr_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'sum': np.zeros_like(p.data), 'step': 0})
                state['step'] += 1
                if wd != 0:
                    g = g + wd * p.data
                state['sum'] += g ** 2
                step_size = lr / (1 + (state['step'] - 1) * lr_decay)
                p.data -= step_size * g / (np.sqrt(state['sum']) + eps)


# ---------------------------------------------------------------------------
# RMSProp
# ---------------------------------------------------------------------------

class RMSProp(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, alpha, eps, wd, momentum = group['lr'], group['alpha'], group['eps'], group['weight_decay'], group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'square_avg': np.zeros_like(p.data), 'momentum_buffer': np.zeros_like(p.data)})
                if wd != 0:
                    g = g + wd * p.data
                state['square_avg'] = alpha * state['square_avg'] + (1 - alpha) * g ** 2
                avg = state['square_avg']
                if momentum != 0:
                    state['momentum_buffer'] = momentum * state['momentum_buffer'] + g / (np.sqrt(avg) + eps)
                    p.data -= lr * state['momentum_buffer']
                else:
                    p.data -= lr * g / (np.sqrt(avg) + eps)


# ---------------------------------------------------------------------------
# Lion
# ---------------------------------------------------------------------------

class Lion(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.99), weight_decay=0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'exp_avg': np.zeros_like(p.data)})
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                update = np.sign(b1 * state['exp_avg'] + (1 - b1) * g)
                p.data *= (1 - lr * wd)
                p.data -= lr * update


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------

class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.05, ns_iterations=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, ns_iterations=ns_iterations)
        super().__init__(params, defaults)

    def newton_schulz(self, G, iterations=5):
        X = G / (np.linalg.norm(G) + 1e-7)
        for _ in range(iterations):
            A = X @ X.T
            X = 1.5 * X - 0.5 * A @ X
        return X

    def step(self):
        for group in self.param_groups:
            lr, momentum, wd, ns_iter = group['lr'], group['momentum'], group['weight_decay'], group['ns_iterations']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'momentum_buffer': np.zeros_like(p.data)})
                buf = state['momentum_buffer']
                buf = momentum * buf + g
                if group['nesterov']:
                    g = g + momentum * buf
                else:
                    g = buf
                state['momentum_buffer'] = buf
                if p.data.ndim >= 2:
                    orth = self.newton_schulz(g, ns_iter)
                    p.data -= lr * orth
                else:
                    p.data -= lr * g
                if wd != 0:
                    p.data *= (1 - lr * wd)


# ---------------------------------------------------------------------------
# SAM
# ---------------------------------------------------------------------------

class SAM(Optimizer):
    def __init__(self, base_optimizer, rho=0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho

    def step(self):
        grad_norm = self._grad_norm()
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                eps = self.rho * p.grad.data / (grad_norm + 1e-12)
                p.data += eps
        self.base_optimizer.step()
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                eps = self.rho * p.grad.data / (grad_norm + 1e-12)
                p.data -= eps

    def _grad_norm(self):
        norm = 0.0
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norm += np.sum(p.grad.data ** 2)
        return np.sqrt(norm)

    def zero_grad(self):
        self.base_optimizer.zero_grad()


# ---------------------------------------------------------------------------
# Sophia
# ---------------------------------------------------------------------------

class Sophia(Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.965, 0.99), eps=1e-12, weight_decay=0.1, k=10, clip_threshold=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k=k, clip_threshold=clip_threshold)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, wd, eps = group['lr'], group['weight_decay'], group['eps']
            b1, b2 = group['betas']
            clip = group['clip_threshold']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'step': 0, 'exp_avg': np.zeros_like(p.data), 'hessian': np.ones_like(p.data)})
                state['step'] += 1
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                if state['step'] % group['k'] == 0:
                    state['hessian'] = b2 * state['hessian'] + (1 - b2) * g ** 2
                m_hat = state['exp_avg'] / (1 - b1 ** state['step'])
                h_hat = state['hessian'] / (1 - b2 ** state['step'])
                update = np.clip(m_hat / (h_hat + eps), -clip, clip)
                p.data -= lr * (update + wd * p.data)


# ---------------------------------------------------------------------------
# Adan
# ---------------------------------------------------------------------------

class Adan(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.98, 0.92, 0.99), eps=1e-8, weight_decay=0.02):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, wd, eps = group['lr'], group['weight_decay'], group['eps']
            b1, b2, b3 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'step': 0, 'prev_grad': np.zeros_like(p.data), 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data)})
                state['step'] += 1
                prev_grad = state['prev_grad']
                state['prev_grad'] = g.copy()
                g_diff = g - prev_grad
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * (g + b2 * g_diff)
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * (g + b2 * g_diff) ** 2
                update = state['exp_avg'] / (np.sqrt(state['exp_avg_sq']) + eps)
                p.data -= lr * (update + wd * p.data)


# ---------------------------------------------------------------------------
# SOAP
# ---------------------------------------------------------------------------

class SOAP(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, update_every=10):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, update_every=update_every)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd, update_every = group['lr'], group['eps'], group['weight_decay'], group['update_every']
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'step': 0, 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data)})
                state['step'] += 1
                if wd != 0:
                    g = g + wd * p.data
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                if p.data.ndim >= 2 and state['step'] % update_every == 0:
                    U, S, Vt = np.linalg.svd(g, full_matrices=False)
                    L = U @ np.diag(S) @ U.T
                    R = Vt.T @ np.diag(S) @ Vt
                    g_projected = L @ g @ R
                    state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g_projected ** 2
                else:
                    state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                p.data -= lr * state['exp_avg'] / bias_correction1 / (np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps)


# ---------------------------------------------------------------------------
# LAMB
# ---------------------------------------------------------------------------

class LAMB(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'step': 0, 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data)})
                state['step'] += 1
                if wd != 0:
                    g = g + wd * p.data
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                update = state['exp_avg'] / bias_correction1 / (np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps)
                w_norm = np.linalg.norm(p.data) + eps
                u_norm = np.linalg.norm(update) + eps
                p.data -= lr * (w_norm / u_norm) * update


# ---------------------------------------------------------------------------
# LARS
# ---------------------------------------------------------------------------

class LARS(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0, eps=1e-8, trust_coefficient=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, wd, eps, tc = group['lr'], group['weight_decay'], group['eps'], group['trust_coefficient']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {'momentum_buffer': np.zeros_like(p.data)})
                if wd != 0:
                    g = g + wd * p.data
                buf = state['momentum_buffer']
                buf = momentum * buf + g
                trust = np.linalg.norm(p.data) / (np.linalg.norm(buf) + eps)
                p.data -= lr * tc * trust * buf


# ---------------------------------------------------------------------------
# LR Schedulers
# ---------------------------------------------------------------------------

class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self._step_count = 0
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]

    def step(self, epoch=None):
        self._step_count += 1
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        self._update_lr()

    def _update_lr(self):
        lr = self.get_lr()
        for i, (group, base_lr) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            group['lr'] = lr * base_lr / max(self.base_lrs[0], 1e-30)

    def get_lr(self):
        raise NotImplementedError

    def state_dict(self):
        return {'last_epoch': self.last_epoch, 'base_lrs': self.base_lrs}

    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']


class ConstantLR(LRScheduler):
    def __init__(self, optimizer, factor=1.0, last_epoch=-1):
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.factor


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma=0.9, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.gamma ** self.last_epoch


class LinearWarmup(LRScheduler):
    def __init__(self, optimizer, warmup_steps=0, total_steps=1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self.last_epoch / max(1, self.warmup_steps)
        return max(0.0, (self.total_steps - self.last_epoch) / max(1, self.total_steps - self.warmup_steps))


class WarmupCosine(LRScheduler):
    def __init__(self, optimizer, warmup_steps=0, total_steps=1, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return self.last_epoch / max(1, self.warmup_steps)
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (1.0 - self.min_lr) * (1.0 + math.cos(math.pi * progress))


class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer, T_0=10, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return 1.0
        if self.T_mult == 1:
            return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_0))
        return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_i))


class CyclicLR(LRScheduler):
    def __init__(self, optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=2000, step_size_down=None, mode='triangular', last_epoch=-1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
        x = self.last_epoch / (self.step_size_up + self.step_size_down) - (cycle - 1)
        if x <= self.step_size_up / (self.step_size_up + self.step_size_down):
            relative = x * (self.step_size_up + self.step_size_down) / self.step_size_up
        else:
            relative = 1.0 - (x - self.step_size_up / (self.step_size_up + self.step_size_down)) / (self.step_size_down / (self.step_size_up + self.step_size_down))
        if self.mode == 'triangular':
            scale_fn = lambda x: x
        elif self.mode == 'triangular2':
            scale_fn = lambda x: x / (2 ** (cycle - 1))
        else:
            scale_fn = lambda x: 1.0
        return (self.max_lr - self.base_lr) * max(0, scale_fn(relative)) + self.base_lr


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, min_lr=0):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best = None
        self.num_bad_epochs = 0
        self.mode = 'min' if mode == 'min' else 'max'

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
        else:
            if self.mode == 'min':
                improved = metrics < self.best - self.threshold
            else:
                improved = metrics > self.best + self.threshold
            if improved:
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
                if self.num_bad_epochs >= self.patience:
                    for group in self.optimizer.param_groups:
                        new_lr = max(group['lr'] * self.factor, self.min_lr)
                        group['lr'] = new_lr
                    self.num_bad_epochs = 0


class PolynomialLR(LRScheduler):
    def __init__(self, optimizer, total_iters=1, power=1.0, last_epoch=-1, min_lr=0):
        self.total_iters = total_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return max(self.min_lr, (1.0 - self.last_epoch / self.total_iters) ** self.power)


class OneCycleLR(LRScheduler):
    def __init__(self, optimizer, max_lr=0.01, total_steps=1, pct_start=0.3, anneal_strategy='cos', last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.total_steps * self.pct_start:
            return step / (self.total_steps * self.pct_start)
        else:
            progress = (step - self.total_steps * self.pct_start) / (self.total_steps * (1 - self.pct_start))
            if self.anneal_strategy == 'cos':
                return 0.5 * (1 + math.cos(math.pi * progress))
            return max(0.0, 1.0 - progress)


class ChainedScheduler(LRScheduler):
    def __init__(self, schedulers, last_epoch=-1):
        self.schedulers = schedulers
        super().__init__(schedulers[0].optimizer, last_epoch)

    def get_lr(self):
        return self.schedulers[-1].get_lr()

    def step(self, epoch=None):
        for s in self.schedulers:
            s.step(epoch)


# ---------------------------------------------------------------------------
# ScheduleFreeAdam
# ---------------------------------------------------------------------------

class ScheduleFreeAdam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, beta=0.9, c=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, beta=beta, c=c)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            beta, c = group['beta'], group['c']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state.setdefault(id(p), {
                    'theta': p.data.copy(),
                    'z': np.zeros_like(p.data),
                    'exp_avg': np.zeros_like(p.data),
                    'exp_avg_sq': np.zeros_like(p.data),
                    'step': 0
                })
                state['step'] += 1
                y = beta * state['theta'] + (1 - beta) * state['z']
                g = p.grad.data
                if wd != 0:
                    g = g + wd * y
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                step_size = lr / bias_correction1
                denom = np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps
                grad_update = state['exp_avg'] / denom
                state['z'] = state['z'] - lr * grad_update
                state['theta'] = (1 - c) * state['theta'] + c * state['z']
                p.data = state['theta'].copy()

    def eval(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(id(p))
                if state is not None:
                    p.data = state['theta'].copy()


# ---------------------------------------------------------------------------
# Prodigy
# ---------------------------------------------------------------------------

class Prodigy(Optimizer):
    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, growth_factor=2.0, d0=1e-6, d_largest=10.0, k=0.5, safety_factor=0.85):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, growth_factor=growth_factor, d0=d0, d_largest=d_largest, k=k, safety_factor=safety_factor)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            d0 = group['d0']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {
                    'step': 0, 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data),
                    'd': d0, 'd_sum': 0.0, 'prev_x': p.data.copy(), 'prev_g': np.zeros_like(p.data)
                })
                state['step'] += 1
                if wd != 0:
                    g = g + wd * p.data
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                m_hat = state['exp_avg'] / bias_correction1
                v_hat = state['exp_avg_sq'] / bias_correction2
                if state['step'] > 1:
                    x_diff = state['prev_x'] - p.data
                    g_diff = state['prev_g'] - g
                    numerator = np.abs(np.sum(x_diff * g_diff))
                    denominator = np.sum(g_diff ** 2) + eps
                    d_hat = numerator / denominator
                    d_hat = max(d_hat.item() if hasattr(d_hat, 'item') else float(d_hat), d0)
                    state['d'] = min(state['d'] * group['growth_factor'], max(d_hat, d0))
                state['d_sum'] += state['d'] ** (-2)
                adapt_lr = group['safety_factor'] * state['d'] / (np.sqrt(state['d_sum'] * v_hat) + eps)
                update = adapt_lr * m_hat
                state['prev_x'] = p.data.copy()
                state['prev_g'] = g.copy()
                p.data -= lr * update


# ---------------------------------------------------------------------------
# GaLore
# ---------------------------------------------------------------------------

class GaLore(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, rank=128, update_every=200):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rank=rank, update_every=update_every)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            rank = group['rank']
            update_every = group['update_every']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g = g + wd * p.data
                state = self.state.setdefault(id(p), {
                    'step': 0, 'P': None, 'exp_avg': None, 'exp_avg_sq': None
                })
                state['step'] += 1
                r = min(rank, min(g.shape))
                if p.data.ndim >= 2 and (state['P'] is None or state['step'] % update_every == 0):
                    U, S, Vt = np.linalg.svd(g, full_matrices=False)
                    state['P'] = U[:, :r].T
                if state['exp_avg'] is None:
                    state['exp_avg'] = np.zeros((r, g.shape[-1]) if g.ndim >= 2 else (r,))
                    state['exp_avg_sq'] = np.zeros_like(state['exp_avg'])
                if state['P'] is not None and g.ndim >= 2:
                    g_proj = state['P'] @ g
                    state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g_proj
                    state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g_proj ** 2
                    bias_correction1 = 1 - b1 ** state['step']
                    bias_correction2 = 1 - b2 ** state['step']
                    denom = np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps
                    update_proj = state['exp_avg'] / denom
                    update_full = state['P'].T @ update_proj
                else:
                    state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                    state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                    bias_correction1 = 1 - b1 ** state['step']
                    bias_correction2 = 1 - b2 ** state['step']
                    denom = np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps
                    update_full = state['exp_avg'] / denom
                p.data -= lr * update_full


# ---------------------------------------------------------------------------
# Flora
# ---------------------------------------------------------------------------

class Flora(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, rank=128):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rank=rank)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            rank = group['rank']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g = g + wd * p.data
                state = self.state.setdefault(id(p), {
                    'step': 0, 'A': None, 'B': None,
                    'exp_avg': None, 'exp_avg_sq': None
                })
                state['step'] += 1
                if state['A'] is None and g.ndim >= 2:
                    m, n = g.shape
                    r = min(rank, min(m, n))
                    state['A'] = np.random.randn(m, r) / np.sqrt(r)
                    state['B'] = np.random.randn(n, r) / np.sqrt(r)
                    state['exp_avg'] = np.zeros((r, r))
                    state['exp_avg_sq'] = np.zeros((r, r))
                if state['A'] is not None and g.ndim >= 2:
                    g_compressed = state['A'].T @ g @ state['B']
                    state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g_compressed
                    state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g_compressed ** 2
                    bias_correction1 = 1 - b1 ** state['step']
                    bias_correction2 = 1 - b2 ** state['step']
                    denom = np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps
                    state_update = state['exp_avg'] / denom
                    update_full = state['A'] @ state_update @ state['B'].T
                else:
                    if state['exp_avg'] is None:
                        state['exp_avg'] = np.zeros_like(g)
                        state['exp_avg_sq'] = np.zeros_like(g)
                    state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                    state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                    bias_correction1 = 1 - b1 ** state['step']
                    bias_correction2 = 1 - b2 ** state['step']
                    denom = np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps
                    update_full = state['exp_avg'] / denom
                p.data -= lr * update_full


# ---------------------------------------------------------------------------
# OneBitAdam
# ---------------------------------------------------------------------------

class OneBitAdam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup_steps=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup_steps=warmup_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            warmup_steps = group['warmup_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state.setdefault(id(p), {
                    'step': 0, 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data),
                    'residual': np.zeros_like(p.data), 'scale': 1.0
                })
                state['step'] += 1
                if wd != 0:
                    g = g + wd * p.data
                if state['step'] > warmup_steps:
                    g_effective = g + state['residual']
                    g_norm = np.linalg.norm(g_effective)
                    state['scale'] = g_norm / max(np.sqrt(np.sum(g_effective.size)), eps)
                    binarized = np.sign(g_effective) * state['scale']
                    state['residual'] = g_effective - binarized
                    g = binarized
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                step_size = lr / bias_correction1
                denom = np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps
                p.data -= step_size * state['exp_avg'] / denom


# ---------------------------------------------------------------------------
# PowerSGD
# ---------------------------------------------------------------------------

class PowerSGD(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, rank=4, powerSGD_iterations=1, start_powerSGD_iter=1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rank=rank, powerSGD_iterations=powerSGD_iterations, start_powerSGD_iter=start_powerSGD_iter)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group['lr'], group['eps'], group['weight_decay']
            b1, b2 = group['betas']
            rank = group['rank']
            n_iter = group['powerSGD_iterations']
            start_iter = group['start_powerSGD_iter']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g = g + wd * p.data
                state = self.state.setdefault(id(p), {
                    'step': 0, 'exp_avg': np.zeros_like(p.data), 'exp_avg_sq': np.zeros_like(p.data),
                    'P': None, 'Q': None
                })
                state['step'] += 1
                if p.data.ndim >= 2 and state['step'] >= start_iter:
                    m, n = g.shape
                    r = min(rank, min(m, n))
                    if state['P'] is None or state['P'].shape != (m, r):
                        state['P'] = np.random.randn(m, r) / np.sqrt(r)
                    P = state['P']
                    Q = None
                    for _ in range(n_iter):
                        Q = g.T @ P
                        Q = np.linalg.qr(Q)[0]
                        P = g @ Q
                        P = np.linalg.qr(P)[0]
                    state['P'] = P
                    g_approx = P @ Q.T
                    state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g_approx
                    state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g_approx ** 2
                else:
                    state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * g
                    state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * g ** 2
                bias_correction1 = 1 - b1 ** state['step']
                bias_correction2 = 1 - b2 ** state['step']
                denom = np.sqrt(state['exp_avg_sq'] / bias_correction2) + eps
                p.data -= lr * state['exp_avg'] / bias_correction1 / denom


# ---------------------------------------------------------------------------
# LRFinder (Learning Rate Finder)
# ---------------------------------------------------------------------------

class LRFinder:
    def __init__(self, model, optimizer, loss_fn, lr_min=1e-7, lr_max=10.0, num_steps=100, beta=0.98):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.num_steps = num_steps
        self.beta = beta
        self.lrs = []
        self.losses = []
        self.best_loss = float('inf')
        self._backup_state = None

    def range_test(self, data_loader):
        self._backup()
        lr_schedule = np.geomspace(self.lr_min, self.lr_max, self.num_steps)
        avg_loss = 0.0
        for step, (x_batch, y_batch) in enumerate(range(self.num_steps)):
            lr = float(lr_schedule[step])
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            self.optimizer.zero_grad()
            if hasattr(x_batch, 'data'):
                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)
            else:
                output = self.model(Tensor(x_batch, requires_grad=True))
                loss = self.loss_fn(output, Tensor(y_batch) if not isinstance(y_batch, Tensor) else y_batch)
            loss_val = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
            avg_loss = self.beta * avg_loss + (1 - self.beta) * loss_val
            smoothed = avg_loss / (1 - self.beta ** (step + 1))
            self.lrs.append(lr)
            self.losses.append(smoothed)
            if smoothed < self.best_loss:
                self.best_loss = smoothed
            self.optimizer.zero_grad()
            loss.backward() if hasattr(loss, 'backward') else None
            self.optimizer.step()
            if smoothed > 4 * self.best_loss:
                break
        self._restore()
        return self.suggest_lr()

    def suggest_lr(self, num_points=20):
        if len(self.losses) < num_points:
            return self.lrs[np.argmin(self.losses)]
        loss_arr = np.array(self.losses)
        lr_arr = np.array(self.lrs)
        log_lr = np.log10(lr_arr)
        min_idx = np.argmin(loss_arr)
        end = min(min_idx + num_points, len(loss_arr))
        start = max(end - num_points, 0)
        segment_lr = log_lr[start:end]
        segment_loss = loss_arr[start:end]
        grads = np.diff(segment_loss) / np.diff(segment_lr)
        min_grad_idx = np.argmin(grads)
        return 10 ** float(segment_lr[min_grad_idx])

    def _backup(self):
        self._backup_state = {}
        for i, group in enumerate(self.optimizer.param_groups):
            self._backup_state[i] = group['lr']

    def _restore(self):
        if self._backup_state:
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self._backup_state[i]


# ---------------------------------------------------------------------------
# DoReMiScheduler
# ---------------------------------------------------------------------------

class DoReMiScheduler:
    def __init__(self, optimizer, num_domains, reference_loss=None, smoothing=0.9, lr_factor=1.0):
        self.optimizer = optimizer
        self.num_domains = num_domains
        self.reference_loss = reference_loss if reference_loss is not None else np.ones(num_domains)
        self.smoothing = smoothing
        self.lr_factor = lr_factor
        self.domain_weights = np.ones(num_domains) / num_domains
        self._ema_excess = np.zeros(num_domains)
        self._step_count = 0

    def step(self, domain_losses):
        self._step_count += 1
        domain_losses = np.array(domain_losses, dtype=np.float64)
        excess_loss = domain_losses - self.reference_loss
        self._ema_excess = self.smoothing * self._ema_excess + (1 - self.smoothing) * excess_loss
        e = self._ema_excess - np.max(self._ema_excess)
        self.domain_weights = np.exp(e) / np.exp(e).sum() * self.num_domains
        for group in self.optimizer.param_groups:
            group['lr'] = group.get('base_lr', group['lr']) * self.lr_factor

    def get_domain_weights(self):
        return self.domain_weights.copy()

    def state_dict(self):
        return {
            'domain_weights': self.domain_weights.copy(),
            'ema_excess': self._ema_excess.copy(),
            'step_count': self._step_count
        }

    def load_state_dict(self, state_dict):
        self.domain_weights = state_dict['domain_weights'].copy()
        self._ema_excess = state_dict['ema_excess'].copy()
        self._step_count = state_dict['step_count']
