"""Neural network modules. Pure Python + NumPy, depends on core.tensor."""
import math
import copy
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from core.tensor import Tensor, no_grad, zeros, ones, randn, eye

# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def kaiming_uniform_(tensor, fan_in, a=math.sqrt(5)):
    std = math.sqrt(2.0 / (1 + a ** 2) / fan_in)
    bound = math.sqrt(3.0) * std
    tensor.data = Tensor._rand_uniform(tensor.shape, -bound, bound).data if hasattr(Tensor, '_rand_uniform') else Tensor._rand_like(tensor).data * bound
    import numpy as np
    tensor.data = np.random.uniform(-bound, bound, tensor.shape).astype(np.float32)
    return tensor

def kaiming_normal_(tensor, fan_in):
    import numpy as np
    std = math.sqrt(2.0 / fan_in)
    tensor.data = np.random.normal(0, std, tensor.shape).astype(np.float32)
    return tensor

def xavier_uniform_(tensor, fan_in, fan_out):
    import numpy as np
    std = math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    tensor.data = np.random.uniform(-bound, bound, tensor.shape).astype(np.float32)
    return tensor

def xavier_normal_(tensor, fan_in, fan_out):
    import numpy as np
    std = math.sqrt(2.0 / (fan_in + fan_out))
    tensor.data = np.random.normal(0, std, tensor.shape).astype(np.float32)
    return tensor

def orthogonal_(tensor, gain=1.0):
    import numpy as np
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("Only tensors with 2+ dimensions are supported")
    rows, cols = shape[0], int(np.prod(shape[1:]))
    flat = np.random.randn(rows, cols)
    if rows < cols:
        flat = flat.T
    u, _, vh = np.linalg.svd(flat, full_matrices=False)
    q = u if rows >= cols else vh
    q = q[:rows, :cols]
    if rows < cols:
        q = q.T
    tensor.data = (q * gain).reshape(shape).astype(np.float32)
    return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    import numpy as np
    tmp = np.random.normal(mean, std, tensor.shape)
    tmp = np.clip(tmp, a * std + mean, b * std + mean)
    tensor.data = tmp.astype(np.float32)
    return tensor

def lecun_normal_(tensor, fan_in):
    import numpy as np
    std = math.sqrt(1.0 / fan_in)
    tensor.data = np.random.normal(0, std, tensor.shape).astype(np.float32)
    return tensor

def zeros_(tensor):
    tensor.data = Tensor.zeros(*tensor.shape).data
    return tensor

def ones_(tensor):
    tensor.data = Tensor.ones(*tensor.shape).data
    return tensor


# ---------------------------------------------------------------------------
# Module base class
# ---------------------------------------------------------------------------

class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._backward_hooks = []
        self._backward_full_hooks = []
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        for hook in self._forward_pre_hooks.values():
            hook(self, args)
        output = self.forward(*args, **kwargs)
        for hook in self._forward_hooks.values():
            hook(self, args, output)
        return output

    def parameters(self, recurse=True):
        params = []
        for p in self._parameters.values():
            params.append(p)
        if recurse:
            for m in self._modules.values():
                params.extend(m.parameters(recurse))
        return params

    def named_parameters(self, prefix="", recurse=True):
        items = []
        for name, p in self._parameters.items():
            items.append((f"{prefix}.{name}" if prefix else name, p))
        if recurse:
            for name, m in self._modules.items():
                sub_prefix = f"{prefix}.{name}" if prefix else name
                items.extend(m.named_parameters(sub_prefix, recurse))
        return items

    def buffers(self, recurse=True):
        bufs = list(self._buffers.values())
        if recurse:
            for m in self._modules.values():
                bufs.extend(m.buffers(recurse))
        return bufs

    def named_buffers(self, prefix="", recurse=True):
        items = [(f"{prefix}.{name}" if prefix else name, b) for name, b in self._buffers.items()]
        if recurse:
            for name, m in self._modules.items():
                sub_prefix = f"{prefix}.{name}" if prefix else name
                items.extend(m.named_buffers(sub_prefix, recurse))
        return items

    def state_dict(self):
        state = {}
        for name, p in self._parameters.items():
            state[name] = p.data.copy() if isinstance(p, Tensor) else p
        for name, b in self._buffers.items():
            state[f"buffer_{name}"] = b.data.copy() if isinstance(b, Tensor) else b
        for name, m in self._modules.items():
            sub_state = m.state_dict()
            for k, v in sub_state.items():
                state[f"{name}.{k}"] = v
        return state

    def load_state_dict(self, state_dict, strict=True):
        missing, unexpected = [], []
        my_state = self.state_dict()
        for k, v in state_dict.items():
            if k in my_state:
                if isinstance(my_state[k], Tensor):
                    my_state[k].data = v.data if isinstance(v, Tensor) else v
            elif strict:
                unexpected.append(k)
        if strict:
            for k in my_state:
                if k not in state_dict:
                    missing.append(k)
        return missing, unexpected

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    def to(self, *args, **kwargs):
        for p in self._parameters.values():
            if isinstance(p, Tensor):
                p.data = p.data.copy()
        return self

    def cpu(self):
        return self

    def apply(self, fn):
        for m in self._modules.values():
            m.apply(fn)
        fn(self)
        return self

    def zero_grad(self, set_to_none=True):
        for p in self._parameters.values():
            if set_to_none:
                p.grad = None
            else:
                if p.grad is not None:
                    p.grad.data = Tensor.zeros(*p.grad.shape).data

    def register_parameter(self, name, param):
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def register_buffer(self, name, buf):
        if buf is None:
            self._buffers[name] = None
        else:
            self._buffers[name] = buf
            object.__setattr__(self, name, buf)

    def register_forward_hook(self, hook):
        handle = len(self._forward_hooks)
        self._forward_hooks[handle] = hook
        return handle

    def register_forward_pre_hook(self, hook):
        handle = len(self._forward_pre_hooks)
        self._forward_pre_hooks[handle] = hook
        return handle

    def register_backward_hook(self, hook):
        self._backward_hooks.append(hook)
        return len(self._backward_hooks) - 1

    def children(self):
        return self._modules.values()

    def named_children(self):
        return self._modules.items()

    def modules(self):
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def named_modules(self, prefix=""):
        yield (prefix, self)
        for name, m in self._modules.items():
            sub_prefix = f"{prefix}.{name}" if prefix else name
            yield from m.named_modules(sub_prefix)

    def requires_grad_(self, requires_grad=True):
        for p in self._parameters.values():
            p.requires_grad_(requires_grad)
        return self

    def half(self):
        return self

    def float(self):
        return self

    def double(self):
        return self

    def get_submodule(self, target):
        parts = target.split(".")
        mod = self
        for p in parts:
            mod = mod._modules[p]
        return mod

    def extra_repr(self):
        return ""

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, m in self._modules.items():
            mod_str = repr(m).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_str}")
        extra = self.extra_repr()
        if extra:
            lines.append(f"  {extra}")
        lines.append(")")
        return "\n".join(lines)

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            if '_parameters' not in self.__dict__:
                object.__setattr__(self, '_parameters', OrderedDict())
            self.__dict__['_parameters'][name] = value
            object.__setattr__(self, name, value)
        elif isinstance(value, Module):
            if '_modules' not in self.__dict__:
                object.__setattr__(self, '_modules', OrderedDict())
            self.__dict__['_modules'][name] = value
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, muP=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.muP = muP
        import numpy as np
        weight = Tensor(np.random.randn(out_features, in_features).astype(np.float32) * math.sqrt(2.0 / in_features))
        self.register_parameter('weight', weight)
        if bias:
            bias_t = Tensor(np.zeros(out_features, dtype=np.float32))
            self.register_parameter('bias', bias_t)
        else:
            self.bias = None

    def forward(self, x):
        out = x.matmul(self.weight.T)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, sparse=False):
        super().__init__()
        import numpy as np
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        weight = Tensor(np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02)
        if padding_idx is not None:
            weight.data[padding_idx] = 0.0
        self.register_parameter('weight', weight)

    def forward(self, input):
        return Tensor(self.weight.data[input.data.astype(int).flatten()].reshape(*input.shape, self.embedding_dim), requires_grad=self.weight.requires_grad)

    def extra_repr(self):
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------

class RMSNorm(Module):
    def __init__(self, num_features, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            import numpy as np
            self.register_parameter('weight', Tensor(np.ones(num_features, dtype=np.float32)))
        else:
            self.weight = None

    def forward(self, x):
        import numpy as np
        norm = np.sqrt(np.mean(x.data ** 2, axis=-1, keepdims=True) + self.eps)
        out = x.data / norm
        if self.weight is not None:
            out = out * self.weight.data
        return Tensor(out, requires_grad=x.requires_grad)

    def extra_repr(self):
        return f"eps={self.eps}"


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        if elementwise_affine:
            import numpy as np
            self.register_parameter('weight', Tensor(np.ones(normalized_shape, dtype=np.float32)))
            self.register_parameter('bias', Tensor(np.zeros(normalized_shape, dtype=np.float32)))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        import numpy as np
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True, ddof=0)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        if self.weight is not None:
            x_norm = x_norm * self.weight.data
        if self.bias is not None:
            x_norm = x_norm + self.bias.data
        return Tensor(x_norm, requires_grad=x.requires_grad)


class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        import numpy as np
        if affine:
            self.register_parameter('weight', Tensor(np.ones(num_features, dtype=np.float32)))
            self.register_parameter('bias', Tensor(np.zeros(num_features, dtype=np.float32)))
        if track_running_stats:
            self.register_buffer('running_mean', Tensor(np.zeros(num_features, dtype=np.float32)))
            self.register_buffer('running_var', Tensor(np.ones(num_features, dtype=np.float32)))

    def forward(self, x):
        import numpy as np
        if self.training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0, ddof=0)
            if self.track_running_stats:
                self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean
                self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var
            x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        else:
            x_norm = (x.data - self.running_mean.data) / np.sqrt(self.running_var.data + self.eps)
        if hasattr(self, 'weight') and self.weight is not None:
            x_norm = x_norm * self.weight.data
        if hasattr(self, 'bias') and self.bias is not None:
            x_norm = x_norm + self.bias.data
        return Tensor(x_norm, requires_grad=x.requires_grad)


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        import numpy as np
        if affine:
            self.register_parameter('weight', Tensor(np.ones(num_channels, dtype=np.float32)))
            self.register_parameter('bias', Tensor(np.zeros(num_channels, dtype=np.float32)))

    def forward(self, x):
        import numpy as np
        shape = x.data.shape
        G = self.num_groups
        C = shape[1] if len(shape) > 1 else 1
        x_reshaped = x.data.reshape(shape[0], G, -1, *shape[2:])
        mean = x_reshaped.mean(axis=(2, 3), keepdims=True)
        var = x_reshaped.var(axis=(2, 3), keepdims=True, ddof=0)
        x_norm = (x_reshaped - mean) / np.sqrt(var + self.eps)
        x_norm = x_norm.reshape(shape)
        if hasattr(self, 'weight'):
            x_norm = x_norm * self.weight.data
        if hasattr(self, 'bias'):
            x_norm = x_norm + self.bias.data
        return Tensor(x_norm, requires_grad=x.requires_grad)


class InstanceNorm(Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        import numpy as np
        if affine:
            self.register_parameter('weight', Tensor(np.ones(num_features, dtype=np.float32)))
            self.register_parameter('bias', Tensor(np.zeros(num_features, dtype=np.float32)))

    def forward(self, x):
        import numpy as np
        mean = x.data.mean(axis=(2, 3), keepdims=True)
        var = x.data.var(axis=(2, 3), keepdims=True, ddof=0)
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        if hasattr(self, 'weight'):
            x_norm = x_norm * self.weight.data.reshape(1, -1, 1, 1)
        if hasattr(self, 'bias'):
            x_norm = x_norm + self.bias.data.reshape(1, -1, 1, 1)
        return Tensor(x_norm, requires_grad=x.requires_grad)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        import numpy as np
        if not self.training or self.p == 0:
            return x
        mask = np.random.binomial(1, 1 - self.p, x.shape).astype(np.float32) / (1 - self.p)
        return Tensor(x.data * mask, requires_grad=x.requires_grad)


class Dropout2d(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        import numpy as np
        if not self.training or self.p == 0:
            return x
        shape = x.data.shape
        mask = np.random.binomial(1, 1 - self.p, (shape[0], shape[1], 1, 1)).astype(np.float32) / (1 - self.p)
        return Tensor(x.data * mask, requires_grad=x.requires_grad)


class AlphaDropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        import numpy as np
        if not self.training or self.p == 0:
            return x
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        keep_prob = 1 - self.p
        mask = np.random.binomial(1, keep_prob, x.shape).astype(np.float32)
        a = alpha * (keep_prob * (1 - keep_prob)) ** 0.5
        b = -a * keep_prob ** 0.5
        noise = mask * a + b
        out = mask * x.data + (1 - mask) * noise
        return Tensor(out * scale, requires_grad=x.requires_grad)


# ---------------------------------------------------------------------------
# Convolution layers
# ---------------------------------------------------------------------------

class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        import numpy as np
        fan_in = in_channels // groups * kernel_size
        weight = Tensor(np.random.randn(out_channels, in_channels // groups, kernel_size).astype(np.float32) * math.sqrt(2.0 / fan_in))
        self.register_parameter('weight', weight)
        if bias:
            self.register_parameter('bias', Tensor(np.zeros(out_channels, dtype=np.float32)))
        else:
            self.bias = None

    def forward(self, x):
        import numpy as np
        B, C, L = x.data.shape
        K = self.kernel_size
        P = self.padding
        S = self.stride
        D = self.dilation
        L_out = (L + 2 * P - D * (K - 1) - 1) // S + 1
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (P, P))) if P > 0 else x.data
        cols = np.zeros((B, self.in_channels // self.groups, K, L_out))
        for i in range(K):
            start = i * D
            cols[:, :, i, :] = x_padded[:, :, start:start + L_out * S:S]
        out = np.zeros((B, self.out_channels, L_out))
        for g in range(self.groups):
            w = self.weight.data[g * self.out_channels // self.groups:(g + 1) * self.out_channels // self.groups]
            c_in = cols[:, g * self.in_channels // self.groups:(g + 1) * self.in_channels // self.groups].reshape(B, -1, L_out)
            out[:, g * self.out_channels // self.groups:(g + 1) * self.out_channels // self.groups] = np.einsum('bik,bjk->bij', w.reshape(self.out_channels // self.groups, -1), c_in)
        if self.bias is not None:
            out = out + self.bias.data.reshape(1, -1, 1)
        return Tensor(out, requires_grad=x.requires_grad)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        import numpy as np
        KH, KW = self.kernel_size
        fan_in = in_channels // groups * KH * KW
        weight = Tensor(np.random.randn(out_channels, in_channels // groups, KH, KW).astype(np.float32) * math.sqrt(2.0 / fan_in))
        self.register_parameter('weight', weight)
        if bias:
            self.register_parameter('bias', Tensor(np.zeros(out_channels, dtype=np.float32)))
        else:
            self.bias = None

    def forward(self, x):
        import numpy as np
        B, C, H, W = x.data.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding
        DH, DW = self.dilation
        H_out = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
        W_out = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (PH, PH), (PW, PW))) if PH > 0 or PW > 0 else x.data
        cols = np.zeros((B, self.in_channels // self.groups, KH, KW, H_out, W_out))
        for i in range(KH):
            for j in range(KW):
                h_start = i * DH
                w_start = j * DW
                cols[:, :, i, j, :, :] = x_padded[:, :, h_start:h_start + H_out * SH:SH, w_start:w_start + W_out * SW:SW]
        G = self.groups
        Cg = self.in_channels // G
        Og = self.out_channels // G
        cols = cols.reshape(B, G, Cg * KH * KW, H_out * W_out)
        out = np.zeros((B, self.out_channels, H_out * W_out))
        for g in range(G):
            w = self.weight.data[g * Og:(g + 1) * Og].reshape(Og, -1)
            out[:, g * Og:(g + 1) * Og] = np.einsum('ij,bjk->bik', w, cols[:, g])
        out = out.reshape(B, self.out_channels, H_out, W_out)
        if self.bias is not None:
            out = out + self.bias.data.reshape(1, -1, 1, 1)
        return Tensor(out, requires_grad=x.requires_grad)


class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride, stride)
        if isinstance(padding, int): padding = (padding, padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation, dilation)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        import numpy as np
        KD, KH, KW = kernel_size
        fan_in = in_channels // groups * KD * KH * KW
        weight = Tensor(np.random.randn(out_channels, in_channels // groups, KD, KH, KW).astype(np.float32) * math.sqrt(2.0 / fan_in))
        self.register_parameter('weight', weight)
        if bias:
            self.register_parameter('bias', Tensor(np.zeros(out_channels, dtype=np.float32)))
        else:
            self.bias = None

    def forward(self, x):
        import numpy as np
        B, C, D, H, W = x.data.shape
        KD, KH, KW = self.kernel_size
        SD, SH, SW = self.stride
        PD, PH, PW = self.padding
        DD, DH, DW = self.dilation
        D_out = (D + 2 * PD - DD * (KD - 1) - 1) // SD + 1
        H_out = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
        W_out = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1
        x_padded = np.pad(x.data, ((0,0),(0,0),(PD,PD),(PH,PH),(PW,PW))) if any(p>0 for p in self.padding) else x.data
        out = np.zeros((B, self.out_channels, D_out, H_out, W_out))
        for kd in range(KD):
            for kh in range(KH):
                for kw in range(KW):
                    d_s = kd * DD
                    h_s = kh * DH
                    w_s = kw * DW
                    slab = x_padded[:, :, d_s:d_s+D_out*SD:SD, h_s:h_s+H_out*SH:SH, w_s:w_s+W_out*SW:SW]
                    for co in range(self.out_channels):
                        out[:, co] += np.sum(slab * self.weight.data[co], axis=(1,2,3)) if slab.ndim > 4 else np.einsum('bijk,ijk->bi', slab, self.weight.data[co])
        if self.bias is not None:
            out = out + self.bias.data.reshape(1, -1, 1, 1, 1)
        return Tensor(out, requires_grad=x.requires_grad)


class ConvTransposed1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        import numpy as np
        weight = Tensor(np.random.randn(in_channels, out_channels // groups, kernel_size).astype(np.float32) * 0.02)
        self.register_parameter('weight', weight)
        if bias:
            self.register_parameter('bias', Tensor(np.zeros(out_channels, dtype=np.float32)))
        else:
            self.bias = None

    def forward(self, x):
        import numpy as np
        B, C, L_in = x.data.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        L_out = (L_in - 1) * S - 2 * P + K + self.output_padding
        out = np.zeros((B, self.out_channels, L_out))
        x_t = x.data.reshape(B, self.groups, -1, L_in)
        for i in range(K):
            start = i - P
            for j in range(L_in):
                pos = start + j * S
                if 0 <= pos < L_out:
                    out[:, :, pos] += np.einsum('bgc,goc->bo', x_t[:, :, :, j], self.weight.data[:, :, i])
        if self.bias is not None:
            out = out + self.bias.data.reshape(1, -1, 1)
        return Tensor(out, requires_grad=x.requires_grad)


class ConvTransposed2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        import numpy as np
        weight = Tensor(np.random.randn(in_channels, out_channels // groups, *kernel_size).astype(np.float32) * 0.02)
        self.register_parameter('weight', weight)
        if bias:
            self.register_parameter('bias', Tensor(np.zeros(out_channels, dtype=np.float32)))
        else:
            self.bias = None

    def forward(self, x):
        import numpy as np
        B, C, H_in, W_in = x.data.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding
        H_out = (H_in - 1) * SH - 2 * PH + KH + self.output_padding
        W_out = (W_in - 1) * SW - 2 * PW + KW + self.output_padding
        out = np.zeros((B, self.out_channels, H_out, W_out))
        x_t = x.data.reshape(B, self.groups, -1, H_in, W_in)
        for ki in range(KH):
            for kj in range(KW):
                for hi in range(H_in):
                    for wi in range(W_in):
                        ho = hi * SH - PH + ki
                        wo = wi * SW - PW + kj
                        if 0 <= ho < H_out and 0 <= wo < W_out:
                            out[:, :, ho, wo] += np.einsum('bgc,goc->bo', x_t[:, :, :, hi, wi], self.weight.data[:, :, ki, kj])
        if self.bias is not None:
            out = out + self.bias.data.reshape(1, -1, 1, 1)
        return Tensor(out, requires_grad=x.requires_grad)


class ConvTransposed3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride, stride)
        if isinstance(padding, int): padding = (padding, padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        import numpy as np
        weight = Tensor(np.random.randn(in_channels, out_channels // groups, *kernel_size).astype(np.float32) * 0.02)
        self.register_parameter('weight', weight)
        if bias:
            self.register_parameter('bias', Tensor(np.zeros(out_channels, dtype=np.float32)))
        else:
            self.bias = None

    def forward(self, x):
        import numpy as np
        B, C, D_in, H_in, W_in = x.data.shape
        KD, KH, KW = self.kernel_size
        SD, SH, SW = self.stride
        PD, PH, PW = self.padding
        D_out = (D_in - 1) * SD - 2 * PD + KD + self.output_padding
        H_out = (H_in - 1) * SH - 2 * PH + KH + self.output_padding
        W_out = (W_in - 1) * SW - 2 * PW + KW + self.output_padding
        out = np.zeros((B, self.out_channels, D_out, H_out, W_out))
        return Tensor(out, requires_grad=x.requires_grad)


class DepthwiseConv2d(Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        import numpy as np
        weight = Tensor(np.random.randn(in_channels, 1, *kernel_size).astype(np.float32) * 0.02)
        self.register_parameter('weight', weight)

    def forward(self, x):
        return Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.in_channels, bias=False)(x)


class SeparableConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = DepthwiseConv2d(in_channels, kernel_size, stride, padding)
        self.pointwise = Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ---------------------------------------------------------------------------
# Pooling layers
# ---------------------------------------------------------------------------

class AvgPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if isinstance(self.stride, int): self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        import numpy as np
        B, C, H, W = x.data.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1
        x_padded = np.pad(x.data, ((0,0),(0,0),(PH,PH),(PW,PW))) if any(p>0 for p in self.padding) else x.data
        out = np.zeros((B, C, H_out, W_out))
        for i in range(KH):
            for j in range(KW):
                out += x_padded[:, :, i:i+H_out*SH:SH, j:j+W_out*SW:SW]
        return Tensor(out / (KH * KW), requires_grad=x.requires_grad)


class MaxPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if isinstance(self.stride, int): self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x):
        import numpy as np
        B, C, H, W = x.data.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding
        H_out = (H + 2 * PH - KH) // SH + 1
        W_out = (W + 2 * PW - KW) // SW + 1
        x_padded = np.pad(x.data, ((0,0),(0,0),(PH,PH),(PW,PW)), constant_values=-1e9) if any(p>0 for p in self.padding) else x.data
        out = np.full((B, C, H_out, W_out), -1e9)
        for i in range(KH):
            for j in range(KW):
                out = np.maximum(out, x_padded[:, :, i:i+H_out*SH:SH, j:j+W_out*SW:SW])
        return Tensor(out, requires_grad=x.requires_grad)


class AdaptiveAvgPool(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def forward(self, x):
        import numpy as np
        B, C, H, W = x.data.shape
        OH, OW = self.output_size
        kernel_h = H // OH
        kernel_w = W // OW
        out = x.data.reshape(B, C, OH, kernel_h, OW, kernel_w).mean(axis=(3, 5))
        return Tensor(out, requires_grad=x.requires_grad)


class AdaptiveMaxPool(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def forward(self, x):
        import numpy as np
        B, C, H, W = x.data.shape
        OH, OW = self.output_size
        kernel_h = H // OH
        kernel_w = W // OW
        out = x.data.reshape(B, C, OH, kernel_h, OW, kernel_w).max(axis=(3, 5))
        return Tensor(out, requires_grad=x.requires_grad)


# ---------------------------------------------------------------------------
# Upsample
# ---------------------------------------------------------------------------

class Upsample(Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        import numpy as np
        if self.scale_factor is not None:
            sf = self.scale_factor if isinstance(self.scale_factor, tuple) else (self.scale_factor, self.scale_factor)
            out = np.repeat(np.repeat(x.data, int(sf[0]), axis=2), int(sf[1]), axis=3)
        else:
            out = x.data
        return Tensor(out, requires_grad=x.requires_grad)


# ---------------------------------------------------------------------------
# Container layers
# ---------------------------------------------------------------------------

class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)


class Unflatten(Module):
    def __init__(self, dim, unflattened_size):
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, x):
        shape = list(x.shape)
        shape[self.dim:self.dim+1] = list(self.unflattened_size) if isinstance(self.unflattened_size, tuple) else [self.unflattened_size]
        return x.reshape(*shape)


class Identity(Module):
    def forward(self, x):
        return x


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(*list(self._modules.values())[idx])
        return list(self._modules.values())[idx]


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                self._modules[str(i)] = m

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def __setitem__(self, idx, module):
        self._modules[str(idx)] = module

    def append(self, module):
        self._modules[str(len(self._modules))] = module

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for k, v in modules.items():
                self._modules[k] = v

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self._modules[key] = module

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class ParameterList(Module):
    def __init__(self, parameters=None):
        super().__init__()
        if parameters is not None:
            for i, p in enumerate(parameters):
                self._parameters[str(i)] = p

    def __len__(self):
        return len(self._parameters)

    def __getitem__(self, idx):
        return list(self._parameters.values())[idx]


class ParameterDict(Module):
    def __init__(self, parameters=None):
        super().__init__()
        if parameters is not None:
            for k, v in parameters.items():
                self._parameters[k] = v

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, param):
        self._parameters[key] = param


class LazyLinear(Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features
        self.linear = None

    def forward(self, x):
        if self.linear is None:
            in_features = x.shape[-1]
            self.linear = Linear(in_features, self.out_features)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Weight reparameterization
# ---------------------------------------------------------------------------

class WeightNorm(Module):
    def __init__(self, module, name='weight', dim=0):
        super().__init__()
        self.module = module
        self.name = name
        self.dim = dim

    def forward(self, x):
        import numpy as np
        w = getattr(self.module, self.name)
        norm = np.sqrt(np.sum(w.data ** 2, axis=self.dim, keepdims=True) + 1e-8)
        setattr(self.module, self.name, Tensor(w.data / norm, requires_grad=w.requires_grad))
        out = self.module(x)
        setattr(self.module, self.name, w)
        return out


class SpectralNorm(Module):
    def __init__(self, module, name='weight', n_power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.u = None

    def forward(self, x):
        import numpy as np
        w = getattr(self.module, self.name)
        shape = w.data.shape
        mat = w.data.reshape(shape[0], -1)
        if self.u is None or self.u.shape[0] != shape[0]:
            self.u = np.random.randn(shape[0]).astype(np.float32)
        with no_grad():
            for _ in range(self.n_power_iterations):
                v = mat.T @ self.u
                v = v / (np.linalg.norm(v) + 1e-8)
                u = mat @ v
                u = u / (np.linalg.norm(u) + 1e-8)
            self.u = u
        sigma = u @ mat @ v
        setattr(self.module, self.name, Tensor(w.data / sigma, requires_grad=w.requires_grad))
        out = self.module(x)
        setattr(self.module, self.name, w)
        return out


# ---------------------------------------------------------------------------
# Activations as Modules
# ---------------------------------------------------------------------------

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class ReLU6(Module):
    def forward(self, x):
        return x.relu6()

class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return x.leaky_relu(self.alpha)

class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        import numpy as np
        self.register_parameter('weight', Tensor(np.full(num_parameters, init, dtype=np.float32)))
    def forward(self, x):
        import numpy as np
        return Tensor(np.where(x.data > 0, x.data, self.weight.data * x.data), requires_grad=x.requires_grad)

class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return x.elu(self.alpha)

class SELU(Module):
    def forward(self, x):
        return x.selu()

class GELU(Module):
    def forward(self, x):
        return x.gelu()

class SiLU(Module):
    def forward(self, x):
        return x.silu()

class Mish(Module):
    def forward(self, x):
        import numpy as np
        sp = np.log1p(np.exp(x.data))
        return Tensor(x.data * np.tanh(sp), requires_grad=x.requires_grad)

class Hardswish(Module):
    def forward(self, x):
        return x.hardswish()

class GLU(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        import numpy as np
        a, b = np.split(x.data, 2, axis=self.dim)
        sig = 1 / (1 + np.exp(-np.clip(b, -500, 500)))
        return Tensor(a * sig, requires_grad=x.requires_grad)

class SwiGLU(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        import numpy as np
        a, b = np.split(x.data, 2, axis=self.dim)
        sig = 1 / (1 + np.exp(-np.clip(b, -500, 500)))
        return Tensor(a * b * sig, requires_grad=x.requires_grad)

class GEGLU(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        import numpy as np
        a, b = np.split(x.data, 2, axis=self.dim)
        c = 0.7978845608028654
        k = 0.044715
        t = np.tanh(c * (b + k * b ** 3))
        return Tensor(a * 0.5 * b * (1 + t), requires_grad=x.requires_grad)

class ReGLU(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        import numpy as np
        a, b = np.split(x.data, 2, axis=self.dim)
        return Tensor(a * np.maximum(0, b), requires_grad=x.requires_grad)

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.softmax(self.dim)

class LogSoftmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.log_softmax(self.dim)
