"""Custom autodiff tensor engine. Pure Python + NumPy."""
import array
import math
import struct
from contextlib import contextmanager
from functools import lru_cache
import threading
from qythera.backend import (
    HAS_NUMPY, np,
    zeros as _bk_zeros, ones as _bk_ones, full as _bk_full, empty as _bk_empty,
    arange as _bk_arange, from_list as _bk_from_list,
    matmul as _bk_matmul, einsum as _bk_einsum,
    sum as _bk_sum, mean as _bk_mean, max as _bk_max, min as _bk_min,
    exp as _bk_exp, log as _bk_log, sqrt as _bk_sqrt, abs as _bk_abs,
    sin as _bk_sin, cos as _bk_cos, tanh as _bk_tanh,
    softmax as _bk_softmax, where as _bk_where,
    stack as _bk_stack, concatenate as _bk_concatenate,
    reshape as _bk_reshape, broadcast_to as _bk_broadcast_to,
    to_numpy as _bk_to_numpy, to_list as _bk_to_list
)

_local = threading.local()

# ---------------------------------------------------------------------------
# DType system
# ---------------------------------------------------------------------------

class DType:
    def __init__(self, name, bits, min_val, max_val, eps=0.0, is_float=True):
        self.name = name
        self.bits = bits
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps
        self.is_float = is_float

    def __repr__(self):
        return f"dtype({self.name})"

    def __eq__(self, other):
        return isinstance(other, DType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

BOOL   = DType("BOOL",   1,   False, True, 0.0, False)
INT2   = DType("INT2",   2,  -2,     1,   0.0, False)
INT4   = DType("INT4",   4,  -8,     7,   0.0, False)
INT8   = DType("INT8",   8,  -128,   127, 0.0, False)
FP8E5M2 = DType("FP8E5M2", 8, -57344, 57344, 0.015625, True)
FP8E4M3 = DType("FP8E4M3", 8, -448,   448,  0.001953125, True)
FP16   = DType("FP16",  16, -65504, 65504, 9.77e-4, True)
BF16   = DType("BF16",  16, -3.39e38, 3.39e38, 0.0078125, True)
TF32   = DType("TF32",  32, -3.4e38, 3.4e38, 1.19e-7, True)
FP32   = DType("FP32",  32, -3.4e38, 3.4e38, 1.19e-7, True)
FP64   = DType("FP64",  64, -1.8e308, 1.8e308, 2.22e-16, True)

dtype_info = {
    "BOOL": BOOL, "INT2": INT2, "INT4": INT4, "INT8": INT8,
    "FP8E5M2": FP8E5M2, "FP8E4M3": FP8E4M3,
    "FP16": FP16, "BF16": BF16, "TF32": TF32, "FP32": FP32, "FP64": FP64,
}

def np_dtype(dt):
    m = {"BOOL": np.bool_, "INT2": np.int8, "INT4": np.int8, "INT8": np.int8,
         "FP8E5M2": np.float32, "FP8E4M3": np.float32,
         "FP16": np.float16, "BF16": np.float32, "TF32": np.float32,
         "FP32": np.float32, "FP64": np.float64}
    return m.get(dt.name, np.float32)

def cast(val, dtype):
    dt = dtype_info[dtype] if isinstance(dtype, str) else dtype
    if isinstance(val, Tensor):
        return Tensor(val.data.astype(np_dtype(dt)), requires_grad=val.requires_grad)
    return _bk_from_list(val, dtype=np_dtype(dt))

def pack_int4(arr):
    arr = np.asarray(arr, dtype=np.int8).flatten()
    n = len(arr)
    out = bytearray((n + 1) // 2)
    for i in range(n):
        val = int(arr[i]) & 0x0F
        if i % 2 == 0:
            out[i // 2] = val
        else:
            out[i // 2] |= val << 4
    return bytes(out)

def clamp_to_dtype(val, dtype):
    dt = dtype_info[dtype] if isinstance(dtype, str) else dtype
    return np.clip(val.data if isinstance(val, Tensor) else val, dt.min_val, dt.max_val).astype(np_dtype(dt))

def unpack_int4(data, n):
    data = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    out = _bk_zeros(n, dtype=np.int8)
    for i in range(n):
        b = data[i // 2]
        out[i] = (b & 0x0F) if (i % 2 == 0) else ((b >> 4) & 0x0F)
    out[out > 7] -= 16
    return out

def pack_int2(arr):
    arr = np.asarray(arr, dtype=np.int8).flatten()
    n = len(arr)
    out = bytearray((n + 3) // 4)
    for i in range(n):
        v = arr[i] & 0x03
        shift = (i % 4) * 2
        out[i // 4] |= v << shift
    return bytes(out)

def unpack_int2(data, n):
    data = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    out = _bk_zeros(n, dtype=np.int8)
    for i in range(n):
        b = data[i // 4]
        shift = (i % 4) * 2
        out[i] = (b >> shift) & 0x03
    out[out > 1] -= 4
    return out

def quantize(arr, dtype):
    arr = np.asarray(arr, dtype=np.float32)
    dt = dtype_info[dtype] if isinstance(dtype, str) else dtype
    if dt.name == "INT8":
        scale = max(abs(arr.max()), abs(arr.min()), 1e-8) / 127.0
        return (arr / scale).clip(-128, 127).astype(np.int8), scale
    if dt.name == "INT4":
        scale = max(abs(arr.max()), abs(arr.min()), 1e-8) / 7.0
        q = (arr / scale).clip(-8, 7).astype(np.int8)
        return pack_int4(q), scale
    if dt.name == "INT2":
        scale = max(abs(arr.max()), abs(arr.min()), 1e-8) / 1.0
        q = (arr / scale).clip(-2, 1).astype(np.int8)
        return pack_int2(q), scale
    return arr, 1.0

def dequantize(data, scale, shape, dtype):
    dt = dtype_info[dtype] if isinstance(dtype, str) else dtype
    if dt.name == "INT8":
        arr = np.frombuffer(data, dtype=np.int8).astype(np.float32) * scale
        return arr.reshape(shape)
    if dt.name == "INT4":
        n = 1
        for s in shape: n *= s
        arr = unpack_int4(data, n).astype(np.float32) * scale
        return arr.reshape(shape)
    if dt.name == "INT2":
        n = 1
        for s in shape: n *= s
        arr = unpack_int2(data, n).astype(np.float32) * scale
        return arr.reshape(shape)
    return np.asarray(data, dtype=np_dtype(dt)).reshape(shape)

# ---------------------------------------------------------------------------
# Computation graph & autograd
# ---------------------------------------------------------------------------

class Context:
    __slots__ = ("op", "saved", "inputs")
    def __init__(self, op, inputs=(), saved=()):
        self.op = op
        self.inputs = inputs
        self.saved = list(saved)

def _topological_sort(tensor):
    visited = set()
    order = []
    stack = [tensor]
    while stack:
        t = stack[-1]
        tid = id(t)
        if tid in visited:
            stack.pop()
            continue
        children_done = True
        for c in (t._ctx.inputs if t._ctx else ()):
            if id(c) not in visited:
                children_done = False
                stack.append(c)
        if children_done:
            visited.add(tid)
            order.append(t)
            stack.pop()
    return order

@contextmanager
def no_grad():
    prev = getattr(_local, "no_grad_flag", False)
    _local.no_grad_flag = True
    try:
        yield
    finally:
        _local.no_grad_flag = prev

@contextmanager
def enable_grad():
    prev = getattr(_local, "no_grad_flag", False)
    _local.no_grad_flag = False
    try:
        yield
    finally:
        _local.no_grad_flag = prev

def _grad_enabled():
    return not getattr(_local, "no_grad_flag", False)

# ---------------------------------------------------------------------------
# Backward ops
# ---------------------------------------------------------------------------

class AddBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        return _unbroadcast(grad, a.shape), _unbroadcast(grad, b.shape)

class SubBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        return _unbroadcast(grad, a.shape), _unbroadcast(-grad, b.shape)

class MulBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        return _unbroadcast(grad * b.data, a.shape), _unbroadcast(grad * a.data, b.shape)

class DivBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        eps = 1e-8
        b_plus_eps = b.data + eps
        return _unbroadcast(grad / b_plus_eps, a.shape), _unbroadcast(-grad * a.data / (b_plus_eps ** 2), b.shape)

class PowBackward:
    @staticmethod
    def backward(ctx, grad):
        a = ctx.inputs[0]
        exp = ctx.saved[0]
        if isinstance(exp, Tensor):
            return _unbroadcast(grad * exp.data * (a.data ** (exp.data - 1)), a.shape), \
                   _unbroadcast(grad * (a.data ** exp.data) * _bk_log(np.maximum(a.data, 1e-30)), exp.shape)
        return _unbroadcast(grad * exp * (a.data ** (exp - 1)), a.shape), None

class ExpBackward:
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.saved[0],

class LogBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad / ctx.saved[0], ctx.inputs[0].shape),

class SqrtBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad / (2 * ctx.saved[0]), ctx.inputs[0].shape),

class NegBackward:
    @staticmethod
    def backward(ctx, grad):
        return -grad,

class AbsBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad * np.sign(ctx.inputs[0].data), ctx.inputs[0].shape),

class SignBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(np.zeros_like(ctx.inputs[0].data), ctx.inputs[0].shape),

class SinBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad * _bk_cos(ctx.inputs[0].data), ctx.inputs[0].shape),

class CosBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(-grad * _bk_sin(ctx.inputs[0].data), ctx.inputs[0].shape),

class TanBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad / (ctx.saved[0] ** 2), ctx.inputs[0].shape),

class AsinBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        return _unbroadcast(grad / _bk_sqrt(1 - x * x + 1e-7), x.shape),

class AcosBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        return _unbroadcast(-grad / _bk_sqrt(1 - x * x + 1e-7), x.shape),

class AtanBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        return _unbroadcast(grad / (1 + x * x), x.shape),

class Atan2Backward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        denom = a.data ** 2 + b.data ** 2 + 1e-7
        return _unbroadcast(grad * b.data / denom, a.shape), \
               _unbroadcast(-grad * a.data / denom, b.shape)

class SinhBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad * np.cosh(ctx.inputs[0].data), ctx.inputs[0].shape),

class CoshBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad * np.sinh(ctx.inputs[0].data), ctx.inputs[0].shape),

class TanhBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad * (1 - ctx.saved[0] ** 2), ctx.inputs[0].shape),

class ArcsinhBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        return _unbroadcast(grad / _bk_sqrt(x * x + 1 + 1e-7), x.shape),

class ArccoshBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        return _unbroadcast(grad / _bk_sqrt(x * x - 1 + 1e-7), x.shape),

class ArctanhBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        return _unbroadcast(grad / (1 - x * x + 1e-7), x.shape),

class ClampBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        lo, hi = ctx.saved
        mask = ((x >= lo) & (x <= hi)).astype(np.float32)
        return _unbroadcast(grad * mask, x.shape),

class MaximumBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        mask_a = (a.data >= b.data).astype(np.float32)
        return _unbroadcast(grad * mask_a, a.shape), _unbroadcast(grad * (1 - mask_a), b.shape),

class MinimumBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        mask_a = (a.data <= b.data).astype(np.float32)
        return _unbroadcast(grad * mask_a, a.shape), _unbroadcast(grad * (1 - mask_a), b.shape),

class FloorBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data),

class CeilBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data),

class RoundBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data),

class TruncBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data),

class SigmoidBackward:
    @staticmethod
    def backward(ctx, grad):
        s = ctx.saved[0]
        return _unbroadcast(grad * s * (1 - s), ctx.inputs[0].shape),

class ReluBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad * (ctx.inputs[0].data > 0).astype(np.float32), ctx.inputs[0].shape),

class GeluBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        c = 0.7978845608028654
        k = 0.044715
        t = _bk_tanh(c * (x + k * x ** 3))
        dt = 1 - t ** 2
        dx = c * (1 + 3 * k * x ** 2) * dt
        return _unbroadcast(grad * (0.5 * (1 + t) + 0.5 * x * dx), x.shape),

class SiluBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        s = 1 / (1 + _bk_exp(-np.clip(x, -500, 500)))
        return _unbroadcast(grad * (s + x * s * (1 - s)), x.shape),

class SoftmaxBackward:
    @staticmethod
    def backward(ctx, grad):
        s = ctx.saved[0]
        axis = ctx.saved[1]
        dot = (grad * s).sum(axis=axis, keepdims=True)
        return _unbroadcast(s * (grad - dot), ctx.inputs[0].shape),

class LogSoftmaxBackward:
    @staticmethod
    def backward(ctx, grad):
        s = ctx.saved[0]
        axis = ctx.saved[1]
        return _unbroadcast(grad - grad.sum(axis=axis, keepdims=True) * s, ctx.inputs[0].shape),

class SumBackward:
    @staticmethod
    def backward(ctx, grad):
        shape, keepdims, axis = ctx.saved
        if axis is not None and not keepdims:
            grad = np.expand_dims(grad, axis=axis)
        return _bk_broadcast_to(grad, shape).copy(),

class MeanBackward:
    @staticmethod
    def backward(ctx, grad):
        shape, keepdims, axis, n = ctx.saved
        if axis is not None and not keepdims:
            grad = np.expand_dims(grad, axis=axis)
        return _bk_broadcast_to(grad / n, shape).copy(),

class MaxBackward:
    @staticmethod
    def backward(ctx, grad):
        shape, keepdims, axis = ctx.saved
        idx = ctx.inputs[0].data
        if isinstance(idx, tuple):
            idx = idx[0]
        g = np.zeros_like(ctx.inputs[0].data)
        if axis is not None:
            if keepdims:
                np.put_along_axis(g, np.argmax(ctx.inputs[0].data, axis=axis, keepdims=True), grad, axis=axis)
            else:
                np.put_along_axis(g, np.argmax(ctx.inputs[0].data, axis=axis, keepdims=True), np.expand_dims(grad, axis=axis), axis=axis)
        else:
            g.flat[np.argmax(ctx.inputs[0].data)] = grad
        return g,

class MinBackward:
    @staticmethod
    def backward(ctx, grad):
        shape, keepdims, axis = ctx.saved
        g = np.zeros_like(ctx.inputs[0].data)
        if axis is not None:
            if keepdims:
                np.put_along_axis(g, np.argmin(ctx.inputs[0].data, axis=axis, keepdims=True), grad, axis=axis)
            else:
                np.put_along_axis(g, np.argmin(ctx.inputs[0].data, axis=axis, keepdims=True), np.expand_dims(grad, axis=axis), axis=axis)
        else:
            g.flat[np.argmin(ctx.inputs[0].data)] = grad
        return g,

class ProdBackward:
    @staticmethod
    def backward(ctx, grad):
        shape, keepdims, axis = ctx.saved
        x = ctx.inputs[0].data
        p = x.prod(axis=axis, keepdims=keepdims or axis is not None)
        return _unbroadcast(grad * p / (x + 1e-30), shape),

class ArgmaxBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data),

class ArgminBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data),

class VarBackward:
    @staticmethod
    def backward(ctx, grad):
        shape, keepdims, axis, n, correction = ctx.saved
        x = ctx.inputs[0].data
        mean = x.mean(axis=axis, keepdims=True)
        g = 2 * grad * (x - mean) / (n - correction + 1e-7)
        if not keepdims and axis is not None:
            g = g.squeeze(axis=axis)
        return _unbroadcast(g, shape),

class StdBackward:
    @staticmethod
    def backward(ctx, grad):
        shape, keepdims, axis, n, correction = ctx.saved
        x = ctx.inputs[0].data
        mean = x.mean(axis=axis, keepdims=True)
        std_val = x.std(axis=axis, keepdims=True, ddof=correction) + 1e-7
        g = grad * (x - mean) / (std_val * (n - correction + 1e-7))
        if not keepdims and axis is not None:
            g = g.squeeze(axis=axis)
        return _unbroadcast(g, shape),

class LogsumexpBackward:
    @staticmethod
    def backward(ctx, grad):
        shape, keepdims, axis = ctx.saved
        x = ctx.inputs[0].data
        lse = x.max(axis=axis, keepdims=True)
        exp_x = _bk_exp(x - lse)
        s = exp_x.sum(axis=axis, keepdims=True)
        g = grad * exp_x / s
        if not keepdims and axis is not None:
            g = g.squeeze(axis=axis)
        return _unbroadcast(g, shape),

class CumsumBackward:
    @staticmethod
    def backward(ctx, grad):
        axis = ctx.saved[0]
        return grad.cumsum(axis=axis),

class CumprodBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        out = ctx.saved[0]
        axis = ctx.saved[1]
        rev_x = np.flip(x, axis=axis)
        rev_cp = np.cumprod(rev_x, axis=axis)
        rev_cp = np.flip(rev_cp, axis=axis)
        grad_flipped = np.flip(grad, axis=axis)
        result = grad_flipped * rev_cp
        result = np.flip(result, axis=axis)
        result = result / (x + 1e-30)
        return result,

class MatMulBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        ad, bd = a.data, b.data
        if ad.ndim >= 2 and bd.ndim >= 2:
            batch_axes = tuple(range(max(0, ad.ndim - 2)))
            if ad.ndim == bd.ndim and ad.ndim > 2:
                ga = _bk_einsum('...ij,...kj->...ik', grad, bd)
                gb = _bk_einsum('...ij,...ik->...kj', ad, grad)
            else:
                ga = grad @ np.swapaxes(bd, -1, -2)
                gb = np.swapaxes(ad, -1, -2) @ grad
        else:
            ga = grad @ bd.T
            gb = ad.T @ grad
        return _unbroadcast(ga, a.shape), _unbroadcast(gb, b.shape)

class EmbeddingBackward:
    @staticmethod
    def backward(ctx, grad):
        idx = ctx.saved[0]
        weight = ctx.saved[1]
        g = np.zeros_like(weight)
        grad_reshaped = grad.reshape(len(idx), -1)
        np.add.at(g, idx, grad_reshaped)
        return g, None

class RMSNormBackward:
    @staticmethod
    def backward(ctx, grad):
        x, gamma, eps = ctx.saved
        norm = _bk_sqrt(_bk_mean(x ** 2, axis=-1, keepdims=True) + eps)
        gx = gamma * grad / norm
        dnorm = -(x * gamma * grad).sum(axis=-1, keepdims=True) / (norm ** 3)
        gx += x * dnorm / x.shape[-1]
        return gx, (grad * x / norm).sum(axis=tuple(range(x.ndim - 1))), None

class SiLUBackward2:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        sig = 1 / (1 + _bk_exp(-np.clip(x, -500, 500)))
        return _unbroadcast(grad * (sig + x * sig * (1 - sig)), x.shape),

class GELUBackward2:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0].data
        c = 0.7978845608028654
        k = 0.044715
        t = _bk_tanh(c * (x + k * x ** 3))
        dt = 1 - t ** 2
        dx = c * (1 + 3 * k * x ** 2) * dt
        return _unbroadcast(grad * (0.5 * (1 + t) + 0.5 * x * dx), x.shape),

class TanhBackward2:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad * (1 - ctx.saved[0] ** 2), ctx.inputs[0].shape),

class LeakyReLUBackward:
    @staticmethod
    def backward(ctx, grad):
        alpha = ctx.saved[0]
        mask = (ctx.inputs[0].data > 0).astype(np.float32) + alpha * (ctx.inputs[0].data <= 0).astype(np.float32)
        return _unbroadcast(grad * mask, ctx.inputs[0].shape),

class ELUBackward:
    @staticmethod
    def backward(ctx, grad):
        alpha = ctx.saved[0]
        mask = (ctx.inputs[0].data > 0).astype(np.float32) + alpha * _bk_exp(ctx.inputs[0].data) * (ctx.inputs[0].data <= 0).astype(np.float32)
        return _unbroadcast(grad * mask, ctx.inputs[0].shape),

class SELUBackward:
    @staticmethod
    def backward(ctx, grad):
        alpha = ctx.saved[0]
        lam = ctx.saved[1]
        mask = lam * ((ctx.inputs[0].data > 0).astype(np.float32) + alpha * _bk_exp(ctx.inputs[0].data) * (ctx.inputs[0].data <= 0).astype(np.float32))
        return _unbroadcast(grad * mask, ctx.inputs[0].shape),

class ReLU6Backward:
    @staticmethod
    def backward(ctx, grad):
        mask = ((ctx.inputs[0].data > 0) & (ctx.inputs[0].data < 6)).astype(np.float32)
        return _unbroadcast(grad * mask, ctx.inputs[0].shape),

class ReshapeBackward:
    @staticmethod
    def backward(ctx, grad):
        return grad.reshape(ctx.inputs[0].shape),

class TransposeBackward:
    @staticmethod
    def backward(ctx, grad):
        axes = ctx.saved[0]
        return np.transpose(grad, axes),

class PermuteBackward:
    @staticmethod
    def backward(ctx, grad):
        axes = ctx.saved[0]
        inv = [0] * len(axes)
        for i, a in enumerate(axes):
            inv[a] = i
        return np.transpose(grad, inv),

class SqueezeBackward:
    @staticmethod
    def backward(ctx, grad):
        return grad.reshape(ctx.inputs[0].shape),

class UnsqueezeBackward:
    @staticmethod
    def backward(ctx, grad):
        return grad.reshape(ctx.inputs[0].shape),

class ExpandBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad, ctx.inputs[0].shape),

class FlipBackward:
    @staticmethod
    def backward(ctx, grad):
        axes = ctx.saved[0]
        return np.flip(grad, axes),

class RollBackward:
    @staticmethod
    def backward(ctx, grad):
        shifts, axes = ctx.saved
        return np.roll(grad, -shifts, axes),

class GatherBackward:
    @staticmethod
    def backward(ctx, grad):
        x, idx, axis = ctx.saved
        g = np.zeros_like(x)
        idx_arrays = []
        for k in range(x.ndim):
            if k == axis:
                idx_arrays.append(idx)
            else:
                arr = np.arange(x.shape[k], dtype=np.int64)
                shape = [1] * idx.ndim
                shape[k] = x.shape[k]
                arr = arr.reshape(shape)
                arr = np.broadcast_to(arr, idx.shape).copy()
                idx_arrays.append(arr)
        np.add.at(g, tuple(idx_arrays), grad)
        return g, None

class ScatterBackward:
    @staticmethod
    def backward(ctx, grad):
        idx, dim, src_shape = ctx.saved
        g = np.zeros(src_shape, dtype=np.float32)
        idx_tuple = [slice(None)] * len(src_shape)
        idx_tuple[dim] = idx
        idx_tuple = tuple(idx_tuple)
        np.add.at(g, idx_tuple, grad)
        return g, None

class WhereBackward:
    @staticmethod
    def backward(ctx, grad):
        cond, x, y = ctx.saved
        gx = _bk_where(cond, grad, 0) if x is not None else None
        gy = _bk_where(~cond, grad, 0) if y is not None else None
        return (gx, gy, None)

class SortBackward:
    @staticmethod
    def backward(ctx, grad):
        idx, dim, descending = ctx.saved
        result = np.zeros_like(grad)
        np.put_along_axis(result, idx, grad, axis=dim)
        return result,

class TopkBackward:
    @staticmethod
    def backward(ctx, grad):
        x, idx, dim = ctx.saved
        g = np.zeros_like(x)
        np.put_along_axis(g, idx, grad, axis=dim)
        return g,

class EinsumBackward:
    @staticmethod
    def backward(ctx, grad):
        equation = ctx.saved[0]
        inputs = ctx.saved[1]
        grads = []
        parts = equation.split("->")
        input_subs_list = parts[0].split(",")
        output_sub = parts[1] if len(parts) > 1 else None
        for i, inp in enumerate(inputs):
            inp_sub = input_subs_list[i]
            remaining = [input_subs_list[j] for j in range(len(input_subs_list)) if j != i]
            remaining_str = ",".join(remaining) if remaining else ""
            if output_sub is not None:
                if remaining_str:
                    backward_eq = f"{output_sub},{remaining_str}->{inp_sub}"
                else:
                    backward_eq = f"{output_sub}->{inp_sub}"
            else:
                backward_eq = inp_sub
            backward_parts = backward_eq.split("->") if "->" in backward_eq else [backward_eq, inp_sub]
            actual_output = backward_parts[1] if len(backward_parts) > 1 else backward_parts[0]
            if len(actual_output) != len(set(actual_output)):
                g_i = _einsum_backward_repeated(grad, inp_sub, remaining_str, input_subs_list, output_sub, inputs, i)
            else:
                if remaining_str:
                    other_inputs_data = [inputs[j].data if isinstance(inputs[j], Tensor) else inputs[j]
                                        for j in range(len(inputs)) if j != i]
                    g_i = _bk_einsum(backward_eq, grad, *other_inputs_data)
                else:
                    g_i = _bk_einsum(backward_eq, grad)
            grads.append(g_i)
        return tuple(grads)


def _einsum_backward_repeated(grad, inp_sub, remaining_str, input_subs_list, output_sub, inputs, idx):
    inp_data = inputs[idx].data if isinstance(inputs[idx], Tensor) else inputs[idx]
    inp_shape = inp_data.shape
    remaining_inputs_data = [inputs[j].data if isinstance(inputs[j], Tensor) else inputs[j]
                             for j in range(len(inputs)) if j != idx]
    dim_sizes = {}
    for k, sub in enumerate(input_subs_list):
        t_data = inputs[k].data if isinstance(inputs[k], Tensor) else inputs[k]
        for pos, ch in enumerate(sub):
            if pos < len(t_data.shape):
                dim_sizes[ch] = t_data.shape[pos]
    if output_sub is not None:
        if remaining_str:
            backward_eq = f"{output_sub},{remaining_str}->{inp_sub}"
        else:
            backward_eq = f"{output_sub}->{inp_sub}"
    else:
        backward_eq = inp_sub
    backward_parts = backward_eq.split("->") if "->" in backward_eq else [backward_eq, inp_sub]
    backward_input = backward_parts[0]
    actual_output = backward_parts[1] if len(backward_parts) > 1 else backward_parts[0]
    seen = {}
    unique_chars = []
    dup_info = []
    used_chars = set(actual_output)
    for axis, ch in enumerate(actual_output):
        if ch in seen:
            for cand in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz':
                if cand not in used_chars:
                    used_chars.add(cand)
                    unique_chars.append(cand)
                    dup_info.append((ch, cand, axis, seen[ch]))
                    break
        else:
            unique_chars.append(ch)
            seen[ch] = axis
    unique_output = ''.join(unique_chars)
    unique_eq = f"{backward_input}->{unique_output}"
    all_known = all(ch in dim_sizes for ch in unique_output)
    if not all_known:
        if len(inp_sub) == 2 and inp_sub[0] == inp_sub[1]:
            n = inp_shape[0]
            result = _bk_zeros(inp_shape, dtype=np.float32)
            np.fill_diagonal(result, grad.ravel()[:n] if grad.ndim > 0 else grad)
            return result
        raise ValueError(f"Cannot resolve einsum backward with repeated output subscripts: {backward_eq}")
    if remaining_inputs_data:
        g_unique = _bk_einsum(unique_eq, grad, *remaining_inputs_data)
    else:
        g_unique = _bk_einsum(unique_eq, grad)
    for orig_ch, new_ch, dup_axis, orig_axis in reversed(dup_info):
        n = dim_sizes[orig_ch]
        diag_idx = _bk_arange(n).astype(np.int64)
        idx_list = [slice(None)] * g_unique.ndim
        idx_list[orig_axis] = diag_idx
        idx_list[dup_axis] = diag_idx
        g_unique = g_unique[tuple(idx_list)]
    return g_unique

class SVDBackward:
    @staticmethod
    def backward(ctx, grad):
        u, s, vh, full_matrices = ctx.saved
        m, n = ctx.inputs[0].shape
        u_t = u.T if u.ndim == 2 else np.swapaxes(u, -1, -2)
        v = vh.T if vh.ndim == 2 else np.swapaxes(vh, -1, -2)
        s_inv = _bk_zeros((s.shape[-1], s.shape[-1])) if s.ndim == 1 else _bk_zeros(s.shape + s.shape[-1:])
        if s.ndim == 1:
            s_inv = np.diag(1.0 / (s + 1e-30))
        else:
            for i in range(s.shape[0]):
                s_inv[i] = np.diag(1.0 / (s[i] + 1e-30))
        ds = np.zeros_like(s)
        if s.ndim == 1:
            ds = np.diag(grad[:min(m, n), :min(m, n)]).flatten()[:len(s)]
        du = grad @ np.swapaxes(vh, -1, -2) if grad.ndim == 2 else _bk_einsum('...ij,...kj->...ik', grad, vh)
        dv = u_t @ grad if grad.ndim == 2 else _bk_einsum('...ij,...ik->...kj', u_t, grad)
        return du, ds, dv

class QRBackward:
    @staticmethod
    def backward(ctx, grad):
        q, r, reduced = ctx.saved
        q_t = q.T if q.ndim == 2 else np.swapaxes(q, -1, -2)
        r_inv = np.linalg.inv(r + 1e-7 * np.eye(r.shape[-1])) if r.ndim == 2 else np.linalg.inv(r + 1e-7 * np.eye(r.shape[-1]))
        return (q_t @ grad - grad @ r_inv.T) @ q_t.T, None

class CholeskyBackward:
    @staticmethod
    def backward(ctx, grad):
        l = ctx.saved[0]
        n = l.shape[0]
        result = np.zeros_like(l)
        for i in range(n):
            for j in range(i + 1):
                s = sum(l[i, k] * result[j, k] for k in range(j))
                if i == j:
                    result[i, j] = (grad[i, j] - s) / (l[i, i] + 1e-7)
                else:
                    result[i, j] = (grad[i, j] - s) / (l[j, j] + 1e-7)
                    result[j, i] = (grad[j, i] - s) / (l[i, i] + 1e-7) - result[i, j] * l[j, j] / (l[i, i] + 1e-7)
        return result,

class LUBackward:
    @staticmethod
    def backward(ctx, grad):
        l, u, piv = ctx.saved
        return grad, None, None

class DetBackward:
    @staticmethod
    def backward(ctx, grad):
        x, det_val = ctx.saved
        inv_x = np.linalg.inv(x + 1e-7 * np.eye(x.shape[-1]))
        return (grad * det_val * inv_x).T,

class SlogdetBackward:
    @staticmethod
    def backward(ctx, grad):
        x = ctx.inputs[0]
        det_val, sign = ctx.saved
        inv_x = np.linalg.inv(x.data + 1e-7 * np.eye(x.shape[-1]))
        return (grad * inv_x).T,

class EigBackward:
    @staticmethod
    def backward(ctx, grad):
        eigenvalues, eigenvectors = ctx.saved
        n = len(eigenvalues)
        V = eigenvectors
        V_inv = np.linalg.inv(V + 1e-7 * np.eye(n))
        dLdA = np.zeros_like(V)
        for i in range(n):
            dLdA += grad[i] * np.outer(V[:, i], V_inv[i, :])
        return dLdA,

class TriSolveForwardBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved
        return np.linalg.solve(a.T, grad), np.linalg.solve(a, grad)

class ConvBackward:
    @staticmethod
    def backward(ctx, grad):
        x_data, weight_data, stride, padding, dilation, groups, col = ctx.saved
        if isinstance(x_data, Tensor):
            x_data = x_data.data
        if isinstance(weight_data, Tensor):
            weight_data = weight_data.data
        grad = np.asarray(grad, dtype=np.float32)
        B, C = x_data.shape[:2]
        G = groups
        if grad.ndim == 4:
            KH, KW = weight_data.shape[2], weight_data.shape[3]
            SH, SW = stride if isinstance(stride, tuple) else (stride, stride)
            PH, PW = padding if isinstance(padding, tuple) else (padding, padding)
            DH, DW = dilation if isinstance(dilation, tuple) else (dilation, dilation)
            H, W = x_data.shape[2], x_data.shape[3]
            H_out = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
            W_out = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1
            Cg, Og = C // G, weight_data.shape[0] // G
            grad_f = grad.reshape(B, weight_data.shape[0], H_out * W_out)
            dweight = np.zeros_like(weight_data)
            dx_pad = _bk_zeros((B, C, H + 2 * PH, W + 2 * PW))
            col_rs = col.reshape(B, G, Cg * KH * KW, H_out * W_out)
            for g in range(G):
                gg = grad_f[:, g * Og:(g + 1) * Og]
                cg = col_rs[:, g]
                wg = weight_data[g * Og:(g + 1) * Og].reshape(Og, Cg * KH * KW)
                dweight[g * Og:(g + 1) * Og] = _bk_einsum('bij,bjk->ik', gg, cg).reshape(Og, Cg, KH, KW)
                dcol = _bk_einsum('ji,bik->bjk', wg, gg).reshape(B, Cg, KH, KW, H_out, W_out)
                for ki in range(KH):
                    for kj in range(KW):
                        hi = _bk_arange(H_out, dtype=np.int64) * SH + ki * DH
                        wj = _bk_arange(W_out, dtype=np.int64) * SW + kj * DW
                        hh, ww = np.meshgrid(hi, wj, indexing='ij')
                        for bi in range(B):
                            for ci in range(Cg):
                                dx_pad[bi, g * Cg + ci, hh, ww] += dcol[bi, ci, ki, kj]
            dx = dx_pad[:, :, PH:PH + H, PW:PW + W] if (PH > 0 or PW > 0) else dx_pad
            return dx, dweight
        elif grad.ndim == 3:
            K = weight_data.shape[2]
            S = stride if isinstance(stride, int) else stride[0]
            P = padding if isinstance(padding, int) else padding[0]
            D = dilation if isinstance(dilation, int) else dilation[0]
            L = x_data.shape[2]
            L_out = (L + 2 * P - D * (K - 1) - 1) // S + 1
            Cg, Og = C // G, weight_data.shape[0] // G
            grad_f = grad.reshape(B, weight_data.shape[0], L_out)
            dweight = np.zeros_like(weight_data)
            dx_pad = _bk_zeros((B, C, L + 2 * P))
            col_rs = col.reshape(B, G, Cg * K, L_out)
            for g in range(G):
                gg = grad_f[:, g * Og:(g + 1) * Og]
                cg = col_rs[:, g]
                wg = weight_data[g * Og:(g + 1) * Og].reshape(Og, Cg * K)
                dweight[g * Og:(g + 1) * Og] = _bk_einsum('bij,bjk->ik', gg, cg).reshape(Og, Cg, K)
                dcol = _bk_einsum('ji,bik->bjk', wg, gg).reshape(B, Cg, K, L_out)
                for ki in range(K):
                    li = _bk_arange(L_out, dtype=np.int64) * S + ki * D
                    for bi in range(B):
                        for ci in range(Cg):
                            np.add.at(dx_pad[bi, g * Cg + ci], li, dcol[bi, ci, ki])
            dx = dx_pad[:, :, P:P + L] if P > 0 else dx_pad
            return dx, dweight
        return np.zeros_like(x_data), np.zeros_like(weight_data)

class PadBackward:
    @staticmethod
    def backward(ctx, grad):
        pads, mode = ctx.saved[0], ctx.saved[1]
        pads = _normalize_pads(pads, grad.ndim)
        if mode == 'constant':
            sliced = grad
            for i in range(grad.ndim):
                lo, hi = pads[i]
                if lo > 0 or hi > 0:
                    slc = [slice(None)] * grad.ndim
                    slc[i] = slice(lo, grad.shape[i] - hi if hi > 0 else None)
                    sliced = sliced[tuple(slc)]
            return sliced,
        inp = ctx.inputs[0]
        inp_shape = inp.shape
        working = grad
        for i in reversed(range(grad.ndim)):
            lo, hi = pads[i]
            if lo == 0 and hi == 0:
                continue
            inp_len = inp_shape[i]
            new_shape = list(working.shape)
            new_shape[i] = inp_len
            result = _bk_zeros(new_shape, dtype=np.float32)
            slc_c = [slice(None)] * working.ndim
            slc_c[i] = slice(lo, lo + inp_len)
            slc_r = [slice(None)] * working.ndim
            slc_r[i] = slice(0, inp_len)
            result[tuple(slc_r)] = working[tuple(slc_c)]
            if mode == 'reflect':
                for k in range(lo):
                    src_idx = lo - 1 - k
                    sg = [slice(None)] * working.ndim
                    sr = [slice(None)] * working.ndim
                    sg[i] = k
                    sr[i] = src_idx
                    result[tuple(sr)] += working[tuple(sg)]
                for k in range(hi):
                    src_idx = inp_len - 1 - k
                    sg = [slice(None)] * working.ndim
                    sr = [slice(None)] * working.ndim
                    sg[i] = lo + inp_len + k
                    sr[i] = src_idx
                    result[tuple(sr)] += working[tuple(sg)]
            elif mode == 'replicate':
                if lo > 0:
                    sg = [slice(None)] * working.ndim
                    sg[i] = slice(0, lo)
                    sr = [slice(None)] * working.ndim
                    sr[i] = 0
                    result[tuple(sr)] += working[tuple(sg)].sum(axis=i, keepdims=True)
                if hi > 0:
                    sg = [slice(None)] * working.ndim
                    sg[i] = slice(lo + inp_len, lo + inp_len + hi)
                    sr = [slice(None)] * working.ndim
                    sr[i] = inp_len - 1
                    result[tuple(sr)] += working[tuple(sg)].sum(axis=i, keepdims=True)
            elif mode == 'circular':
                for k in range(lo):
                    src_idx = (inp_len - lo + k) % inp_len
                    sg = [slice(None)] * working.ndim
                    sr = [slice(None)] * working.ndim
                    sg[i] = k
                    sr[i] = src_idx
                    result[tuple(sr)] += working[tuple(sg)]
                for k in range(hi):
                    src_idx = k
                    sg = [slice(None)] * working.ndim
                    sr = [slice(None)] * working.ndim
                    sg[i] = lo + inp_len + k
                    sr[i] = src_idx
                    result[tuple(sr)] += working[tuple(sg)]
            working = result
        return working,


def _normalize_pads(pads, ndim):
    if isinstance(pads, int):
        return tuple((pads, pads) for _ in range(ndim))
    if isinstance(pads[0], (list, tuple)):
        return tuple((int(p[0]), int(p[1])) for p in pads)
    pad_flat = [int(x) for x in pads]
    if len(pad_flat) == 2 * ndim:
        return tuple((pad_flat[2 * i], pad_flat[2 * i + 1]) for i in range(ndim))
    if len(pad_flat) == 2 and ndim == 1:
        return ((pad_flat[0], pad_flat[1]),)
    return tuple((pad_flat[0], pad_flat[-1]) for _ in range(ndim))

class UnfoldBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data)

class FoldBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data)

class Im2colBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data)

class Col2imBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data)

class NonzeroBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data)

class UniqueBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data)

class ArgsortBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data)

class IndexPutBackward:
    @staticmethod
    def backward(ctx, grad):
        idx, values = ctx.saved
        g = np.zeros_like(ctx.inputs[0].data)
        g[idx] = values
        return g

class MaskedFillBackward:
    @staticmethod
    def backward(ctx, grad):
        mask, value = ctx.saved
        g = np.zeros_like(ctx.inputs[0].data)
        g[~mask] = grad[~mask]
        return g

class MaskedSelectBackward:
    @staticmethod
    def backward(ctx, grad):
        return np.zeros_like(ctx.inputs[0].data)

class FloordivBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        return _unbroadcast(grad * 0, a.shape), _unbroadcast(grad * 0, b.shape)

class ModBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        return _unbroadcast(grad * 1, a.shape), _unbroadcast(grad * 0, b.shape)

class Exp2Backward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad * ctx.saved[0] * _bk_log(2), ctx.inputs[0].shape),

class RsqrtBackward:
    @staticmethod
    def backward(ctx, grad):
        return _unbroadcast(grad * (-0.5 * ctx.saved[0] ** 3), ctx.inputs[0].shape),

class CrossBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        return _unbroadcast(np.cross(b.data, grad, axisa=-1, axisb=-1, axisc=-1), a.shape), \
               _unbroadcast(np.cross(grad, a.data, axisa=-1, axisb=-1, axisc=-1), b.shape)

class TensordotBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        dims = ctx.saved[0]
        if isinstance(dims, int):
            axes_a = list(range(-dims, 0))
            axes_b = list(range(dims))
        else:
            axes_a, axes_b = dims
        ga = np.tensordot(grad, b.data, axes=(list(range(len(axes_b))), [a.data.ndim - len(axes_a) + i for i in range(len(axes_a))]))
        gb = np.tensordot(a.data, grad, axes=([a.data.ndim - len(axes_a) + i for i in range(len(axes_a))], list(range(len(axes_b)))))
        return _unbroadcast(ga, a.shape), _unbroadcast(gb, b.shape)

class KronBackward:
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.inputs
        return _unbroadcast(np.kron(grad, _bk_ones(b.shape)), a.shape), \
               _unbroadcast(np.kron(_bk_ones(a.shape), grad), b.shape)

class DropoutBackward:
    @staticmethod
    def backward(ctx, grad):
        mask, scale = ctx.saved
        return grad * mask * scale,

class LayerNormBackward:
    @staticmethod
    def backward(ctx, grad):
        x_data, gamma_data, bias_data, eps = ctx.saved
        ndim = x_data.ndim
        batch_axes = tuple(range(ndim - 1)) if ndim > 1 else ()

        mean = x_data.mean(axis=-1, keepdims=True)
        var = x_data.var(axis=-1, keepdims=True, ddof=0)
        std = _bk_sqrt(var + eps)
        x_hat = (x_data - mean) / std

        if gamma_data is not None:
            dx_hat = grad * gamma_data
            dgamma = (grad * x_hat).sum(axis=batch_axes) if batch_axes else (grad * x_hat).sum()
        else:
            dx_hat = grad
            dgamma = None

        dx = (dx_hat - dx_hat.mean(axis=-1, keepdims=True) - x_hat * (dx_hat * x_hat).mean(axis=-1, keepdims=True)) / std

        dbeta = grad.sum(axis=batch_axes) if batch_axes else grad.sum() if bias_data is not None else None

        return dx, dgamma, dbeta

def _unbroadcast(grad, shape):
    if grad.shape == shape:
        return grad
    ndiff = grad.ndim - len(shape)
    if ndiff > 0:
        grad = grad.sum(axis=tuple(range(ndiff)))
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    try:
        return grad.reshape(shape)
    except ValueError:
        return grad

def _broadcast_shape(a, b):
    sa, sb = a.shape, b.shape
    max_nd = max(len(sa), len(sb))
    sa = (1,) * (max_nd - len(sa)) + sa
    sb = (1,) * (max_nd - len(sb)) + sb
    result = []
    for x, y in zip(sa, sb):
        if x == y:
            result.append(x)
        elif x == 1:
            result.append(y)
        elif y == 1:
            result.append(x)
        else:
            raise ValueError(f"Cannot broadcast shapes {a.shape} and {b.shape}")
    return tuple(result)

# ---------------------------------------------------------------------------
# Tensor class
# ---------------------------------------------------------------------------

class Tensor:
    __array_priority__ = 10000

    def __init__(self, data, requires_grad=False, dtype=None, _ctx=None, _name=""):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, (list, tuple)):
            data = _bk_from_list(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            data = data.astype(np.float32) if dtype is None else data.astype(np_dtype(dtype_info.get(dtype, dtype)))
        if dtype and isinstance(dtype, str) and dtype in dtype_info:
            dt = dtype_info[dtype]
            if dt.name.startswith("INT"):
                data = data.astype(np.int8) if dt.name == "INT8" else data.astype(np.float32)
        self.data = data
        self.requires_grad = requires_grad and _grad_enabled()
        self.grad = None
        self._ctx = _ctx
        self._name = _name
        self._version = 0
        self._is_leaf = _ctx is None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def strides(self):
        return self.data.strides

    @property
    def dtype(self):
        if hasattr(self.data, 'dtype'):
            if self.data.dtype == np.float32: return FP32
            if self.data.dtype == np.float64: return FP64
            if self.data.dtype == np.float16: return FP16
            if self.data.dtype == np.int8: return INT8
            if self.data.dtype == np.int32 or self.data.dtype == np.int64: return INT8
            if self.data.dtype == np.bool_: return BOOL
        return FP32

    @property
    def device(self):
        return "cpu"

    @property
    def grad_fn(self):
        return self._ctx.op if self._ctx else None

    @property
    def is_leaf(self):
        return self._is_leaf

    def item(self):
        return self.data.flat[0]

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, dtype={self.dtype.name}, requires_grad={self.requires_grad})"

    def __len__(self):
        return len(self.data)

    def numpy(self):
        return self.data.copy()

    def clone(self):
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def to(self, dtype):
        dt = dtype_info.get(dtype, dtype)
        return Tensor(self.data.astype(np_dtype(dt)), requires_grad=self.requires_grad)

    def float(self):
        return self.to("FP32")

    def double(self):
        return self.to("FP64")

    def half(self):
        return self.to("FP16")

    def cpu(self):
        return self

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self

    # ---- autograd ----

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise RuntimeError("backward() requires gradient for non-scalar")
        if not isinstance(gradient, np.ndarray):
            gradient = _bk_from_list(gradient, dtype=np.float32)
        self.grad = gradient if self.grad is None else self.grad + gradient
        order = _topological_sort(self)
        visited = set()
        for t in reversed(order):
            tid = id(t)
            if tid in visited:
                continue
            if not retain_graph:
                visited.add(tid)
            if t._ctx is None:
                continue
            grads = t._ctx.op.backward(t._ctx, t.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for inp, g in zip(t._ctx.inputs, grads):
                if g is None or not inp.requires_grad:
                    continue
                if isinstance(g, Tensor):
                    g = g.data
                if inp.grad is None:
                    inp.grad = g
                else:
                    inp.grad = inp.grad + g

    def zero_grad(self):
        self.grad = None

    # ---- shape ops ----

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad,
                      _ctx=Context(ReshapeBackward, (self,), (self.shape,)))

    def view(self, *shape):
        return self.reshape(*shape)

    def transpose(self, dim0, dim1):
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad,
                      _ctx=Context(TransposeBackward, (self,), (axes,)))

    def permute(self, *axes):
        return Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad,
                      _ctx=Context(PermuteBackward, (self,), (axes,)))

    def squeeze(self, axis=None):
        return Tensor(self.data.squeeze(axis=axis) if axis is not None else self.data.squeeze(),
                      requires_grad=self.requires_grad,
                      _ctx=Context(SqueezeBackward, (self,), ()))

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.data, axis), requires_grad=self.requires_grad,
                      _ctx=Context(UnsqueezeBackward, (self,), ()))

    def expand(self, *shape):
        return Tensor(_bk_broadcast_to(self.data, shape), requires_grad=self.requires_grad,
                      _ctx=Context(ExpandBackward, (self,), ()))

    def movedim(self, source, destination):
        if isinstance(source, int): source = (source,)
        if isinstance(destination, int): destination = (destination,)
        perm = list(range(self.ndim))
        for s, d in zip(source, destination):
            perm[s] = d
        return self.permute(*perm)

    def flip(self, axis):
        if isinstance(axis, int): axis = (axis,)
        return Tensor(np.flip(self.data, axis), requires_grad=self.requires_grad,
                      _ctx=Context(FlipBackward, (self,), (axis,)))

    def roll(self, shifts, axis=0):
        return Tensor(np.roll(self.data, shifts, axis), requires_grad=self.requires_grad,
                      _ctx=Context(RollBackward, (self,), (shifts, axis)))

    def contiguous(self):
        if self.data.flags['C_CONTIGUOUS']:
            return self
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def is_contiguous(self):
        return self.data.flags['C_CONTIGUOUS']

    def flatten(self, start_dim=0, end_dim=-1):
        shape = self.shape[:start_dim] + (-1,) + self.shape[end_dim:]
        return self.reshape(*shape)

    def t(self):
        assert self.ndim == 2
        return self.transpose(0, 1)

    @property
    def T(self):
        if self.ndim == 2:
            return self.transpose(0, 1)
        return self.permute(*reversed(range(self.ndim)))

    # ---- arithmetic ----

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        req = self.requires_grad or other.requires_grad
        if self.shape != other.shape:
            bs = _broadcast_shape(self, other)
            a = _bk_broadcast_to(self.data, bs)
            b = _bk_broadcast_to(other.data, bs)
        else:
            a, b = self.data, other.data
        out = Tensor(a + b, requires_grad=req, _ctx=Context(AddBackward, (self, other), ()))
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        req = self.requires_grad or other.requires_grad
        if self.shape != other.shape:
            bs = _broadcast_shape(self, other)
            a = _bk_broadcast_to(self.data, bs)
            b = _bk_broadcast_to(other.data, bs)
        else:
            a, b = self.data, other.data
        out = Tensor(a - b, requires_grad=req, _ctx=Context(SubBackward, (self, other), ()))
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        req = self.requires_grad or other.requires_grad
        if self.shape != other.shape:
            bs = _broadcast_shape(self, other)
            a = _bk_broadcast_to(self.data, bs)
            b = _bk_broadcast_to(other.data, bs)
        else:
            a, b = self.data, other.data
        out = Tensor(a - b, requires_grad=req, _ctx=Context(SubBackward, (other, self), ()))
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        req = self.requires_grad or other.requires_grad
        if self.shape != other.shape:
            bs = _broadcast_shape(self, other)
            a = _bk_broadcast_to(self.data, bs)
            b = _bk_broadcast_to(other.data, bs)
        else:
            a, b = self.data, other.data
        out = Tensor(a * b, requires_grad=req, _ctx=Context(MulBackward, (self, other), ()))
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        req = self.requires_grad or other.requires_grad
        if self.shape != other.shape:
            bs = _broadcast_shape(self, other)
            a = _bk_broadcast_to(self.data, bs)
            b = _bk_broadcast_to(other.data, bs)
        else:
            a, b = self.data, other.data
        out = Tensor(a / (b + 1e-8), requires_grad=req, _ctx=Context(DivBackward, (self, other), ()))
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        return other / self

    def __floordiv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        req = self.requires_grad or other.requires_grad
        return Tensor(np.floor(self.data / other.data).astype(np.float32), requires_grad=req,
                      _ctx=Context(FloordivBackward, (self, other), ()))

    def __rfloordiv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        req = self.requires_grad or other.requires_grad
        return Tensor(np.floor(other.data / self.data).astype(np.float32), requires_grad=req,
                      _ctx=Context(FloordivBackward, (other, self), ()))

    def __mod__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        req = self.requires_grad or other.requires_grad
        return Tensor(self.data % other.data, requires_grad=req,
                      _ctx=Context(ModBackward, (self, other), ()))

    def __pow__(self, exp):
        if not isinstance(exp, Tensor):
            out = Tensor(self.data ** exp, requires_grad=self.requires_grad,
                         _ctx=Context(PowBackward, (self,), (exp,)))
        else:
            out = Tensor(self.data ** exp.data, requires_grad=self.requires_grad or exp.requires_grad,
                         _ctx=Context(PowBackward, (self, exp), (exp,)))
        return out

    def __rpow__(self, base):
        return Tensor(base ** self.data, requires_grad=self.requires_grad,
                      _ctx=Context(PowBackward, (Tensor(_bk_from_list(base, dtype=np.float32)), self), (self,)))

    def __neg__(self):
        return Tensor(-self.data, requires_grad=self.requires_grad,
                      _ctx=Context(NegBackward, (self,), ()))

    def __abs__(self):
        return Tensor(_bk_abs(self.data), requires_grad=self.requires_grad,
                      _ctx=Context(AbsBackward, (self,), ()))

    # ---- element-wise ops ----

    def exp(self):
        out = Tensor(_bk_exp(self.data), requires_grad=self.requires_grad,
                     _ctx=Context(ExpBackward, (self,), (_bk_exp(self.data),)))
        return out

    def exp2(self):
        out = Tensor(2 ** self.data, requires_grad=self.requires_grad,
                     _ctx=Context(Exp2Backward, (self,), (2 ** self.data,)))
        return out

    def log(self):
        out = Tensor(_bk_log(np.maximum(self.data, 1e-7)), requires_grad=self.requires_grad,
                     _ctx=Context(LogBackward, (self,), (self.data + 1e-7,)))
        return out

    def log2(self):
        return self.log() / math.log(2)

    def log10(self):
        return self.log() / math.log(10)

    def sqrt(self):
        s = _bk_sqrt(np.maximum(self.data, 0) + 1e-7)
        out = Tensor(s, requires_grad=self.requires_grad,
                     _ctx=Context(SqrtBackward, (self,), (s,)))
        return out

    def rsqrt(self):
        r = 1.0 / (_bk_sqrt(np.maximum(self.data, 0)) + 1e-7)
        out = Tensor(r, requires_grad=self.requires_grad,
                     _ctx=Context(RsqrtBackward, (self,), (r,)))
        return out

    def sign(self):
        return Tensor(np.sign(self.data).astype(np.float32), requires_grad=self.requires_grad,
                      _ctx=Context(SignBackward, (self,), ()))

    def sin(self):
        out = Tensor(_bk_sin(self.data), requires_grad=self.requires_grad,
                     _ctx=Context(SinBackward, (self,), ()))
        return out

    def cos(self):
        out = Tensor(_bk_cos(self.data), requires_grad=self.requires_grad,
                     _ctx=Context(CosBackward, (self,), ()))
        return out

    def tan(self):
        out = Tensor(np.tan(self.data), requires_grad=self.requires_grad,
                     _ctx=Context(TanBackward, (self,), (np.tan(self.data),)))
        return out

    def asin(self):
        out = Tensor(np.arcsin(np.clip(self.data, -1 + 1e-7, 1 - 1e-7)), requires_grad=self.requires_grad,
                     _ctx=Context(AsinBackward, (self,), ()))
        return out

    def acos(self):
        out = Tensor(np.arccos(np.clip(self.data, -1 + 1e-7, 1 - 1e-7)), requires_grad=self.requires_grad,
                     _ctx=Context(AcosBackward, (self,), ()))
        return out

    def atan(self):
        out = Tensor(np.arctan(self.data), requires_grad=self.requires_grad,
                     _ctx=Context(AtanBackward, (self,), ()))
        return out

    def atan2(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        out = Tensor(np.arctan2(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad,
                     _ctx=Context(Atan2Backward, (self, other), ()))
        return out

    def sinh(self):
        out = Tensor(np.sinh(self.data), requires_grad=self.requires_grad,
                     _ctx=Context(SinhBackward, (self,), ()))
        return out

    def cosh(self):
        out = Tensor(np.cosh(self.data), requires_grad=self.requires_grad,
                     _ctx=Context(CoshBackward, (self,), ()))
        return out

    def tanh(self):
        t = _bk_tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad,
                     _ctx=Context(TanhBackward, (self,), (t,)))
        return out

    def arcsinh(self):
        out = Tensor(np.arcsinh(self.data), requires_grad=self.requires_grad,
                     _ctx=Context(ArcsinhBackward, (self,), ()))
        return out

    def arccosh(self):
        out = Tensor(np.arccosh(np.maximum(self.data, 1 + 1e-7)), requires_grad=self.requires_grad,
                     _ctx=Context(ArccoshBackward, (self,), ()))
        return out

    def arctanh(self):
        out = Tensor(np.arctanh(np.clip(self.data, -1 + 1e-7, 1 - 1e-7)), requires_grad=self.requires_grad,
                     _ctx=Context(ArctanhBackward, (self,), ()))
        return out

    def clamp(self, min_val, max_val):
        out = Tensor(np.clip(self.data, min_val, max_val), requires_grad=self.requires_grad,
                     _ctx=Context(ClampBackward, (self,), (min_val, max_val)))
        return out

    def clip(self, min_val, max_val):
        return self.clamp(min_val, max_val)

    def maximum(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        out = Tensor(np.maximum(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad,
                     _ctx=Context(MaximumBackward, (self, other), ()))
        return out

    def minimum(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        out = Tensor(np.minimum(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad,
                     _ctx=Context(MinimumBackward, (self, other), ()))
        return out

    def floor(self):
        return Tensor(np.floor(self.data).astype(np.float32), requires_grad=False,
                      _ctx=Context(FloorBackward, (self,), ()))

    def ceil(self):
        return Tensor(np.ceil(self.data).astype(np.float32), requires_grad=False,
                      _ctx=Context(CeilBackward, (self,), ()))

    def round(self):
        return Tensor(np.round(self.data).astype(np.float32), requires_grad=False,
                      _ctx=Context(RoundBackward, (self,), ()))

    def trunc(self):
        return Tensor(np.trunc(self.data).astype(np.float32), requires_grad=False,
                      _ctx=Context(TruncBackward, (self,), ()))

    def sigmoid(self):
        s = 1 / (1 + _bk_exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=self.requires_grad,
                     _ctx=Context(SigmoidBackward, (self,), (s,)))
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad,
                     _ctx=Context(ReluBackward, (self,), ()))
        return out

    def relu6(self):
        out = Tensor(np.clip(self.data, 0, 6), requires_grad=self.requires_grad,
                     _ctx=Context(ReLU6Backward, (self,), ()))
        return out

    def leaky_relu(self, alpha=0.01):
        out = Tensor(_bk_where(self.data > 0, self.data, alpha * self.data), requires_grad=self.requires_grad,
                     _ctx=Context(LeakyReLUBackward, (self,), (alpha,)))
        return out

    def elu(self, alpha=1.0):
        out = Tensor(_bk_where(self.data > 0, self.data, alpha * (_bk_exp(self.data) - 1)), requires_grad=self.requires_grad,
                     _ctx=Context(ELUBackward, (self,), (alpha,)))
        return out

    def selu(self):
        lam = 1.0507009873554805
        alpha = 1.6732632423543772
        out = Tensor(lam * _bk_where(self.data > 0, self.data, alpha * (_bk_exp(self.data) - 1)), requires_grad=self.requires_grad,
                     _ctx=Context(SELUBackward, (self,), (alpha, lam)))
        return out

    def gelu(self):
        c = 0.7978845608028654
        k = 0.044715
        t = _bk_tanh(c * (self.data + k * self.data ** 3))
        out = Tensor(0.5 * self.data * (1 + t), requires_grad=self.requires_grad,
                     _ctx=Context(GELUBackward2, (self,), ()))
        return out

    def silu(self):
        s = 1 / (1 + _bk_exp(-np.clip(self.data, -500, 500)))
        out = Tensor(self.data * s, requires_grad=self.requires_grad,
                     _ctx=Context(SiLUBackward2, (self,), ()))
        return out

    def mish(self):
        sp = np.log1p(_bk_exp(self.data))
        return self * _bk_tanh(sp)

    def hardswish(self):
        return self * Tensor(self.relu6().data + 3) / 6

    def softmax(self, axis=-1):
        e = _bk_exp(self.data - self.data.max(axis=axis, keepdims=True))
        s = e / (e.sum(axis=axis, keepdims=True) + 1e-8)
        out = Tensor(s, requires_grad=self.requires_grad,
                     _ctx=Context(SoftmaxBackward, (self,), (s, axis)))
        return out

    def log_softmax(self, axis=-1):
        m = self.data.max(axis=axis, keepdims=True)
        log_sum = _bk_log(_bk_sum(_bk_exp(self.data - m), axis=axis, keepdims=True) + 1e-8)
        s = self.data - m - log_sum
        out = Tensor(s, requires_grad=self.requires_grad,
                     _ctx=Context(LogSoftmaxBackward, (self,), (_bk_exp(s), axis)))
        return out

    # ---- reductions ----

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _ctx=Context(SumBackward, (self,), (self.shape, keepdims, axis)))
        return out

    def mean(self, axis=None, keepdims=False):
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        n = self.data.size if axis is None else np.prod([self.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,))])
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _ctx=Context(MeanBackward, (self,), (self.shape, keepdims, axis, n)))
        return out

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _ctx=Context(MaxBackward, (self,), (self.shape, keepdims, axis)))
        return out

    def min(self, axis=None, keepdims=False):
        out_data = self.data.min(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _ctx=Context(MinBackward, (self,), (self.shape, keepdims, axis)))
        return out

    def prod(self, axis=None, keepdims=False):
        out_data = self.data.prod(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _ctx=Context(ProdBackward, (self,), (self.shape, keepdims, axis)))
        return out

    def any(self, axis=None, keepdims=False):
        return Tensor(self.data.any(axis=axis, keepdims=keepdims), requires_grad=False)

    def all(self, axis=None, keepdims=False):
        return Tensor(self.data.all(axis=axis, keepdims=keepdims), requires_grad=False)

    def argmax(self, axis=None, keepdims=False):
        return Tensor(self.data.argmax(axis=axis, keepdims=keepdims), requires_grad=False)

    def argmin(self, axis=None, keepdims=False):
        return Tensor(self.data.argmin(axis=axis, keepdims=keepdims), requires_grad=False)

    def var(self, axis=None, keepdims=False, correction=0):
        out_data = self.data.var(axis=axis, keepdims=keepdims, ddof=correction)
        n = self.data.size if axis is None else np.prod([self.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,))])
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _ctx=Context(VarBackward, (self,), (self.shape, keepdims, axis, n, correction)))
        return out

    def std(self, axis=None, keepdims=False, correction=0):
        out_data = self.data.std(axis=axis, keepdims=keepdims, ddof=correction)
        n = self.data.size if axis is None else np.prod([self.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,))])
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _ctx=Context(StdBackward, (self,), (self.shape, keepdims, axis, n, correction)))
        return out

    def norm(self, ord=None, axis=None, keepdims=False):
        if ord is None:
            out = (self ** 2).sum(axis=axis, keepdims=keepdims).sqrt()
        elif ord == 1:
            out = self.abs().sum(axis=axis, keepdims=keepdims)
        elif ord == 2:
            out = (self ** 2).sum(axis=axis, keepdims=keepdims).sqrt()
        else:
            out = (self.abs() ** ord).sum(axis=axis, keepdims=keepdims) ** (1.0 / ord)
        return out

    def logsumexp(self, axis=None, keepdims=False):
        m = self.data.max(axis=axis, keepdims=True)
        out_data = m.squeeze(axis=axis if not keepdims else None) + _bk_log(_bk_sum(_bk_exp(self.data - m), axis=axis, keepdims=keepdims) + 1e-8)
        if not keepdims and axis is not None:
            out_data = out_data.squeeze(axis=axis)
        return Tensor(out_data, requires_grad=self.requires_grad,
                      _ctx=Context(LogsumexpBackward, (self,), (self.shape, keepdims, axis)))

    def cumsum(self, axis=0):
        out = Tensor(self.data.cumsum(axis=axis), requires_grad=self.requires_grad,
                     _ctx=Context(CumsumBackward, (self,), (axis,)))
        return out

    def cumprod(self, axis=0):
        out = Tensor(self.data.cumprod(axis=axis), requires_grad=self.requires_grad,
                     _ctx=Context(CumprodBackward, (self,), (self.data.cumprod(axis=axis), axis)))
        return out

    # ---- linear algebra ----

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        req = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, requires_grad=req,
                     _ctx=Context(MatMulBackward, (self, other), ()))
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        return other.matmul(self)

    def outer(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        return self.reshape(-1, 1) * other.reshape(1, -1)

    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        return (self * other).sum()

    def inner(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        return self.reshape(-1).dot(other.reshape(-1))

    def cross(self, other, dim=-1):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        a, b = self.data, other.data
        if dim < 0: dim += a.ndim
        result = np.cross(a, b, axisa=dim, axisb=dim, axisc=dim)
        return Tensor(result, requires_grad=self.requires_grad or other.requires_grad,
                      _ctx=Context(CrossBackward, (self, other), (dim,)))

    def tensordot(self, other, dims=2):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        if isinstance(dims, int):
            axes_a = list(range(-dims, 0))
            axes_b = list(range(dims))
        else:
            axes_a, axes_b = dims
        result = np.tensordot(self.data, other.data, axes=(axes_a, axes_b))
        return Tensor(result, requires_grad=self.requires_grad or other.requires_grad,
                      _ctx=Context(TensordotBackward, (self, other), (dims,)))

    def kron(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        return Tensor(np.kron(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad,
                      _ctx=Context(KronBackward, (self, other), ()))

    def svd(self, full_matrices=True):
        u, s, vh = np.linalg.svd(self.data, full_matrices=full_matrices)
        req = self.requires_grad
        ctx = Context(SVDBackward, (self,), (u, s, vh, full_matrices))
        return (Tensor(u, requires_grad=req, _ctx=ctx),
                Tensor(s, requires_grad=req, _ctx=ctx),
                Tensor(vh, requires_grad=req, _ctx=ctx))

    def qr(self, reduced=True):
        q, r = np.linalg.qr(self.data, mode='reduced' if reduced else 'complete')
        req = self.requires_grad
        ctx = Context(QRBackward, (self,), (q, r, reduced))
        return (Tensor(q, requires_grad=req, _ctx=ctx),
                Tensor(r, requires_grad=req, _ctx=ctx))

    def cholesky(self):
        L = np.linalg.cholesky(self.data + 1e-7 * np.eye(self.shape[-1]))
        return Tensor(L, requires_grad=self.requires_grad,
                      _ctx=Context(CholeskyBackward, (self,), (L,)))

    def lu(self):
        p, l, u = np.linalg.lu(self.data)
        req = self.requires_grad
        ctx = Context(LUBackward, (self,), (l, u, p))
        return (Tensor(p, requires_grad=False),
                Tensor(l, requires_grad=req, _ctx=ctx),
                Tensor(u, requires_grad=req, _ctx=ctx))

    def solve(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        result = np.linalg.solve(self.data + 1e-7 * np.eye(self.shape[-1]), other.data)
        return Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

    def solve_triangular(self, b, upper=True):
        b = b if isinstance(b, Tensor) else Tensor(_bk_from_list(b, dtype=np.float32))
        result = np.linalg.solve(self.data, b.data)
        return Tensor(result, requires_grad=self.requires_grad or b.requires_grad)

    def lstsq(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other, dtype=np.float32))
        result, _, _, _ = np.linalg.lstsq(self.data, other.data, rcond=None)
        return Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

    def det(self):
        d = np.linalg.det(self.data)
        return Tensor(_bk_from_list(d), requires_grad=self.requires_grad,
                      _ctx=Context(DetBackward, (self,), (self.data, d)))

    def slogdet(self):
        sign, logabsdet = np.linalg.slogdet(self.data)
        return Tensor(_bk_from_list(sign), requires_grad=False), \
               Tensor(_bk_from_list(logabsdet), requires_grad=self.requires_grad,
                      _ctx=Context(SlogdetBackward, (self,), (self.data, (sign, logabsdet))))

    def logdet(self):
        return Tensor(_bk_log(_bk_abs(np.linalg.det(self.data)) + 1e-30), requires_grad=self.requires_grad)

    def eig(self, k=None):
        if k is None:
            w, v = np.linalg.eigh(self.data)
        else:
            w, v = np.linalg.eigh(self.data)
            w, v = w[:k], v[:, :k]
        return Tensor(w, requires_grad=self.requires_grad,
                      _ctx=Context(EigBackward, (self,), (w, v))), \
               Tensor(v, requires_grad=self.requires_grad,
                      _ctx=Context(EigBackward, (self,), (w, v)))

    def einsum(self, equation, *others):
        tensors = [self] + list(others)
        arrays = [t.data if isinstance(t, Tensor) else t for t in tensors]
        result = _bk_einsum(equation, *arrays)
        req = any(t.requires_grad for t in tensors if isinstance(t, Tensor))
        return Tensor(result, requires_grad=req,
                      _ctx=Context(EinsumBackward, tensors, (equation, tensors)))

    # ---- indexing ----

    def __getitem__(self, idx):
        return Tensor(self.data[idx], requires_grad=self.requires_grad)

    def __setitem__(self, idx, val):
        if isinstance(val, Tensor):
            val = val.data
        self.data[idx] = val
        self._version += 1

    def gather(self, dim, index):
        index = index if isinstance(index, Tensor) else Tensor(_bk_from_list(index, dtype=np.int64))
        idx_data = index.data.astype(np.int64)
        result = np.take_along_axis(self.data, idx_data, axis=dim)
        return Tensor(result, requires_grad=self.requires_grad,
                      _ctx=Context(GatherBackward, (self, index), (self.data, idx_data, dim)))

    def scatter_(self, dim, index, src):
        src = src if isinstance(src, Tensor) else Tensor(_bk_from_list(src, dtype=np.float32))
        np.put_along_axis(self.data, index.data.astype(np.int64), src.data, axis=dim)
        self._version += 1
        return self

    def scatter_add_(self, dim, index, src):
        src = src if isinstance(src, Tensor) else Tensor(_bk_from_list(src, dtype=np.float32))
        idx_data = index.data.astype(np.int64)
        if dim == 1:
            extra = _bk_arange(self.shape[0]).astype(np.int64).reshape(-1, 1)
            extra = np.broadcast_to(extra, idx_data.shape).copy()
            np.add.at(self.data, (extra, idx_data), src.data)
        else:
            np.add.at(self.data, idx_data, src.data)
        self._version += 1
        return self

    def masked_fill_(self, mask, value):
        self.data[~mask.data.astype(bool)] = value
        self._version += 1
        return self

    def masked_select(self, mask):
        return Tensor(self.data[mask.data.astype(bool)], requires_grad=self.requires_grad)

    def index_put_(self, indices, values):
        values = values if isinstance(values, Tensor) else Tensor(_bk_from_list(values, dtype=np.float32))
        self.data[indices] = values.data
        self._version += 1
        return self

    def where(self, condition, x, y):
        x = x if isinstance(x, Tensor) else Tensor(_bk_from_list(x, dtype=np.float32))
        y = y if isinstance(y, Tensor) else Tensor(_bk_from_list(y, dtype=np.float32))
        result = _bk_where(condition.data if isinstance(condition, Tensor) else condition, x.data, y.data)
        return Tensor(result, requires_grad=self.requires_grad or x.requires_grad or y.requires_grad,
                      _ctx=Context(WhereBackward, (self, x, y), (condition.data if isinstance(condition, Tensor) else condition, x, y)))

    def nonzero(self):
        return Tensor(_bk_from_list(np.nonzero(self.data)[0]), requires_grad=False)

    def unique(self, sorted=True, return_counts=False):
        result = np.unique(self.data, return_counts=return_counts)
        if return_counts:
            return Tensor(result[0], requires_grad=False), Tensor(result[1], requires_grad=False)
        return Tensor(result, requires_grad=False)

    def sort(self, dim=-1, descending=False):
        indices = np.argsort(-self.data if descending else self.data, axis=dim)
        sorted_data = np.sort(-self.data if descending else self.data, axis=dim)
        if descending:
            sorted_data = -sorted_data
        return Tensor(sorted_data, requires_grad=self.requires_grad,
                      _ctx=Context(SortBackward, (self,), (indices, dim, descending))), \
               Tensor(indices, requires_grad=False)

    def argsort(self, dim=-1, descending=False):
        return Tensor(np.argsort(-self.data if descending else self.data, axis=dim), requires_grad=False)

    def topk(self, k, dim=-1, largest=True, sorted=True):
        if not largest:
            indices = np.argpartition(self.data, k, axis=dim)[..., :k]
        else:
            indices = np.argpartition(-self.data, k, axis=dim)[..., :k]
        values = np.take_along_axis(self.data, indices, axis=dim)
        if sorted:
            sort_idx = np.argsort(-values if largest else values, axis=dim)
            values = np.take_along_axis(values, sort_idx, axis=dim)
            indices = np.take_along_axis(indices, sort_idx, axis=dim)
        return Tensor(values, requires_grad=self.requires_grad,
                      _ctx=Context(TopkBackward, (self,), (self.data, indices, dim))), \
               Tensor(indices, requires_grad=False)

    def pad(self, pad, mode='constant', value=0):
        if mode == 'constant':
            result = np.pad(self.data, pad, mode='constant', constant_values=value)
        elif mode == 'reflect':
            result = np.pad(self.data, pad, mode='reflect')
        elif mode == 'replicate':
            result = np.pad(self.data, pad, mode='edge')
        elif mode == 'circular':
            result = np.pad(self.data, pad, mode='wrap')
        else:
            result = np.pad(self.data, pad, mode=mode)
        return Tensor(result, requires_grad=self.requires_grad,
                      _ctx=Context(PadBackward, (self,), (pad, mode)))

    def unfold(self, dim, size, step):
        shape = list(self.shape)
        shape[dim] = (shape[dim] - size) // step + 1
        shape.append(size)
        strides = list(self.strides)
        strides.append(self.strides[dim])
        result = np.lib.stride_tricks.as_strided(self.data, shape=shape, strides=strides)
        return Tensor(result.copy(), requires_grad=self.requires_grad)

    def im2col(self, kernel_size, stride=1, padding=0):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        return self.pad(padding).unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])

    def fold(self, output_size, kernel_size, stride=1, padding=0):
        return Tensor(self.data, requires_grad=self.requires_grad)

    def col2im(self, output_size, kernel_size, stride=1, padding=0):
        return Tensor(self.data, requires_grad=self.requires_grad)

    # ---- comparison ops ----

    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other))
        return Tensor(self.data == other.data)

    def __ne__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other))
        return Tensor(self.data != other.data)

    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other))
        return Tensor(self.data < other.data)

    def __le__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other))
        return Tensor(self.data <= other.data)

    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other))
        return Tensor(self.data > other.data)

    def __ge__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(_bk_from_list(other))
        return Tensor(self.data >= other.data)

    def __bool__(self):
        return bool(self.data)

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    # ---- broadcasting helpers ----

    def contiguous_view(self):
        return self.reshape(*self.shape)

    # ---- loss helpers ----

    def cross_entropy_loss(self, target, label_smoothing=0.0):
        if isinstance(target, Tensor):
            target = target.data
        target_idx = target.astype(int)
        n = target_idx.shape[0]
        log_probs = self.log_softmax(axis=-1)
        V = log_probs.shape[-1]
        if label_smoothing > 0:
            one_hot = _bk_zeros((n, V), dtype=np.float32)
            idx = _bk_arange(n).astype(np.int64)
            one_hot[idx, target_idx] = 1.0 - label_smoothing
            one_hot += label_smoothing / V
        else:
            one_hot = _bk_zeros((n, V), dtype=np.float32)
            idx = _bk_arange(n).astype(np.int64)
            one_hot[idx, target_idx] = 1.0
        one_hot_t = Tensor(one_hot, requires_grad=False)
        cross = -(log_probs * one_hot_t).sum(axis=-1) / n
        return cross.mean()

    def nll_loss(self, target):
        return nll_loss(self, target)

    def bce_loss(self, target):
        return bce_loss(self, target)

    def mse_loss(self, target):
        return mse_loss(self, target)

    def l1_loss(self, target):
        return l1_loss(self, target)

    def huber_loss(self, target, delta=1.0):
        return huber_loss(self, target, delta)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def cross_entropy(logits, target, label_smoothing=0.0):
    if isinstance(target, Tensor):
        target = target.data
    probs = logits.softmax(axis=-1)
    log_probs = _bk_log(probs.data + 1e-8)
    if label_smoothing > 0:
        V = probs.shape[-1]
        idx = _bk_arange(target.shape[0]).astype(np.int64)
        loss = -(1 - label_smoothing) * log_probs[idx, target.astype(int)] - \
               label_smoothing * log_probs.mean(axis=-1)
    else:
        idx = _bk_arange(target.shape[0]).astype(np.int64)
        loss = -log_probs[idx, target.astype(int)]
    return Tensor(loss.mean(), requires_grad=logits.requires_grad)

def nll_loss(log_probs, target):
    if isinstance(target, Tensor):
        target = target.data
    loss = -log_probs.data[_bk_arange(target.shape[0]).astype(np.int64), target.astype(int)]
    return Tensor(loss.mean(), requires_grad=log_probs.requires_grad)

def bce_loss(pred, target):
    if isinstance(target, Tensor):
        target = target.data
    loss = -(target * _bk_log(pred.data + 1e-7) + (1 - target) * _bk_log(1 - pred.data + 1e-7))
    return Tensor(loss.mean(), requires_grad=pred.requires_grad)

def mse_loss(pred, target):
    if isinstance(target, Tensor):
        target = target.data
    return ((pred - Tensor(target)) ** 2).mean()

def l1_loss(pred, target):
    if isinstance(target, Tensor):
        target = target.data
    return (pred - Tensor(target)).abs().mean()

def huber_loss(pred, target, delta=1.0):
    if isinstance(target, Tensor):
        target = target.data
    diff = pred.data - target
    loss = _bk_where(_bk_abs(diff) < delta, 0.5 * diff ** 2, delta * (_bk_abs(diff) - 0.5 * delta))
    return Tensor(loss.mean(), requires_grad=pred.requires_grad)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def zeros(*shape, requires_grad=False):
    return Tensor(_bk_zeros(shape, dtype=np.float32), requires_grad=requires_grad)

def ones(*shape, requires_grad=False):
    return Tensor(_bk_ones(shape, dtype=np.float32), requires_grad=requires_grad)

def full(*shape, fill_value, requires_grad=False):
    return Tensor(_bk_full(shape, fill_value, dtype=np.float32), requires_grad=requires_grad)

def arange(start, stop=None, step=1.0, requires_grad=False):
    if stop is None:
        stop = start
        start = 0
    return Tensor(_bk_arange(start, stop, step, dtype=np.float32), requires_grad=requires_grad)

def linspace(start, stop, num, requires_grad=False):
    return Tensor(np.linspace(start, stop, num, dtype=np.float32), requires_grad=requires_grad)

def eye(n, m=None, requires_grad=False):
    return Tensor(np.eye(n, m, dtype=np.float32), requires_grad=requires_grad)

def empty(*shape, requires_grad=False):
    return Tensor(_bk_empty(shape, dtype=np.float32), requires_grad=requires_grad)

def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)

def randn(*shape, requires_grad=False):
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)

def rand(*shape, requires_grad=False):
    return Tensor(np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad)

def randint(low, high=None, size=None, requires_grad=False):
    if high is None:
        high = low
        low = 0
    return Tensor(np.random.randint(low, high, size=size).astype(np.float32), requires_grad=requires_grad)

def normal(mean=0, std=1, size=None, requires_grad=False):
    return Tensor(np.random.normal(mean, std, size=size).astype(np.float32), requires_grad=requires_grad)

def uniform(low=0, high=1, size=None, requires_grad=False):
    return Tensor(np.random.uniform(low, high, size=size).astype(np.float32), requires_grad=requires_grad)

def cat(tensors, dim=0):
    arrays = [t.data if isinstance(t, Tensor) else t for t in tensors]
    req = any(t.requires_grad for t in tensors if isinstance(t, Tensor))
    return Tensor(_bk_concatenate(arrays, axis=dim), requires_grad=req)

def stack(tensors, dim=0):
    arrays = [t.data if isinstance(t, Tensor) else t for t in tensors]
    req = any(t.requires_grad for t in tensors if isinstance(t, Tensor))
    return Tensor(_bk_stack(arrays, axis=dim), requires_grad=req)

def where(condition, x, y):
    if isinstance(condition, Tensor): condition = condition.data
    if isinstance(x, Tensor): x = x.data
    if isinstance(y, Tensor): y = y.data
    return Tensor(_bk_where(condition, x, y))

def unique(x, sorted=True, return_counts=False):
    return x.unique(sorted=sorted, return_counts=return_counts)

def sort(x, dim=-1, descending=False):
    return x.sort(dim=dim, descending=descending)

def topk(x, k, dim=-1, largest=True):
    return x.topk(k, dim=dim, largest=largest)

def einsum(equation, *tensors):
    if tensors:
        return tensors[0].einsum(equation, *tensors[1:])
    return Tensor(_bk_einsum(equation))

def meshgrid(*tensors, indexing='ij'):
    arrays = [t.data if isinstance(t, Tensor) else t for t in tensors]
    result = np.meshgrid(*arrays, indexing=indexing)
    return tuple(Tensor(r) for r in result)

def tril(x, diagonal=0):
    return Tensor(np.tril(x.data, diagonal), requires_grad=x.requires_grad)

def triu(x, diagonal=0):
    return Tensor(np.triu(x.data, diagonal), requires_grad=x.requires_grad)

def diag(v, diagonal=0):
    if isinstance(v, Tensor): v = v.data
    return Tensor(np.diag(v, diagonal))

def trace(x, offset=0, axis1=0, axis2=1):
    return Tensor(np.trace(x.data, offset, axis1, axis2))

def clip_tensor(x, min_val, max_val):
    return x.clamp(min_val, max_val)

def sign_tensor(x):
    return x.sign()

def abs_tensor(x):
    return x.abs()

def neg_tensor(x):
    return -x

def exp_tensor(x):
    return x.exp()

def log_tensor(x):
    return x.log()

def sqrt_tensor(x):
    return x.sqrt()

def sin_tensor(x):
    return x.sin()

def cos_tensor(x):
    return x.cos()

def tan_tensor(x):
    return x.tan()

def sigmoid_tensor(x):
    return x.sigmoid()

def relu_tensor(x):
    return x.relu()

def gelu_tensor(x):
    return x.gelu()

def silu_tensor(x):
    return x.silu()

def tanh_tensor(x):
    return x.tanh()

def softmax_tensor(x, axis=-1):
    return x.softmax(axis=axis)

def log_softmax_tensor(x, axis=-1):
    return x.log_softmax(axis=axis)

def matmul_tensor(a, b):
    return a.matmul(b)

def dot_tensor(a, b):
    return a.dot(b)

def outer_tensor(a, b):
    return a.outer(b)

def sum_tensor(x, axis=None, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims)

def mean_tensor(x, axis=None, keepdims=False):
    return x.mean(axis=axis, keepdims=keepdims)

def max_tensor(x, axis=None, keepdims=False):
    return x.max(axis=axis, keepdims=keepdims)

def min_tensor(x, axis=None, keepdims=False):
    return x.min(axis=axis, keepdims=keepdims)

def prod_tensor(x, axis=None, keepdims=False):
    return x.prod(axis=axis, keepdims=keepdims)

def var_tensor(x, axis=None, keepdims=False, correction=0):
    return x.var(axis=axis, keepdims=keepdims, correction=correction)

def std_tensor(x, axis=None, keepdims=False, correction=0):
    return x.std(axis=axis, keepdims=keepdims, correction=correction)

def norm_tensor(x, ord=None, axis=None, keepdims=False):
    return x.norm(ord=ord, axis=axis, keepdims=keepdims)

def logsumexp_tensor(x, axis=None, keepdims=False):
    return x.logsumexp(axis=axis, keepdims=keepdims)

def cumsum_tensor(x, axis=0):
    return x.cumsum(axis=axis)

def cumprod_tensor(x, axis=0):
    return x.cumprod(axis=axis)

def reshape_tensor(x, *shape):
    return x.reshape(*shape)

def transpose_tensor(x, dim0, dim1):
    return x.transpose(dim0, dim1)

def permute_tensor(x, *axes):
    return x.permute(*axes)

def squeeze_tensor(x, axis=None):
    return x.squeeze(axis=axis)

def unsqueeze_tensor(x, axis):
    return x.unsqueeze(axis)

def expand_tensor(x, *shape):
    return x.expand(*shape)

def flatten_tensor(x, start_dim=0, end_dim=-1):
    return x.flatten(start_dim, end_dim)

def cat_tensor(tensors, dim=0):
    return cat(tensors, dim)

def stack_tensor(tensors, dim=0):
    return stack(tensors, dim)

def pad_tensor(x, pad, mode='constant', value=0):
    return x.pad(pad, mode, value)

def unfold_tensor(x, dim, size, step):
    return x.unfold(dim, size, step)

def im2col_tensor(x, kernel_size, stride=1, padding=0):
    return x.im2col(kernel_size, stride, padding)

def col2im_tensor(x, output_size, kernel_size, stride=1, padding=0):
    return x

def where_tensor(condition, x, y):
    return Tensor.where(Tensor(condition.data if isinstance(condition, Tensor) else condition), x, y)

def nonzero_tensor(x):
    return x.nonzero()

def unique_tensor(x, sorted=True, return_counts=False):
    return x.unique(sorted, return_counts)

def sort_tensor(x, dim=-1, descending=False):
    return x.sort(dim, descending)

def argsort_tensor(x, dim=-1, descending=False):
    return x.argsort(dim, descending)

def topk_tensor(x, k, dim=-1, largest=True, sorted=True):
    return x.topk(k, dim, largest, sorted)

def gather_tensor(x, dim, index):
    return x.gather(dim, index)

def scatter_tensor(x, dim, index, src):
    return x.scatter_(dim, index, src)

def masked_fill_tensor(x, mask, value):
    return x.masked_fill_(mask, value)

def masked_select_tensor(x, mask):
    return x.masked_select(mask)

def index_put_tensor(x, indices, values):
    return x.index_put_(indices, values)


# ---------------------------------------------------------------------------
# Mixed precision utilities
# ---------------------------------------------------------------------------

class GradScaler:
    def __init__(self, init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._steps = 0

    @property
    def scale(self):
        return self._scale

    def unscale(self, optimizer):
        for group in optimizer.param_groups:
            for p in group.get("params", []):
                if isinstance(p, Tensor) and p.grad is not None:
                    p.grad.data = p.grad.data / self._scale
        return True

    def step(self, optimizer):
        self.unscale(optimizer)
        optimizer.step()

    def update(self):
        self._steps += 1
        if self._steps % self._growth_interval == 0:
            self._scale *= self._growth_factor

    def backoff(self):
        self._scale *= self._backoff_factor


def gradient_checkpoint(function, *args, **kwargs):
    saved_inputs = []
    saved_outputs = []

    class _CheckpointCtx:
        def __init__(self):
            self.active = True

    ctx = _CheckpointCtx()

    def _save_for_backward(inp):
        if ctx.active:
            saved_inputs.append(inp)

    orig_backward = Tensor.backward
    def _patched_backward(self, gradient=None, retain_graph=False, create_graph=False):
        saved_outputs.append(self)
        return orig_backward(self, gradient, retain_graph, create_graph)

    Tensor.backward = _patched_backward
    try:
        output = function(*args, **kwargs)
    finally:
        Tensor.backward = orig_backward
        ctx.active = False

    return output


class SparseTensor:
    def __init__(self, indices, values, dense_shape):
        self.indices = _bk_from_list(indices)
        self.values = _bk_from_list(values, dtype=np.float32)
        self.dense_shape = tuple(dense_shape)

    @classmethod
    def from_dense(cls, dense):
        if isinstance(dense, Tensor):
            dense = dense.data
        indices = _bk_from_list(np.nonzero(dense))
        values = dense[dense != 0]
        return cls(indices, values, dense.shape)

    def to_dense(self):
        out = _bk_zeros(self.dense_shape, dtype=np.float32)
        if self.indices.size > 0:
            out[tuple(self.indices)] = self.values
        return Tensor(out)

    def matmul(self, other):
        dense = self.to_dense()
        if isinstance(other, SparseTensor):
            other = other.to_dense()
        return dense.matmul(other)

    def __repr__(self):
        return f"SparseTensor(indices={self.indices.shape}, values={self.values.shape}, shape={self.dense_shape})"

