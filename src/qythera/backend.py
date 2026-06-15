import math
import array as _array
from functools import reduce
from operator import mul

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


def get_backend():
    return 'numpy' if HAS_NUMPY else 'python'


class PyArray:
    def __init__(self, data, shape):
        self.data = data
        self.shape = shape
        self.ndim = len(shape)
        self.size = reduce(mul, shape, 1)
        self.dtype = 'float32'

    def _flat_index(self, indices):
        idx = 0
        stride = 1
        for i in reversed(range(self.ndim)):
            idx += indices[i] * stride
            stride *= self.shape[i]
        return idx

    def _get(self, indices):
        return self.data[self._flat_index(indices)]

    def _set(self, indices, value):
        self.data[self._flat_index(indices)] = value

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self._get(key)
        return self._get((key,))

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self._set(key, value)
        else:
            self._set((key,), value)

    def tolist(self):
        if self.ndim == 1:
            return list(self.data)
        result = []
        inner_size = reduce(mul, self.shape[1:], 1) if self.ndim > 1 else 1
        for i in range(self.shape[0]):
            sub = PyArray(
                list(self.data[i * inner_size:(i + 1) * inner_size]),
                self.shape[1:]
            )
            result.append(sub.tolist())
        return result

    def __repr__(self):
        return f"PyArray({self.tolist()}, shape={self.shape})"


DTYPE_MAP = {'float32': 'f', 'float64': 'd', 'int32': 'i', 'int64': 'l', 'int': 'l'}


def _make_py_array(shape, fill=0.0):
    size = reduce(mul, shape, 1)
    return PyArray(_array.array('d', [fill] * size), shape)


def _broadcast_shape(shapes):
    max_ndim = max(len(s) for s in shapes)
    result = []
    for i in range(max_ndim):
        dim = 1
        for s in shapes:
            idx = len(s) - max_ndim + i
            if idx >= 0:
                if s[idx] != 1 and s[idx] != dim:
                    if dim == 1:
                        dim = s[idx]
                    elif s[idx] != dim:
                        raise ValueError(f"Cannot broadcast shapes {shapes}")
        result.append(dim)
    return tuple(result)


def _broadcast_to(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    result = _make_py_array(target_shape)
    strides = [1] * arr.ndim
    for i in range(arr.ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * arr.shape[i + 1]
    target_strides = [1] * len(target_shape)
    for i in range(len(target_shape) - 2, -1, -1):
        target_strides[i] = target_strides[i + 1] * target_shape[i + 1]

    def _fill(target_idx, arr_idx, dim):
        if dim == arr.ndim:
            val = arr._get(tuple(arr_idx))
            result._set(tuple(target_idx), val)
        else:
            arr_dim = arr.shape[dim]
            target_dim = target_shape[dim]
            for i in range(target_dim):
                ai = arr_idx[:]
                if arr_dim == 1:
                    ai[dim] = 0
                else:
                    ai[dim] = i
                ti = target_idx[:]
                ti[dim] = i
                _fill(ti, ai, dim + 1)

    _fill([0] * len(target_shape), [0] * arr.ndim, 0)
    return result


def zeros(shape, dtype='float32'):
    if HAS_NUMPY:
        return np.zeros(shape, dtype=dtype)
    return _make_py_array(shape, 0.0)


def ones(shape, dtype='float32'):
    if HAS_NUMPY:
        return np.ones(shape, dtype=dtype)
    return _make_py_array(shape, 1.0)


def full(shape, fill_value, dtype='float32'):
    if HAS_NUMPY:
        return np.full(shape, fill_value, dtype=dtype)
    return _make_py_array(shape, fill_value)


def empty(shape, dtype='float32'):
    if HAS_NUMPY:
        return np.empty(shape, dtype=dtype)
    return _make_py_array(shape, 0.0)


def arange(start, stop=None, step=1, dtype='float32'):
    if stop is None:
        stop = start
        start = 0
    if HAS_NUMPY:
        return np.arange(start, stop, step, dtype=dtype)
    vals = []
    v = start
    while v < stop:
        vals.append(v)
        v += step
    return PyArray(_array.array('d', vals), (len(vals),))


def from_list(data, dtype='float32'):
    if HAS_NUMPY:
        return np.array(data, dtype=dtype)
    flat = []

    def _flatten(d):
        if isinstance(d, (list, tuple)):
            for item in d:
                _flatten(item)
        else:
            flat.append(float(d))

    _flatten(data)
    shape = []
    d = data
    while isinstance(d, (list, tuple)):
        shape.append(len(d))
        d = d[0] if len(d) > 0 else 0
    shape = tuple(shape) if shape else (0,)
    return PyArray(_array.array('d', flat), shape)


def to_numpy(array):
    if HAS_NUMPY:
        if isinstance(array, PyArray):
            return np.array(array.data, dtype=np.float64).reshape(array.shape)
        return array
    return array


def to_list(array):
    if isinstance(array, PyArray):
        return array.tolist()
    if HAS_NUMPY:
        return array.tolist()
    return array


def matmul(a, b):
    if HAS_NUMPY:
        return np.matmul(a, b)
    if isinstance(a, PyArray) and isinstance(b, PyArray):
        if a.ndim == 1 and b.ndim == 1:
            result = sum(a.data[i] * b.data[i] for i in range(a.shape[0]))
            return PyArray(_array.array('d', [result]), (1,))
        if b.ndim == 1:
            b = PyArray(list(b.data), (b.shape[0], 1))
        if a.ndim == 1:
            a = PyArray(list(a.data), (1, a.shape[0]))
        m, k1 = a.shape[0], a.shape[1]
        k2, n = b.shape[0], b.shape[1]
        if k1 != k2:
            raise ValueError(f"Incompatible dimensions: {a.shape} and {b.shape}")
        flat = [0.0] * (m * n)
        for i in range(m):
            for j in range(n):
                s = 0.0
                for k in range(k1):
                    s += a._get((i, k)) * b._get((k, j))
                flat[i * n + j] = s
        return PyArray(_array.array('d', flat), (m, n))
    return a @ b


def einsum(subscripts, *operands):
    if HAS_NUMPY:
        return np.einsum(subscripts, *operands)
    raise NotImplementedError("einsum requires numpy")


def sum(arr, axis=None, keepdims=False):
    if HAS_NUMPY:
        return np.sum(arr, axis=axis, keepdims=keepdims)
    if isinstance(arr, PyArray):
        if axis is None:
            return sum(arr.data)
        result_shape = tuple(s for i, s in enumerate(arr.shape) if i != axis)
        if keepdims:
            result_shape = result_shape[:axis] + (1,) + result_shape[axis:]
        result = _make_py_array(result_shape, 0.0)
        inner_size = reduce(mul, arr.shape[axis + 1:], 1) if axis + 1 < arr.ndim else 1
        outer_size = reduce(mul, arr.shape[:axis], 1) if axis > 0 else 1
        axis_size = arr.shape[axis]
        for o in range(outer_size):
            for i in range(axis_size):
                for j in range(inner_size):
                    src_idx = list(range(len(arr.shape)))
                    outer_idx = []
                    tmp = o
                    for d in range(axis - 1, -1, -1):
                        outer_idx.insert(0, tmp % arr.shape[d])
                        tmp //= arr.shape[d]
                    for d in range(axis):
                        src_idx[d] = outer_idx[d] if d < len(outer_idx) else 0
                    src_idx[axis] = i
                    tmp2 = j
                    for d in range(len(arr.shape) - 1, axis, -1):
                        src_idx[d] = tmp2 % arr.shape[d]
                        tmp2 //= arr.shape[d]
                    target_idx = list(range(len(result_shape)))
                    tmp3 = o
                    for d in range(len(result_shape) - (0 if keepdims else 1), -1, -1):
                        if d < axis:
                            target_idx[d] = tmp3 % result_shape[d]
                            tmp3 //= result_shape[d]
                        elif d >= axis and not keepdims:
                            target_idx[d] = tmp3 % result_shape[d] if d < len(result_shape) else 0
                            tmp3 //= result_shape[d] if d < len(result_shape) else 1
                    val = arr._get(tuple(src_idx))
                    curr = result._get(tuple(target_idx))
                    result._set(tuple(target_idx), curr + val)
        return result
    raise TypeError(f"Unsupported type: {type(arr)}")


def mean(arr, axis=None, keepdims=False):
    if HAS_NUMPY:
        return np.mean(arr, axis=axis, keepdims=keepdims)
    s = sum(arr, axis=axis, keepdims=keepdims)
    if axis is None:
        count = arr.size if isinstance(arr, PyArray) else 1
    else:
        count = arr.shape[axis] if isinstance(arr, PyArray) else 1
    if isinstance(s, PyArray):
        return PyArray(_array.array('d', [v / count for v in s.data]), s.shape)
    return s / count


def max(arr, axis=None, keepdims=False):
    if HAS_NUMPY:
        return np.max(arr, axis=axis, keepdims=keepdims)
    if isinstance(arr, PyArray):
        if axis is None:
            return max(arr.data)
        result_shape = tuple(s for i, s in enumerate(arr.shape) if i != axis)
        if keepdims:
            result_shape = result_shape[:axis] + (1,) + result_shape[axis:]
        result = _make_py_array(result_shape, float('-inf'))
        for flat_idx in range(arr.size):
            idx = []
            tmp = flat_idx
            for d in reversed(range(arr.ndim)):
                idx.insert(0, tmp % arr.shape[d])
                tmp //= arr.shape[d]
            val = arr._get(tuple(idx))
            target_idx = [idx[i] for i in range(arr.ndim) if i != axis]
            curr = result._get(tuple(target_idx))
            result._set(tuple(target_idx), max(curr, val))
        return result
    raise TypeError(f"Unsupported type: {type(arr)}")


def min(arr, axis=None, keepdims=False):
    if HAS_NUMPY:
        return np.min(arr, axis=axis, keepdims=keepdims)
    if isinstance(arr, PyArray):
        if axis is None:
            return min(arr.data)
        result_shape = tuple(s for i, s in enumerate(arr.shape) if i != axis)
        if keepdims:
            result_shape = result_shape[:axis] + (1,) + result_shape[axis:]
        result = _make_py_array(result_shape, float('inf'))
        for flat_idx in range(arr.size):
            idx = []
            tmp = flat_idx
            for d in reversed(range(arr.ndim)):
                idx.insert(0, tmp % arr.shape[d])
                tmp //= arr.shape[d]
            val = arr._get(tuple(idx))
            target_idx = [idx[i] for i in range(arr.ndim) if i != axis]
            curr = result._get(tuple(target_idx))
            result._set(tuple(target_idx), min(curr, val))
        return result
    raise TypeError(f"Unsupported type: {type(arr)}")


def exp(arr):
    if HAS_NUMPY:
        return np.exp(arr)
    if isinstance(arr, PyArray):
        return PyArray(_array.array('d', [math.exp(v) for v in arr.data]), arr.shape)
    return math.exp(arr)


def log(arr):
    if HAS_NUMPY:
        return np.log(arr)
    if isinstance(arr, PyArray):
        return PyArray(_array.array('d', [math.log(v) for v in arr.data]), arr.shape)
    return math.log(arr)


def sqrt(arr):
    if HAS_NUMPY:
        return np.sqrt(arr)
    if isinstance(arr, PyArray):
        return PyArray(_array.array('d', [math.sqrt(v) for v in arr.data]), arr.shape)
    return math.sqrt(arr)


def abs(arr):
    if HAS_NUMPY:
        return np.abs(arr)
    if isinstance(arr, PyArray):
        return PyArray(_array.array('d', [abs(v) for v in arr.data]), arr.shape)
    return abs(arr)


def sin(arr):
    if HAS_NUMPY:
        return np.sin(arr)
    if isinstance(arr, PyArray):
        return PyArray(_array.array('d', [math.sin(v) for v in arr.data]), arr.shape)
    return math.sin(arr)


def cos(arr):
    if HAS_NUMPY:
        return np.cos(arr)
    if isinstance(arr, PyArray):
        return PyArray(_array.array('d', [math.cos(v) for v in arr.data]), arr.shape)
    return math.cos(arr)


def tanh(arr):
    if HAS_NUMPY:
        return np.tanh(arr)
    if isinstance(arr, PyArray):
        return PyArray(_array.array('d', [math.tanh(v) for v in arr.data]), arr.shape)
    return math.tanh(arr)


def softmax(arr, axis=-1):
    if HAS_NUMPY:
        return np.softmax(arr, axis=axis)
    if not isinstance(arr, PyArray):
        raise TypeError(f"Unsupported type: {type(arr)}")
    if axis < 0:
        axis = arr.ndim + axis
    m = max(arr, axis=axis, keepdims=True)
    e = exp(arr - m)
    s = sum(e, axis=axis, keepdims=True)
    return e / s


def where(condition, x, y):
    if HAS_NUMPY:
        return np.where(condition, x, y)
    if isinstance(condition, PyArray):
        result = PyArray(_array.array('d', [0.0] * condition.size), condition.shape)
        for i in range(condition.size):
            result.data[i] = x.data[i] if condition.data[i] else y.data[i]
        return result
    return x if condition else y


def stack(arrays, axis=0):
    if HAS_NUMPY:
        return np.stack(arrays, axis=axis)
    if not arrays:
        raise ValueError("Need at least one array to stack")
    first = arrays[0]
    if not isinstance(first, PyArray):
        raise TypeError(f"Unsupported type: {type(first)}")
    new_shape = list(first.shape)
    new_shape.insert(axis, len(arrays))
    result = _make_py_array(tuple(new_shape))
    for i, arr in enumerate(arrays):
        target_idx = [slice(None)] * result.ndim
        target_idx[axis] = i
        source_data = arr.data
        dest_offset = i * first.size
        result.data[dest_offset:dest_offset + first.size] = source_data
    return result


def concatenate(arrays, axis=0):
    if HAS_NUMPY:
        return np.concatenate(arrays, axis=axis)
    if not arrays:
        raise ValueError("Need at least one array to concatenate")
    first = arrays[0]
    if not isinstance(first, PyArray):
        raise TypeError(f"Unsupported type: {type(first)}")
    new_shape = list(first.shape)
    new_shape[axis] = sum(a.shape[axis] for a in arrays)
    result = _make_py_array(tuple(new_shape))
    offset = 0
    for arr in arrays:
        chunk_size = arr.size
        result.data[offset:offset + chunk_size] = arr.data
        offset += chunk_size
    return result


def reshape(arr, shape):
    if HAS_NUMPY:
        return np.reshape(arr, shape)
    if isinstance(arr, PyArray):
        return PyArray(arr.data, shape)
    raise TypeError(f"Unsupported type: {type(arr)}")


def broadcast_to(arr, shape):
    if HAS_NUMPY:
        return np.broadcast_to(arr, shape)
    if isinstance(arr, PyArray):
        return _broadcast_to(arr, shape)
    raise TypeError(f"Unsupported type: {type(arr)}")
