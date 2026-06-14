"""Tests for custom autodiff tensor engine."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.autodiff.tensor import Tensor


class TestTensor:
    def test_create(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert np.allclose(t.data, [1, 2, 3])

    def test_add(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        assert np.allclose(c.data, [5, 7, 9])

    def test_mul(self):
        a = Tensor([2, 3, 4])
        b = Tensor([5, 6, 7])
        c = a * b
        assert np.allclose(c.data, [10, 18, 28])

    def test_matmul(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a.matmul(b)
        assert c.shape == (2, 2)
        assert np.allclose(c.data, [[19, 22], [43, 50]])

    def test_softmax(self):
        t = Tensor([1.0, 2.0, 3.0])
        s = t.softmax()
        assert np.allclose(s.data.sum(), 1.0, atol=1e-5)

    def test_backward(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a * b
        c.backward()
        assert np.allclose(a.grad.data, [3.0])
        assert np.allclose(b.grad.data, [2.0])

    def test_relu(self):
        t = Tensor([-1, 0, 1, 2])
        r = t.relu()
        assert np.allclose(r.data, [0, 0, 1, 2])

    def test_sigmoid(self):
        t = Tensor([0.0])
        s = t.sigmoid()
        assert np.allclose(s.data, 0.5, atol=1e-5)

    def test_rmsnorm(self):
        t = Tensor([1.0, 2.0, 3.0])
        n = t.rmsnorm()
        assert abs(n.data.mean() - 0) < 1.0  # roughly centered

    def test_sum(self):
        t = Tensor([1, 2, 3, 4])
        assert t.sum().item() == 10.0

    def test_mean(self):
        t = Tensor([1, 2, 3, 4])
        assert t.mean().item() == 2.5

    def test_pow(self):
        t = Tensor([2, 3, 4])
        r = t ** 2
        assert np.allclose(r.data, [4, 9, 16])

    def test_transpose(self):
        t = Tensor([[1, 2], [3, 4]])
        tt = t.transpose()
        assert tt.shape == (2, 2)
        assert np.allclose(tt.data, [[1, 3], [2, 4]])
