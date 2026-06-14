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
        assert np.allclose((Tensor([1,2,3]) + Tensor([4,5,6])).data, [5,7,9])

    def test_mul(self):
        assert np.allclose((Tensor([2,3,4]) * Tensor([5,6,7])).data, [10,18,28])

    def test_matmul(self):
        c = Tensor([[1,2],[3,4]]).matmul(Tensor([[5,6],[7,8]]))
        assert c.shape == (2, 2)
        assert np.allclose(c.data, [[19,22],[43,50]])

    def test_softmax(self):
        s = Tensor([1.0, 2.0, 3.0]).softmax()
        assert abs(s.data.sum() - 1.0) < 1e-5

    def test_backward(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        (a * b).backward()
        assert np.allclose(a.grad.data, [3.0])
        assert np.allclose(b.grad.data, [2.0])

    def test_relu(self):
        assert np.allclose(Tensor([-1, 0, 1, 2]).relu().data, [0, 0, 1, 2])

    def test_sigmoid(self):
        assert abs(Tensor([0.0]).sigmoid().data[0] - 0.5) < 1e-5

    def test_rmsnorm(self):
        t = Tensor([1.0, 2.0, 3.0])
        n = t.rmsnorm()
        assert n.shape == (3,)

    def test_sum(self):
        assert Tensor([1, 2, 3, 4]).sum().item() == 10.0

    def test_mean(self):
        assert Tensor([1, 2, 3, 4]).mean().item() == 2.5

    def test_pow(self):
        assert np.allclose((Tensor([2, 3, 4]) ** 2).data, [4, 9, 16])

    def test_transpose(self):
        t = Tensor([[1, 2], [3, 4]])
        tt = t.transpose()
        assert tt.shape == (2, 2)
        assert np.allclose(tt.data, [[1, 3], [2, 4]])

    def test_chained_backward(self):
        a = Tensor([5.0], requires_grad=True)
        c = (a - 2.0) ** 2
        c.backward()
        assert isinstance(a.grad, Tensor)
        # grad of (a-2)^2 w.r.t. a is 2*(a-2) = 2*3 = 6
        assert abs(a.grad.data[0] - 6.0) < 0.1
