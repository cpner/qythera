import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.autodiff.tensor import Tensor

class TestTensor:
    def test_add(self): assert np.allclose((Tensor([1,2,3])+Tensor([4,5,6])).data, [5,7,9])
    def test_mul(self): assert np.allclose((Tensor([2,3,4])*Tensor([5,6,7])).data, [10,18,28])
    def test_matmul(self): assert np.allclose(Tensor([[1,2],[3,4]]).matmul(Tensor([[5,6],[7,8]])).data, [[19,22],[43,50]])
    def test_backward(self):
        a = Tensor([2.0], requires_grad=True); b = Tensor([3.0], requires_grad=True)
        c = a * b; c.backward()
        assert np.allclose(a.grad.data, [3.0])
    def test_relu(self): assert np.allclose(Tensor([-1,0,1,2]).relu().data, [0,0,1,2])
    def test_sigmoid(self): assert abs(Tensor([0.0]).sigmoid().data[0] - 0.5) < 1e-5
    def test_softmax(self): assert abs(Tensor([1,2,3]).softmax().data.sum() - 1.0) < 1e-5
    def test_log(self): assert abs(Tensor([1.0]).log().data[0]) < 1e-5
    def test_mean(self): assert Tensor([1,2,3,4]).mean().item() == 2.5
    def test_sum(self): assert Tensor([1,2,3,4]).sum().item() == 10.0
    def test_pow(self): assert np.allclose((Tensor([2,3,4])**2).data, [4,9,16])
    def test_training(self):
        w = Tensor([0.5], requires_grad=True)
        x = Tensor(np.arange(10, dtype=np.float32))
        Y = 2 * x.data + 1
        for _ in range(200):
            pred = x * w
            loss = ((pred - Tensor(Y))**2).mean()
            w.grad = None; loss.backward()
            w.data = w.data - 0.01 * w.grad
        assert abs(w.data[0] - 2.0) < 0.5
