from core.nn.module import Module
from core.nn.linear import Linear
from core.autodiff.tensor import Tensor


class SwiGLU(Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        self.w1 = Linear(dim, intermediate, bias=False)
        self.w2 = Linear(intermediate, dim, bias=False)
        self.w3 = Linear(dim, intermediate, bias=False)

    def forward(self, x):
        return self.w2(self.w1(x).silu() * self.w3(x))


class FeedForward(Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        self.w1 = Linear(dim, intermediate, bias=False)
        self.w2 = Linear(intermediate, dim, bias=False)

    def forward(self, x):
        return self.w2(self.w1(x).gelu())
