from core.nn.module import Module
from core.autodiff.tensor import Tensor


class Dropout(Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        return x.dropout(p=self.p, training=self.training)
