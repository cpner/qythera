import numpy as np
from core.nn.module import Module, Parameter
from core.autodiff.tensor import Tensor

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, indices):
        if isinstance(indices, Tensor): indices = indices.data.astype(int)
        return Tensor(self.weight.data[indices], requires_grad=True)
