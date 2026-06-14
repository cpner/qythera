"""Computation graph for tracking operations during forward pass."""

from typing import List, Optional
from core.autodiff.tensor import Tensor


class ComputationGraph:
    """Tracks computation history for reverse-mode autodiff.
    
    Builds a DAG of operations during forward pass,
    then traverses it in reverse for gradient computation.
    """
    
    def __init__(self):
        self.nodes: List[Tensor] = []
        self._enabled = True

    def track(self, tensor: Tensor):
        if self._enabled and tensor.requires_grad:
            self.nodes.append(tensor)

    def backward(self, loss: Tensor, retain_graph=False):
        """Run backward pass from loss tensor."""
        if loss.grad is None:
            loss.grad = Tensor.ones_like(loss)

        visited = set()
        queue = [loss]
        order = []

        while queue:
            node = queue.pop(0)
            if id(node) in visited:
                continue
            visited.add(id(node))
            order.append(node)
            for child in node._prev:
                if id(child) not in visited:
                    queue.append(child)

        for node in reversed(order):
            node._backward()

        if not retain_graph:
            self.clear()

    def clear(self):
        self.nodes.clear()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.clear()
