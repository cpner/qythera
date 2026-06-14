"""Backward pass utilities for gradient computation."""

from core.autodiff.tensor import Tensor
from typing import List


def backward_pass(loss: Tensor, tensors: List[Tensor] = None, retain_graph=False):
    """Compute gradients of loss with respect to given tensors.
    
    Args:
        loss: Scalar loss tensor to differentiate from
        tensors: List of tensors to compute gradients for (if None, all)
        retain_graph: If True, don't clear the computation graph
    """
    if loss.grad is None:
        loss.grad = Tensor.ones_like(loss)

    visited = set()
    order = []
    queue = [loss]

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
        for node in order:
            node._prev = []


def zero_grad(tensors: List[Tensor]):
    """Reset gradients for a list of tensors."""
    for t in tensors:
        t.grad = None
