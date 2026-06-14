"""Qythera Autodiff Engine - Custom automatic differentiation from scratch."""

from core.autodiff.tensor import Tensor
from core.autodiff.graph import ComputationGraph
from core.autodiff.backward import backward_pass
from core.autodiff.optim import SGD, Adam, AdamW

__all__ = ["Tensor", "ComputationGraph", "backward_pass", "SGD", "Adam", "AdamW"]
