import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class AttentionVisualizer:
    """Plot attention weights as text heatmap."""

    def explain(self, model: Any, input_tokens: List[str], layer: int = 0) -> dict:
        """Visualize attention from model.

        Args:
            model: must expose get_attention(input_tokens) -> np.ndarray [layers, heads, seq, seq]
            input_tokens: list of token strings
            layer: which layer to visualize
        """
        attn = model.get_attention(input_tokens)
        weights = attn[layer]  # heads x seq x seq
        avg_weights = np.mean(weights, axis=0)  # seq x seq
        heatmap = self._format_heatmap(avg_weights, input_tokens)
        return {
            "attention_matrix": avg_weights,
            "heatmap_text": heatmap,
            "layer": layer,
            "tokens": input_tokens,
        }

    def _format_heatmap(self, matrix: np.ndarray, tokens: List[str]) -> str:
        n = len(tokens)
        col_width = max(6, max(len(t) for t in tokens) + 1)
        header = " " * col_width + "".join(t.rjust(col_width) for t in tokens)
        rows = [header]
        for i, tok in enumerate(tokens):
            row = tok.rjust(col_width)
            for j in range(n):
                val = matrix[i, j]
                block = self._value_to_block(val)
                row += block.rjust(col_width)
            rows.append(row)
        return "\n".join(rows)

    def _value_to_block(self, val: float) -> str:
        if val < 0.01:
            return " "
        elif val < 0.1:
            return "░"
        elif val < 0.3:
            return "▒"
        elif val < 0.6:
            return "▓"
        else:
            return "█"


class IntegratedGradients:
    """Compute attributions via integrated gradients from baseline to input."""

    def __init__(self, n_steps: int = 30):
        self.n_steps = n_steps

    def explain(self, model: Any, input_array: np.ndarray, target: int) -> dict:
        """Compute integrated gradients.

        Args:
            model: must expose forward(x) -> np.ndarray (logits)
            input_array: numpy array input
            target: target class index
        """
        baseline = np.zeros_like(input_array)
        alphas = np.linspace(0, 1, self.n_steps + 1)
        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (input_array - baseline)
            grad = self._compute_gradient(model, interpolated, target)
            gradients.append(grad)
        avg_grads = np.mean(gradients, axis=0)
        attributions = (input_array - baseline) * avg_grads
        return {
            "attributions": attributions,
            "input": input_array,
            "target": target,
            "n_steps": self.n_steps,
            "magnitude": float(np.sum(np.abs(attributions))),
        }

    def _compute_gradient(self, model: Any, x: np.ndarray, target: int) -> np.ndarray:
        eps = 1e-5
        grad = np.zeros_like(x)
        flat = x.flatten()
        orig_shape = x.shape
        for i in range(flat.size):
            flat_plus = flat.copy()
            flat_plus[i] += eps
            flat_minus = flat.copy()
            flat_minus[i] -= eps
            logits_plus = model.forward(flat_plus.reshape(orig_shape))
            logits_minus = model.forward(flat_minus.reshape(orig_shape))
            grad.flat[i] = (logits_plus[target] - logits_minus[target]) / (2 * eps)
        return grad


class GradientXInput:
    """Element-wise product of gradient and input for saliency maps."""

    def explain(self, model: Any, input_array: np.ndarray, target: int) -> dict:
        """Compute Gradient x Input saliency.

        Args:
            model: must expose forward(x) -> np.ndarray
            input_array: input tensor
            target: class index for gradient
        """
        grad = self._compute_gradient(model, input_array, target)
        saliency = input_array * grad
        return {
            "saliency": saliency,
            "gradient": grad,
            "input": input_array,
            "target": target,
            "max_saliency": float(np.max(np.abs(saliency))),
        }

    def _compute_gradient(self, model: Any, x: np.ndarray, target: int) -> np.ndarray:
        eps = 1e-5
        grad = np.zeros_like(x, dtype=float)
        flat = x.flatten().astype(float)
        orig_shape = x.shape
        for i in range(flat.size):
            flat_plus = flat.copy()
            flat_plus[i] += eps
            flat_minus = flat.copy()
            flat_minus[i] -= eps
            l_plus = model.forward(flat_plus.reshape(orig_shape))
            l_minus = model.forward(flat_minus.reshape(orig_shape))
            grad.flat[i] = (l_plus[target] - l_minus[target]) / (2 * eps)
        return grad


class LogitLens:
    """Project intermediate residual stream to vocabulary logits."""

    def __init__(self, projection: Optional[np.ndarray] = None):
        self.projection = projection

    def explain(self, model: Any, input_array: np.ndarray, position: int = 0) -> dict:
        """Project hidden state to vocabulary.

        Args:
            model: must expose get_residual(input, layer) -> np.ndarray
            input_array: input tokens or embeddings
            position: token position to analyze
        """
        results = []
        n_layers = getattr(model, "n_layers", 10)
        for layer_idx in range(n_layers):
            hidden = model.get_residual(input_array, layer_idx)
            if hidden.ndim > 1:
                hidden = hidden[position]
            if self.projection is not None:
                logits = hidden @ self.projection.T
            else:
                logits = hidden
            top_k_idx = np.argsort(-logits)[:5]
            top_k_vals = logits[top_k_idx]
            results.append({
                "layer": layer_idx,
                "top_tokens": top_k_idx.tolist(),
                "top_logits": top_k_vals.tolist(),
            })
        return {"per_layer": results, "position": position}

    def get_convergence_point(self, results: dict) -> int:
        per_layer = results["per_layer"]
        for i in range(1, len(per_layer)):
            prev = set(per_layer[i - 1]["top_tokens"])
            curr = set(per_layer[i]["top_tokens"])
            if prev == curr:
                return per_layer[i]["layer"]
        return per_layer[-1]["layer"]


class CausalTracing:
    """Patch activations and measure causal effect on output."""

    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level

    def explain(self, model: Any, input_array: np.ndarray, target: int, layer: int = 0) -> dict:
        """Trace causal effect by patching a layer's activation.

        Args:
            model: must expose forward(x), get_residual(x, layer), set_residual(x, layer, val)
            input_array: input
            target: output token/class to measure
            layer: layer to patch
        """
        baseline_logits = model.forward(input_array)
        baseline_prob = self._softmax(baseline_logits)[target]
        hidden = model.get_residual(input_array, layer)
        corrupted = hidden + self.noise_level * np.random.randn(*hidden.shape)
        patched_logits = model.forward_from_residual(corrupted, layer)
        patched_prob = self._softmax(patched_logits)[target]
        effect = baseline_prob - patched_prob
        restore_logits = model.forward_from_residual(hidden, layer)
        restore_prob = self._softmax(restore_logits)[target]
        restoration = patched_prob - restore_prob
        return {
            "baseline_prob": float(baseline_prob),
            "patched_prob": float(patched_prob),
            "restored_prob": float(restore_prob),
            "causal_effect": float(effect),
            "restoration_delta": float(restoration),
            "layer": layer,
            "target": target,
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
