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


class AttentionRollout:
    """Multiply attention matrices across layers for rollout visualization."""

    def rollout(self, attention_matrices):
        result = attention_matrices[0]
        for attn in attention_matrices[1:]:
            if result.ndim == 4:
                result = np.einsum('bhij,bhjk->bhik', result, attn)
            else:
                result = np.einsum('bij,bjk->bik', result, attn)
        return result


class LogitLens:
    """Project intermediate residual stream to vocabulary logits."""

    def __init__(self, model, lm_head):
        self.model = model
        self.lm_head = lm_head

    def project(self, hidden_states):
        return self.lm_head(hidden_states)


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


class SHAPExplainer:
    """SHAP Shapley values via sampling."""
    def __init__(self, model):
        self.model = model

    def explain(self, input_tokens, num_samples=100):
        base_value = np.zeros_like(input_tokens, dtype=np.float32)
        shap_values = np.zeros_like(input_tokens, dtype=np.float32)
        n_features = len(input_tokens)
        for _ in range(num_samples):
            mask = np.random.randint(0, 2, size=n_features)
            nonzero = np.where(mask)[0]
            if len(nonzero) == 0:
                continue
            included = input_tokens[nonzero]
            score = np.mean(included) if len(included) > 0 else 0
            for idx in nonzero:
                shap_values[idx] += score / num_samples
        return shap_values


class ProbingClassifier:
    """Linear probe on each layer's activations."""
    def __init__(self, model):
        self.model = model
        self.probes = {}

    def train_probe(self, layer_idx, activations, labels):
        W = np.random.randn(activations.shape[-1], len(np.unique(labels))) * 0.01
        b = np.zeros(W.shape[1])
        lr = 0.01
        for _ in range(100):
            logits = activations @ W + b
            exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
            grad_w = activations.T @ (probs - np.eye(W.shape[1])[labels]) / len(labels)
            grad_b = (probs - np.eye(W.shape[1])[labels]).mean(axis=0)
            W -= lr * grad_w
            b -= lr * grad_b
        self.probes[layer_idx] = (W, b)

    def evaluate_probe(self, layer_idx, test_activations, test_labels):
        W, b = self.probes[layer_idx]
        logits = test_activations @ W + b
        preds = np.argmax(logits, axis=-1)
        return np.mean(preds == test_labels)
