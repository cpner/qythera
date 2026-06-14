"""Model merging algorithms."""

import numpy as np
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, field
from collections import OrderedDict


def _get_param_keys(models: List[Dict]) -> List[str]:
    keys = set()
    for m in models:
        keys.update(m.keys())
    return sorted(keys)


def _ensure_same_shape(params: List[Dict], keys: List[str]) -> bool:
    for key in keys:
        shapes = [p[key].shape for p in params if key in p]
        if len(set(shapes)) > 1:
            return False
    return True


@dataclass
class MergeConfig:
    base_model_idx: int = 0
    density: float = 0.5
    scale: float = 1.0
    normalize: bool = True


def task_arithmetic(
    models: List[Dict[str, np.ndarray]],
    weights: List[float],
    base_idx: int = 0,
    density: float = 1.0,
) -> Dict[str, np.ndarray]:
    if len(models) != len(weights):
        raise ValueError("models and weights must have same length")

    base = models[base_idx]
    keys = _get_param_keys(models)
    merged = {}

    for key in keys:
        base_param = base.get(key, np.zeros_like(list(models[0].values())[0]))
        delta = np.zeros_like(base_param, dtype=np.float64)
        for i, (model, w) in enumerate(zip(models, weights)):
            if i == base_idx:
                continue
            if key in model:
                d = (model[key].astype(np.float64) - base_param.astype(np.float64))
                if density < 1.0:
                    threshold = np.percentile(np.abs(d), (1 - density) * 100)
                    mask = np.abs(d) >= threshold
                    d = d * mask
                delta += w * d
        merged[key] = (base_param.astype(np.float64) + delta).astype(base_param.dtype)

    return merged


def ties_merge(
    models: List[Dict[str, np.ndarray]],
    weights: List[float],
    base_idx: int = 0,
    density: float = 0.5,
) -> Dict[str, np.ndarray]:
    if len(models) != len(weights):
        raise ValueError("models and weights must have same length")

    base = models[base_idx]
    keys = _get_param_keys(models)
    merged = {}

    for key in keys:
        base_param = base.get(key, np.zeros_like(list(models[0].values())[0]))
        deltas = []
        for i, (model, w) in enumerate(zip(models, weights)):
            if i == base_idx:
                continue
            if key in model:
                deltas.append(w * (model[key].astype(np.float64) - base_param.astype(np.float64)))

        if not deltas:
            merged[key] = base_param
            continue

        stacked = np.stack(deltas, axis=0)
        abs_sum = np.sum(np.abs(stacked), axis=0)
        threshold = np.percentile(abs_sum, (1 - density) * 100)
        mask = abs_sum >= threshold

        sign_votes = np.sign(stacked)
        majority_sign = np.sign(np.sum(sign_votes, axis=0))
        consistent_mask = mask & (sign_votes == majority_sign).all(axis=0)

        result = base_param.astype(np.float64)
        for i, d in enumerate(deltas):
            layer_mask = consistent_mask
            result += np.where(layer_mask, d, 0)
        merged[key] = result.astype(base_param.dtype)

    return merged


def dare_merge(
    models: List[Dict[str, np.ndarray]],
    weights: List[float],
    base_idx: int = 0,
    density: float = 0.5,
    scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    if len(models) != len(weights):
        raise ValueError("models and weights must have same length")

    base = models[base_idx]
    keys = _get_param_keys(models)
    merged = {}

    for key in keys:
        base_param = base.get(key, np.zeros_like(list(models[0].values())[0]))
        combined_delta = np.zeros_like(base_param, dtype=np.float64)

        for i, (model, w) in enumerate(zip(models, weights)):
            if i == base_idx:
                continue
            if key in model:
                d = (model[key].astype(np.float64) - base_param.astype(np.float64))

                num_elements = d.size
                num_keep = int(num_elements * density)
                flat_d = d.flatten()

                if num_keep < num_elements:
                    threshold = np.percentile(np.abs(flat_d), (1 - density) * 100)
                    mask = np.abs(d) >= threshold
                    flat_mask = mask.flatten()
                    indices = np.where(flat_mask)[0]
                    drop_indices = np.random.choice(
                        indices, size=max(0, len(indices) - num_keep), replace=False
                    )
                    drop_mask = np.zeros(num_elements, dtype=bool)
                    drop_mask[drop_indices] = True
                    d = np.where(drop_mask.reshape(d.shape), 0, d)
                    d = d * (num_elements / max(num_keep, 1))

                combined_delta += w * d

        merged[key] = (base_param.astype(np.float64) + scale * combined_delta).astype(base_param.dtype)

    return merged


def slerp(
    p0: np.ndarray,
    p1: np.ndarray,
    t: float,
) -> np.ndarray:
    flat0 = p0.flatten().astype(np.float64)
    flat1 = p1.flatten().astype(np.float64)

    norm0 = np.linalg.norm(flat0)
    norm1 = np.linalg.norm(flat1)

    if norm0 < 1e-10 or norm1 < 1e-10:
        return ((1 - t) * flat0 + t * flat1).astype(p0.dtype).reshape(p0.shape)

    unit0 = flat0 / norm0
    unit1 = flat1 / norm1

    dot = np.clip(np.dot(unit0, unit1), -1.0, 1.0)
    omega = np.arccos(dot)

    if abs(omega) < 1e-6:
        result = (1 - t) * flat0 + t * flat1
    else:
        sin_omega = np.sin(omega)
        coeff0 = np.sin((1 - t) * omega) / sin_omega
        coeff1 = np.sin(t * omega) / sin_omega
        result = coeff0 * flat0 + coeff1 * flat1

    return result.astype(p0.dtype).reshape(p0.shape)


def slerp_merge(
    models: List[Dict[str, np.ndarray]],
    weights: List[float],
) -> Dict[str, np.ndarray]:
    if len(models) != len(weights):
        raise ValueError("models and weights must have same length")
    if len(models) < 2:
        raise ValueError("need at least 2 models for SLERP")

    keys = _get_param_keys(models)
    merged = {}

    for key in keys:
        params = [m[key] for m in models if key in m]
        w_params = [(m[key], w) for m, w in zip(models, weights) if key in m]

        if len(w_params) < 2:
            merged[key] = w_params[0][0] if w_params else params[0]
            continue

        result = w_params[0][0].astype(np.float64)
        cumulative_t = 0.0

        for i in range(1, len(w_params)):
            p_i = w_params[i][0].astype(np.float64)
            t_i = w_params[i][1] / (weights[0] + sum(w for _, w in w_params[:i]))
            result = slerp(result, p_i, t_i)

        merged[key] = result.astype(params[0].dtype)

    return merged


def model_soup(
    models: List[Dict[str, np.ndarray]],
    weights: Optional[List[float]] = None,
) -> Dict[str, np.ndarray]:
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    keys = _get_param_keys(models)
    merged = {}

    for key in keys:
        acc = np.zeros_like(list(models[0].values())[0], dtype=np.float64)
        for model, w in zip(models, weights):
            if key in model:
                acc += w * model[key].astype(np.float64)
        merged[key] = acc.astype(list(models[0].values())[0].dtype)

    return merged


MERGE_METHODS = {
    "task_arithmetic": task_arithmetic,
    "ties": ties_merge,
    "dare": dare_merge,
    "slerp": slerp_merge,
    "model_soup": model_soup,
}


def merge(
    models: List[Dict[str, np.ndarray]],
    weights: List[float],
    method: str = "task_arithmetic",
    **kwargs,
) -> Dict[str, np.ndarray]:
    if method not in MERGE_METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(MERGE_METHODS.keys())}")

    fn = MERGE_METHODS[method]
    import inspect
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    filtered_kwargs = {}
    for k, v in kwargs.items():
        if k in params:
            filtered_kwargs[k] = v

    if method in ("task_arithmetic", "ties", "dare"):
        return fn(models, weights, **filtered_kwargs)
    elif method == "slerp":
        return fn(models, weights, **filtered_kwargs)
    elif method == "model_soup":
        return fn(models, weights, **filtered_kwargs)
    else:
        return fn(models, weights, **filtered_kwargs)


class ModelMerger:
    def __init__(self, method: str = "task_arithmetic", **config):
        self.method = method
        self.config = config
        self.history: List[Dict] = []

    def merge(
        self,
        models: List[Dict[str, np.ndarray]],
        weights: List[float],
    ) -> Dict[str, np.ndarray]:
        result = merge(models, weights, self.method, **self.config)
        self.history.append({
            "method": self.method,
            "num_models": len(models),
            "weights": weights,
            "num_params": len(result),
        })
        return result

    def multi_merge(
        self,
        layers: List[List[Dict[str, np.ndarray]]],
        layer_weights: List[List[float]],
        methods: List[str],
    ) -> List[Dict[str, np.ndarray]]:
        if len(layers) != len(methods):
            raise ValueError("layers and methods must have same length")

        results = []
        for layer_models, layer_w, method in zip(layers, layer_weights, methods):
            results.append(merge(layer_models, layer_w, method))
        return results
