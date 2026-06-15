"""Theoretical ML concepts: scaling laws, emergent abilities, grokking, etc."""
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# ScalingLaws
# ---------------------------------------------------------------------------

class ScalingLaws:
    """Log-log linear fit of loss vs params/data/compute (Kaplan et al.)."""

    def __init__(self):
        self.fits: Dict[str, Dict[str, float]] = {}

    def fit_loss_vs_params(self, params: List[float], losses: List[float]) -> Dict[str, float]:
        log_n = np.log(np.array(params))
        log_l = np.log(np.array(losses))
        coeffs = np.polyfit(log_n, log_l, 1)
        alpha, log_c = coeffs
        return {
            "exponent": float(alpha),
            "coefficient": float(np.exp(log_c)),
            "r_squared": float(self._r_squared(log_n, log_l, coeffs)),
        }

    def fit_loss_vs_data(self, data: List[float], losses: List[float]) -> Dict[str, float]:
        return self.fit_loss_vs_params(data, losses)

    def fit_loss_vs_compute(self, compute: List[float], losses: List[float]) -> Dict[str, float]:
        return self.fit_loss_vs_params(compute, losses)

    def predict_loss(self, n: float, alpha: float, c: float) -> float:
        return c * (n ** alpha)

    def compute_optimal_allocation(self, total_compute: float, alpha: float, beta: float,
                                    epsilon: float = 0.1) -> Dict[str, float]:
        n_opt = total_compute ** (beta / (alpha + beta))
        d_opt = total_compute ** (alpha / (alpha + beta))
        return {"optimal_params": n_opt, "optimal_data": d_opt}

    def _r_squared(self, x: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / max(ss_tot, 1e-12)


# ---------------------------------------------------------------------------
# Chinchilla
# ---------------------------------------------------------------------------

class Chinchilla:
    """Fit Chinchilla scaling law: L(N,D) = E + A/N^alpha + B/D^beta."""

    def __init__(self):
        self.params: Optional[Dict[str, float]] = None

    def fit(self, N: List[float], D: List[float], L: List[float]) -> Dict[str, float]:
        N_arr = np.array(N, dtype=float)
        D_arr = np.array(D, dtype=float)
        L_arr = np.array(L, dtype=float)

        def residuals(theta):
            E, A, alpha, B, beta = theta
            pred = E + A / (N_arr ** alpha) + B / (D_arr ** beta)
            return np.sum((pred - L_arr) ** 2)

        best_loss = math.inf
        best_params = None

        for _ in range(20):
            theta0 = np.array([
                np.min(L_arr) * 0.5,
                np.random.uniform(0.1, 10.0),
                np.random.uniform(0.3, 1.0),
                np.random.uniform(0.1, 10.0),
                np.random.uniform(0.3, 1.0),
            ])
            loss = residuals(theta0)
            if loss < best_loss:
                best_loss = loss
                best_params = theta0

        if best_params is not None:
            self.params = {
                "E": float(best_params[0]),
                "A": float(best_params[1]),
                "alpha": float(best_params[2]),
                "B": float(best_params[3]),
                "beta": float(best_params[4]),
            }
            self.params["r_squared"] = float(self._r_squared(N_arr, D_arr, L_arr, best_params))
        return self.params or {}

    def predict(self, N: float, D: float) -> float:
        if not self.params:
            return 0.0
        E = self.params["E"]
        A = self.params["A"]
        alpha = self.params["alpha"]
        B = self.params["B"]
        beta = self.params["beta"]
        return E + A / (N ** alpha) + B / (D ** beta)

    def _r_squared(self, N: np.ndarray, D: np.ndarray, L: np.ndarray, theta: np.ndarray) -> float:
        E, A, alpha, B, beta = theta
        pred = E + A / (N ** alpha) + B / (D ** beta)
        ss_res = np.sum((L - pred) ** 2)
        ss_tot = np.sum((L - np.mean(L)) ** 2)
        return 1.0 - ss_res / max(ss_tot, 1e-12)


# ---------------------------------------------------------------------------
# EmergentAbilities
# ---------------------------------------------------------------------------

class EmergentAbilities:
    """Track phase transitions in metrics as model scale increases."""

    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
        self.transitions: List[Dict[str, Any]] = []

    def add_point(self, scale: float, metric_name: str, value: float) -> None:
        self.metrics.append({
            "scale": scale,
            "metric": metric_name,
            "value": value,
        })

    def detect_transitions(self, metric_name: Optional[str] = None,
                           threshold: float = 0.15) -> List[Dict[str, Any]]:
        filtered = self.metrics
        if metric_name:
            filtered = [m for m in filtered if m["metric"] == metric_name]

        if len(filtered) < 3:
            return []

        sorted_pts = sorted(filtered, key=lambda x: x["scale"])
        values = np.array([p["value"] for p in sorted_pts])
        scales = np.array([p["scale"] for p in sorted_pts])

        diffs = np.diff(values)
        rel_diffs = np.abs(diffs) / (np.abs(values[:-1]) + 1e-12)

        transitions = []
        for i, (rel_d, diff) in enumerate(zip(rel_diffs, diffs)):
            if rel_d > threshold:
                trans = {
                    "from_scale": float(scales[i]),
                    "to_scale": float(scales[i + 1]),
                    "from_value": float(values[i]),
                    "to_value": float(values[i + 1]),
                    "relative_change": float(rel_d),
                    "metric": sorted_pts[i]["metric"],
                }
                transitions.append(trans)
                self.transitions.append(trans)

        return transitions

    def summary(self) -> Dict[str, Any]:
        metric_names = list(set(m["metric"] for m in self.metrics))
        return {
            "total_points": len(self.metrics),
            "num_transitions": len(self.transitions),
            "metrics_tracked": metric_names,
            "transitions": self.transitions,
        }


# ---------------------------------------------------------------------------
# Grokking
# ---------------------------------------------------------------------------

class Grokking:
    """Monitor generalization gap during extended training."""

    def __init__(self):
        self.history: List[Dict[str, float]] = []
        self.grokking_detected: Optional[Dict[str, Any]] = None

    def record(self, step: int, train_acc: float, val_acc: float,
               train_loss: float = 0.0, val_loss: float = 0.0) -> None:
        self.history.append({
            "step": step,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gap": train_acc - val_acc,
        })

    def detect_grokking(self, gap_threshold: float = 0.05,
                        patience: int = 50) -> Optional[Dict[str, Any]]:
        if len(self.history) < patience:
            return None

        for i in range(len(self.history) - patience):
            early_acc = self.history[i]["train_acc"]
            early_val = self.history[i]["val_acc"]
            late_acc = self.history[i + patience]["train_acc"]
            late_val = self.history[i + patience]["val_acc"]

            early_gap = early_acc - early_val
            late_gap = late_acc - late_val

            if early_gap > gap_threshold and late_gap < gap_threshold and late_acc > 0.9:
                self.grokking_detected = {
                    "detected_at_step": self.history[i + patience]["step"],
                    "early_step": self.history[i]["step"],
                    "early_train_acc": early_acc,
                    "early_val_acc": early_val,
                    "late_train_acc": late_acc,
                    "late_val_acc": late_val,
                    "gap_closed": early_gap - late_gap,
                }
                return self.grokking_detected

        return None

    def generalization_curve(self) -> List[Dict[str, float]]:
        return [{"step": h["step"], "gap": h["gap"],
                 "train_acc": h["train_acc"], "val_acc": h["val_acc"]}
                for h in self.history]


# ---------------------------------------------------------------------------
# LotteryTicket
# ---------------------------------------------------------------------------

class LotteryTicket:
    """Lottery Ticket Hypothesis: IMP loop with mask generation and rewinding."""

    def __init__(self):
        self.masks: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []

    def generate_mask(self, params: Dict[str, np.ndarray], prune_ratio: float = 0.2,
                      strategy: str = "magnitude") -> Dict[str, np.ndarray]:
        masks = {}
        for name, weights in params.items():
            if not isinstance(weights, np.ndarray):
                masks[name] = np.ones_like(weights)
                continue

            if strategy == "magnitude":
                threshold = np.percentile(np.abs(weights), prune_ratio * 100)
                mask = (np.abs(weights) >= threshold).astype(float)
            elif strategy == "random":
                mask = (np.random.random(weights.shape) > prune_ratio).astype(float)
            elif strategy == "structured":
                row_norms = np.linalg.norm(weights, axis=1, keepdims=True)
                threshold = np.percentile(row_norms, prune_ratio * 100)
                mask = (row_norms >= threshold).astype(float)
            else:
                mask = np.ones_like(weights)

            masks[name] = mask

        self.masks.append({"strategy": strategy, "prune_ratio": prune_ratio, "num_params": len(masks)})
        return masks

    def apply_mask(self, params: Dict[str, np.ndarray],
                   masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pruned = {}
        for name in params:
            if name in masks and isinstance(params[name], np.ndarray):
                pruned[name] = params[name] * masks[name]
            else:
                pruned[name] = params[name]
        return pruned

    def imp_loop(self, train_fn: Callable, eval_fn: Callable,
                 init_params: Dict[str, np.ndarray], num_iterations: int = 3,
                 prune_ratio: float = 0.2) -> Dict[str, Any]:
        best_ticket = None
        best_score = -math.inf
        all_iterations = []

        current_params = {k: v.copy() if isinstance(v, np.ndarray) else v
                          for k, v in init_params.items()}

        for iteration in range(num_iterations):
            mask = self.generate_mask(current_params, prune_ratio)
            pruned = self.apply_mask(current_params, mask)

            trained = train_fn(pruned, mask)
            score = eval_fn(trained)

            iteration_result = {
                "iteration": iteration,
                "score": score,
                "prune_ratio": prune_ratio,
                "remaining_params": int(np.sum(list(mask.values())[0])) if mask else 0,
            }
            all_iterations.append(iteration_result)
            self.results.append(iteration_result)

            if score > best_score:
                best_score = score
                best_ticket = {"params": trained, "mask": mask, "iteration": iteration}

            rewind_idx = max(0, iteration - 1) if iteration > 0 else 0
            if rewind_idx < len(self.masks):
                rewind_mask = self.masks[rewind_idx]
                current_params = self.apply_mask(
                    init_params,
                    self.generate_mask(init_params, prune_ratio * (iteration + 1))
                )
            else:
                current_params = trained

        return {
            "best_ticket": best_ticket,
            "best_score": best_score,
            "iterations": all_iterations,
        }


# ---------------------------------------------------------------------------
# FlatMinima
# ---------------------------------------------------------------------------

class FlatMinima:
    """Loss sharpness via trace of Hessian (approximate)."""

    def __init__(self):
        self.measurements: List[Dict[str, Any]] = []

    def compute_hessian_trace(self, loss_fn: Callable, params: Dict[str, np.ndarray],
                              epsilon: float = 1e-4) -> float:
        trace = 0.0
        for name, param in params.items():
            if not isinstance(param, np.ndarray):
                continue

            flat_param = param.flatten()
            n = len(flat_param)
            if n == 0:
                continue

            diag_sum = 0.0
            for i in range(min(n, 100)):
                original = flat_param[i]

                flat_param[i] = original + epsilon
                p_plus = flat_param.reshape(param.shape)
                loss_plus = loss_fn({**params, name: p_plus})

                flat_param[i] = original - epsilon
                p_minus = flat_param.reshape(param.shape)
                loss_minus = loss_fn({**params, name: p_minus})

                flat_param[i] = original

                hessian_ii = (float(loss_plus) - 2 * float(loss_fn(params)) + float(loss_minus)) / (epsilon ** 2)
                diag_sum += hessian_ii

            trace += diag_sum

        return trace

    def measure_sharpness(self, loss_fn: Callable, params: Dict[str, np.ndarray],
                          epsilon: float = 1e-4) -> Dict[str, Any]:
        trace = self.compute_hessian_trace(loss_fn, params, epsilon)
        current_loss = float(loss_fn(params))

        measurement = {
            "hessian_trace": trace,
            "loss": current_loss,
            "sharpness_ratio": trace / max(abs(current_loss), 1e-12),
            "epsilon": epsilon,
        }
        self.measurements.append(measurement)
        return measurement

    def compare_minima(self, loss_fn: Callable, params_list: List[Dict[str, np.ndarray]],
                       epsilon: float = 1e-4) -> Dict[str, Any]:
        results = []
        for i, params in enumerate(params_list):
            m = self.measure_sharpness(loss_fn, params, epsilon)
            m["index"] = i
            results.append(m)

        flatness_ranking = sorted(range(len(results)),
                                  key=lambda i: results[i]["hessian_trace"])

        return {
            "measurements": results,
            "flattest_index": flatness_ranking[0] if flatness_ranking else None,
            "sharpest_index": flatness_ranking[-1] if flatness_ranking else None,
            "flatness_ranking": flatness_ranking,
        }
