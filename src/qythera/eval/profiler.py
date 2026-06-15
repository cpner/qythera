"""Profiling and analysis tools for Qythera. Pure Python + NumPy."""
import cProfile
import io
import math
import os
import pstats
import random
import time
import tracemalloc
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# cProfile wrapper
# ---------------------------------------------------------------------------

def profile_function(fn: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Profile a function call and return stats."""
    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        success = True
    except Exception as e:
        result = e
        success = False
    end_time = time.perf_counter()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    profile_output = stream.getvalue()

    return {
        "success": success,
        "result": result,
        "elapsed_ms": round((end_time - start_time) * 1000, 2),
        "total_calls": stats.total_calls,
        "profile_output": profile_output,
    }


# ---------------------------------------------------------------------------
# BottleneckDetector
# ---------------------------------------------------------------------------

class BottleneckDetector:
    """Find top-K slowest operations from profiling data."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def analyze(self, fn: Callable, *args, top_k: int = 10, **kwargs) -> Dict[str, Any]:
        profiler = cProfile.Profile()
        profiler.enable()
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            success = True
        except Exception as e:
            result = e
            success = False
        elapsed = time.perf_counter() - start
        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('tottime')
        stats.print_stats(top_k + 5)
        raw = stream.getvalue()

        bottlenecks = []
        for func_info, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:top_k]:
            filename, line, func_name = func_info
            bottlenecks.append({
                "file": filename,
                "line": line,
                "function": func_name,
                "ncalls": nc,
                "tottime_ms": round(tt * 1000, 3),
                "cumtime_ms": round(ct * 1000, 3),
            })

        record = {
            "success": success,
            "elapsed_ms": round(elapsed * 1000, 2),
            "bottlenecks": bottlenecks,
            "raw_profile": raw,
        }
        self.records.append(record)
        return record

    def summary(self) -> str:
        if not self.records:
            return "No profiling records."
        last = self.records[-1]
        lines = [f"Elapsed: {last['elapsed_ms']}ms, Success: {last['success']}"]
        for b in last["bottlenecks"][:5]:
            lines.append(f"  {b['function']} @ {b['file']}:{b['line']} - {b['tottime_ms']}ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MemoryProfiler
# ---------------------------------------------------------------------------

class MemoryProfiler:
    """Track allocations via tracemalloc."""

    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []

    def start(self):
        tracemalloc.start()

    def stop(self):
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def snapshot(self, label: str = "") -> Dict[str, Any]:
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        snap = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()

        top_stats = snap.statistics('lineno')[:10]
        allocations = []
        for stat in top_stats:
            allocations.append({
                "file": str(stat.traceback),
                "size_kb": round(stat.size / 1024, 2),
                "count": stat.count,
            })

        record = {
            "label": label,
            "current_memory_kb": round(current / 1024, 2),
            "peak_memory_kb": round(peak / 1024, 2),
            "allocations": allocations,
            "timestamp": time.time(),
        }
        self.snapshots.append(record)
        return record

    def diff(self, label_a: str = "", label_b: str = "") -> Optional[Dict[str, Any]]:
        if len(self.snapshots) < 2:
            return None
        a = None
        b = None
        for s in self.snapshots:
            if s["label"] == label_a or (label_a == "" and a is None):
                a = s
            if s["label"] == label_b or (label_b == "" and b is None and s is not a):
                b = s
        if a and b:
            return {
                "memory_delta_kb": b["current_memory_kb"] - a["current_memory_kb"],
                "peak_delta_kb": b["peak_memory_kb"] - a["peak_memory_kb"],
            }
        return None


# ---------------------------------------------------------------------------
# AutoML
# ---------------------------------------------------------------------------

class AutoML:
    """Grid search with early stopping on validation loss plateau."""

    def __init__(self, param_grid: Optional[Dict[str, List]] = None, patience: int = 5):
        self.param_grid = param_grid or {}
        self.patience = patience
        self.results: List[Dict[str, Any]] = []

    def grid_search(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        max_epochs_per_trial: int = 20,
        val_loss_threshold: float = 1e-6,
    ) -> Dict[str, Any]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = self._generate_combos(values)

        best_val = math.inf
        best_params = None
        all_results = []

        for combo in combos:
            params = dict(zip(keys, combo))
            best_trial_val = math.inf
            no_improve_count = 0
            epoch_losses = []

            for epoch in range(max_epochs_per_trial):
                train_loss = train_fn(params, epoch)
                val_loss = eval_fn(params, epoch)
                epoch_losses.append({"epoch": epoch, "train": train_loss, "val": val_loss})

                if val_loss < best_trial_val - val_loss_threshold:
                    best_trial_val = val_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= self.patience:
                    break

            result = {
                "params": params,
                "best_val_loss": best_trial_val,
                "epochs_run": len(epoch_losses),
                "history": epoch_losses,
            }
            all_results.append(result)

            if best_trial_val < best_val:
                best_val = best_trial_val
                best_params = params

        self.results = all_results
        return {
            "best_params": best_params,
            "best_val_loss": best_val,
            "total_trials": len(all_results),
            "all_results": all_results,
        }

    def _generate_combos(self, values: List[List]) -> List[List]:
        if not values:
            return [[]]
        result = []
        for v in values[0]:
            for rest in self._generate_combos(values[1:]):
                result.append([v] + rest)
        return result


# ---------------------------------------------------------------------------
# ModelAnalyzer
# ---------------------------------------------------------------------------

class ModelAnalyzer:
    """Count parameters, estimate FLOPs, memory usage."""

    def analyze(self, model: Any) -> Dict[str, Any]:
        param_count = 0
        param_details = []

        if hasattr(model, 'parameters'):
            for name, param in model.parameters():
                n = 1
                for s in param.shape:
                    n *= s
                param_count += n
                param_details.append({
                    "name": name,
                    "shape": list(param.shape),
                    "params": n,
                    "size_kb": round(n * 4 / 1024, 2),
                })

        embedding_params = sum(p["params"] for p in param_details if "embed" in p["name"].lower())
        attention_params = sum(p["params"] for p in param_details if any(k in p["name"].lower() for k in ["attn", "qkv", "attention"]))
        ffn_params = sum(p["params"] for p in param_details if any(k in p["name"].lower() for k in ["ffn", "mlp", "feed"]))

        estimated_memory_mb = round(param_count * 4 / (1024 * 1024), 2)
        estimated_memory_bytes = param_count * 4

        return {
            "total_params": param_count,
            "total_params_human": self._human_readable(param_count),
            "embedding_params": embedding_params,
            "attention_params": attention_params,
            "ffn_params": ffn_params,
            "estimated_memory_mb": estimated_memory_mb,
            "estimated_memory_bytes": estimated_memory_bytes,
            "num_layers": sum(1 for p in param_details if "layer" in p["name"].lower()),
            "param_details": param_details,
        }

    def estimate_flops(self, seq_len: int, batch_size: int = 1, model_info: Optional[Dict] = None) -> Dict[str, int]:
        params = model_info.get("total_params", 10_000_000) if model_info else 10_000_000
        forward_flops = 2 * params * seq_len * batch_size
        backward_flops = 2 * forward_flops
        total = forward_flops + backward_flops
        return {
            "forward_flops": forward_flops,
            "backward_flops": backward_flops,
            "total_flops": total,
            "flops_per_token": 2 * params,
        }

    def _human_readable(self, n: int) -> str:
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        return str(n)


# ---------------------------------------------------------------------------
# DatasetInspector
# ---------------------------------------------------------------------------

class DatasetInspector:
    """Token distribution and sequence length statistics."""

    def analyze(self, sequences: List[List[int]], vocab_size: int = 0) -> Dict[str, Any]:
        lengths = [len(s) for s in sequences]

        token_counts = defaultdict(int)
        for seq in sequences:
            for tok in seq:
                token_counts[tok] += 1

        total_tokens = sum(token_counts.values())
        freq_dist = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        top_tokens = freq_dist[:20]
        top_token_probs = [(t, c, c / max(total_tokens, 1)) for t, c in top_tokens]

        unique_tokens = len(token_counts)
        coverage = unique_tokens / max(vocab_size, 1) if vocab_size > 0 else 0

        if token_counts:
            probs = np.array([c / total_tokens for c in token_counts.values()])
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
        else:
            entropy = 0

        return {
            "num_sequences": len(sequences),
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "vocab_coverage": round(coverage, 4),
            "token_entropy_bits": round(float(entropy), 4),
            "length_stats": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "mean": round(float(np.mean(lengths)), 2) if lengths else 0,
                "median": round(float(np.median(lengths)), 2) if lengths else 0,
                "std": round(float(np.std(lengths)), 2) if lengths else 0,
            },
            "top_tokens": [{"token": t, "count": c, "prob": round(p, 6)} for t, c, p in top_token_probs],
            "length_histogram": self._histogram(lengths),
        }

    def _histogram(self, values: List[int], bins: int = 10) -> List[Dict[str, Any]]:
        if not values:
            return []
        min_v, max_v = min(values), max(values)
        if min_v == max_v:
            return [{"range": f"{min_v}", "count": len(values)}]
        bin_edges = np.linspace(min_v, max_v, bins + 1)
        hist, _ = np.histogram(values, bins=bin_edges)
        result = []
        for i in range(len(hist)):
            lo = int(bin_edges[i])
            hi = int(bin_edges[i + 1])
            result.append({"range": f"{lo}-{hi}", "count": int(hist[i])})
        return result


# ---------------------------------------------------------------------------
# NAS (Neural Architecture Search)
# ---------------------------------------------------------------------------

class NAS:
    """Neural Architecture Search with random search, DARTS-style, and evolutionary methods."""

    def __init__(self, search_space: Optional[Dict[str, List]] = None):
        self.search_space = search_space or {}
        self.history: List[Dict[str, Any]] = []

    def random_search(
        self,
        eval_fn: Callable,
        num_samples: int = 50,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        rng = random.Random(seed)
        best_score = -math.inf
        best_arch = None
        results = []

        for _ in range(num_samples):
            arch = {k: rng.choice(v) for k, v in self.search_space.items()}
            score = eval_fn(arch)
            results.append({"arch": arch, "score": score})
            self.history.append({"method": "random", "arch": arch, "score": score})
            if score > best_score:
                best_score = score
                best_arch = arch

        return {"best_arch": best_arch, "best_score": best_score, "trials": results}

    def darts_search(
        self,
        eval_fn: Callable,
        num_ops: int = 8,
        num_nodes: int = 4,
        iterations: int = 30,
        temperature: float = 1.0,
        temp_decay: float = 0.95,
    ) -> Dict[str, Any]:
        ops = list(range(num_ops))
        arch_params = {}
        for node in range(num_nodes):
            for prev in range(node + 1):
                weights = np.ones(num_ops) / num_ops
                arch_params[(prev, node)] = weights

        for iteration in range(iterations):
            total_score = 0.0
            for (prev, node), weights in arch_params.items():
                sampled_op = rng_choice(ops, weights)
                candidate = {
                    "node": node,
                    "prev": prev,
                    "op": sampled_op,
                    "iteration": iteration,
                }
                score = eval_fn(candidate)
                total_score += score
                weights[sampled_op] += 0.1
                weights /= weights.sum()

            temperature *= temp_decay
            self.history.append({
                "method": "darts",
                "iteration": iteration,
                "score": total_score,
                "temperature": temperature,
            })

        best_arch = {}
        for (prev, node), weights in arch_params.items():
            best_arch[(prev, node)] = int(np.argmax(weights))

        return {
            "best_arch": best_arch,
            "final_weights": {str(k): v.tolist() for k, v in arch_params.items()},
            "iterations": iterations,
        }

    def evolutionary_search(
        self,
        eval_fn: Callable,
        population_size: int = 20,
        generations: int = 15,
        mutation_rate: float = 0.3,
        elite_fraction: float = 0.2,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        rng = random.Random(seed)
        elite_count = max(1, int(population_size * elite_fraction))

        population = [
            {k: rng.choice(v) for k, v in self.search_space.items()}
            for _ in range(population_size)
        ]

        best_overall_score = -math.inf
        best_overall_arch = None

        for gen in range(generations):
            scored = []
            for arch in population:
                score = eval_fn(arch)
                scored.append((arch, score))
                self.history.append({
                    "method": "evolutionary",
                    "generation": gen,
                    "score": score,
                })

            scored.sort(key=lambda x: x[1], reverse=True)

            if scored[0][1] > best_overall_score:
                best_overall_score = scored[0][1]
                best_overall_arch = scored[0][0].copy()

            elites = [a.copy() for a, _ in scored[:elite_count]]

            new_pop = list(elites)
            while len(new_pop) < population_size:
                parent = rng.choice(elites).copy()
                child = self._mutate(parent, rng, mutation_rate)
                new_pop.append(child)

            population = new_pop

        return {
            "best_arch": best_overall_arch,
            "best_score": best_overall_score,
            "generations": generations,
        }

    def _mutate(self, arch: Dict, rng: random.Random, rate: float) -> Dict:
        mutated = arch.copy()
        for k, v in self.search_space.items():
            if rng.random() < rate:
                mutated[k] = rng.choice(v)
        return mutated


def rng_choice(items: list, weights: np.ndarray) -> int:
    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return items[i]
    return items[-1]


# ---------------------------------------------------------------------------
# MetaLearning
# ---------------------------------------------------------------------------

class MetaLearning:
    """Model-Agnostic Meta-Learning (MAML): inner-loop fine-tune, outer-loop minimize."""

    def __init__(self, inner_lr: float = 0.01, outer_lr: float = 0.001,
                 inner_steps: int = 5, num_tasks: int = 10):
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.num_tasks = num_tasks

    def maml(
        self,
        theta: Dict[str, Any],
        get_task_batch: Callable,
        loss_fn: Callable,
        num_outer_epochs: int = 100,
    ) -> Dict[str, Any]:
        meta_grad_accum = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0
                           for k, v in theta.items()}

        history = []
        for outer_epoch in range(num_outer_epochs):
            task_losses = []

            for task_idx in range(self.num_tasks):
                support, query = get_task_batch(task_idx)
                theta_prime = {k: v.copy() if isinstance(v, np.ndarray) else v
                               for k, v in theta.items()}

                for inner_step in range(self.inner_steps):
                    grad = loss_fn(theta_prime, support)
                    for k in theta_prime:
                        if isinstance(theta_prime[k], np.ndarray) and k in grad:
                            theta_prime[k] = theta_prime[k] - self.inner_lr * grad[k]

                query_grad = loss_fn(theta_prime, query)
                for k in meta_grad_accum:
                    if k in query_grad:
                        meta_grad_accum[k] += query_grad[k]

                task_loss = loss_fn(theta_prime, query)
                if isinstance(task_loss, dict):
                    task_losses.append(sum(v for v in task_loss.values() if isinstance(v, (int, float))))
                else:
                    task_losses.append(float(task_loss))

            for k in theta:
                if isinstance(theta[k], np.ndarray) and k in meta_grad_accum:
                    theta[k] = theta[k] - self.outer_lr * meta_grad_accum[k] / self.num_tasks

            avg_loss = sum(task_losses) / max(len(task_losses), 1)
            history.append({"epoch": outer_epoch, "avg_loss": avg_loss})

        return {"theta": theta, "history": history, "num_outer_epochs": num_outer_epochs}


# ---------------------------------------------------------------------------
# ContinualLearning
# ---------------------------------------------------------------------------

class ContinualLearning:
    """Elastic Weight Consolidation (EWC) regularization for continual learning."""

    def __init__(self, lambda_ewc: float = 1000.0, num_tasks: int = 5):
        self.lambda_ewc = lambda_ewc
        self.num_tasks = num_tasks
        self.fisher_information: Dict[str, Any] = {}
        self.optimal_params: Dict[str, Any] = {}

    def compute_fisher(self, theta: Dict[str, Any], loss_fn: Callable,
                       data: Any, num_samples: int = 200) -> Dict[str, np.ndarray]:
        fisher = {k: np.zeros_like(v) if isinstance(v, np.ndarray) else 0.0
                  for k, v in theta.items()}

        for _ in range(num_samples):
            grad = loss_fn(theta, data)
            for k in fisher:
                if k in grad and isinstance(grad[k], np.ndarray):
                    fisher[k] += grad[k] ** 2

        for k in fisher:
            if isinstance(fisher[k], np.ndarray):
                fisher[k] /= num_samples
        return fisher

    def update(self, theta: Dict[str, Any], loss_fn: Callable,
               data: Any, task_idx: int) -> Dict[str, Any]:
        current_loss = loss_fn(theta, data)

        ewc_penalty = 0.0
        if self.optimal_params:
            for k in theta:
                if k in self.optimal_params and k in self.fisher_information:
                    if isinstance(theta[k], np.ndarray):
                        diff = theta[k] - self.optimal_params[k]
                        ewc_penalty += float(np.sum(self.fisher_information[k] * diff ** 2))

        total_loss = current_loss + (self.lambda_ewc * ewc_penalty / 2.0)

        grad = loss_fn(theta, data)
        for k in theta:
            if isinstance(theta[k], np.ndarray) and k in grad:
                ewc_grad = self.lambda_ewc * self.fisher_information.get(k, np.zeros_like(theta[k])) * (theta[k] - self.optimal_params.get(k, np.zeros_like(theta[k])))
                grad[k] = grad[k] + ewc_grad
                theta[k] = theta[k] - 0.001 * grad[k]

        self.fisher_information = self.compute_fisher(theta, loss_fn, data)
        self.optimal_params = {k: v.copy() if isinstance(v, np.ndarray) else v
                               for k, v in theta.items()}

        return {
            "theta": theta,
            "loss": float(total_loss) if isinstance(total_loss, (int, float)) else total_loss,
            "task_idx": task_idx,
            "ewc_penalty": ewc_penalty,
        }

    def train_sequence(self, theta: Dict[str, Any], loss_fn: Callable,
                       get_task_data: Callable) -> List[Dict[str, Any]]:
        results = []
        for task_idx in range(self.num_tasks):
            data = get_task_data(task_idx)
            result = self.update(theta, loss_fn, data, task_idx)
            theta = result["theta"]
            results.append(result)
        return results
