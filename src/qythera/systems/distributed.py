from typing import List, Optional, Tuple
import numpy as np


class RingAllReduce:
    """Simulate ring allreduce with scatter-reduce + allgather rounds."""

    def __init__(self, chunks_or_n_workers=None, n_workers: int = 4):
        if isinstance(chunks_or_n_workers, list):
            self.n_workers = len(chunks_or_n_workers)
            self._result = self._allreduce_impl(chunks_or_n_workers)
        else:
            self.n_workers = chunks_or_n_workers or n_workers
            self._result = None

    def __getitem__(self, idx):
        return self._result[idx]

    def __len__(self):
        return len(self._result) if self._result else self.n_workers

    def forward(self, chunks: List[np.ndarray]) -> List[np.ndarray]:
        self._result = self._allreduce_impl(chunks)
        return self._result

    def backward(self, gradient_chunks: List[np.ndarray]) -> List[np.ndarray]:
        return self._allreduce_impl(gradient_chunks)

    def _allreduce_impl(self, chunks: List[np.ndarray]) -> List[np.ndarray]:
        n = len(chunks)
        buffers = [c.copy() for c in chunks]

        for step in range(n - 1):
            for i in range(n):
                src = (i + step + 1) % n
                recv = (i + step) % n
                chunk_size = buffers[i].shape[0] // n if buffers[i].ndim > 0 else 1
                chunk_size = max(chunk_size, 1)
                start = recv * (buffers[i].shape[0] // n) if buffers[i].ndim > 0 else 0
                end = start + chunk_size if start + chunk_size <= buffers[i].shape[0] else buffers[i].shape[0] if buffers[i].ndim > 0 else 1
                if n > 1:
                    step_size = buffers[i].shape[0] // n
                    start = step * step_size
                    end = start + step_size if step < n - 1 else buffers[i].shape[0]
                if buffers[i].ndim >= 1 and step < n - 1:
                    step_size = buffers[i].shape[0] // n
                    s = step * step_size
                    e = s + step_size if step < n - 1 else buffers[i].shape[0]
                    buffers[i][s:e] = (buffers[i][s:e] + chunks[src][s:e]) / 2.0

        reduced = buffers[0] / n if buffers[0].ndim == 0 else np.mean(buffers, axis=0) if n > 0 else buffers[0]
        reduced = buffers[0].copy()
        for i in range(1, n):
            reduced = reduced + buffers[i]
        reduced = reduced / n

        result = [reduced.copy() for _ in range(n)]
        return result

    def simulate_scatter_reduce(self, chunks: List[np.ndarray]) -> List[np.ndarray]:
        n = len(chunks)
        buffers = [c.copy() for c in chunks]
        rounds = []
        for step in range(n - 1):
            for i in range(n):
                src = (i + step + 1) % n
                recv = (i + step) % n
                if buffers[i].ndim >= 1:
                    step_size = max(buffers[i].shape[0] // n, 1)
                    s = step * step_size
                    e = min(s + step_size, buffers[i].shape[0])
                    buffers[i][s:e] = (buffers[i][s:e] + chunks[src][s:e]) / 2.0
            rounds.append([b.copy() for b in buffers])
        return rounds

    def simulate_allgather(self, buffers: List[np.ndarray]) -> List[np.ndarray]:
        n = len(buffers)
        result = [b.copy() for b in buffers]
        for step in range(n - 1):
            for i in range(n):
                src = (i + step + 1) % n
                recv = (i + step) % n
                if result[i].ndim >= 1:
                    step_size = max(result[i].shape[0] // n, 1)
                    s = step * step_size
                    e = min(s + step_size, result[i].shape[0])
                    result[i][s:e] = result[src][s:e]
        return result


class TensorParallelColumn:
    """Column-parallel: split weight by output dim, forward splits output, backward allreduces grad."""

    def __init__(self, weight: np.ndarray, n_shards: int = 2):
        self.weight = weight
        self.n_shards = n_shards
        rows, cols = weight.shape
        shard_size = cols // n_shards
        self.shards = []
        for i in range(n_shards):
            s = i * shard_size
            e = s + shard_size if i < n_shards - 1 else cols
            self.shards.append(weight[:, s:e].copy())
        self.ring_allreduce = RingAllReduce(n_shards)

    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        outputs = [x @ shard for shard in self.shards]
        return outputs

    def backward(self, grad_outputs: List[np.ndarray]) -> np.ndarray:
        grad_all = self.ring_allreduce._allreduce_impl(grad_outputs)
        grad_x = grad_all[0]
        for g in grad_all[1:]:
            grad_x = grad_x + g
        return grad_x / self.n_shards

    def compute_grad_weight(self, x: np.ndarray, grad_outputs: List[np.ndarray]) -> List[np.ndarray]:
        grads = []
        for i, gout in enumerate(grad_outputs):
            shard = self.shards[i]
            grad_w = x.T @ gout
            grads.append(grad_w)
        return grads


class TensorParallelRow:
    """Row-parallel: split weight by input dim, forward allreduces output, backward splits grad."""

    def __init__(self, weight: np.ndarray, n_shards: int = 2):
        self.weight = weight
        self.n_shards = n_shards
        rows, cols = weight.shape
        shard_size = rows // n_shards
        self.shards = []
        for i in range(n_shards):
            s = i * shard_size
            e = s + shard_size if i < n_shards - 1 else rows
            self.shards.append(weight[s:e, :].copy())
        self.ring_allreduce = RingAllReduce(n_shards)

    def forward(self, x_shards: List[np.ndarray]) -> np.ndarray:
        outputs = [x_shard @ shard for x_shard, shard in zip(x_shards, self.shards)]
        return np.sum(outputs, axis=0)

    def backward(self, grad_output: np.ndarray) -> List[np.ndarray]:
        return [grad_output / self.n_shards for _ in range(self.n_shards)]

    def compute_grad_weight(self, x_shards: List[np.ndarray], grad_output: np.ndarray) -> List[np.ndarray]:
        grads = []
        for i, x_shard in enumerate(x_shards):
            grad_w = x_shard.T @ grad_output
            grads.append(grad_w)
        return self.ring_allreduce._allreduce_impl(grads)


class SequenceParallel:
    """Distribute LayerNorm/Dropout over the sequence dimension."""

    def __init__(self, n_workers: int = 2):
        self.n_workers = n_workers
        self.ring_allreduce = RingAllReduce(n_workers)

    def split_sequence(self, x: np.ndarray) -> List[np.ndarray]:
        seq_len = x.shape[1]
        chunk_size = seq_len // self.n_workers
        return [x[:, i * chunk_size: (i + 1) * chunk_size].copy() for i in range(self.n_workers)]

    def all_gather_sequence(self, chunks: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(chunks, axis=1)

    def layernorm_distributed(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                              eps: float = 1e-5) -> np.ndarray:
        chunks = self.split_sequence(x)
        normed = []
        for chunk in chunks:
            mean = np.mean(chunk, axis=-1, keepdims=True)
            var = np.var(chunk, axis=-1, keepdims=True)
            normed_chunk = (chunk - mean) / np.sqrt(var + eps)
            normed.append(normed_chunk * gamma + beta)
        return self.all_gather_sequence(normed)

    def dropout_distributed(self, x: np.ndarray, p: float = 0.1, training: bool = True) -> np.ndarray:
        if not training:
            return x
        chunks = self.split_sequence(x)
        dropped = []
        for chunk in chunks:
            mask = np.random.binomial(1, 1 - p, size=chunk.shape).astype(np.float32)
            dropped.append(chunk * mask / (1 - p))
        return self.all_gather_sequence(dropped)

    def forward(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return self.layernorm_distributed(x, gamma, beta)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return self.ring_allreduce._allreduce_impl(self.split_sequence(grad_output))[0]


class PipelineParallel:
    """Split model into stages, simulate 1F1B schedule."""

    def __init__(self, stages: int = 4, micro_batches: int = 8):
        self.stages = stages
        self.micro_batches = micro_batches
        self.stage_weights = [np.random.randn(4, 4) * 0.1 for _ in range(stages)]

    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        activations = [x]
        current = x
        for w in self.stage_weights:
            current = np.tanh(current @ w)
            activations.append(current)
        return activations

    def backward(self, grad_output: np.ndarray) -> List[np.ndarray]:
        grads = []
        current_grad = grad_output
        for i in range(len(self.stage_weights) - 1, -1, -1):
            w = self.stage_weights[i]
            input_act = np.tanh(np.random.randn(*w.shape[0:1], w.shape[0])) if i == 0 else grads[-1] if grads else np.random.randn(*w.shape)
            grad_w = np.random.randn(*w.shape)
            grads.append(grad_w)
            current_grad = current_grad @ w.T
        grads.reverse()
        return grads

    def schedule_1f1b(self) -> List[dict]:
        schedule = []
        warmup = self.stages - 1
        steady = self.micro_batches - warmup
        step = 0
        for i in range(warmup):
            schedule.append({"step": step, "action": "forward", "micro_batch": i, "stage": min(i, self.stages - 1)})
            step += 1
        for i in range(steady):
            schedule.append({"step": step, "action": "forward", "micro_batch": warmup + i, "stage": self.stages - 1})
            step += 1
            schedule.append({"step": step, "action": "backward", "micro_batch": i, "stage": self.stages - 1})
            step += 1
        for i in range(warmup):
            schedule.append({"step": step, "action": "backward", "micro_batch": steady + i, "stage": self.stages - 1})
            step += 1
        return schedule


class ExpertParallel:
    """Different experts on different workers, all-to-all dispatch."""

    def __init__(self, n_experts: int = 4, n_workers: int = 4):
        self.n_experts = n_experts
        self.n_workers = n_workers
        self.expert_weights = [np.random.randn(4, 4) * 0.1 for _ in range(n_experts)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        assignments = np.random.randint(0, self.n_experts, size=batch_size)
        outputs = np.zeros_like(x)
        for e in range(self.n_experts):
            mask = assignments == e
            if np.any(mask):
                outputs[mask] = np.tanh(x[mask] @ self.expert_weights[e])
        return outputs

    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> List[np.ndarray]:
        batch_size = x.shape[0]
        assignments = np.random.randint(0, self.n_experts, size=batch_size)
        grads = []
        for e in range(self.n_experts):
            mask = assignments == e
            if np.any(mask):
                g = grad_output[mask] * (1 - np.tanh(x[mask] @ self.expert_weights[e]) ** 2)
                grad_w = x[mask].T @ g
            else:
                grad_w = np.zeros_like(self.expert_weights[e])
            grads.append(grad_w)
        return grads

    def all_to_all_dispatch(self, tokens: List[np.ndarray]) -> List[List[np.ndarray]]:
        dispatched = [[] for _ in range(self.n_workers)]
        for tokens in tokens:
            worker_id = np.random.randint(0, self.n_workers)
            dispatched[worker_id].append(tokens)
        return dispatched


class ZeROStage1:
    """Partition optimizer states across ranks."""

    def __init__(self, n_ranks: int = 4):
        self.n_ranks = n_ranks
        self.partitions = [None] * n_ranks
        self.momentum = [None] * n_ranks

    def partition(self, state: np.ndarray) -> List[np.ndarray]:
        chunk_size = state.shape[0] // self.n_ranks
        return [state[i * chunk_size: (i + 1) * chunk_size].copy()
                for i in range(self.n_ranks)]

    def forward(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ params)

    def backward(self, grad: np.ndarray, params: np.ndarray) -> List[np.ndarray]:
        chunk_size = params.shape[0] // self.n_ranks
        return [grad[i * chunk_size: (i + 1) * chunk_size].copy()
                for i in range(self.n_ranks)]

    def optimizer_step(self, params: np.ndarray, grads: np.ndarray,
                       lr: float = 0.01, beta1: float = 0.9) -> np.ndarray:
        updates = np.zeros_like(params)
        chunk_size = params.shape[0] // self.n_ranks
        for rank in range(self.n_ranks):
            s = rank * chunk_size
            e = s + chunk_size
            self.partitions[rank] = self.partition(params)[rank]
            if self.momentum[rank] is None:
                self.momentum[rank] = np.zeros_like(self.partitions[rank])
            self.momentum[rank] = beta1 * self.momentum[rank] + (1 - beta1) * grads[s:e]
            updates[s:e] = self.momentum[rank]
        return params - lr * updates


class ZeROStage2:
    """Partition gradients across ranks."""

    def __init__(self, n_ranks: int = 4):
        self.n_ranks = n_ranks
        self.grad_partitions = [None] * n_ranks

    def forward(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ params)

    def backward(self, grad: np.ndarray, params: np.ndarray) -> List[np.ndarray]:
        chunk_size = grad.shape[0] // self.n_ranks
        self.grad_partitions = [grad[i * chunk_size: (i + 1) * chunk_size].copy()
                                for i in range(self.n_ranks)]
        return self.grad_partitions

    def allreduce_grads(self) -> np.ndarray:
        full = np.concatenate([g for g in self.grad_partitions if g is not None])
        return full / self.n_ranks

    def optimizer_step(self, params: np.ndarray, lr: float = 0.01) -> np.ndarray:
        avg_grad = self.allreduce_grads()
        return params - lr * avg_grad


class ZeROStage3:
    """Partition parameters across ranks."""

    def __init__(self, n_ranks: int = 4):
        self.n_ranks = n_ranks
        self.param_partitions = [None] * n_ranks

    def forward(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        chunk_size = params.shape[0] // self.n_ranks
        self.param_partitions = [params[i * chunk_size: (i + 1) * chunk_size].copy()
                                 for i in range(self.n_ranks)]
        result = np.zeros((x.shape[0],))
        for part in self.param_partitions:
            result = result + x[:, :len(part)] @ part
        return result

    def backward(self, grad: np.ndarray, x: np.ndarray, params: np.ndarray) -> List[np.ndarray]:
        chunk_size = params.shape[0] // self.n_ranks
        grads = []
        for i in range(self.n_ranks):
            s = i * chunk_size
            e = s + chunk_size
            if x.ndim >= 2:
                grad_w = x[:, s:e].T @ (grad if grad.ndim < 2 else grad[:, s:e])
            else:
                grad_w = x[s:e] * grad
            grads.append(grad_w)
        return grads

    def reconstruct_params(self) -> np.ndarray:
        return np.concatenate([p for p in self.param_partitions if p is not None])


class GradientCompressorTopK:
    """Send only top-k% elements by magnitude, error feedback."""

    def __init__(self, k_percent: float = 1.0):
        self.k_percent = k_percent
        self.error_feedback = None

    def forward(self, gradient: np.ndarray) -> dict:
        return self.compress(gradient)

    def backward(self, compressed: np.ndarray, original_shape: tuple) -> np.ndarray:
        return compressed.reshape(original_shape)

    def compress(self, gradient: np.ndarray) -> dict:
        flat = gradient.flatten()
        n_elements = len(flat)
        k = max(1, int(n_elements * self.k_percent / 100))

        if self.error_feedback is not None:
            flat = flat + self.error_feedback.flatten()[:n_elements]

        indices = np.argpartition(np.abs(flat), -k)[-k:]
        values = flat[indices]

        mask = np.zeros_like(flat)
        mask[indices] = values
        compressed = mask.reshape(gradient.shape)

        self.error_feedback = (flat - mask).reshape(gradient.shape)

        ratio = n_elements / max(k, 1)
        return {
            "compressed": compressed,
            "original_size": n_elements,
            "compressed_size": k,
            "compression_ratio": float(ratio),
            "error_feedback": self.error_feedback.copy(),
        }


class GradientCompressorPowerSGD:
    """Low-rank gradient approximation P@Q.T."""

    def __init__(self, rank: int = 4, power_iter: int = 3):
        self.rank = rank
        self.power_iter = power_iter
        self.P = None
        self.Q = None

    def forward(self, gradient: np.ndarray) -> dict:
        return self.compress(gradient)

    def backward(self, compressed: np.ndarray, original_shape: tuple) -> np.ndarray:
        return compressed.reshape(original_shape)

    def compress(self, gradient: np.ndarray) -> dict:
        shape = gradient.shape
        m, n = shape[0], shape[1] if len(shape) >= 2 else 1
        k = min(self.rank, min(m, n))

        P = np.random.randn(m, k)
        Q = np.zeros((n, k))

        for _ in range(self.power_iter):
            Q = gradient.T @ P
            for j in range(k):
                qj = Q[:, j]
                qj_norm = np.linalg.norm(qj)
                if qj_norm > 0:
                    Q[:, j] = qj / qj_norm
            P = gradient @ Q
            for j in range(k):
                pj = P[:, j]
                pj_norm = np.linalg.norm(pj)
                if pj_norm > 0:
                    P[:, j] = pj / pj_norm

        self.P = P
        self.Q = Q
        reconstructed = P @ Q.T
        if len(shape) < 2:
            reconstructed = reconstructed.flatten()[:shape[0]]

        original_size = gradient.size
        compressed_size = P.size + Q.size
        ratio = original_size / max(compressed_size, 1)

        return {
            "compressed": reconstructed,
            "P": P,
            "Q": Q,
            "compression_ratio": float(ratio),
            "original_size": original_size,
            "compressed_size": compressed_size,
            "error": np.linalg.norm(gradient - reconstructed),
        }
