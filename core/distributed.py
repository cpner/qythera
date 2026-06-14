from typing import List, Optional

import numpy as np


class DataParallel:
    """Simulate gradient allreduce across data-parallel workers."""

    def __init__(self, n_workers: int = 2):
        self.n_workers = n_workers
        self.gradients = [None] * n_workers

    def forward(self, data_chunks: List[np.ndarray], weights: np.ndarray) -> List[np.ndarray]:
        """Simulate forward pass: each worker computes loss on its chunk."""
        losses = []
        for chunk in data_chunks:
            loss = float(np.mean((chunk - weights) ** 2))
            losses.append(loss)
        return losses

    def backward(self, data_chunks: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """Compute gradients on each chunk, then allreduce (mean)."""
        grads = []
        for chunk in data_chunks:
            grad = 2.0 * (weights - chunk) / len(chunk)
            grads.append(grad)
        self.gradients = grads
        allreduced = np.mean(grads, axis=0)
        return allreduced

    def step(self, weights: np.ndarray, lr: float = 0.01) -> np.ndarray:
        if self.gradients[0] is None:
            return weights
        return weights - lr * np.mean(self.gradients, axis=0)


class RingAllReduce:
    """Simulate ring allreduce with list of arrays."""

    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers

    def forward(self, chunks: List[np.ndarray]) -> np.ndarray:
        """Reduce-scatter + allgather simulation."""
        reduced = np.sum(chunks, axis=0) / self.n_workers
        return reduced

    def backward(self, gradient_chunks: List[np.ndarray]) -> List[np.ndarray]:
        """Allreduce gradients and return identical reduced gradient per worker."""
        total = np.sum(gradient_chunks, axis=0)
        avg = total / len(gradient_chunks)
        return [avg.copy() for _ in range(self.n_workers)]

    def simulate_ring(self, chunks: List[np.ndarray]) -> dict:
        """Show step-by-step ring reduction."""
        buffers = [c.copy() for c in chunks]
        steps = []
        n = len(buffers)
        for step in range(n - 1):
            for i in range(n):
                send_idx = (i + step + 1) % n
                recv_idx = (i + step) % n
                buffers[i] = (buffers[i] + chunks[send_idx]) / 2
            steps.append([b.copy() for b in buffers])
        final = buffers[0]
        return {"steps": steps, "final": final, "n_steps": n - 1}


class TensorParallel:
    """Column and row split simulation for tensor parallelism."""

    def __init__(self, n_shards: int = 2):
        self.n_shards = n_shards

    def column_split(self, weight: np.ndarray) -> List[np.ndarray]:
        """Split weight matrix along columns."""
        cols = weight.shape[1]
        chunk_size = cols // self.n_shards
        shards = []
        for i in range(self.n_shards):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_shards - 1 else cols
            shards.append(weight[:, start:end].copy())
        return shards

    def row_split(self, weight: np.ndarray) -> List[np.ndarray]:
        """Split weight matrix along rows."""
        rows = weight.shape[0]
        chunk_size = rows // self.n_shards
        shards = []
        for i in range(self.n_shards):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_shards - 1 else rows
            shards.append(weight[start:end, :].copy())
        return shards

    def forward(self, input_array: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Column parallel forward: split input, multiply shards, gather."""
        shards = self.column_split(weight)
        row_size = input_array.shape[0] // self.n_shards
        outputs = []
        for i in range(self.n_shards):
            chunk = input_array[i * row_size: (i + 1) * row_size]
            outputs.append(chunk @ shards[i])
        return np.concatenate(outputs, axis=0)

    def backward(self, grad_output: np.ndarray, shards: List[np.ndarray]) -> np.ndarray:
        """Distributed gradient computation."""
        grad_parts = []
        for shard in shards:
            grad_parts.append(grad_output @ shard.T)
        return np.concatenate(grad_parts, axis=0)


class ZeROStage:
    """ZeRO Stage 1/2/3 partition simulation."""

    def __init__(self, stage: int = 1, n_partitions: int = 4):
        self.stage = stage
        self.n_partitions = n_partitions

    def partition_optimizer_state(self, state: np.ndarray) -> List[np.ndarray]:
        """Stage 1: partition optimizer states."""
        chunk_size = len(state) // self.n_partitions
        return [state[i * chunk_size: (i + 1) * chunk_size].copy()
                for i in range(self.n_partitions)]

    def partition_gradients(self, grad: np.ndarray) -> List[np.ndarray]:
        """Stage 1+2: partition gradients."""
        chunk_size = len(grad) // self.n_partitions
        return [grad[i * chunk_size: (i + 1) * chunk_size].copy()
                for i in range(self.n_partitions)]

    def partition_parameters(self, params: np.ndarray) -> List[np.ndarray]:
        """Stage 3: partition parameters."""
        chunk_size = len(params) // self.n_partitions
        return [params[i * chunk_size: (i + 1) * chunk_size].copy()
                for i in range(self.n_partitions)]

    def forward(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Simulate forward with partitioned parameters."""
        parts = self.partition_parameters(params)
        results = [part @ data[i] for i, part in enumerate(parts) if i < len(data)]
        return np.sum(results) if results else np.array(0.0)

    def backward(self, grad: np.ndarray) -> List[np.ndarray]:
        """Simulate backward with partitioned gradients."""
        if self.stage >= 2:
            return self.partition_gradients(grad)
        chunk_size = len(grad) // self.n_partitions
        return [grad[i * chunk_size: (i + 1) * chunk_size].copy()
                for i in range(self.n_partitions)]

    def reconstruct(self, parts: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts)


class GradientCompressor:
    """TopK and PowerSGD gradient compression simulation."""

    def __init__(self, k: int = 10, rank: int = 4):
        self.k = k
        self.rank = rank

    def topk_compress(self, gradient: np.ndarray) -> dict:
        flat = gradient.flatten()
        indices = np.argpartition(np.abs(flat), -self.k)[-self.k:]
        values = flat[indices]
        mask = np.zeros_like(flat)
        mask[indices] = values
        compressed = mask.reshape(gradient.shape)
        ratio = len(flat) / max(self.k, 1)
        return {
            "compressed": compressed,
            "original_size": flat.size,
            "compressed_size": self.k,
            "compression_ratio": float(ratio),
        }

    def power_sgd_compress(self, gradient: np.ndarray) -> dict:
        shape = gradient.shape
        if len(shape) < 2:
            flat = gradient.flatten()
            if len(flat) <= self.rank:
                return {
                    "compressed": flat,
                    "P": flat[:self.rank],
                    "Q": np.ones(self.rank),
                    "compression_ratio": 1.0,
                }
            P = flat[:self.rank].copy()
            Q = flat[:self.rank].copy()
        else:
            m, n = shape[0], shape[1] if len(shape) > 1 else 1
            k = min(self.rank, min(m, n))
            P = np.random.randn(m, k)
            for _ in range(3):
                Q = gradient.T @ P
                P = gradient @ Q
            norm_P = np.linalg.norm(P, axis=0, keepdims=True)
            norm_P[norm_P == 0] = 1
            P = P / norm_P
        original_size = gradient.size
        reconstructed = P @ Q.T if len(shape) >= 2 else P * Q
        if len(shape) < 2:
            reconstructed = reconstructed.flatten()[:original_size]
        else:
            reconstructed = reconstructed[:shape[0], :shape[1]]
        ratio = original_size / max(P.size + Q.size, 1)
        return {
            "compressed": reconstructed,
            "P": P,
            "Q": Q if len(shape) >= 2 else Q,
            "compression_ratio": float(ratio),
            "original_size": original_size,
            "compressed_size": P.size + Q.size,
        }

    def forward(self, gradient: np.ndarray, method: str = "topk") -> dict:
        if method == "topk":
            return self.topk_compress(gradient)
        elif method == "power_sgd":
            return self.power_sgd_compress(gradient)
        return {"compressed": gradient, "compression_ratio": 1.0}

    def backward(self, compressed: np.ndarray, original_shape: tuple) -> np.ndarray:
        return compressed.reshape(original_shape)
