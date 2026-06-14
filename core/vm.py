"""Tensor virtual machine for executing low-level tensor operations."""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import heapq


@dataclass
class Instruction:
    opcode: str
    operands: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    destination: Optional[str] = None

    def __repr__(self):
        dest = f"{self.destination} = " if self.destination else ""
        return f"{dest}{self.opcode}({', '.join(self.operands)})"


@dataclass
class Register:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype = np.float32
    value: Optional[np.ndarray] = None

    def allocate(self):
        if self.value is None:
            self.value = np.zeros(self.shape, dtype=self.dtype)
        return self.value

    def free(self):
        self.value = None


class MemoryManager:
    def __init__(self, max_buffers: int = 1024):
        self.max_buffers = max_buffers
        self.pool: Dict[int, np.ndarray] = {}
        self.free_list: List[Tuple[int, Tuple[int, ...], np.dtype]] = []
        self.access_order: OrderedDict = OrderedDict()
        self.allocation_count = 0
        self.eviction_count = 0

    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> int:
        for i, (buf_id, buf_shape, buf_dtype) in enumerate(self.free_list):
            if buf_shape == shape and buf_dtype == dtype:
                self.free_list.pop(i)
                self.access_order[buf_id] = time.monotonic()
                return buf_id

        buf_id = self.allocation_count
        self.allocation_count += 1
        self.pool[buf_id] = np.zeros(shape, dtype=dtype)
        self.access_order[buf_id] = time.monotonic()
        self._evict_if_needed()
        return buf_id

    def free(self, buf_id: int):
        if buf_id in self.pool:
            buf = self.pool[buf_id]
            self.free_list.append((buf_id, buf.shape, buf.dtype))
            self.access_order.pop(buf_id, None)

    def get(self, buf_id: int) -> Optional[np.ndarray]:
        if buf_id in self.pool:
            self.access_order[buf_id] = time.monotonic()
            return self.pool[buf_id]
        return None

    def set(self, buf_id: int, value: np.ndarray):
        self.pool[buf_id] = value
        self.access_order[buf_id] = time.monotonic()

    def _evict_if_needed(self):
        while len(self.pool) > self.max_buffers:
            oldest_id = min(self.access_order, key=self.access_order.get)
            self.free(oldest_id)
            self.eviction_count += 1


class VMState:
    def __init__(self, memory_size: int = 1024):
        self.registers: Dict[str, np.ndarray] = {}
        self.memory = MemoryManager(max_buffers=memory_size)
        self.buffer_map: Dict[str, int] = {}
        self.stack: List[np.ndarray] = []
        self.call_stack: List[List[Instruction]] = []

    def set_register(self, name: str, value: np.ndarray):
        self.registers[name] = value

    def get_register(self, name: str) -> np.ndarray:
        if name not in self.registers:
            raise KeyError(f"Register '{name}' not found")
        return self.registers[name]

    def has_register(self, name: str) -> bool:
        return name in self.registers

    def alloc_buffer(self, name: str, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> int:
        buf_id = self.memory.allocate(shape, dtype)
        self.buffer_map[name] = buf_id
        return buf_id

    def free_buffer(self, name: str):
        if name in self.buffer_map:
            self.memory.free(self.buffer_map.pop(name))


class Profiler:
    def __init__(self):
        self.instruction_times: Dict[str, List[float]] = defaultdict(list)
        self.total_time: float = 0.0
        self.instruction_counts: Dict[str, int] = defaultdict(int)

    def record(self, opcode: str, duration: float):
        self.instruction_times[opcode].append(duration)
        self.instruction_counts[opcode] += 1
        self.total_time += duration

    def summary(self) -> Dict[str, Any]:
        stats = {}
        for opcode, times in self.instruction_times.items():
            stats[opcode] = {
                "count": self.instruction_counts[opcode],
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
            }
        stats["_total"] = {
            "total_time": self.total_time,
            "total_instructions": sum(self.instruction_counts.values()),
        }
        return stats

    def reset(self):
        self.instruction_times.clear()
        self.total_time = 0.0
        self.instruction_counts.clear()


class TensorVM:
    def __init__(self, memory_size: int = 1024):
        self.state = VMState(memory_size)
        self.profiler = Profiler()
        self.dispatch_table = self._build_dispatch()

    def _build_dispatch(self) -> Dict[str, callable]:
        return {
            "ALLOC": self._alloc,
            "FREE": self._free,
            "LOAD": self._load,
            "STORE": self._store,
            "MATMUL": self._matmul,
            "ATTENTION": self._attention,
            "ROPE": self._rope,
            "EMBED": self._embed,
            "NORM": self._norm,
            "ADD": self._add,
            "MUL": self._mul,
            "SOFTMAX": self._softmax,
            "SWIGLU": self._swiglu,
            "LOSS": self._loss,
            "SAMPLE": self._sample,
        }

    def execute(self, instructions: List[Instruction]) -> Dict[str, np.ndarray]:
        for instr in instructions:
            start = time.monotonic()
            handler = self.dispatch_table.get(instr.opcode)
            if handler is None:
                raise ValueError(f"Unknown opcode: {instr.opcode}")
            handler(instr)
            duration = time.monotonic() - start
            self.profiler.record(instr.opcode, duration)
        return self.state.registers

    def _resolve_operand(self, name: str) -> np.ndarray:
        if self.state.has_register(name):
            return self.state.get_register(name)
        if name in self.state.buffer_map:
            return self.state.memory.get(self.state.buffer_map[name])
        try:
            return np.array(float(name), dtype=np.float32)
        except ValueError:
            raise ValueError(f"Cannot resolve operand: {name}")

    def _set_dest(self, instr: Instruction, value: np.ndarray):
        dest = instr.destination or instr.operands[-1]
        self.state.set_register(dest, value)

    def _alloc(self, instr: Instruction):
        shape = tuple(instr.metadata.get("shape", [1]))
        dtype = np.dtype(instr.metadata.get("dtype", "float32"))
        name = instr.operands[0]
        self.state.alloc_buffer(name, shape, dtype)

    def _free(self, instr: Instruction):
        name = instr.operands[0]
        self.state.free_buffer(name)

    def _load(self, instr: Instruction):
        name = instr.operands[0]
        source = instr.metadata.get("source")
        if source is not None:
            arr = np.array(source, dtype=instr.metadata.get("dtype", "float32"))
        else:
            shape = instr.metadata.get("shape", [1])
            arr = np.random.randn(*shape).astype(instr.metadata.get("dtype", "float32"))
        self._set_dest(instr, arr)

    def _store(self, instr: Instruction):
        src = self._resolve_operand(instr.operands[0])
        name = instr.operands[1] if len(instr.operands) > 1 else instr.destination
        if name in self.state.buffer_map:
            self.state.memory.set(self.state.buffer_map[name], src)
        else:
            self.state.set_register(name, src)

    def _matmul(self, instr: Instruction):
        a = self._resolve_operand(instr.operands[0])
        b = self._resolve_operand(instr.operands[1])
        self._set_dest(instr, np.matmul(a, b))

    def _attention(self, instr: Instruction):
        q = self._resolve_operand(instr.operands[0])
        k = self._resolve_operand(instr.operands[1])
        v = self._resolve_operand(instr.operands[2])
        scale = instr.metadata.get("scale", 1.0 / np.sqrt(q.shape[-1]))
        scores = np.matmul(q, k.T) * scale
        weights = self._softmax_array(scores)
        self._set_dest(instr, np.matmul(weights, v))

    def _softmax_array(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _rope(self, instr: Instruction):
        x = self._resolve_operand(instr.operands[0])
        pos = instr.metadata.get("position", 0)
        dim = x.shape[-1]
        freqs = 1.0 / (10000.0 ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        t = pos * freqs
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        out = x.copy()
        out[..., 0::2] = x[..., 0::2] * cos_t - x[..., 1::2] * sin_t
        out[..., 1::2] = x[..., 0::2] * sin_t + x[..., 1::2] * cos_t
        self._set_dest(instr, out)

    def _embed(self, instr: Instruction):
        indices = self._resolve_operand(instr.operands[0])
        vocab_size = instr.metadata.get("vocab_size", 1000)
        embed_dim = instr.metadata.get("embed_dim", 64)
        weights = self._resolve_operand(instr.operands[1]) if len(instr.operands) > 1 else np.random.randn(vocab_size, embed_dim).astype(np.float32)
        indices = indices.astype(int)
        self._set_dest(instr, weights[indices])

    def _norm(self, instr: Instruction):
        x = self._resolve_operand(instr.operands[0])
        eps = instr.metadata.get("eps", 1e-5)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self._set_dest(instr, (x - mean) / np.sqrt(var + eps))

    def _add(self, instr: Instruction):
        a = self._resolve_operand(instr.operands[0])
        b = self._resolve_operand(instr.operands[1])
        self._set_dest(instr, a + b)

    def _mul(self, instr: Instruction):
        a = self._resolve_operand(instr.operands[0])
        b = self._resolve_operand(instr.operands[1])
        self._set_dest(instr, a * b)

    def _softmax(self, instr: Instruction):
        x = self._resolve_operand(instr.operands[0])
        self._set_dest(instr, self._softmax_array(x))

    def _swiglu(self, instr: Instruction):
        gate = self._resolve_operand(instr.operands[0])
        up = self._resolve_operand(instr.operands[1])
        silu = gate / (1.0 + np.exp(-gate))
        self._set_dest(instr, silu * up)

    def _loss(self, instr: Instruction):
        pred = self._resolve_operand(instr.operands[0])
        target = self._resolve_operand(instr.operands[1])
        loss_type = instr.metadata.get("type", "cross_entropy")
        if loss_type == "cross_entropy":
            probs = self._softmax_array(pred)
            log_probs = np.log(probs + 1e-8)
            target_onehot = np.zeros_like(probs)
            target_onehot[np.arange(len(target)), target.astype(int)] = 1.0
            loss = -np.sum(target_onehot * log_probs, axis=-1)
            self._set_dest(instr, np.mean(loss))
        elif loss_type == "mse":
            self._set_dest(instr, np.mean((pred - target) ** 2))
        else:
            self._set_dest(instr, np.mean(np.abs(pred - target)))

    def _sample(self, instr: Instruction):
        logits = self._resolve_operand(instr.operands[0])
        temperature = instr.metadata.get("temperature", 1.0)
        top_k = instr.metadata.get("top_k", 0)
        probs = self._softmax_array(logits / temperature)
        if top_k > 0:
            top_indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_indices] = probs[top_indices]
            probs = mask / np.sum(mask)
        flat_probs = probs.flatten()
        flat_probs = flat_probs / flat_probs.sum()
        sample_idx = np.random.choice(len(flat_probs), p=flat_probs)
        self._set_dest(instr, np.array(sample_idx, dtype=np.int32))


def create_sample_program() -> List[Instruction]:
    return [
        Instruction("LOAD", ["embeddings"], {"shape": [1000, 64], "dtype": "float32"}),
        Instruction("LOAD", ["tokens"], {"source": [1, 5, 3, 7], "dtype": "int32"}),
        Instruction("EMBED", ["tokens", "embeddings"], destination="embedded"),
        Instruction("NORM", ["embedded"], destination="normed"),
        Instruction("LOAD", ["Wq"], {"shape": [64, 64], "dtype": "float32"}),
        Instruction("LOAD", ["Wk"], {"shape": [64, 64], "dtype": "float32"}),
        Instruction("LOAD", ["Wv"], {"shape": [64, 64], "dtype": "float32"}),
        Instruction("MATMUL", ["normed", "Wq"], destination="q"),
        Instruction("MATMUL", ["normed", "Wk"], destination="k"),
        Instruction("MATMUL", ["normed", "Wv"], destination="v"),
        Instruction("ATTENTION", ["q", "k", "v"], destination="attn_out"),
        Instruction("ADD", ["attn_out", "embedded"], destination="residual"),
        Instruction("SOFTMAX", ["residual"], destination="logits"),
        Instruction("SAMPLE", ["logits"], {"temperature": 0.8}, destination="output"),
    ]


if __name__ == "__main__":
    vm = TensorVM()
    program = create_sample_program()
    results = vm.execute(program)
    print("Execution complete!")
    print(f"Registers: {list(results.keys())}")
    for name, val in results.items():
        print(f"  {name}: shape={val.shape}, dtype={val.dtype}")
    print(f"\nProfiler summary:")
    for opcode, stats in vm.profiler.summary().items():
        print(f"  {opcode}: {stats}")
