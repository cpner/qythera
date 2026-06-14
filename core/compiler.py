"""Graph-based IR compiler with optimizations and JIT compilation."""

import numpy as np
import time
import hashlib
import textwrap
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict


@dataclass
class GraphNode:
    opcode: str
    inputs: List[str] = field(default_factory=list)
    output: str = ""
    shape: Tuple[int, ...] = ()
    dtype: np.dtype = np.float32
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: str = ""

    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"{self.opcode}_{id(self)}"

    def __repr__(self):
        return f"GraphNode({self.opcode}, inputs={self.inputs}, out={self.output})"


class GraphIR:
    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.node_map: Dict[str, GraphNode] = {}
        self.inputs: Set[str] = set()
        self.outputs: Set[str] = set()

    def add_node(self, node: GraphNode):
        self.nodes.append(node)
        self.node_map[node.output] = node
        for inp in node.inputs:
            if inp not in self.node_map:
                self.inputs.add(inp)

    def remove_node(self, node_id: str):
        self.nodes = [n for n in self.nodes if n.node_id != node_id]
        self.node_map = {k: v for k, v in self.node_map.items() if v.node_id != node_id}

    def topological_sort(self) -> List[GraphNode]:
        in_degree = defaultdict(int)
        dependents = defaultdict(list)

        for node in self.nodes:
            if node.node_id not in in_degree:
                in_degree[node.node_id] = 0
            for inp in node.inputs:
                if inp in self.node_map:
                    dep_id = self.node_map[inp].node_id
                    dependents[dep_id].append(node.node_id)
                    in_degree[node.node_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_nodes = []

        while queue:
            nid = queue.pop(0)
            node = next((n for n in self.nodes if n.node_id == nid), None)
            if node:
                sorted_nodes.append(node)
            for dep in dependents[nid]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        return sorted_nodes

    def get_users(self, name: str) -> List[GraphNode]:
        return [n for n in self.nodes if name in n.inputs]

    def replace_usage(self, old_name: str, new_name: str):
        for node in self.nodes:
            node.inputs = [new_name if x == old_name else x for x in node.inputs]


class ConstantFolding:
    def __init__(self):
        self.folded_count = 0

    def optimize(self, graph: GraphIR) -> GraphIR:
        constants = {}
        to_remove = []

        for node in graph.topological_sort():
            if node.opcode in ("CONST", "LOAD_CONST"):
                constants[node.output] = node.metadata.get("value")
            elif node.opcode in ("ADD", "MUL", "SUB", "DIV"):
                if all(inp in constants for inp in node.inputs):
                    vals = [constants[inp] for inp in node.inputs]
                    result = self._eval(node.opcode, vals)
                    const_node = GraphNode(
                        opcode="CONST",
                        output=node.output,
                        shape=node.shape,
                        dtype=node.dtype,
                        metadata={"value": result},
                    )
                    graph.node_map[node.output] = const_node
                    to_remove.append(node.node_id)
                    constants[node.output] = result
                    self.folded_count += 1

        for nid in to_remove:
            graph.remove_node(nid)

        return graph

    def _eval(self, opcode: str, vals):
        if opcode == "ADD":
            return vals[0] + vals[1]
        elif opcode == "MUL":
            return vals[0] * vals[1]
        elif opcode == "SUB":
            return vals[0] - vals[1]
        elif opcode == "DIV":
            return vals[0] / vals[1]
        return None


class DeadCodeElimination:
    def __init__(self):
        self.removed_count = 0

    def optimize(self, graph: GraphIR) -> GraphIR:
        used = set()
        queue = list(graph.outputs)

        while queue:
            name = queue.pop()
            if name in used:
                continue
            used.add(name)
            if name in graph.node_map:
                node = graph.node_map[name]
                for inp in node.inputs:
                    if inp not in used:
                        queue.append(inp)

        to_remove = []
        for node in graph.nodes:
            if node.output not in used and node.output not in graph.outputs:
                to_remove.append(node.node_id)
                self.removed_count += 1

        for nid in to_remove:
            graph.remove_node(nid)

        return graph


class OperatorFusion:
    def __init__(self):
        self.fused_count = 0

    def optimize(self, graph: GraphIR) -> GraphIR:
        fusible = {"ADD", "MUL", "SUB", "DIV", "RELU", "SIGMOID", "TANH"}
        changed = True

        while changed:
            changed = False
            sorted_nodes = graph.topological_sort()

            for node in sorted_nodes:
                if node.opcode not in fusible:
                    continue

                users = graph.get_users(node.output)
                if len(users) != 1:
                    continue

                user = users[0]
                if user.opcode not in fusible:
                    continue

                if node.output not in user.inputs:
                    continue

                fused = GraphNode(
                    opcode=f"{node.opcode}_{user.opcode}",
                    inputs=list(node.inputs) + [i for i in user.inputs if i != node.output],
                    output=user.output,
                    shape=user.shape,
                    dtype=user.dtype,
                    metadata={"fused_ops": [node.opcode, user.opcode]},
                )

                graph.replace_usage(node.output, fused.output)
                graph.remove_node(node.node_id)
                graph.remove_node(user.node_id)
                graph.add_node(fused)
                self.fused_count += 1
                changed = True
                break

        return graph


@dataclass
class LivenessInterval:
    name: str
    start: int
    end: int


class MemoryPlanner:
    def __init__(self):
        self.buffer_reuses = 0

    def optimize(self, graph: GraphIR) -> GraphIR:
        intervals = self._compute_liveness(graph)
        allocations = self._assign_buffers(intervals)
        return allocations

    def _compute_liveness(self, graph: GraphIR) -> List[LivenessInterval]:
        sorted_nodes = graph.topological_sort()
        name_first_use = {}
        name_last_use = {}

        for i, node in enumerate(sorted_nodes):
            for inp in node.inputs:
                if inp not in name_first_use:
                    name_first_use[inp] = i
                name_last_use[inp] = i
            if node.output not in name_first_use:
                name_first_use[node.output] = i
            name_last_use[node.output] = i

        intervals = []
        for name in set(list(name_first_use.keys()) + list(name_last_use.keys())):
            if name in name_first_use and name in name_last_use:
                intervals.append(LivenessInterval(
                    name=name,
                    start=name_first_use[name],
                    end=name_last_use[name],
                ))

        return intervals

    def _assign_buffers(self, intervals: List[LivenessInterval]) -> Dict[str, int]:
        sorted_intervals = sorted(intervals, key=lambda x: x.start)
        buffer_map = {}
        free_buffers = []

        for interval in sorted_intervals:
            free_buffers = [b for b in free_buffers if b[1] > interval.start]

            if free_buffers:
                buf_id, _ = free_buffers.pop(0)
                buffer_map[interval.name] = buf_id
                free_buffers.append((buf_id, interval.end))
                self.buffer_reuses += 1
            else:
                buf_id = len(buffer_map)
                buffer_map[interval.name] = buf_id
                free_buffers.append((buf_id, interval.end))

        return buffer_map


class KernelGenerator:
    def __init__(self):
        self.generated_code = ""

    def generate(self, graph: GraphIR) -> str:
        sorted_nodes = graph.topological_sort()
        lines = ["def compiled_kernel(inputs):"]
        lines.append("    import numpy as np")
        lines.append("    _out = {}")

        for node in sorted_nodes:
            if node.opcode == "CONST":
                lines.append(f'    _out["{node.output}"] = np.array({repr(node.metadata.get("value", 0))}, dtype="{node.dtype}")')
            elif node.opcode == "INPUT":
                lines.append(f'    _out["{node.output}"] = inputs.get("{node.output}", np.zeros({node.shape}, dtype="{node.dtype}"))')
            elif node.opcode == "MATMUL":
                a, b = node.inputs[0], node.inputs[1]
                lines.append(f'    _out["{node.output}"] = np.matmul(_out["{a}"], _out["{b}"])')
            elif node.opcode == "ADD":
                a, b = node.inputs[0], node.inputs[1]
                lines.append(f'    _out["{node.output}"] = _out["{a}"] + _out["{b}"]')
            elif node.opcode == "MUL":
                a, b = node.inputs[0], node.inputs[1]
                lines.append(f'    _out["{node.output}"] = _out["{a}"] * _out["{b}"]')
            elif node.opcode == "RELU":
                a = node.inputs[0]
                lines.append(f'    _out["{node.output}"] = np.maximum(0, _out["{a}"])')
            elif node.opcode == "SOFTMAX":
                a = node.inputs[0]
                lines.append(f'    _x = _out["{a}"]')
                lines.append(f'    _x_max = np.max(_x, axis=-1, keepdims=True)')
                lines.append(f'    _e_x = np.exp(_x - _x_max)')
                lines.append(f'    _out["{node.output}"] = _e_x / np.sum(_e_x, axis=-1, keepdims=True)')
            elif node.opcode == "NORM":
                a = node.inputs[0]
                eps = node.metadata.get("eps", 1e-5)
                lines.append(f'    _x = _out["{a}"]')
                lines.append(f'    _mean = np.mean(_x, axis=-1, keepdims=True)')
                lines.append(f'    _var = np.var(_x, axis=-1, keepdims=True)')
                lines.append(f'    _out["{node.output}"] = (_x - _mean) / np.sqrt(_var + {eps})')
            else:
                lines.append(f'    # Unhandled: {node.opcode} -> {node.output}')

        last_node = sorted_nodes[-1] if sorted_nodes else None
        if last_node:
            lines.append(f'    return _out["{last_node.output}"]')
        else:
            lines.append("    return {}")

        self.generated_code = "\n".join(lines)
        return self.generated_code

    def compile_and_load(self, graph: GraphIR) -> Callable:
        code = self.generate(graph)
        namespace = {}
        exec(code, namespace)
        return namespace.get("compiled_kernel", lambda x: {})


class JITCompiler:
    def __init__(self):
        self.cache: OrderedDict = OrderedDict()
        self.max_cache_size = 128
        self.cache_hits = 0
        self.cache_misses = 0
        self.kernel_gen = KernelGenerator()

    def _cache_key(self, graph: GraphIR, input_shapes: Dict[str, Tuple]) -> str:
        graph_str = "|".join(
            f"{n.opcode}:{','.join(n.inputs)}:{n.output}"
            for n in graph.topological_sort()
        )
        shape_str = "|".join(f"{k}:{v}" for k, v in sorted(input_shapes.items()))
        return hashlib.md5(f"{graph_str}:{shape_str}".encode()).hexdigest()

    def compile(self, graph: GraphIR, input_shapes: Optional[Dict[str, Tuple]] = None) -> Callable:
        if input_shapes is None:
            input_shapes = {}

        key = self._cache_key(graph, input_shapes)

        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]

        self.cache_misses += 1
        compiled = self.kernel_gen.compile_and_load(graph)

        def cached_fn(*args, **kwargs):
            return compiled(*args, **kwargs)

        self.cache[key] = cached_fn
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)

        return cached_fn

    def clear_cache(self):
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def stats(self) -> Dict[str, int]:
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


class Compiler:
    def __init__(self, optimize: bool = True):
        self.optimize = optimize
        self.passes = [
            ConstantFolding(),
            DeadCodeElimination(),
            OperatorFusion(),
        ]
        self.memory_planner = MemoryPlanner()
        self.jit = JITCompiler()

    def compile(self, graph: GraphIR) -> Callable:
        if self.optimize:
            for pass_ in self.passes:
                graph = pass_.optimize(graph)

        self.memory_planner.optimize(graph)
        return self.jit.compile(graph)

    def compile_and_run(self, graph: GraphIR, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        fn = self.compile(graph)
        return fn(inputs)
