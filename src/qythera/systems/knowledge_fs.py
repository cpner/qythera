"""Knowledge graph filesystem with URI scheme, HNSW index, and transaction log."""

import numpy as np
import time
import re
import json
import hashlib
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
import heapq


@dataclass
class KnowledgeNode:
    content: str
    embedding: np.ndarray = field(default_factory=lambda: np.random.randn(64).astype(np.float32))
    relations: Dict[str, List[str]] = field(default_factory=dict)
    timestamps: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: str = ""

    def __post_init__(self):
        if not self.node_id:
            self.node_id = hashlib.md5(self.content.encode()).hexdigest()[:12]
        if "created" not in self.timestamps:
            self.timestamps["created"] = time.time()
        if "modified" not in self.timestamps:
            self.timestamps["modified"] = time.time()

    def add_relation(self, rel_type: str, target_id: str):
        if rel_type not in self.relations:
            self.relations[rel_type] = []
        if target_id not in self.relations[rel_type]:
            self.relations[rel_type].append(target_id)
        self.timestamps["modified"] = time.time()

    def remove_relation(self, rel_type: str, target_id: str):
        if rel_type in self.relations:
            self.relations[rel_type] = [t for t in self.relations[rel_type] if t != target_id]
            self.timestamps["modified"] = time.time()


class TransactionLog:
    def __init__(self):
        self.entries: List[Dict] = []
        self.position = 0

    def append(self, operation: str, path: str, data: Optional[Dict] = None):
        entry = {
            "position": self.position,
            "operation": operation,
            "path": path,
            "data": data,
            "timestamp": time.time(),
        }
        self.entries.append(entry)
        self.position += 1
        return entry

    def get_entries(self, after: Optional[int] = None) -> List[Dict]:
        if after is None:
            return self.entries
        return [e for e in self.entries if e["position"] > after]

    def replay(self, fs: 'KnowledgeFileSystem'):
        for entry in self.entries:
            if entry["operation"] == "write":
                if entry["data"]:
                    node = KnowledgeNode(
                        content=entry["data"].get("content", ""),
                        node_id=entry["data"].get("node_id", ""),
                    )
                    fs._write_raw(entry["path"], node)
            elif entry["operation"] == "delete":
                fs._delete_raw(entry["path"])

    def compact(self, up_to: int):
        self.entries = [e for e in self.entries if e["position"] > up_to]


@dataclass
class HNSWNode:
    vector: np.ndarray
    node_id: str
    level: int = 0
    neighbors: Dict[int, List[str]] = field(default_factory=dict)


class HNSWIndex:
    def __init__(self, dim: int = 64, max_connections: int = 16, ef_construction: int = 32):
        self.dim = dim
        self.max_connections = max_connections
        self.ef_construction = ef_construction
        self.nodes: Dict[str, HNSWNode] = {}
        self.entry_point: Optional[str] = None
        self.max_level = 0

    def add(self, node_id: str, vector: np.ndarray):
        if len(vector.shape) > 1:
            vector = vector.flatten()[:self.dim]
        if len(vector) < self.dim:
            vector = np.pad(vector, (0, self.dim - len(vector)))

        level = int(np.random.geometric(0.5))
        level = min(level, 16)

        hnsw_node = HNSWNode(vector=vector.astype(np.float32), node_id=node_id, level=level)
        self.nodes[node_id] = hnsw_node

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = level
            return

        current = self.entry_point
        for lvl in range(self.max_level, level, -1):
            if current in self.nodes:
                neighbors = self._get_neighbors(current, lvl)
                if neighbors:
                    current = min(neighbors, key=lambda n: self._distance(vector, self.nodes[n].vector))

        for lvl in range(min(level, self.max_level), -1, -1):
            neighbors = self._search_layer(vector, current, self.ef_construction, lvl)
            self._connect(node_id, neighbors, lvl)
            current = neighbors[0] if neighbors else current

        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if len(query.shape) > 1:
            query = query.flatten()[:self.dim]
        if len(query) < self.dim:
            query = np.pad(query, (0, self.dim - len(query)))
        query = query.astype(np.float32)

        if self.entry_point is None:
            return []

        current = self.entry_point
        for lvl in range(self.max_level, 0, -1):
            neighbors = self._search_layer(query, current, 1, lvl)
            if neighbors:
                current = neighbors[0]

        results = self._search_layer(query, current, self.ef_construction, 0)
        results = sorted(results, key=lambda n: self._distance(query, self.nodes[n].vector))
        return [(nid, self._distance(query, self.nodes[nid].vector)) for nid in results[:k]]

    def _search_layer(self, query: np.ndarray, entry: str, ef: int, level: int) -> List[str]:
        if entry not in self.nodes:
            return []

        candidates = []
        visited = {entry}
        dist = self._distance(query, self.nodes[entry].vector)
        heapq.heappush(candidates, (dist, entry))

        results = []
        while candidates and len(results) < ef:
            d, current = heapq.heappop(candidates)
            results.append(current)

            for neighbor_id in self._get_neighbors(current, level):
                if neighbor_id in visited or neighbor_id not in self.nodes:
                    continue
                visited.add(neighbor_id)
                nd = self._distance(query, self.nodes[neighbor_id].vector)
                if nd < d or len(results) < ef:
                    heapq.heappush(candidates, (nd, neighbor_id))

        return results

    def _get_neighbors(self, node_id: str, level: int) -> List[str]:
        if node_id in self.nodes:
            return self.nodes[node_id].neighbors.get(level, [])
        return []

    def _connect(self, node_id: str, neighbors: List[str], level: int):
        if node_id not in self.nodes:
            return
        self.nodes[node_id].neighbors[level] = neighbors[:self.max_connections]
        for nid in neighbors[:self.max_connections]:
            if nid in self.nodes:
                if level not in self.nodes[nid].neighbors:
                    self.nodes[nid].neighbors[level] = []
                if node_id not in self.nodes[nid].neighbors[level]:
                    self.nodes[nid].neighbors[level].append(node_id)
                    if len(self.nodes[nid].neighbors[level]) > self.max_connections:
                        self.nodes[nid].neighbors[level] = self.nodes[nid].neighbors[level][:self.max_connections]

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(np.sqrt(np.sum(diff * diff)))


class VFS:
    def __init__(self):
        self.root = defaultdict(dict)

    def _navigate(self, path_parts: List[str]) -> dict:
        current = self.root
        for part in path_parts:
            if part not in current:
                current[part] = defaultdict(dict)
            current = current[part]
        return current

    def get(self, path: str) -> Optional[Any]:
        parts = [p for p in path.split('/') if p]
        if not parts:
            return self.root
        current = self.root
        for part in parts:
            if part not in current:
                return None
            current = current[part]
        return current

    def set(self, path: str, value: Any):
        parts = [p for p in path.split('/') if p]
        if not parts:
            return
        current = self.root
        for part in parts[:-1]:
            if part not in current:
                current[part] = defaultdict(dict)
            current = current[part]
        current[parts[-1]] = value

    def delete(self, path: str) -> bool:
        parts = [p for p in path.split('/') if p]
        if not parts:
            return False
        current = self.root
        for part in parts[:-1]:
            if part not in current:
                return False
            current = current[part]
        if parts[-1] in current:
            del current[parts[-1]]
            return True
        return False

    def list(self, path: str = "") -> List[str]:
        node = self.get(path)
        if node is None:
            return []
        if isinstance(node, dict):
            return list(node.keys())
        return []

    def exists(self, path: str) -> bool:
        return self.get(path) is not None


def parse_uri(uri: str) -> Tuple[str, str, str]:
    match = re.match(r'knowledge://([^/]+)(?:/([^/]+))?(?:/(.+))?', uri)
    if not match:
        raise ValueError(f"Invalid URI: {uri}")
    domain = match.group(1)
    subdomain = match.group(2) or ""
    concept = match.group(3) or ""
    return domain, subdomain, concept


class KnowledgeFileSystem:
    def __init__(self):
        self.vfs = VFS()
        self.index = HNSWIndex()
        self.transaction_log = TransactionLog()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.mount_points: Dict[str, str] = {}

    def mount(self, path: str, domain: str):
        self.mount_points[path] = domain
        self.transaction_log.append("mount", path, {"domain": domain})

    def read(self, path: str) -> Optional[KnowledgeNode]:
        node = self.vfs.get(path)
        if isinstance(node, KnowledgeNode):
            return node
        return None

    def write(self, path: str, content: str, metadata: Optional[Dict] = None) -> KnowledgeNode:
        node = KnowledgeNode(content=content, metadata=metadata or {})
        self._write_raw(path, node)
        self.transaction_log.append("write", path, {
            "content": content,
            "node_id": node.node_id,
        })
        return node

    def _write_raw(self, path: str, node: KnowledgeNode):
        self.vfs.set(path, node)
        self.nodes[node.node_id] = node
        self.index.add(node.node_id, node.embedding)

    def delete(self, path: str) -> bool:
        node = self.read(path)
        if node:
            self._delete_raw(path)
            self.transaction_log.append("delete", path, {"node_id": node.node_id})
            return True
        return False

    def _delete_raw(self, path: str):
        node = self.read(path)
        if node:
            self.nodes.pop(node.node_id, None)
        self.vfs.delete(path)

    def link(self, source_path: str, target_path: str, rel_type: str = "related"):
        source = self.read(source_path)
        target = self.read(target_path)
        if source and target:
            source.add_relation(rel_type, target.node_id)
            target.add_relation(f"reverse_{rel_type}", source.node_id)
            self.transaction_log.append("link", source_path, {
                "target": target_path,
                "rel_type": rel_type,
            })

    def search(self, query: str, k: int = 5) -> List[Tuple[str, KnowledgeNode, float]]:
        query_embedding = np.random.randn(64).astype(np.float32)
        results = self.index.search(query_embedding, k)

        output = []
        for node_id, dist in results:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                path = self._find_path(node_id)
                output.append((path, node, dist))
        return output

    def _find_path(self, node_id: str) -> str:
        for path, node in self.vfs_get_all():
            if isinstance(node, KnowledgeNode) and node.node_id == node_id:
                return path
        return ""

    def vfs_get_all(self):
        def _walk(node, path=""):
            if isinstance(node, KnowledgeNode):
                yield path, node
            elif isinstance(node, dict):
                for key, val in node.items():
                    new_path = f"{path}/{key}" if path else key
                    yield from _walk(val, new_path)
        yield from _walk(self.vfs.root)

    def open(self, path: str, mode: str = "r") -> 'FileHandle':
        return FileHandle(self, path, mode)

    def commit(self):
        self.transaction_log.append("commit", "", {})

    def rollback(self, to_position: int = 0):
        recent = self.transaction_log.get_entries(to_position)
        for entry in reversed(recent):
            if entry["operation"] == "write":
                self._delete_raw(entry["path"])
            elif entry["operation"] == "delete":
                if entry.get("data", {}).get("node_id"):
                    pass

    def stats(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self.nodes),
            "total_paths": len(list(self.vfs_get_all())),
            "transaction_count": len(self.transaction_log.entries),
            "index_size": len(self.index.nodes),
            "mount_points": list(self.mount_points.keys()),
        }


class FileHandle:
    def __init__(self, fs: KnowledgeFileSystem, path: str, mode: str):
        self.fs = fs
        self.path = path
        self.mode = mode
        self.node = None

    def __enter__(self):
        if "r" in self.mode:
            self.node = self.fs.read(self.path)
        elif "w" in self.mode or "a" in self.mode:
            self.node = self.fs.read(self.path) or KnowledgeNode(content="")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.node and ("w" in self.mode):
            self.fs.write(self.path, self.node.content, self.node.metadata)
        return False

    def read(self) -> Optional[str]:
        if self.node:
            return self.node.content
        return None

    def write(self, content: str):
        if self.node:
            self.node.content = content
            self.node.timestamps["modified"] = time.time()

    def tell(self) -> int:
        if self.node:
            return len(self.node.content)
        return 0

    def seek(self, offset: int):
        pass


if __name__ == "__main__":
    print("=== KnowledgeFileSystem Demo ===\n")

    fs = KnowledgeFileSystem()

    print("1. Writing nodes:")
    fs.write("concepts/transformer", "Transformer architecture with self-attention")
    fs.write("concepts/attention", "Scaled dot-product attention mechanism")
    fs.write("concepts/embedding", "Token embedding layer mapping vocab to vectors")
    fs.write("concepts/ffn", "Feed-forward network in transformer blocks")
    print(f"   Stats: {fs.stats()}")

    print("\n2. Reading nodes:")
    node = fs.read("concepts/transformer")
    print(f"   transformer: {node.content[:40]}...")

    print("\n3. Linking concepts:")
    fs.link("concepts/transformer", "concepts/attention", "contains")
    fs.link("concepts/transformer", "concepts/embedding", "uses")
    fs.link("concepts/transformer", "concepts/ffn", "contains")
    t = fs.read("concepts/transformer")
    print(f"   transformer relations: {list(t.relations.keys())}")

    print("\n4. Searching (cosine similarity):")
    results = fs.search("attention mechanism", k=3)
    for path, node, dist in results:
        print(f"   {path}: {node.content[:30]}... (dist={dist:.3f})")

    print("\n5. URI parsing:")
    domain, subdomain, concept = parse_uri("knowledge://ml/transformers/self-attention")
    print(f"   domain={domain}, subdomain={subdomain}, concept={concept}")

    print("\n6. File handle usage:")
    with fs.open("concepts/test", "w") as f:
        f.write("Test content for file handle")
    with fs.open("concepts/test", "r") as f:
        print(f"   Read: {f.read()}")

    print("\n7. Transaction log:")
    print(f"   Total transactions: {len(fs.transaction_log.entries)}")
    for entry in fs.transaction_log.get_entries()[:5]:
        print(f"   [{entry['operation']}] {entry['path']}")
