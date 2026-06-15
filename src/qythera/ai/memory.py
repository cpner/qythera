"""Memory systems for Qythera. Pure Python + NumPy."""
import math
import time
import hashlib
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from qythera.nn import Module, Linear
from qythera.tensor import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


def _tfidf_matrix(docs: List[List[str]], vocab: Dict[str, int]) -> np.ndarray:
    n = len(docs)
    v = len(vocab)
    tf = np.zeros((n, v), dtype=np.float64)
    df = np.zeros(v, dtype=np.float64)
    for i, doc in enumerate(docs):
        counts = defaultdict(int)
        for tok in doc:
            if tok in vocab:
                counts[vocab[tok]] += 1
        total = max(1, sum(counts.values()))
        for tid, c in counts.items():
            tf[i, tid] = c / total
            df[tid] += 1
    idf = np.zeros(v, dtype=np.float64)
    for j in range(v):
        idf[j] = math.log((n + 1) / (df[j] + 1)) + 1
    return tf * idf


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], 'little'))
    vec = rng.randn(dim).astype(np.float64)
    vec /= max(np.linalg.norm(vec), 1e-12)
    return vec


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """Store/retrieve conversations via TF-IDF."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.entries: List[Dict[str, Any]] = []
        self._vocab: Dict[str, int] = {}
        self._docs: List[List[str]] = []
        self._matrix: Optional[np.ndarray] = None

    def _rebuild_index(self):
        self._vocab = {}
        for doc in self._docs:
            for tok in doc:
                if tok not in self._vocab:
                    self._vocab[tok] = len(self._vocab)
        if self._docs and self._vocab:
            self._matrix = _tfidf_matrix(self._docs, self._vocab)
        else:
            self._matrix = None

    def store(self, role: str, content: str, metadata: Optional[Dict] = None):
        entry = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "id": len(self.entries),
        }
        self.entries.append(entry)
        self._docs.append(_tokenize(content))
        self._rebuild_index()
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size:]
            self._docs = self._docs[-self.max_size:]
            self._rebuild_index()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._docs or self._matrix is None:
            return []
        q_toks = _tokenize(query)
        q_vec = np.zeros(len(self._vocab), dtype=np.float64)
        for tok in q_toks:
            if tok in self._vocab:
                q_vec[self._vocab[tok]] += 1
        total = max(1, sum(q_vec))
        q_vec /= total
        scores = self._matrix @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                e = dict(self.entries[idx])
                e["score"] = float(scores[idx])
                results.append(e)
        return results

    def clear(self):
        self.entries.clear()
        self._docs.clear()
        self._matrix = None
        self._vocab.clear()


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """Fact embeddings with cosine similarity retrieval."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.facts: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None

    def store(self, fact: str, label: Optional[str] = None, metadata: Optional[Dict] = None):
        vec = _hash_embed(fact, self.dim)
        self.facts.append({
            "fact": fact,
            "label": label or fact[:40],
            "metadata": metadata or {},
            "timestamp": time.time(),
        })
        if self.vectors is None:
            self.vectors = vec.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)])

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.facts) == 0:
            return []
        q_vec = _hash_embed(query, self.dim)
        sims = np.array([_cosine_sim(q_vec, self.vectors[i]) for i in range(len(self.facts))])
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if sims[idx] > 0:
                e = dict(self.facts[idx])
                e["score"] = float(sims[idx])
                results.append(e)
        return results

    def clear(self):
        self.facts.clear()
        self.vectors = None


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

class WorkingMemory:
    """Fixed-size context window management."""

    def __init__(self, max_tokens: int = 2048, tokenizer=None):
        self.max_tokens = max_tokens
        self.buffer: deque = deque()
        self.total_tokens = 0
        self._tokenizer = tokenizer

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        return len(text.split())

    def store(self, role: str, content: str, metadata: Optional[Dict] = None):
        tokens = self._count_tokens(content)
        entry = {
            "role": role,
            "content": content,
            "tokens": tokens,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        self.buffer.append(entry)
        self.total_tokens += tokens
        while self.total_tokens > self.max_tokens and self.buffer:
            removed = self.buffer.popleft()
            self.total_tokens -= removed["tokens"]

    def retrieve(self, query: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        items = list(self.buffer)
        if query is None:
            return items[-top_k:]
        q_toks = set(_tokenize(query))

        def score(entry):
            entry_toks = set(_tokenize(entry["content"]))
            return len(q_toks & entry_toks)

        items.sort(key=score, reverse=True)
        return items[:top_k]

    def get_context(self) -> str:
        return "\n".join(f"{e['role']}: {e['content']}" for e in self.buffer)

    def clear(self):
        self.buffer.clear()
        self.total_tokens = 0


# ---------------------------------------------------------------------------
# ProceduralMemory
# ---------------------------------------------------------------------------

class ProceduralMemory:
    """Stored reasoning chains with template matching."""

    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {}

    def store(self, name: str, steps: List[str], description: str = "", tags: Optional[List[str]] = None):
        self.templates[name] = {
            "steps": list(steps),
            "description": description,
            "tags": tags or [],
            "usage_count": 0,
            "timestamp": time.time(),
        }

    def retrieve(self, problem: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_tags = set(_tokenize(problem))
        scored = []
        for name, tmpl in self.templates.items():
            tmpl_tags = set(tmpl["tags"])
            desc_toks = set(_tokenize(tmpl["description"]))
            name_toks = set(_tokenize(name))
            overlap = len(q_tags & (tmpl_tags | desc_toks | name_toks))
            if overlap > 0:
                scored.append((overlap, name, tmpl))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, name, tmpl in scored[:top_k]:
            results.append({"name": name, **tmpl})
            self.templates[name]["usage_count"] += 1
        return results

    def get_chain(self, name: str) -> Optional[List[str]]:
        tmpl = self.templates.get(name)
        if tmpl:
            tmpl["usage_count"] += 1
            return list(tmpl["steps"])
        return None

    def remove(self, name: str):
        self.templates.pop(name, None)


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------

class LongTermMemory:
    """Hierarchical memory: recent entries full text, older entries summarized."""

    def __init__(self, full_threshold: int = 50, summary_max: int = 200):
        self.full_threshold = full_threshold
        self.summary_max = summary_max
        self.recent: List[Dict[str, Any]] = []
        self.archived: List[Dict[str, Any]] = []

    def store(self, role: str, content: str, metadata: Optional[Dict] = None):
        entry = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        self.recent.append(entry)
        if len(self.recent) > self.full_threshold:
            overflow = self.recent[:len(self.recent) - self.summary_max]
            self.recent = self.recent[len(overflow):]
            for e in overflow:
                summary = e["content"][:120] + ("..." if len(e["content"]) > 120 else "")
                self.archived.append({**e, "content": summary, "summarized": True})

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_toks = set(_tokenize(query))

        def score(entry):
            return len(q_toks & set(_tokenize(entry["content"])))

        all_entries = self.archived + self.recent
        scored = sorted(all_entries, key=score, reverse=True)
        results = []
        for e in scored[:top_k]:
            r = dict(e)
            r["source"] = "archived" if e.get("summarized") else "recent"
            results.append(r)
        return results

    def clear(self):
        self.recent.clear()
        self.archived.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "recent_count": len(self.recent),
            "archived_count": len(self.archived),
            "full_threshold": self.full_threshold,
        }


# ---------------------------------------------------------------------------
# AssociativeRecall
# ---------------------------------------------------------------------------

class AssociativeRecall:
    """Hopfield-style pattern retrieval."""

    def __init__(self, dim: int = 64, temperature: float = 1.0):
        self.dim = dim
        self.temperature = temperature
        self.patterns: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None

    def store(self, content: str, metadata: Optional[Dict] = None):
        vec = _hash_embed(content, self.dim)
        self.patterns.append({
            "content": content,
            "metadata": metadata or {},
            "vector": vec,
            "timestamp": time.time(),
        })
        if self._matrix is None:
            self._matrix = vec.reshape(-1, 1)
        else:
            self._matrix = np.hstack([self._matrix, vec.reshape(-1, 1)])

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.patterns or self._matrix is None:
            return []
        q_vec = _hash_embed(query, self.dim)

        energies = -(self._matrix.T @ q_vec) / self.temperature
        exp_energies = np.exp(energies - np.max(energies))
        probs = exp_energies / max(np.sum(exp_energies), 1e-12)
        top_idx = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if probs[idx] > 1e-8:
                e = dict(self.patterns[idx])
                e.pop("vector", None)
                e["probability"] = float(probs[idx])
                results.append(e)
        return results

    def clear(self):
        self.patterns.clear()
        self._matrix = None


# ---------------------------------------------------------------------------
# DNC - Differentiable Neural Computer
# ---------------------------------------------------------------------------

class DNC(Module):
    """Differentiable Neural Computer with external memory matrix."""

    def __init__(self, input_size: int, hidden_size: int, memory_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory: Optional[Tensor] = None
        self.controller = Linear(input_size + memory_size, hidden_size)
        self.read_head = Linear(hidden_size, memory_size)
        self.write_head = Linear(hidden_size, memory_size)
        self.erase_head = Linear(hidden_size, memory_size)
        self.add_head = Linear(hidden_size, memory_size)

    def _init_memory(self, batch_size: int):
        self.memory = Tensor(np.zeros((batch_size, self.memory_size, self.memory_size), dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        import numpy as np
        batch, seq_len, _ = x.shape
        self._init_memory(batch)
        output = []
        prev_read = np.zeros((batch, self.memory_size), dtype=np.float32)

        for t in range(seq_len):
            xt = x.data[:, t, :]
            controller_input = np.concatenate([xt, prev_read], axis=-1)
            h = self.controller(Tensor(controller_input.astype(np.float32)))

            r_weights = self.read_head(h).softmax(axis=-1)
            read_vec = r_weights.matmul(self.memory)
            prev_read = read_vec.data.reshape(batch, self.memory_size)

            erase = self.erase_head(h).softmax(axis=-1)
            add = self.add_head(h).softmax(axis=-1)
            write_weights = self.write_head(h).softmax(axis=-1)

            erase_term = Tensor((1.0 - erase.data * write_weights.data).astype(np.float32))
            self.memory = Tensor((self.memory.data * erase_term.data).astype(np.float32))
            outer = write_weights.data.reshape(batch, self.memory_size, 1) * add.data.reshape(batch, 1, self.memory_size)
            self.memory = Tensor((self.memory.data + outer).astype(np.float32))

            step_out = np.concatenate([h.data, prev_read], axis=-1)
            output.append(Tensor(step_out.astype(np.float32)))

        return Tensor(np.stack([o.data for o in output], axis=1))


# ---------------------------------------------------------------------------
# MemoryAugmentedTransformer
# ---------------------------------------------------------------------------

class MemoryAugmentedTransformer(Module):
    """Transformer with external memory matrix for augmented recall."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or {}
        self.d_model = cfg.get("d_model", 256)
        self.memory_size = cfg.get("memory_size", 128)
        self.memory: Optional[Tensor] = None
        self.key_proj = Linear(self.d_model, self.d_model)
        self.query_proj = Linear(self.d_model, self.d_model)
        self.value_proj = Linear(self.d_model, self.d_model)
        self.output_proj = Linear(self.d_model * 2, self.d_model)
        self.write_key_head = Linear(self.d_model, self.memory_size)
        self.write_val_head = Linear(self.d_model, self.d_model)

    def _init_memory(self, batch_size: int):
        self.memory = Tensor(np.zeros((batch_size, self.memory_size, self.d_model), dtype=np.float32))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        import numpy as np
        if self.memory is None:
            self._init_memory(x.shape[0])

        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        scale = math.sqrt(self.d_model)

        mem_k = self.key_proj(self.memory)
        scores = q.matmul(mem_k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = Tensor((scores.data + mask.data * -1e9).astype(np.float32))

        attn = scores.softmax(axis=-1)
        r = attn.matmul(self.memory)

        write_w = self.write_key_head(x).softmax(axis=-1)
        write_v = self.write_val_head(x)
        outer = np.einsum('bsm,bsd->bmd', write_w.data, write_v.data)
        self.memory = Tensor((self.memory.data + outer).astype(np.float32))

        combined = Tensor(np.concatenate([x.data, r.data], axis=-1).astype(np.float32))
        out = self.output_proj(combined)
        return out
