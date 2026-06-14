
import os, json, numpy as np
from typing import List, Tuple, Optional

class VectorStore:
    def __init__(self, dim=384, path="./vstore"):
        self.dim, self.path = dim, path
        self.vecs, self.docs, self.meta = [], [], []
        try:
            import faiss
            self.index = faiss.IndexFlatIP(dim)
            self._has_faiss = True
        except: self.index, self._has_faiss = None, False
        if os.path.exists(os.path.join(path, "data.json")): self.load()

    def _embed(self, texts):
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_encoder'):
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            return np.array(self._encoder.encode(texts, normalize_embeddings=True), dtype=np.float32)
        except: return np.random.randn(len(texts), self.dim).astype(np.float32)

    def add(self, docs, metadata=None):
        embs = self._embed(docs)
        if self._has_faiss and self.index: self.index.add(embs)
        self.vecs.extend(embs.tolist())
        self.docs.extend(docs)
        self.meta.extend(metadata or [{}]*len(docs))

    def search(self, query, k=5) -> List[Tuple[str, float]]:
        if not self.docs: return []
        q = self._embed([query])
        if self._has_faiss and self.index and self.index.ntotal > 0:
            D, I = self.index.search(q, min(k, len(self.docs)))
            return [(self.docs[i], float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]
        scores = [(i, float(np.dot(q[0], np.array(v)))) for i, v in enumerate(self.vecs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.docs[i], s) for i, s in scores[:k]]

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, "data.json"), "w") as f:
            json.dump({"docs": self.docs, "meta": self.meta, "vecs": self.vecs}, f)

    def load(self):
        with open(os.path.join(self.path, "data.json")) as f:
            d = json.load(f)
        self.docs, self.meta, self.vecs = d.get("docs",[]), d.get("meta",[]), d.get("vecs",[])
        if self._has_faiss and self.vecs:
            self.index.add(np.array(self.vecs, dtype=np.float32))

    def __len__(self): return len(self.docs)
