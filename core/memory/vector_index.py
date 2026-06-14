import numpy as np
import os, json
from typing import List, Tuple


class VectorIndex:
    """Custom IVF vector index (no external dependencies).
    
    Uses numpy for all operations.
    Supports:
    - Flat search for small datasets
    - IVF (Inverted File Index) for larger datasets
    - Cosine similarity
    """
    
    def __init__(self, dim=384, n_clusters=16):
        self.dim = dim
        self.n_clusters = n_clusters
        self.vectors = []
        self.ids = []
        self.centroids = None
        self.labels = None
        self._built = False

    def add(self, vectors, ids=None):
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        # Normalize
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        vectors = vectors / (norms + 1e-8)
        
        for i in range(len(vectors)):
            self.vectors.append(vectors[i])
            self.ids.append(ids[i] if ids else len(self.ids))
        self._built = False

    def build_index(self):
        if len(self.vectors) < self.n_clusters:
            self._built = True
            return
        
        X = np.array(self.vectors, dtype=np.float32)
        
        # K-means for IVF
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[indices].copy()
        
        for _ in range(10):
            dists = np.linalg.norm(X[:, None] - self.centroids[None], axis=-1)
            self.labels = np.argmin(dists, axis=-1)
            for c in range(self.n_clusters):
                mask = self.labels == c
                if mask.any():
                    self.centroids[c] = X[mask].mean(axis=0)
                    self.centroids[c] /= np.linalg.norm(self.centroids[c]) + 1e-8
        
        self._built = True

    def search(self, query, k=5) -> List[Tuple[int, float]]:
        if not self.vectors:
            return []
        
        if isinstance(query, list):
            query = np.array(query, dtype=np.float32)
        query = query.reshape(1, -1)
        query = query / (np.linalg.norm(query) + 1e-8)
        
        X = np.array(self.vectors, dtype=np.float32)
        
        if self._built and self.centroids is not None:
            c_dists = np.linalg.norm(query - self.centroids, axis=-1)
            nearest = np.argsort(c_dists)[:min(4, self.n_clusters)]
            mask = np.isin(self.labels, nearest)
            X_search = X[mask]
            search_ids = [i for i, m in enumerate(mask) if m]
        else:
            X_search = X
            search_ids = list(range(len(X)))
        
        sims = (X_search @ query.T).flatten()
        top_k = np.argsort(sims)[-k:][::-1]
        
        return [(search_ids[i], float(sims[i])) for i in top_k]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        data = {
            "dim": self.dim, "vectors": [v.tolist() for v in self.vectors],
            "ids": self.ids, "n_clusters": self.n_clusters,
        }
        with open(os.path.join(path, "index.json"), "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(os.path.join(path, "index.json")) as f:
            data = json.load(f)
        self.dim = data["dim"]
        self.vectors = [np.array(v, dtype=np.float32) for v in data["vectors"]]
        self.ids = data["ids"]
        self.build_index()

    def __len__(self):
        return len(self.vectors)
