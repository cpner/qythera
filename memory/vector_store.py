import os
import json
import numpy as np
from typing import List, Optional, Tuple

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class VectorStore:
    def __init__(self, dimension: int = 384, index_path: Optional[str] = None,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.dimension = dimension
        self.index_path = index_path
        self.documents: List[str] = []
        self.metadata: List[dict] = []

        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None

        self.encoder = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.encoder = SentenceTransformer(model_name)
            except Exception:
                pass

        if index_path and os.path.exists(os.path.join(index_path, "index.faiss")):
            self.load(index_path)

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self.encoder:
            embeddings = self.encoder.encode(texts, normalize_embeddings=True)
            return np.array(embeddings, dtype=np.float32)
        np.random.seed(42)
        return np.random.randn(len(texts), self.dimension).astype(np.float32)

    def add(self, documents: List[str], metadata: Optional[List[dict]] = None):
        embeddings = self._encode(documents)
        if self.index is not None:
            self.index.add(embeddings)
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        if not self.documents or self.index is None:
            return []
        query_emb = self._encode([query])
        scores, indices = self.index.search(query_emb, min(top_k, len(self.documents)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score), self.metadata[idx]))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.index is not None and HAS_FAISS:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.json"), "w") as f:
            json.dump({"documents": self.documents, "metadata": self.metadata}, f)

    def load(self, path: str):
        if HAS_FAISS:
            idx_path = os.path.join(path, "index.faiss")
            if os.path.exists(idx_path):
                self.index = faiss.read_index(idx_path)
        doc_path = os.path.join(path, "documents.json")
        if os.path.exists(doc_path):
            with open(doc_path) as f:
                data = json.load(f)
            self.documents = data.get("documents", [])
            self.metadata = data.get("metadata", [])

    def __len__(self):
        return len(self.documents)
