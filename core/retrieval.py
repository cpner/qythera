"""Retrieval modules: BM25, Dense, Hybrid, ColBERT, InvertedIndex."""

import math
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np


@dataclass
class Document:
    text: str
    doc_id: int = 0
    tokens: List[str] = field(default_factory=list)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.tf: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def add(self, doc: str) -> None:
        doc_id = len(self.documents)
        tokens = self._tokenize(doc)
        document = Document(text=doc, doc_id=doc_id, tokens=tokens)
        self.documents.append(document)
        self.doc_lengths.append(len(tokens))
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        term_counts = Counter(tokens)
        self.tf.append(dict(term_counts))

        seen = set()
        for token in tokens:
            if token not in seen:
                self.doc_freqs[token] += 1
                seen.add(token)

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self.doc_freqs.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        for term in query_tokens:
            if term not in self.tf[doc_idx]:
                continue
            tf_val = self.tf[doc_idx][term]
            idf = self._idf(term)
            numerator = tf_val * (self.k1 + 1)
            denominator = tf_val + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_tokens = self._tokenize(query)
        scores = [(i, self._bm25_score(query_tokens, i)) for i in range(len(self.documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DenseRetrieval:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.rng = np.random.default_rng(42)

    def _embed(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        vec = np.zeros(self.dim)
        for token in tokens:
            h = hash(token) % self.dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def add(self, doc: str) -> None:
        emb = self._embed(doc)
        self.documents.append(doc)
        if self.embeddings is None:
            self.embeddings = emb.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.embeddings is None or len(self.documents) == 0:
            return []
        q_emb = self._embed(query)
        similarities = self.embeddings @ q_emb
        indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in indices]


class HybridRetrieval:
    def __init__(self, bm25_weight: float = 0.5, dense_weight: float = 0.5, dim: int = 128):
        self.bm25 = BM25()
        self.dense = DenseRetrieval(dim=dim)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def add(self, doc: str) -> None:
        self.bm25.add(doc)
        self.dense.add(doc)

    def _min_max_normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return scores
        min_s = min(scores)
        max_s = max(scores)
        rng = max_s - min_s
        if rng == 0:
            return [0.5] * len(scores)
        return [(s - min_s) / rng for s in scores]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        bm25_results = self.bm25.search(query, top_k=len(self.bm25.documents))
        dense_results = self.dense.search(query, top_k=len(self.dense.documents))

        bm25_scores_dict = {doc_id: score for doc_id, score in bm25_results}
        dense_scores_dict = {doc_id: score for doc_id, score in dense_results}

        all_doc_ids = set(bm25_scores_dict.keys()) | set(dense_scores_dict.keys())

        bm25_raw = [bm25_scores_dict.get(did, 0.0) for did in all_doc_ids]
        dense_raw = [dense_scores_dict.get(did, 0.0) for did in all_doc_ids]

        bm25_norm = self._min_max_normalize(bm25_raw)
        dense_norm = self._min_max_normalize(dense_raw)

        combined = []
        for i, did in enumerate(all_doc_ids):
            score = self.bm25_weight * bm25_norm[i] + self.dense_weight * dense_norm[i]
            combined.append((did, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]


class ColBERT:
    def __init__(self, dim: int = 32):
        self.dim = dim
        self.documents: List[str] = []
        self.doc_token_embeddings: List[np.ndarray] = []
        self.rng = np.random.default_rng(42)

    def _embed_tokens(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        if not tokens:
            return np.zeros((0, self.dim))
        embeddings = []
        for token in tokens:
            vec = np.zeros(self.dim)
            h = hash(token) % self.dim
            vec[h] = 1.0
            vec += self.rng.normal(0, 0.01, self.dim)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings)

    def add(self, doc: str) -> None:
        self.documents.append(doc)
        self.doc_token_embeddings.append(self._embed_tokens(doc))

    def _maxsim(self, query_embs: np.ndarray, doc_embs: np.ndarray) -> float:
        if query_embs.shape[0] == 0 or doc_embs.shape[0] == 0:
            return 0.0
        sim_matrix = query_embs @ doc_embs.T
        max_per_query = sim_matrix.max(axis=1)
        return float(max_per_query.sum())

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_embs = self._embed_tokens(query)
        scores = []
        for i, doc_embs in enumerate(self.doc_token_embeddings):
            score = self._maxsim(query_embs, doc_embs)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, List[int]] = defaultdict(list)
        self.documents: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def add(self, doc: str) -> None:
        doc_id = len(self.documents)
        self.documents.append(doc)
        tokens = self._tokenize(doc)
        seen = set()
        for token in tokens:
            if token not in seen:
                self.index[token].append(doc_id)
                seen.add(token)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokens = self._tokenize(query)
        doc_scores: Dict[int, float] = defaultdict(float)
        for token in tokens:
            if token in self.index:
                for doc_id in self.index[token]:
                    doc_scores[doc_id] += 1.0
        results = [(did, score) for did, score in doc_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
