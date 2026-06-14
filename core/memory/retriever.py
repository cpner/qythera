from typing import List, Dict, Tuple
from core.memory.vector_index import VectorIndex
from core.memory.episodic import EpisodicMemory


class HybridRetriever:
    """Combines vector search + episodic search with reciprocal rank fusion."""
    
    def __init__(self, vector_index=None, episodic_memory=None):
        self.vector_index = vector_index or VectorIndex()
        self.episodic_memory = episodic_memory or EpisodicMemory()

    def add_document(self, text: str, metadata: Dict = None):
        """Add a document to the vector index."""
        # Simple embedding: character frequency vector
        emb = self._simple_embed(text)
        self.vector_index.add([emb], ids=[len(self.vector_index)])

    def _simple_embed(self, text: str):
        """Simple frequency-based embedding (placeholder for real embeddings)."""
        import numpy as np
        vec = np.zeros(384, dtype=np.float32)
        for i, ch in enumerate(text[:384]):
            vec[i % 384] += ord(ch) / 1000.0
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Hybrid retrieval with RRF (Reciprocal Rank Fusion)."""
        # Vector search
        q_emb = self._simple_embed(query)
        vector_results = self.vector_index.search(q_emb, k=k*2)
        
        # Episodic search
        episodic_results = self.episodic_memory.search(query, k=k*2)
        
        # Reciprocal Rank Fusion
        rrf_scores = {}
        k_rff = 60
        
        for rank, (idx, score) in enumerate(vector_results):
            key = f"vec_{idx}"
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k_rff + rank + 1)
        
        for rank, result in enumerate(episodic_results):
            conv = result["conversation"]
            key = f"epi_{conv['id']}"
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k_rff + rank + 1)
        
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        results = []
        for key in sorted_keys[:k]:
            results.append({"key": key, "score": rrf_scores[key], "source": "hybrid"})
        
        return results

    def save(self):
        self.vector_index.save("./vector_index")

    def load(self):
        self.vector_index.load("./vector_index")
