import math
from typing import List, Tuple, Dict, Optional
from memory.vector_store import VectorStore
from memory.episodic_memory import EpisodicMemory


class HybridRetriever:
    def __init__(self, vector_store: Optional[VectorStore] = None,
                 episodic_memory: Optional[EpisodicMemory] = None):
        self.vector_store = vector_store or VectorStore()
        self.episodic_memory = episodic_memory or EpisodicMemory()

    def reciprocal_rank_fusion(self, ranked_lists: List[List[Tuple]], k: int = 60) -> List[Tuple]:
        scores = {}
        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list):
                key = item[0] if isinstance(item, tuple) else str(item)
                if key not in scores:
                    scores[key] = {"score": 0, "item": item}
                scores[key]["score"] += 1.0 / (k + rank + 1)
        sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [(item["item"], item["score"]) for item in sorted_items]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        dense_results = self.vector_store.search(query, top_k=top_k * 2)
        dense_ranked = [(r[0], r[1]) for r in dense_results]

        episodic_results = self.episodic_memory.search(query, top_k=top_k * 2)
        episodic_ranked = [(r["conversation"].get("id", ""), r["relevance"]) for r in episodic_results]

        fused = self.reciprocal_rank_fusion([dense_ranked, episodic_ranked])

        results = []
        for item, score in fused[:top_k]:
            if isinstance(item, tuple):
                text = item[0] if len(item) > 0 else ""
                results.append({"text": text, "score": score, "source": "hybrid"})
            else:
                results.append({"text": str(item), "score": score, "source": "hybrid"})
        return results
