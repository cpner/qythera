
from typing import List, Dict
from core.memory.vector_store import VectorStore
from core.memory.episodic import EpisodicMemory

class HybridRetriever:
    def __init__(self, vs=None, em=None):
        self.vs = vs or VectorStore()
        self.em = em or EpisodicMemory()

    def retrieve(self, query, k=5):
        dense = [(t, s, "vector") for t, s in self.vs.search(query, k*2)]
        epi = [(c["messages"][-1]["content"] if c["messages"] else "", 0.5, "episodic") for c in self.em.search(query, k*2)]
        all_r = dense + epi
        all_r.sort(key=lambda x: x[1], reverse=True)
        return all_r[:k]

    def add_document(self, text, metadata=None):
        self.vs.add([text], [metadata or {}])

    def save(self):
        self.vs.save()
