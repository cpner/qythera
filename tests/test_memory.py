import tempfile, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from core.memory.vector_index import VectorIndex
from core.memory.episodic import EpisodicMemory

class TestVectorIndex:
    def test_add_search(self):
        idx = VectorIndex(dim=16)
        for _ in range(10):
            idx.add([np.random.randn(16).astype(np.float32)])
        results = idx.search(np.random.randn(16), k=3)
        assert len(results) == 3

    def test_empty(self):
        idx = VectorIndex()
        assert len(idx) == 0
        assert idx.search(np.zeros(384)) == []

class TestEpisodic:
    def test_conversation(self):
        with tempfile.TemporaryDirectory() as d:
            em = EpisodicMemory(d)
            em.start()
            em.add("user", "Hi")
            em.add("assistant", "Hello!")
            em.end()
            assert len(em.conversations) == 1

    def test_search(self):
        with tempfile.TemporaryDirectory() as d:
            em = EpisodicMemory(d)
            em.start()
            em.add("user", "Python question")
            em.end()
            results = em.search("Python")
            assert len(results) > 0

    def test_recent(self):
        with tempfile.TemporaryDirectory() as d:
            em = EpisodicMemory(d)
            for _ in range(5):
                em.start()
                em.add("user", "msg")
                em.end()
            assert len(em.recent(3)) == 3
