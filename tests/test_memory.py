"""Tests for memory system."""
import tempfile, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.memory.vector_index import VectorIndex
from core.memory.episodic import EpisodicMemory
from core.memory.retriever import HybridRetriever
import numpy as np


class TestVectorIndex:
    def test_add_and_search(self):
        idx = VectorIndex(dim=16)
        vecs = [np.random.randn(16).astype(np.float32) for _ in range(10)]
        idx.add(vecs)
        results = idx.search(np.random.randn(16), k=3)
        assert len(results) == 3

    def test_empty_index(self):
        idx = VectorIndex()
        results = idx.search(np.random.randn(384))
        assert results == []

    def test_len(self):
        idx = VectorIndex()
        assert len(idx) == 0
        idx.add([np.zeros(384)])
        assert len(idx) == 1


class TestEpisodicMemory:
    def test_conversation_flow(self):
        with tempfile.TemporaryDirectory() as d:
            em = EpisodicMemory(d)
            em.start()
            em.add("user", "Hello")
            em.add("assistant", "Hi!")
            em.end()
            assert len(em.conversations) == 1

    def test_search(self):
        with tempfile.TemporaryDirectory() as d:
            em = EpisodicMemory(d)
            em.start()
            em.add("user", "What is Python?")
            em.end()
            results = em.search("Python")
            assert len(results) > 0

    def test_recent(self):
        with tempfile.TemporaryDirectory() as d:
            em = EpisodicMemory(d)
            for i in range(5):
                em.start()
                em.add("user", f"Message {i}")
                em.end()
            assert len(em.recent(3)) == 3


class TestRetriever:
    def test_add_and_retrieve(self):
        r = HybridRetriever()
        r.add_document("Python is a programming language")
        r.add_document("JavaScript is used for web")
        results = r.retrieve("programming")
        assert len(results) > 0
