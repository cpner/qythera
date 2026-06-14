import pytest
import tempfile
import os
from memory.vector_store import VectorStore
from memory.episodic_memory import EpisodicMemory


class TestVectorStore:
    def test_add_and_search(self):
        vs = VectorStore(dimension=32)
        vs.add(["hello world", "foo bar", "test query"])
        results = vs.search("hello", top_k=2)
        assert len(results) > 0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = VectorStore(dimension=32)
            vs.add(["test document"])
            vs.save(tmpdir)
            vs2 = VectorStore(dimension=32)
            vs2.load(tmpdir)
            assert len(vs2) == 1


class TestEpisodicMemory:
    def test_conversation_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            em = EpisodicMemory(tmpdir)
            em.start_conversation()
            em.add_message("user", "Hello")
            em.add_message("assistant", "Hi there!")
            em.end_conversation()
            assert len(em.conversations) == 1

    def test_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            em = EpisodicMemory(tmpdir)
            em.start_conversation()
            em.add_message("user", "What is Python?")
            em.end_conversation()
            results = em.search("Python")
            assert len(results) > 0
