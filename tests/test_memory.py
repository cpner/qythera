
import tempfile, pytest
from core.memory.vector_store import VectorStore
from core.memory.episodic import EpisodicMemory
from core.memory.retriever import HybridRetriever

class TestVectorStore:
    def test_add_search(self):
        vs = VectorStore(dim=32)
        vs.add(["hello world", "foo bar", "test query"])
        results = vs.search("hello", k=2)
        assert len(results) > 0

class TestEpisodic:
    def test_conversation(self):
        with tempfile.TemporaryDirectory() as d:
            em = EpisodicMemory(d)
            em.start()
            em.add("user", "Hi")
            em.add("assistant", "Hello!")
            em.end()
            assert len(em.convs) == 1

class TestRetriever:
    def test_retrieve(self):
        r = HybridRetriever()
        r.add_document("Python is a programming language")
        results = r.retrieve("programming")
        assert len(results) > 0
