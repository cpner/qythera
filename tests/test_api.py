import pytest
from inference.server import InferenceServer


class TestInferenceServer:
    def test_server_creation(self):
        server = InferenceServer()
        assert server.host == "0.0.0.0"
        assert server.port == 8000

    def test_generate(self):
        server = InferenceServer()
        result = server.generate("Hello", max_tokens=5)
        assert isinstance(result, str)
