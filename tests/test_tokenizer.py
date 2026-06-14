
from core.tokenizer import Tokenizer

class TestTokenizer:
    def test_encode_decode(self):
        t = Tokenizer()
        ids = t.encode("Hello world")
        text = t.decode(ids)
        assert "Hello" in text

    def test_vocab_size(self):
        t = Tokenizer()
        assert t.vocab_size > 100

    def test_chat_encode(self):
        t = Tokenizer()
        msgs = [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"}]
        ids = t.encode_chat(msgs)
        assert len(ids) > 5
