import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.tokenizer.bpe import BPETokenizer

class TestTokenizer:
    def test_encode_decode(self):
        tok = BPETokenizer()
        tok.train(["hello world test"], vocab_size=300)
        ids = tok.encode("hello")
        text = tok.decode(ids)
        assert len(ids) > 0
        assert isinstance(text, str)

    def test_chat_encode(self):
        tok = BPETokenizer()
        msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        ids = tok.encode_chat(msgs)
        assert len(ids) > 5

    def test_special_tokens(self):
        tok = BPETokenizer()
        assert "<bos>" in tok.SPECIAL
        assert "<eos>" in tok.SPECIAL
