import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.tokenizer.bpe import BPETokenizer

class TestTokenizer:
    def test_encode_decode(self):
        t = BPETokenizer()
        t.train(["hello world test"], vocab_size=100)
        ids = t.encode("hello")
        assert len(ids) > 0
        text = t.decode(ids)
        assert isinstance(text, str)
    def test_special_tokens(self):
        t = BPETokenizer()
        assert "<bos>" in [v for k,v in t.char_to_id.items()]
    def test_vocab_size(self):
        t = BPETokenizer()
        assert len(t) >= 8
