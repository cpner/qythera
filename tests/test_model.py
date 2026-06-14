
import torch, pytest
from core.config import ModelConfig
from core.model import QytheraModel

class TestModel:
    def test_forward(self, small_cfg):
        m = QytheraModel(small_cfg)
        ids = torch.randint(0, 1000, (1, 16))
        logits, loss = m(ids, labels=ids)
        assert logits.shape[2] == small_cfg.vocab_size

    def test_generate(self, small_cfg):
        m = QytheraModel(small_cfg)
        m.eval()
        ids = torch.randint(0, 1000, (1, 8))
        out = m.generate(ids, max_new=5)
        assert out.shape[1] == 13

    def test_configs(self):
        for name in ["small", "medium", "large", "xlarge"]:
            cfg = getattr(ModelConfig, name)()
            m = QytheraModel(cfg)
            params = sum(p.numel() for p in m.parameters())
            assert params > 0
