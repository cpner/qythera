import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.model import VaelonModel, VaelonConfig
from core.autodiff.tensor import Tensor

class TestVaelonModel:
    def test_small_config(self):
        cfg = VaelonConfig.small()
        assert cfg.hidden_size == 512
        assert cfg.num_layers == 6

    def test_create_model(self):
        cfg = VaelonConfig.small()
        model = VaelonModel(cfg)
        params = sum(p.data.size for p in model.parameters())
        assert params > 1000

    def test_forward(self):
        cfg = VaelonConfig.small()
        model = VaelonModel(cfg)
        ids = Tensor(np.random.randint(0, 100, (1, 16)).astype(np.int32))
        logits, loss, aux = model(ids)
        assert logits.shape[2] == cfg.vocab_size

    def test_forward_with_labels(self):
        cfg = VaelonConfig.small()
        model = VaelonModel(cfg)
        ids = Tensor(np.random.randint(0, 100, (1, 16)).astype(np.int32))
        labels = Tensor(np.random.randint(0, 100, (1, 16)).astype(np.int32))
        _, loss, _ = model(ids, labels)
        assert loss.item() > 0

    def test_causal_mask(self):
        mask = VaelonModel.causal_mask(4)
        assert mask.shape == (1, 1, 4, 4)

    def test_generate(self):
        cfg = VaelonConfig.small()
        model = VaelonModel(cfg)
        gen = VaelonModel.generate_ids(model, [10, 20, 30], max_new=5, temp=1.0)
        assert len(gen) > 3

    def test_parameter_count(self):
        model = VaelonModel(VaelonConfig.small())
        params = sum(p.data.size for p in model.parameters())
        assert params > 1_000_000
