import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.transformer.model import VaelonTransformer, ModelConfig

class TestTransformer:
    def test_create(self):
        config = ModelConfig.tiny()
        model = VaelonTransformer(config)
        assert model.num_params > 0
    def test_forward(self):
        config = ModelConfig.tiny()
        model = VaelonTransformer(config)
        ids = np.random.randint(0, 100, (1, 16)).astype(np.int64)
        logits, _ = model.forward(ids)
        assert logits.shape[2] == config.vocab_size
    def test_generate(self):
        config = ModelConfig.tiny()
        model = VaelonTransformer(config)
        ids = list(range(10))
        output = model.generate(ids, max_new=5, temperature=1.0)
        assert len(output) > len(ids)
    def test_save_load(self):
        import tempfile
        config = ModelConfig.tiny()
        model = VaelonTransformer(config)
        with tempfile.TemporaryDirectory() as d:
            model.save(d)
            model2 = VaelonTransformer.load(d)
            assert model2.num_params == model.num_params
    def test_sizes(self):
        for name in ["tiny", "small", "medium", "large"]:
            config = getattr(ModelConfig, name)()
            model = VaelonTransformer(config)
            assert model.num_params > 0
