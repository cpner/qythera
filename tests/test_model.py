import torch
import pytest
from vaelon.config import VaelonConfig
from vaelon.model import VaelonModel


class TestVaelonModel:
    def test_forward(self, sample_config):
        model = VaelonModel(sample_config)
        input_ids = torch.randint(0, 1000, (1, 16))
        outputs = model(input_ids=input_ids)
        assert outputs.logits is not None
        assert outputs.logits.shape == (1, 16, 1000)

    def test_forward_with_labels(self, sample_config):
        model = VaelonModel(sample_config)
        input_ids = torch.randint(0, 1000, (1, 16))
        labels = torch.randint(0, 1000, (1, 16))
        outputs = model(input_ids=input_ids, labels=labels)
        assert outputs.loss is not None

    def test_generate(self, sample_config):
        model = VaelonModel(sample_config)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 8))
        output = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
        assert output.shape[1] == 18

    def test_model_sizes(self):
        config = VaelonConfig.vaelon_7b()
        assert config.hidden_size == 4096
        assert config.num_layers == 32
