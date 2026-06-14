import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def sample_config():
    from vaelon.config import VaelonConfig
    return VaelonConfig(
        hidden_size=256, intermediate_size=512, num_layers=2,
        num_heads=4, num_kv_heads=2, head_dim=64, num_experts=2,
        num_experts_per_tok=2, vocab_size=1000, max_seq_len=128,
    )
