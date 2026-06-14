
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from core.config import ModelConfig

@pytest.fixture
def small_cfg():
    return ModelConfig.small()
