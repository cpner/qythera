"""Model compatibility layer."""
try:
    from core.layers.transformer import Transformer as VaelonModel
except ImportError:
    VaelonModel = None

try:
    from core.config import ModelConfig as VaelonConfig
except ImportError:
    VaelonConfig = None

try:
    from core.config import Config
except ImportError:
    Config = None

QytheraModel = VaelonModel

__all__ = ["VaelonModel", "VaelonConfig", "QytheraModel", "Config"]
