"""Model configuration for Vaelon transformer."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VaelonConfig:
    """Configuration for Vaelon transformer model.

    Supports scaling from 7B to 70B parameters via Mixture of Experts.
    """

    vocab_size: int = 128256
    max_seq_len: int = 8192
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128

    num_experts: int = 8
    num_experts_per_tok: int = 2

    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    norm_eps: float = 1e-5
    dropout: float = 0.0

    activation_function: str = "silu"
    tie_word_embeddings: bool = False
    use_bias: bool = False

    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    @classmethod
    def vaelon_7b(cls) -> "VaelonConfig":
        return cls(
            hidden_size=4096, intermediate_size=14336, num_layers=32,
            num_heads=32, num_kv_heads=8, head_dim=128,
            num_experts=8, num_experts_per_tok=2,
        )

    @classmethod
    def vaelon_13b(cls) -> "VaelonConfig":
        return cls(
            hidden_size=5120, intermediate_size=13824, num_layers=40,
            num_heads=40, num_kv_heads=8, head_dim=128,
            num_experts=8, num_experts_per_tok=2,
        )

    @classmethod
    def vaelon_70b(cls) -> "VaelonConfig":
        return cls(
            hidden_size=8192, intermediate_size=28672, num_layers=80,
            num_heads=64, num_kv_heads=8, head_dim=128,
            num_experts=64, num_experts_per_tok=8, max_seq_len=32768,
        )

    def get_model_args(self) -> dict:
        return {
            "vocab_size": self.vocab_size, "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size, "num_layers": self.num_layers,
            "num_heads": self.num_heads, "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim, "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok, "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps, "dropout": self.dropout, "max_seq_len": self.max_seq_len,
            "activation_function": self.activation_function,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
