from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json, os

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 11008
    max_seq_len: int = 4096
    num_experts: int = 8
    experts_per_tok: int = 2
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    activation: str = "silu"
    tie_embeddings: bool = False

    @classmethod
    def small(cls): return cls(hidden_size=1024, num_layers=12, num_heads=8, num_kv_heads=2, intermediate_size=2816, num_experts=4, experts_per_tok=2)
    @classmethod
    def medium(cls): return cls(hidden_size=2048, num_layers=24, num_heads=16, num_kv_heads=4, intermediate_size=5504, num_experts=8, experts_per_tok=2)
    @classmethod
    def large(cls): return cls(hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=8, intermediate_size=11008)
    @classmethod
    def xlarge(cls): return cls(hidden_size=8192, num_layers=64, num_heads=64, num_kv_heads=8, intermediate_size=22016, num_experts=16, experts_per_tok=4, max_seq_len=8192)

@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig.large)
    output_dir: str = "./checkpoints"
    batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    lr_scheduler: str = "cosine"
    bf16: bool = True
    gradient_clipping: float = 1.0
    log_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 500

@dataclass
class InferenceConfig:
    model_path: str = "./models/vaelon"
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "auto"
    max_batch_size: int = 32
    max_seq_len: int = 2048
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    quantization: str = "none"
    cache_size: int = 1000

@dataclass
class SafetyConfig:
    toxicity_threshold: float = 0.5
    jailbreak_detection: bool = True
    pii_redaction: bool = True
    content_filter: str = "standard"
    blocked_topics: list = field(default_factory=list)

@dataclass
class MemoryConfig:
    vector_dim: int = 384
    max_entries: int = 100000
    similarity_threshold: float = 0.7
    embedding_model: str = "all-MiniLM-L6-v2"
    persistent: bool = True
    storage_path: str = "./memory_store"

@dataclass
class WebConfig:
    title: str = "Qythera AI"
    theme: str = "dark"
    max_message_length: int = 10000
    streaming: bool = True
    markdown: bool = True
    code_highlight: bool = True
    language: str = "en"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig.large)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    web: WebConfig = field(default_factory=WebConfig)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f: json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f: data = json.load(f)
        cfg = cls()
        if "model" in data: cfg.model = ModelConfig(**{k:v for k,v in data["model"].items() if hasattr(cfg.model, k)})
        if "inference" in data: cfg.inference = InferenceConfig(**{k:v for k,v in data["inference"].items() if hasattr(cfg.inference, k)})
        return cfg

    @classmethod
    def from_preset(cls, name: str) -> "Config":
        presets = {"small": ModelConfig.small(), "medium": ModelConfig.medium(),
                   "large": ModelConfig.large(), "xlarge": ModelConfig.xlarge()}
        cfg = cls()
        if name in presets: cfg.model = presets[name]
        return cfg