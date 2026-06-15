# Qythera

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-00C853)
![NumPy](https://img.shields.io/badge/NumPy-1.22%2B-4DABCF?logo=numpy&logoColor=white)
![Modules](https://img.shields.io/badge/Modules-45-8B5CF6)
![Lines](https://img.shields.io/badge/Lines-29K%2B-06B6D4)

**Production Superintelligence Platform.** Pure Python + NumPy. Zero external AI APIs. Dual backend (pure Python when NumPy unavailable, NumPy when available).

## Architecture

```
tokens → embeddings → N transformer layers → logits → sampling

Each layer: RMSNorm → MHA+RoPE → RMSNorm → SwiGLU → residual
```

## Prerequisites

- Python 3.9 or higher
- NumPy 1.22+ (optional — framework works without it, slower)

## Installation

```bash
git clone https://github.com/cpner/qythera.git
cd qythera
pip install .
```

## Quick Start

```bash
# Run with NumPy (fast)
PYTHONPATH=src python3 -c "
from qythera.model import Transformer, TransformerConfig
from qythera.tensor import Tensor
import numpy as np
model = Transformer(TransformerConfig())
logits = model(Tensor(np.random.randint(0, 100, (1, 16))))
print(logits.shape)
"

# Start server
PYTHONPATH=src python3 -m qythera.inference.cli serve --port 8080
```

## Cross-Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | ✅ Full | All features |
| macOS | ✅ Full | All features |
| Windows | ✅ Full | Signal handling guarded |
| Android/Termux | ✅ Full | Auto-detect RAM |
| Docker | ✅ Full | `pip install .` |

## Core Engine

### Tensor Engine (`tensor.py` — 2,327 lines)
- Custom autodiff with computation graph
- 50+ element-wise ops with exact backward
- Broadcasting, views, stride manipulation
- Linear algebra: matmul, einsum, SVD, QR, Cholesky, LU, eig
- Sparse Tensor COO format
- Forward mode AD (dual numbers)
- GradScaler, gradient checkpointing

### Neural Networks (`nn.py` — 1,557 lines)
- Module base with hooks, state_dict, train/eval
- Linear, Embedding, RMSNorm, LayerNorm, BatchNorm1D/2D/3D
- Conv1D/2D/3D, ConvTransposed, DepthwiseConv, SeparableConv
- 18 activation modules (ReLU through Entmax)
- Dropout, WeightNorm, SpectralNorm

### Transformer (`model.py` — ~2,500 lines)
- **Attention**: MHA, GQA, MQA, Flash Attention, Sliding Window, Linear, Cosformer, AFT, Ring, Infini
- **Architectures**: Decoder-only, Encoder-Decoder, BERT, XLNet, Transformer-XL, ELECTRA, Funnel, Hierarchical, DeBERTa
- **MoE**: Standard, Switch, GShard, GLaM, Mixtral, DeepSeekMoE, Expert Choice, Soft MoE, BASE, MoD, ACT, Universal
- **KV Cache**: Standard, Paged, Prefix, H2O, SnapKV, ScissorHands, INT8 Quantized, AttentionSink
- **Configurations**: Post-norm, Sandwich-norm, Parallel (PaLM), Logit softcapping

### Tokenizer (`tokenizer.py` — 578 lines)
- BPE, WordPiece, Unigram, Tiktoken-style
- BPE-Dropout, Subword Regularization
- Chat template, FIM (PSM/SPM)

### Positional Encodings (`positional.py` — 225 lines)
- RoPE, YaRN, LongRoPE, NTK-aware
- ALiBi, Learned PE, Sinusoidal, T5 relative bias, FIRE

### Optimizers (`optim.py` — 1,200+ lines)
26 optimizers: SGD, Adam, AdamW, RAdam, NAdam, AdaFactor, AdaBelief, Adadelta, Adagrad, RMSProp, Lion, Muon, LAMB, LARS, SAM, Sophia, SOAP, ScheduleFreeAdam, Prodigy, GaLore, Flora, OneBitAdam, PowerSGD, Adan, SGD with Nesterov

10 LR Schedulers: WarmupCosine, CosineAnnealingWarmRestarts, CyclicLR, ReduceLROnPlateau, OneCycleLR, PolynomialLR, ExponentialLR, ConstantLR, LinearWarmup, ChainedScheduler

## Alternative Sequence Models (`models/`)

| Model | File | Description |
|-------|------|-------------|
| Mamba | `mamba.py` | Structured SSM, selective state space |
| RWKV | `rwkv.py` | Linear attention, O(1) inference |
| xLSTM | `xlstm.py` | sLSTM + mLSTM with parallel prefix scan |

## Training (`training/`)

- **Trainer**: gradient accumulation, grad clipping, EMA, loss spike detection, NaN recovery
- **Data pipeline**: TextDataset, DataLoader, MinHash dedup, SimHash, HTML cleaning
- **Distillation**: KD loss, Medusa heads, Speculative decoding
- **Quantization**: Dynamic/Static INT8, INT4, GPTQ, AWQ, SmoothQuant, LLM.int8, QuaRot, EXL2, QAT, SpQR, AQLM

## PEFT (`peft/`)

LoRA, QLoRA, AdaLoRA, DoRA, VeRA, LoftQ, Prefix Tuning, Adapter Layers, IA3, Prompt Tuning, LoRA Manager

## Alignment (`alignment/`)

RewardModel, PPO, DPO, SimPO, ORPO, GRPO, KTO, RLAIF, Constitutional AI, SPIN, ILQL, RejectionSampling, SelfPlay

## Safety (`safety/`)

PII Detection, Jailbreak Detection, Output Filtering, Rate Limiting, DP-SGD, Secure Aggregation, Adversarial Detection, Watermark Verification

## AI Capabilities (`ai/`)

| Module | Classes |
|--------|---------|
| `reasoning.py` | ChainOfThought, SelfConsistency, TreeOfThought, ReACT, PAL, SocraticMethod |
| `agent.py` | ToolRegistry, ReACTLoop, ReflexionAgent, AutoGPT, BabyAGI, MultiAgentDebate |
| `memory.py` | Episodic, Semantic, Working, Procedural, Long-term, DNC, MAT |
| `retrieval.py` | BM25, Dense, Hybrid, ColBERT, HyDE, RAPTOR, SelfRAG, RRF |
| `symbolic.py` | KnowledgeGraph, RDF, SPARQL, FOL, DPLL SAT |
| `planning.py` | A*, MCTS, IDA*, STRIPS |
| `logic.py` | Propositional, FOL, Modal, Temporal, Fuzzy, Probabilistic |
| `world.py` | Physics, CausalDAG, RigidBody, Fluid, Economic, Social, BDI |
| `scientific.py` | SymbolicRegression, SINDy, CausalDiscovery, PINN, BayesianOpt |
| `probabilistic.py` | NaiveBayes, HMM, KalmanFilter, VAE, DiffusionModel, CRF |
| `theoretical.py` | ScalingLaws, Chinchilla, EmergentAbilities, Grokking, LotteryTicket, FlatMinima |

## Evaluation (`eval/`)

Benchmarks: MMLU, HumanEval, GSM8K, HellaSwag, TruthfulQA, BoolQ, PIQA, CommonsenseQA, TriviaQA, MATH, MT-Bench, BERTScore, BLEU, ROUGE, Perplexity, ECE, Pass@k

Interpretability: AttentionViz, IntegratedGradients, SHAP, ProbingClassifiers, CausalTracing, LogitLens, AttentionRollout

Profiler: NAS, MetaLearning, ContinualLearning, AutoML

## Infrastructure

- **Server**: HTTP/1.1, SSE streaming, continuous batching, circuit breaker, metrics
- **CLI**: train, infer, tokenize, quantize, serve, finetune, evaluate
- **Distributed**: Data Parallel, Ring-AllReduce, Tensor/Sequence/Pipeline Parallel, ZeRO 1/2/3, FSDP, AsyncSGD, FedAvg, Gossip
- **Graph Compiler**: Graph IR, constant folding, DCE, operator fusion, JIT
- **Tensor VM**: 15-opcode instruction set, memory manager, profiler
- **Knowledge FS**: VFS, HNSW index, transaction log, garbage collector
- **Custom Language**: Lexer/parser/codegen for model definitions
- **Hardware Detection**: Linux/macOS/Windows auto-config

## API (OpenAI Compatible)

```bash
POST /v1/chat/completions    # Chat completions with streaming
POST /v1/completions         # Text completions
POST /v1/embeddings          # Embeddings
GET  /v1/models              # List models
GET  /health                 # Health check
```

## CLI Commands

```bash
qythera train --data corpus.txt --config config.json --steps 10000
qythera infer --model checkpoint.npz --prompt "Hello"
qythera serve --port 8080
qythera evaluate --benchmarks mmlu gsm8k
qythera tokenize --algorithm bpe --vocab-size 32000
qythera quantize --model checkpoint.npz --bits 4
qythera merge --models model_a.npz model_b.npz --method slerp
```

## Default Model

```python
TransformerConfig(
    vocab_size=32000, embed_dim=256, num_layers=6,
    num_heads=8, num_kv_heads=4, ffn_dim=768,
    max_seq_len=2048
)
# ~21M parameters
```

## Project Structure

```
src/qythera/
├── backend.py          # Dual backend (pure Python + NumPy)
├── tensor.py           # Autodiff engine (2,327 lines)
├── nn.py               # Neural network modules (1,557 lines)
├── optim.py            # 26 optimizers + 10 schedulers
├── positional.py       # RoPE, YaRN, ALiBi, etc.
├── model.py            # Transformer + 30+ attention/MoE variants
├── tokenizer.py        # BPE, WordPiece, Unigram, Tiktoken
├── sampler.py          # 15+ sampling strategies
├── models/             # Mamba, RWKV, xLSTM
├── training/           # trainer, data, distill, quantize
├── peft/               # LoRA, QLoRA, DoRA, VeRA, IA3
├── alignment/          # PPO, DPO, SimPO, ORPO, GRPO, KTO, RLHF
├── inference/          # server, cli, hardware
├── ai/                 # 12 modules: reasoning, memory, retrieval, etc.
├── safety/             # PII, jailbreak, DP-SGD, adversarial
├── eval/               # benchmarks, interpret, profiler
├── graph/              # GCN, GAT, stats
├── multimodal/         # ViT, CLIP, audio
├── systems/            # merge, distributed, VM, compiler, DSL, knowledge_fs
└── web/                # ui.html, sw.js, manifest.json
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License
