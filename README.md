# Qythera

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-00C853)
![NumPy](https://img.shields.io/badge/NumPy-1.22%2B-4DABCF?logo=numpy&logoColor=white)

Production Superintelligence Platform. Pure Python + NumPy. No external AI APIs.

## Prerequisites

- Python 3.9 or higher
- NumPy 1.22 or higher

## Installation

```bash
git clone https://github.com/cpner/qythera.git
cd qythera
pip install .
```

Or install directly from a repository:

```bash
pip install git+https://github.com/cpner/qythera.git
```

## Quick Start

```bash
python3 -m qythera.inference.cli serve --port 8080
```

Open http://localhost:8080 in any browser.

## Cross-Platform

### Linux / macOS

```bash
python3 -m qythera.inference.cli serve --port 8080
```

### Windows

```powershell
python -m qythera.inference.cli serve --port 8080
```

### Android (Termux)

```bash
pkg install python numpy
pip install .
python3 -m qythera.inference.cli serve --port 8080
```

## Architecture

```
tokens -> embeddings -> N transformer layers -> logits -> sampling

Each layer:
  RMSNorm -> Multi-Head Attention + RoPE -> RMSNorm -> SwiGLU FFN -> Residual
```

```
┌─────────────────────────────────────────────────────────┐
│                    Qythera Stack                        │
├─────────────────────────────────────────────────────────┤
│  Web UI (PWA)  │  CLI  │  OpenAI-Compatible HTTP API   │
├─────────────────────────────────────────────────────────┤
│  Inference: server.py, cli.py, hardware.py              │
├─────────────────────────────────────────────────────────┤
│  Model: Transformer (MHA/GQA/MQA) + SwiGLU + MoE      │
├─────────────────────────────────────────────────────────┤
│  Engine: tensor.py (autodiff), nn.py, optim.py         │
├─────────────────────────────────────────────────────────┤
│  NumPy Backend (zero external AI dependencies)          │
└─────────────────────────────────────────────────────────┘
```

## Modules (41 modules, ~25K lines)

### Core Engine
- `tensor.py` — Custom autodiff tensor engine: 50+ ops, broadcasting, einsum, SVD, QR, Cholesky, LU, eig
- `nn.py` — Neural network modules: Linear, Embedding, RMSNorm, LayerNorm, BatchNorm, Conv1D/2D/3D, 15 activations
- `optim.py` — 18 optimizers: Adam, AdamW, Lion, Muon, SOAP, Sophia, SAM + 10 LR schedulers
- `positional.py` — RoPE, YaRN, NTK-aware, LongRoPE, ALiBi, Sinusoidal, T5 bias, FIRE

### Model
- `model.py` — Full transformer: MHA/GQA/MQA, SwiGLU FFN, MoE, KV cache, generate()
- `tokenizer.py` — BPE, WordPiece, Unigram, Tiktoken-style, chat template, FIM
- `sampler.py` — Greedy, TopK, TopP, MinP, Beam Search, Self-Consistency, Watermarking

### Training & Optimization
- `training/data.py` — MMapDataset, StreamingDataset, DataLoader, MinHash dedup, SimHash
- `training/trainer.py` — Training loop with gradient accumulation, mixed precision, checkpointing
- `training/quantize.py` — INT8/INT4/INT2, GPTQ, AWQ, SmoothQuant, LLM.int8
- `training/distill.py` — Knowledge distillation, Medusa heads, MultiToken prediction, Speculative decoding

### Parameter-Efficient Fine-Tuning
- `peft/lora.py` — LoRA, QLoRA, AdaLoRA, DoRA, VeRA, PrefixTuning, IA3, Adapter, PromptTuning

### Alignment
- `alignment/rlhf.py` — PPO, DPO, SimPO, ORPO, GRPO, KTO, RLAIF, SPIN, ILQL

### Inference
- `inference/server.py` — HTTP server, OpenAI API, SSE streaming, batch scheduling
- `inference/cli.py` — CLI: train, infer, tokenize, quantize, serve, evaluate
- `inference/hardware.py` — CPU/RAM/SIMD detection, auto-config

### AI Capabilities
- `ai/agent.py` — Tool calling, ReACT loop, Reflexion, Multi-Agent Debate
- `ai/reasoning.py` — Chain-of-Thought, Self-Consistency, Tree-of-Thought, Constitutional AI
- `ai/memory.py` — Episodic, Semantic, Working, Procedural, Long-term, Hopfield
- `ai/retrieval.py` — BM25, Dense retrieval, Hybrid, ColBERT, InvertedIndex
- `ai/symbolic.py` — Knowledge Graph, SAT solver, First-Order Logic, Symbolic Regression
- `ai/planning.py` — A*, MCTS, IDA*, STRIPS planning
- `ai/logic.py` — Propositional, First-Order, Modal, Temporal, Fuzzy logic
- `ai/world.py` — Physics simulation, Causal DAG, BDI agents
- `ai/knowledge.py` — Knowledge base management and reasoning

### Multimodal & Safety
- `multimodal/vision.py` — ViT, CLIP, Audio STFT+mel, Simple Diffusion
- `safety/filters.py` — Jailbreak detection, PII, Output filtering, Rate limiting, Watermark

### Evaluation & Interpretability
- `eval/benchmarks.py` — MMLU, HumanEval, GSM8K, BLEU, ROUGE, ECE, Arena ELO
- `eval/interpret.py` — Attention viz, Integrated Gradients, Logit Lens, Causal Tracing
- `eval/profiler.py` — cProfile, memory profiler, AutoML, model analyzer

### Graph & Statistics
- `graph/graph_nn.py` — GCN, GraphSAGE, GAT, TransE
- `graph/stats.py` — Entropy, KL/JS divergence, MI, Bayesian, EM, hypothesis tests

### Systems
- `systems/distributed.py` — Data/Tensor Parallel, ZeRO 1/2/3, gradient compression
- `systems/vm.py` — 15-opcode Tensor VM, memory manager, profiler
- `systems/compiler.py` — Graph IR, constant folding, DCE, operator fusion, JIT
- `systems/language.py` — DSL lexer/parser/codegen for model definitions
- `systems/knowledge_fs.py` — Virtual filesystem, HNSW index, URI scheme
- `systems/merge.py` — Task Arithmetic, TIES, DARE, SLERP, Model Soup

### Alternative Architectures
- `models/mamba.py` — State-space Mamba model
- `models/rwkv.py` — RWKV linear-attention model
- `models/xlstm.py` — xLSTM model

## API Reference

Qythera exposes an OpenAI-compatible HTTP API.

### Chat Completions

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qythera-default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help you?"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12}
}
```

### Text Completions

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qythera-default", "prompt": "The capital of France is"}'
```

### Embeddings

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "qythera-default"}'
```

### Health Check

```bash
curl http://localhost:8080/health
```

## CLI Reference

```bash
# Serve the model (HTTP + Web UI)
python3 -m qythera.inference.cli serve --port 8080

# Run inference
python3 -m qythera.inference.cli infer --prompt "Hello"

# Tokenize text
python3 -m qythera.inference.cli tokenize --text "Hello world"

# Quantize a model
python3 -m qythera.inference.cli quantize --bits 4

# Run benchmarks
python3 -m qythera.inference.cli evaluate --benchmark mmlu
```

## Default Model

~21M params: vocab=32000, embed=256, layers=6, heads=8, kv_heads=4, ffn=768, ctx=2048

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/cpner/qythera.git
cd qythera
pip install -e .
PYTHONPATH=src python3 -m qythera.inference.cli serve --port 8080
```

## License

MIT License. See [LICENSE](LICENSE) for details.
