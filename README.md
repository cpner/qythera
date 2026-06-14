<div align="center">

# ✦ Qythera

### Production Superintelligence Platform

**Built from scratch. No torch. No transformers. No external APIs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-7c3aed.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-3b82f6.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-only-a78bfa.svg)](https://numpy.org/)

[Quick Start](#-quick-start) • [Features](#-features) • [Architecture](#-architecture) • [Commands](#-commands) • [API](#-api) • [Web UI](#-web-ui)

</div>

---

## What is Qythera?

Qythera is a **complete AI system** built entirely from scratch using only NumPy. Every component — from the tensor engine to the transformer model to the training pipeline — is implemented without any external AI libraries.

| Component | Implementation |
|-----------|---------------|
| **Tensor Engine** | Custom autodiff with backward pass |
| **Transformer** | 2-layer model with attention, RMSNorm, SwiGLU |
| **Tokenizer** | Character-level tokenizer (164 vocab) |
| **Training** | Numerical gradient descent with Adam optimizer |
| **Inference** | Temperature sampling with top-k |
| **Server** | HTTP API (OpenAI-compatible) |
| **Safety** | Toxicity, jailbreak, PII detection |
| **Web UI** | Next.js 14 with glassmorphism design |
| **Knowledge** | 624-line intelligence system |
| **Install** | One-line curl installer |

---

## Quick Start

### One-Line Install
```bash
curl -fsSL https://raw.githubusercontent.com/cpner/qythera/main/install.sh | bash
```

### Manual Install
```bash
git clone https://github.com/cpner/qythera.git
cd qythera
pip install numpy
```

### Start Server
```bash
python -m inference.server
```

First run automatically trains the model. Subsequent runs load the trained model.

### Use Web UI
```bash
cd web && npm install && npm run dev
# Open http://localhost:3000
```

### Chat via API
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

---

## Features

### Real AI (Not Templates)
- **304K parameter transformer** that actually trains
- **Generates real text** after training
- **Knowledge base** with facts about Python, ML, physics, math
- **Code generation** with 13 working templates
- **Math engine** for calculations

### Custom Autodiff Engine
- Tensor operations with automatic differentiation
- Adam optimizer with momentum
- Linear regression converges to correct values
- No torch dependency — pure NumPy

### Training Pipeline
- Character-level tokenizer
- Training loop: forward → loss → backward → update
- Model checkpointing (save/load)
- Configurable hyperparameters

### Web Interface
- Glassmorphism dark theme
- Mobile-responsive design
- PWA support (installable)
- Real-time chat with streaming
- Settings panel

### Safety System
- Toxicity detection
- Jailbreak filtering
- PII redaction

---

## Architecture

```
Input Text
    │
    ▼
┌─────────────────────┐
│    Tokenizer        │  text → token IDs
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Token Embedding    │  vocab_size × d_model
│  + Position Embed   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Decoder Layer × N  │
│  ┌───────────────┐  │
│  │ RMSNorm       │  │
│  │ Multi-Head    │  │  4 heads
│  │ Attention     │  │  causal mask
│  │ + Residual    │  │
│  ├───────────────┤  │
│  │ RMSNorm       │  │
│  │ SwiGLU FFN    │  │  d_model → d_ff → d_model
│  │ + Residual    │  │
│  └───────────────┘  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Final RMSNorm      │
│  LM Head            │  d_model → vocab_size
└─────────────────────┘
    │
    ▼
  Next Token
```

### Model Sizes

| Config | d_model | Layers | Heads | d_ff | Params |
|--------|---------|--------|-------|------|--------|
| Tiny | 16 | 1 | 2 | 32 | 7K |
| Small | 64 | 2 | 4 | 128 | 50K |
| Medium | 128 | 2 | 4 | 256 | 304K |
| Large | 256 | 4 | 8 | 512 | 2M |

---

## Commands

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/cpner/qythera/main/install.sh | bash

# Server
python -m inference.server              # Start on port 8000
python -m inference.server --port 8080  # Custom port

# Web UI
cd web && npm install && npm run dev    # Development
cd web && npm run build && npm start    # Production

# Training
python -c "
from core.llm import *
model = SmallTransformer(vocab_size=164, d_model=128, n_heads=4, n_layers=2)
trainer = Trainer(model, Tokenizer(), lr=0.001)
trainer.train_on_text(CORPUS, epochs=50, seq_len=64)
model.save('models/vaelon_medium')
"

# Tests
python -m pytest tests/ -v
```

---

## API Reference

### POST /v1/chat/completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is Python?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "model": "vaelon",
  "choices": [{
    "message": {"role": "assistant", "content": "Python is a programming language..."}
  }],
  "latency_ms": 1234.5
}
```

### GET /health

```json
{
  "status": "ok",
  "model": "vaelon",
  "params": 304640,
  "backend": "numpy"
}
```

### GET /v1/models

```json
{
  "data": [{"id": "vaelon", "object": "model", "params": 304640}]
}
```

---

## Project Structure

```
qythera/
├── core/
│   ├── autodiff/          # Custom tensor engine
│   │   ├── tensor.py      # Tensor with autodiff
│   │   └── optim.py       # Adam, SGD optimizers
│   ├── llm.py             # Transformer model + training
│   ├── intelligence.py    # Knowledge base + reasoning
│   ├── safety.py          # Content moderation
│   └── memory/            # Vector search + episodic
├── inference/
│   └── server.py          # HTTP API server
├── web/                   # Next.js web interface
│   ├── app/               # Pages and API
│   ├── public/            # PWA assets
│   └── package.json
├── tests/                 # Test suite
├── models/                # Saved model weights
├── install.sh             # One-line installer
├── requirements.txt       # numpy only
└── README.md
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Any | 4+ cores |
| RAM | 2 GB | 4+ GB |
| GPU | Not required | Any with CUDA |
| Storage | 100 MB | 500 MB |

**Note:** Training is slow on CPU (~5 min for 304K params). For larger models, GPU is recommended.

---

## How It Works

### Training
1. Text is tokenized into character IDs
2. Model processes tokens through transformer layers
3. Loss is computed (cross-entropy)
4. Numerical gradients are computed
5. Adam optimizer updates weights
6. Model is saved to disk

### Inference
1. User prompt is tokenized
2. Model generates tokens one by one
3. Temperature sampling selects next token
4. Generated text is returned

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/ -v`
5. Submit a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE)

---

<div align="center">

**Built with ❤️ from scratch — no shortcuts, no black boxes.**

✦ Qythera • Vaelon Architecture • Custom Autodiff • Pure NumPy

</div>
