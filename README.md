# Qythera

<div align="center">

![Qythera](web/public/logo.svg)

**Production Superintelligence Platform**

[![CI](https://github.com/cpner/qythera/actions/workflows/ci.yml/badge.svg)](https://github.com/cpner/qythera/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[Website](https://github.com/cpner/qythera) | [Documentation](docs/) | [Training Guide](docs/TRAINING.md)

</div>

---

## Quick Start

```bash
# One-command install
curl -fsSL https://raw.githubusercontent.com/cpner/qythera/main/scripts/install.sh | bash

# Or install manually
git clone https://github.com/cpner/qythera.git
cd qythera
pip install -e .

# Start chatting
qythera chat

# Launch web UI
qythera web

# Start inference server
qythera serve
```

## Features

- **Vaelon Architecture**: Custom transformer with Mixture of Experts (7B-70B parameters)
- **Training Pipeline**: Pre-training, SFT, RLHF (DPO/PPO), Constitutional AI
- **Inference Server**: vLLM-compatible API with quantization (AWQ, GPTQ, bitsandbytes)
- **Memory System**: Hybrid retrieval (FAISS + episodic + semantic)
- **Agent Framework**: ReAct reasoning, Plan-Execute, Reflexion with tool use
- **Safety Filters**: Toxicity, jailbreak, and PII detection
- **Web UI**: Beautiful glassmorphism interface with streaming
- **CLI**: Full-featured terminal interface
- **Deployment**: Docker, Kubernetes, Terraform

## Architecture

```
Qythera/
├── vaelon/          # Model architecture (MoE, GQA, RoPE)
├── training/        # Pre-training, SFT, RLHF
├── inference/       # Serving with vLLM
├── memory/          # Vector store, episodic, semantic
├── multimodal/      # Vision, audio, video encoders
├── agent/           # ReAct, Plan-Execute, tools
├── safety/          # Toxicity, jailbreak, PII
├── web/             # Next.js web interface
├── cli/             # Command line interface
└── infra/           # Docker, K8s, Terraform
```

## Model Sizes

| Model | Parameters | Experts | Hidden | Layers |
|-------|-----------|---------|--------|--------|
| Vaelon-7B | 7B | 8 | 4096 | 32 |
| Vaelon-13B | 13B | 8 | 5120 | 40 |
| Vaelon-70B | 70B | 64 | 8192 | 80 |

## Hardware Requirements

| Model | Minimum GPU | Recommended |
|-------|------------|-------------|
| 7B | 1x RTX 4090 (24GB) | 1x A100 40GB |
| 13B | 2x RTX 4090 | 2x A100 80GB |
| 70B | 4x A100 80GB | 8x A100 80GB |

## Training

```bash
# Pre-train from scratch
make train

# Fine-tune with LoRA
python training/finetune/sft_trainer.py --config training/configs/7b_lora.yaml

# DPO alignment
python training/rlhf/dpo_trainer.py
```

## Web Interface

```bash
cd web && npm install && npm run dev
# Open http://localhost:3000
```

## API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## Testing

```bash
pytest tests/ -v
```

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md)

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@software{qythera2024,
  title={Qythera: Production Superintelligence},
  author={Qythera Team},
  year={2024},
  url={https://github.com/cpner/qythera}
}
```
