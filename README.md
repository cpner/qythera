# Qythera

**Production Superintelligence Platform**

Built from scratch. No torch. No transformers. No external APIs.

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/cpner/qythera/main/install.sh | bash
python -m core.inference.server
```

## Features

- Custom autodiff tensor engine
- Vaelon transformer (supports 50K to 150M params)
- BPE tokenizer
- Training pipeline
- Knowledge base with real facts
- Code generation (13 templates)
- Math engine
- Safety filters
- Web UI (glassmorphism, mobile-responsive, PWA)
- CLI interface

## Commands

```bash
python -m core.inference.server    # Start server
cd web && npm install && npm run dev  # Web UI
python cli/main.py chat           # CLI chat
python cli/main.py info           # System info
```

## API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

## Architecture

- **Tensor Engine**: Custom autodiff with numpy
- **Transformer**: GQA, RoPE, RMSNorm, SwiGLU, KV Cache
- **Training**: Numerical gradients with Adam optimizer
- **Tokenizer**: BPE (trained from scratch)
- **Knowledge**: 50+ facts, 13 code templates
- **Safety**: Toxicity, jailbreak, PII detection
- **Web UI**: Next.js 14, glassmorphism, PWA
- **CLI**: Interactive chat, server control

## License

MIT
