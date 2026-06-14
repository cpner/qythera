# Qythera

Production Superintelligence Platform with Vaelon Model Architecture.

## Features
- Custom autodiff tensor engine (no torch.nn)
- Vaelon transformer with MoE, GQA, RoPE
- BPE tokenizer (from scratch)
- Training pipeline
- Inference server (OpenAI-compatible API)
- Memory system (vector index + episodic)
- Safety filters
- Web UI (mobile-responsive, glassmorphism)
- CLI

## Quick Start
```bash
git clone https://github.com/cpner/qythera.git
cd qythera
pip install -e .

# Start server
python -m inference.server

# Open web UI (on any device)
cd web && npm install && npm run dev

# Or use CLI
qythera chat
```

## How It Works
1. **Custom Autodiff Engine**: All tensor operations with automatic differentiation
2. **Vaelon Model**: Transformer with Mixture of Experts for efficient scaling
3. **BPE Tokenizer**: Trained from scratch on your data
4. **Inference Server**: Serves model via REST API
5. **Web UI**: Beautiful glassmorphism interface, works on phones
6. **Memory**: Vector search + conversation history
7. **Safety**: Toxicity, jailbreak, and PII detection

## Hardware
- CPU: Works but slow
- GPU: Recommended for training and inference
- Mobile: Web UI accessible via browser

## License
MIT
