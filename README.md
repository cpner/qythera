# Qythera

Production Superintelligence Platform powered by Vaelon model.

## Quick Start

```bash
git clone https://github.com/cpner/qythera.git
cd qythera
pip install -e .

# Start server (on GPU cloud instance)
python -m inference.server

# Open web UI (on any device)
cd web && npm install && npm run dev

# Or use CLI
qythera chat
```

## Features
- Vaelon MoE Transformer (7B-70B)
- Training pipeline (pretrain, SFT, DPO)
- Inference server with streaming
- Hybrid memory (vector + episodic)
- Safety filters (toxicity, jailbreak, PII)
- Mobile-responsive web UI
- Works on any device via browser
